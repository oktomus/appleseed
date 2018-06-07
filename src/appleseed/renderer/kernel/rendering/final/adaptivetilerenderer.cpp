
//
// This source file is part of appleseed.
// Visit https://appleseedhq.net/ for additional information and resources.
//
// This software is released under the MIT license.
//
// Copyright (c) 2018 Kevin Masson, The appleseedhq Organization
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

// Interface header.
#include "adaptivetilerenderer.h"

// appleseed.renderer headers.
#include "renderer/global/globallogger.h"
#include "renderer/global/globaltypes.h"
#include "renderer/kernel/aov/aovaccumulator.h"
#include "renderer/kernel/aov/imagestack.h"
#include "renderer/kernel/aov/tilestack.h"
#include "renderer/kernel/rendering/isamplerenderer.h"
#include "renderer/kernel/rendering/ishadingresultframebufferfactory.h"
#include "renderer/kernel/rendering/pixelcontext.h"
#include "renderer/kernel/rendering/pixelrendererbase.h"
#include "renderer/kernel/rendering/shadingresultframebuffer.h"
#include "renderer/kernel/rendering/final/variationtracker.h"
#include "renderer/kernel/shading/shadingresult.h"
#include "renderer/modeling/frame/frame.h"
#include "renderer/utility/settingsparsing.h"

// appleseed.foundation headers.
#include "foundation/image/canvasproperties.h"
#include "foundation/image/image.h"
#include "foundation/image/tile.h"
#include "foundation/math/aabb.h"
#include "foundation/math/filter.h"
#include "foundation/math/fastmath.h"
#include "foundation/math/hash.h"
#include "foundation/math/ordering.h"
#include "foundation/math/population.h"
#include "foundation/math/scalar.h"
#include "foundation/math/vector.h"
#include "foundation/platform/arch.h"
#include "foundation/platform/debugger.h"
#include "foundation/platform/types.h"
#include "foundation/utility/autoreleaseptr.h"
#include "foundation/utility/iostreamop.h"
#include "foundation/utility/job.h"
#include "foundation/utility/statistics.h"
#include "foundation/utility/string.h"

// Standard headers.
#include <cassert>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace foundation;
using namespace std;

namespace renderer
{

    //
    // Reference:
    //
    // A Hierarchical Automatic Stopping Condition for Monte Carlo Global Illumination
    // https://jo.dreggn.org/home/2009_stopping.pdf
    //

namespace
{

    const size_t BlockMinAllowedSize = 4;
    const size_t BlockSplittingThreshold = 8;

    //
    // Adaptive tile renderer.
    //

#ifndef NDEBUG

    // Define this symbol to break execution into the debugger
    // when a specific pixel is about to be rendered.
    // #define DEBUG_BREAK_AT_PIXEL Vector2i(0, 0)

#endif

    class AdaptiveTileRenderer
      : public ITileRenderer, public PixelRendererBase
    {
      public:
        AdaptiveTileRenderer(
            const Frame&                        frame,
            ISampleRendererFactory*             sample_renderer_factory,
            IShadingResultFrameBufferFactory*   framebuffer_factory,
            const ParamArray&                   params,
            const size_t                        thread_index)
          : PixelRendererBase(frame, thread_index, params)
          , m_sample_renderer(sample_renderer_factory->create(thread_index))
          , m_aov_accumulators(frame)
          , m_framebuffer_factory(framebuffer_factory)
          , m_params(params)
          , m_total_samples(0)
          , m_total_saved_samples(0)
          , m_max_samples(0)
          , m_total_pixel_converged(0)
          , m_total_pixel(0)
        {
            compute_tile_margins(frame, thread_index == 0);

            if (are_diagnostics_enabled())
            {
                m_variation_aov_index = frame.create_extra_aov_image("variation");
                m_samples_aov_index = frame.create_extra_aov_image("samples");
                m_block_coverage_aov_index = frame.create_extra_aov_image("block-coverage");

                if ((thread_index == 0) &&
                    (m_variation_aov_index == ~size_t(0)
                    || m_samples_aov_index == ~size_t(0)
                    || m_block_coverage_aov_index == ~size_t(0)))
                {
                    RENDERER_LOG_WARNING(
                        "could not create some of the diagnostic aovs, maximum number of aovs (" FMT_SIZE_T ") reached.",
                        MaxAOVCount);
                }
            }
        }

        void release() override
        {
            delete this;
        }

        void print_settings() const override
        {
            RENDERER_LOG_INFO(
                "adaptive tile renderer settings:\n"
                "  min samples                   %s\n"
                "  max samples                   %s\n"
                "  error threshold               %f\n"
                "  splitting threshold           %f\n"
                "  diagnostics                   %s",
                pretty_uint(m_params.m_min_samples).c_str(),
                pretty_uint(m_params.m_max_samples).c_str(),
                m_params.m_error_threshold,
                m_params.m_splitting_threshold,
                are_diagnostics_enabled() ? "on" : "off");

            m_sample_renderer->print_settings();
        }

        void render_tile(
            const Frame&    frame,
            const size_t    tile_x,
            const size_t    tile_y,
            const size_t    pass_hash,
            IAbortSwitch&   abort_switch) override
        {
            const size_t frame_width = frame.image().properties().m_canvas_width;
            const size_t aov_count = frame.aov_images().size();

            // Retrieve frame properties.
            const CanvasProperties& frame_properties = frame.image().properties();
            assert(tile_x < frame_properties.m_tile_count_x);
            assert(tile_y < frame_properties.m_tile_count_y);

            // Retrieve tile properties.
            Tile& tile = frame.image().tile(tile_x, tile_y);
            TileStack aov_tiles = frame.aov_images().tiles(tile_x, tile_y);
            const int tile_origin_x = static_cast<int>(frame_properties.m_tile_width * tile_x);
            const int tile_origin_y = static_cast<int>(frame_properties.m_tile_height * tile_y);
            const size_t tile_index = tile_x + tile_y * (frame_width / static_cast<float>(frame_properties.m_tile_width));

            // Compute the image space bounding box of the pixels to render.
            AABB2i tile_bbox;
            tile_bbox.min.x = tile_origin_x;
            tile_bbox.min.y = tile_origin_y;
            tile_bbox.max.x = tile_origin_x + static_cast<int>(tile.get_width()) - 1;
            tile_bbox.max.y = tile_origin_y + static_cast<int>(tile.get_height()) - 1;
            tile_bbox = AABB2i::intersect(tile_bbox, AABB2i(frame.get_crop_window()));
            if (!tile_bbox.is_valid())
                return;

            // Transform the bounding box to local (tile) space.
            tile_bbox.min.x -= tile_origin_x;
            tile_bbox.min.y -= tile_origin_y;
            tile_bbox.max.x -= tile_origin_x;
            tile_bbox.max.y -= tile_origin_y;

            // Pad the bounding box with tile margins.
            AABB2i padded_tile_bbox;
            padded_tile_bbox.min.x = tile_bbox.min.x - m_margin_width;
            padded_tile_bbox.min.y = tile_bbox.min.y - m_margin_height;
            padded_tile_bbox.max.x = tile_bbox.max.x + m_margin_width;
            padded_tile_bbox.max.y = tile_bbox.max.y + m_margin_height;

            // Inform the pixel renderer that we are about to render a tile.
            on_tile_begin(frame, tile, aov_tiles);

            // Inform the AOV accumulators that we are about to render a tile.
            m_aov_accumulators.on_tile_begin(
                frame,
                tile_x,
                tile_y,
                m_params.m_max_samples);

            // Create the framebuffer into which we will accumulate the samples.
            ShadingResultFrameBuffer* framebuffer =
                m_framebuffer_factory->create(
                    frame,
                    tile_x,
                    tile_y,
                    tile_bbox);
            assert(framebuffer);

            // Create the buffer into which we will accumulate every second samples.
            ShadingResultFrameBuffer* second_framebuffer =
                m_framebuffer_factory->create(
                    frame,
                    tile_x,
                    tile_y,
                    tile_bbox);
            assert(second_framebuffer);

            const size_t pixel_count = framebuffer->get_width() * framebuffer->get_height();

            // Pixel rendering.
            size_t samples_so_far = 0;
            size_t batch_number = 0;

            vector<PixelBlock>  pixel_blocks;
            vector<PixelBlock>  finished_blocks;
            pixel_blocks.emplace_back(padded_tile_bbox);

            while (true)
            {
                const int remaining_samples = m_params.m_max_samples - samples_so_far;

                if (remaining_samples < 1 || abort_switch.is_aborted() || pixel_blocks.size() < 1)
                {
                    if (pixel_blocks.size() > 0)
                        finished_blocks.insert(finished_blocks.end(), pixel_blocks.begin(), pixel_blocks.end());

                    break;
                }

                // Each batch contains 'min' samples.
                const size_t batch_size = min(static_cast<int>(m_params.m_min_samples), remaining_samples);
                batch_number++;

                RENDERER_LOG_DEBUG("T=%s; BN=%s; SSF=%s; RS=%i; BS=%s; RBA=%s; CBA=%s",
                    pretty_uint(tile_index).c_str(),
                    pretty_uint(batch_number).c_str(),
                    pretty_uint(samples_so_far).c_str(),
                    remaining_samples,
                    pretty_uint(batch_size).c_str(),
                    pretty_uint(pixel_blocks.size()).c_str(),
                    pretty_uint(finished_blocks.size()).c_str());

                vector<PixelBlock>  tmp_blocks;

                // For each block.
                for (auto& pb : pixel_blocks)
                {
                    // Draw samples.
                    sample_pixel_block(
                        pb,
                        abort_switch,
                        batch_size,
                        framebuffer,
                        second_framebuffer,
                        tile_bbox,
                        tile_origin_x,
                        tile_origin_y,
                        frame,
                        frame_width,
                        pass_hash,
                        aov_count);

                    if (abort_switch.is_aborted())
                    {
                        finished_blocks.push_back(pb);
                        continue;
                    }

                    if (batch_number <= 3)
                    {
                        tmp_blocks.push_back(pb);
                        continue;
                    }

                    // No point to continue if this is the end
                    if (remaining_samples - batch_size < 1)
                    {
                        finished_blocks.push_back(pb);
                        continue;
                    }

                    // Evaluate error metric.
                    evaluate_pixel_block_error(
                        pb,
                        framebuffer,
                        second_framebuffer);

                    // Take a decision.
                    if (pb.m_converged)
                    {
                        finished_blocks.push_back(pb);
                    }
                    else if (pb.m_splitting_point != 0)
                    {
                        split_pixel_block(pb, pixel_blocks, tmp_blocks);
                    }
                    else
                    {
                        tmp_blocks.push_back(pb);
                    }
                }

                pixel_blocks = tmp_blocks;
                samples_so_far += batch_size;
            }

            // Pixels end.
            // For each block.
            size_t pb_index = 1;
            for (auto& pb : finished_blocks)
            {
                const AABB2i& pb_aabb = pb.m_surface;
                const float pb_pixel_count = (pb_aabb.max.x - pb_aabb.min.x + 1) * (pb_aabb.max.y - pb_aabb.min.y + 1);

                // Update statistics.
                m_total_samples += pb.m_spp * pb_pixel_count;
                m_spp.insert(pb.m_spp);
                m_total_saved_samples += (m_params.m_max_samples - pb.m_spp) * pb_pixel_count;
                m_saved_samples.insert(m_params.m_max_samples - pb.m_spp);
                m_block_error.insert(pb.m_block_error);
                m_total_pixel += pb_pixel_count;
                if (pb.m_converged)
                    m_total_pixel_converged += pb_pixel_count;

                if (!(pb.m_spp > 0))
                {
                    RENDERER_LOG_WARNING(
                        "Block missing some information on tile %s\n"
                        "  min x:           %s\n"
                        "  max x:           %s\n"
                        "  min y:           %s\n"
                        "  max x:           %s\n"
                        "  error:           %f\n"
                        "  samples/pixel:   %s",
                        pretty_uint(tile_index).c_str(),
                        pretty_uint(pb.m_surface.min.x).c_str(),
                        pretty_uint(pb.m_surface.max.x).c_str(),
                        pretty_uint(pb.m_surface.min.y).c_str(),
                        pretty_uint(pb.m_surface.max.y).c_str(),
                        pb.m_block_error,
                        pretty_uint(pb.m_spp).c_str());
                }

                for (int y = pb_aabb.min.y; y <= pb_aabb.max.y; ++y)
                {
                    for (int x = pb_aabb.min.x; x <= pb_aabb.max.x; ++x)
                    {
                        // Retrieve the coordinates of the pixel in the padded tile.
                        const Vector2i pt(x, y);

                        if (!padded_tile_bbox.contains(pt)) continue;

                        // Store diagnostics values in the diagnostics tile.
                        if (are_diagnostics_enabled() && tile_bbox.contains(pt))
                        {
                            Color<float, 5> values;

                            values[0] = pb.m_block_error;
                            values[1] =
                                m_params.m_min_samples == m_params.m_max_samples
                                ? 1.0f
                                : fit(
                                        static_cast<float>(pb.m_spp),
                                        static_cast<float>(m_params.m_min_samples),
                                        static_cast<float>(m_params.m_max_samples),
                                        0.0f, 1.0f);

                            Color3f pb_color = integer_to_color(pass_hash * tile_index + pb_index);
                            values[2] = pb_color[0];
                            values[3] = pb_color[1];
                            values[4] = pb_color[2];

                            m_diagnostics->set_pixel(pt.x, pt.y, values);
                        }
                    }
                }

                pb_index++;
            }

            // Update statistics.
            m_max_samples += pixel_count * m_params.m_max_samples;
            m_block_amount.insert(finished_blocks.size());

            // Develop the framebuffer to the tile.
            framebuffer->develop_to_tile(tile, aov_tiles);

            // Release the framebuffer.
            m_framebuffer_factory->destroy(framebuffer);

            // Inform the AOV accumulators that we are done rendering a tile.
            m_aov_accumulators.on_tile_end(frame, tile_x, tile_y);

            // Inform the pixel renderer that we are done rendering the tile.
            on_tile_end(frame, tile, aov_tiles);
        }

        void on_tile_begin(
            const Frame&            frame,
            Tile&                   tile,
            TileStack&              aov_tiles) override
        {
            PixelRendererBase::on_tile_begin(frame, tile, aov_tiles);

            if (are_diagnostics_enabled())
            {
                m_diagnostics.reset(new Tile(
                    tile.get_width(), tile.get_height(), 5, PixelFormatFloat));
            }
        }

        void on_tile_end(
            const Frame&            frame,
            Tile&                   tile,
            TileStack&              aov_tiles) override
        {
            PixelRendererBase::on_tile_end(frame, tile, aov_tiles);

            if (are_diagnostics_enabled())
            {
                const size_t width = tile.get_width();
                const size_t height = tile.get_height();

                for (size_t y = 0; y < height; ++y)
                {
                    for (size_t x = 0; x < width; ++x)
                    {
                        Color<float, 5> values;
                        m_diagnostics->get_pixel(x, y, values);

                        if (m_variation_aov_index != ~size_t(0))
                            aov_tiles.set_pixel(x, y, m_variation_aov_index, colorize_variation(values[0]));

                        if (m_samples_aov_index != ~size_t(0))
                            aov_tiles.set_pixel(x, y, m_samples_aov_index, colorize_samples(values[1]));

                        if (m_block_coverage_aov_index != ~size_t(0))
                            aov_tiles.set_pixel(x, y, m_block_coverage_aov_index, Color4f(values[2], values[3], values[4], 1.0f));
                    }
                }
            }
        }

        void render_pixel(
            const Frame&                frame,
            Tile&                       tile,
            TileStack&                  aov_tiles,
            const AABB2i&               tile_bbox,
            const size_t                pass_hash,
            const Vector2i&             pi,
            const Vector2i&             pt,
            AOVAccumulatorContainer&    aov_accumulators,
            ShadingResultFrameBuffer&   framebuffer) override
        {
        }

        StatisticsVector get_statistics() const override
        {
            Statistics stats;
            stats.insert("max theorical samples", m_max_samples);
            stats.insert("total samples", m_total_samples);
            stats.insert("samples/pixel", m_spp);
            stats.insert("total samples saved", m_total_saved_samples);
            stats.insert("saved samples/pixel", m_saved_samples);
            stats.insert_percent("saved samples rate", m_total_saved_samples, m_max_samples);
            stats.insert("block error", m_block_error, 4);
            stats.insert("blocks/tile", m_block_amount);
            stats.insert_percent("convergence rate", m_total_pixel_converged, m_total_pixel);

            StatisticsVector vec;
            vec.insert("adaptive tile renderer statistics", stats);
            vec.merge(m_sample_renderer->get_statistics());

            return vec;
        }

        size_t get_max_samples_per_pixel() const override
        {
            return m_params.m_max_samples;
        }

      protected:
        auto_release_ptr<ISampleRenderer>   m_sample_renderer;
        AOVAccumulatorContainer             m_aov_accumulators;
        IShadingResultFrameBufferFactory*   m_framebuffer_factory;
        int                                 m_margin_width;
        int                                 m_margin_height;

        void compute_tile_margins(const Frame& frame, const bool primary)
        {
            m_margin_width = truncate<int>(ceil(frame.get_filter().get_xradius() - 0.5f));
            m_margin_height = truncate<int>(ceil(frame.get_filter().get_yradius() - 0.5f));

            const CanvasProperties& properties = frame.image().properties();
            const size_t padded_tile_width = properties.m_tile_width + 2 * m_margin_width;
            const size_t padded_tile_height = properties.m_tile_height + 2 * m_margin_height;
            const size_t padded_pixel_count = padded_tile_width * padded_tile_height;
            const size_t pixel_count = properties.m_tile_width * properties.m_tile_height;
            const size_t overhead_pixel_count = padded_pixel_count - pixel_count;
            const double wasted_effort = static_cast<double>(overhead_pixel_count) / pixel_count * 100.0;
            const double MaxWastedEffort = 15.0;    // percents

            if (primary)
            {
                RENDERER_LOG(
                    wasted_effort > MaxWastedEffort ? LogMessage::Warning : LogMessage::Info,
                    "rendering effort wasted by tile borders: %s (tile dimensions: %s x %s, tile margins: %s x %s)",
                    pretty_percent(overhead_pixel_count, pixel_count).c_str(),
                    pretty_uint(properties.m_tile_width).c_str(),
                    pretty_uint(properties.m_tile_height).c_str(),
                    pretty_uint(2 * m_margin_width).c_str(),
                    pretty_uint(2 * m_margin_height).c_str());
            }
        }

      private:
        struct Parameters
        {
            const SamplingContext::Mode     m_sampling_mode;
            const size_t                    m_min_samples;
            const size_t                    m_max_samples;
            const float                     m_error_threshold;
            float                           m_splitting_threshold;

            explicit Parameters(const ParamArray& params)
              : m_sampling_mode(get_sampling_context_mode(params))
              , m_min_samples(params.get_required<size_t>("min_samples", 32))
              , m_max_samples(params.get_required<size_t>("max_samples", 512))
              , m_error_threshold(params.get_required<float>("precision", 0.01f))
            {
                m_splitting_threshold = m_error_threshold * 256;
            }
        };

        const Parameters                        m_params;
        size_t                                  m_variation_aov_index;
        size_t                                  m_samples_aov_index;
        size_t                                  m_block_coverage_aov_index;
        size_t                                  m_total_samples;
        Population<uint64>                      m_spp;
        size_t                                  m_total_saved_samples;
        Population<uint64>                      m_saved_samples;
        Population<size_t>                      m_block_amount;
        size_t                                  m_max_samples;
        size_t                                  m_total_pixel_converged;
        size_t                                  m_total_pixel;
        Population<float>                       m_block_error;
        unique_ptr<Tile>                        m_diagnostics;

        struct PixelBlock
        {
            AABB2i                              m_surface;
            size_t                              m_area;
            size_t                              m_width;
            size_t                              m_height;
            size_t                              m_spp;
            float                               m_block_error;
            bool                                m_converged;
            size_t                              m_splitting_point;

            enum Axis
            {
                HORIZONTAL_X,
                VERTICAL_Y,
            };

            Axis                                m_main_axis;

            explicit PixelBlock(
                const AABB2i&       surface)
              : m_surface(surface)
              , m_width(surface.max.x - surface.min.x + 1)
              , m_height(surface.max.y - surface.min.y + 1)
              , m_spp(0)
              , m_block_error(0)
              , m_converged(false)
              , m_splitting_point(0)
            {
                assert(m_surface.is_valid());
                m_area = m_width * m_height;

                if (m_width >= m_height)
                    m_main_axis = Axis::HORIZONTAL_X; // block is wider
                else
                    m_main_axis = Axis::VERTICAL_Y; // block is taller
            }
        };

        void sample_pixel_block(
            PixelBlock&                         pb,
            IAbortSwitch&                       abort_switch,
            const size_t                        batch_size,
            ShadingResultFrameBuffer*           framebuffer,
            ShadingResultFrameBuffer*           second_framebuffer,
            const AABB2i&                       tile_bbox,
            const int                           tile_origin_x,
            const int                           tile_origin_y,
            const Frame&                        frame,
            const size_t                        frame_width,
            const size_t                        pass_hash,
            const size_t                        aov_count)
        {
            // Loop over block pixels.
            for (int y = pb.m_surface.min.y; y <= pb.m_surface.max.y; ++y)
            {
                for (int x = pb.m_surface.min.x; x <= pb.m_surface.max.x; ++x)
                {
                    // Cancel any work done on this tile if rendering is aborted.
                    if (abort_switch.is_aborted())
                        return;

                    // Retrieve the coordinates of the pixel in the padded tile.
                    const Vector2i pt(x, y);

                    // Skip pixels outside the intersection of the padded tile and the crop window.
                    if (!pb.m_surface.contains(pt))
                        continue;

                    const Vector2i pi(tile_origin_x + pt.x, tile_origin_y + pt.y);

#ifdef DEBUG_BREAK_AT_PIXEL

                    // Break in the debugger when this pixel is reached.
                    if (pi == DEBUG_BREAK_AT_PIXEL)
                        BREAKPOINT();

#endif

                    // Render this pixel.
                    {
                        on_pixel_begin(pi, pt, tile_bbox, m_aov_accumulators);

                        const size_t pixel_index = pi.y * frame_width + pi.x;
                        const size_t instance = hash_uint32(static_cast<uint32>(pass_hash + pixel_index * (pb.m_spp + 1)));

                        SamplingContext::RNGType rng(pass_hash, instance);
                        SamplingContext sampling_context(
                            rng,
                            m_params.m_sampling_mode,
                            2,                          // number of dimensions
                            0,                          // number of samples -- unknown
                            instance);                  // initial instance number

                        bool second = false;

                        for (size_t j = 0; j < batch_size; ++j)
                        {
                            // Generate a uniform sample in [0,1)^2.
                            const Vector2d s = sampling_context.next2<Vector2d>();

                            // Compute the sample position in NDC.
                            const Vector2d sample_position = frame.get_sample_position(pi.x + s.x, pi.y + s.y);

                            // Create a pixel context that identifies the pixel and sample currently being rendered.
                            const PixelContext pixel_context(pi, sample_position);

                            // Render the sample.
                            ShadingResult shading_result(aov_count);
                            SamplingContext child_sampling_context(sampling_context);
                            m_sample_renderer->render_sample(
                                child_sampling_context,
                                pixel_context,
                                sample_position,
                                m_aov_accumulators,
                                shading_result);

                            // Ignore invalid samples.
                            if (!shading_result.is_valid())
                            {
                                signal_invalid_sample();
                                continue;
                            }

                            // Merge the sample into the scratch framebuffer.
                            framebuffer->add(
                                static_cast<float>(pt.x + s.x),
                                static_cast<float>(pt.y + s.y),
                                shading_result);

                            if (second)
                            {
                                second_framebuffer->add(
                                        static_cast<float>(pt.x + s.x),
                                        static_cast<float>(pt.y + s.y),
                                        shading_result);
                            }

                            second = !second;
                        }
                        on_pixel_end(pi, pt, tile_bbox, m_aov_accumulators);
                    }
                }
            }

            pb.m_spp += batch_size;
        }

        void evaluate_pixel_block_error(
            PixelBlock&                         pb,
            ShadingResultFrameBuffer*           framebuffer,
            ShadingResultFrameBuffer*           second_framebuffer)
        {
            float error = pb.m_block_error = 0.0f;

            // Determine a splitting point so that two blocks have roughly the same error.
            size_t splitting_point = pb.m_splitting_point = 0;

            AABB2i block_area = pb.m_surface;
            const AABB2i img_area = framebuffer->get_crop_window();
            block_area.min.x = max(0, block_area.min.x);
            block_area.min.y = max(0, block_area.min.y);
            block_area.max.x = min(img_area.max.x, block_area.max.x);
            block_area.max.y = min(img_area.max.y, block_area.max.y);

            const float pixel_count = (block_area.max.x - block_area.min.x + 1) * (block_area.max.y - block_area.min.y + 1);
            const float img_pixel_count = (img_area.max.x - img_area.min.x + 1) * (img_area.max.y - img_area.min.y + 1);

            // Compute scale factor as the block area over the image area.
            double scale_factor = fast_sqrt(pixel_count / img_pixel_count) / pixel_count;
            const double splitting_threshold = (m_params.m_splitting_threshold - m_params.m_error_threshold) * 0.5;

            if (pb.m_width >= BlockSplittingThreshold
                && pb.m_height >= BlockSplittingThreshold)
            {
                if (pb.m_main_axis == PixelBlock::Axis::HORIZONTAL_X)
                {
                    // Loop over block pixels when splitting vertically.
                    for (int x = block_area.min.x; x <= block_area.max.x; ++x)
                    {
                        for (int y = block_area.min.y; y <= block_area.max.y; ++y)
                        {
                            assert(x >= 0 && x < framebuffer->get_width());
                            assert(y >= 0 && y < framebuffer->get_height());

                            const float* main_ptr = framebuffer->pixel(x, y);
                            const float* second_ptr = second_framebuffer->pixel(x, y);

                            error += compute_pixel_error(main_ptr, second_ptr);
                        }
                        if (splitting_point == 0
                            && error * scale_factor >= splitting_threshold
                            && x >= BlockMinAllowedSize && (block_area.max.x - x) <= BlockMinAllowedSize)
                        {
                            splitting_point = x;
                        }
                    }
                }
                else
                {
                    // Loop over block pixels when splitting horizontally.
                    for (int y = block_area.min.y; y <= block_area.max.y; ++y)
                    {
                        for (int x = block_area.min.x; x <= block_area.max.x; ++x)
                        {
                            assert(x >= 0 && x < framebuffer->get_width());
                            assert(y >= 0 && y < framebuffer->get_height());

                            const float* main_ptr = framebuffer->pixel(x, y);
                            const float* second_ptr = second_framebuffer->pixel(x, y);

                            error += compute_pixel_error(main_ptr, second_ptr);
                        }
                        if (splitting_point == 0
                            && error * scale_factor >= splitting_threshold
                            && y >= BlockMinAllowedSize && (block_area.max.y - y) <= BlockMinAllowedSize)
                        {
                            splitting_point = y;
                        }
                    }
                }
            }
            else
            {
                // Loop over block pixels when not splitting.
                for (int x = block_area.min.x; x <= block_area.max.x; ++x)
                {
                    for (int y = block_area.min.y; y <= block_area.max.y; ++y)
                    {
                        assert(x >= 0 && x < framebuffer->get_width());
                        assert(y >= 0 && y < framebuffer->get_height());

                        const float* main_ptr = framebuffer->pixel(x, y);
                        const float* second_ptr = second_framebuffer->pixel(x, y);

                        error += compute_pixel_error(main_ptr, second_ptr);
                    }
                }
            }

            pb.m_block_error = error * scale_factor;

            if (pb.m_block_error <= m_params.m_error_threshold)
                pb.m_converged = true;
            else if (pb.m_block_error <= m_params.m_splitting_threshold)
            {
                if (splitting_point != 0)
                {
                    pb.m_splitting_point = splitting_point;
                }
                else if ( pb.m_width >= BlockSplittingThreshold
                    && pb.m_height >= BlockSplittingThreshold)
                {
                    if (pb.m_main_axis == PixelBlock::Axis::HORIZONTAL_X)
                        pb.m_splitting_point = pb.m_surface.min.x + static_cast<int>(pb.m_width * 0.5f - 0.5f);
                    else
                        pb.m_splitting_point = pb.m_surface.min.y + static_cast<int>(pb.m_height * 0.5f - 0.5f);
                }
                else
                {
                    pb.m_splitting_point = 0;
                }
            }
            else
            {
                pb.m_splitting_point = 0;
            }
        }

        void split_pixel_block(
            PixelBlock&                         pb,
            vector<PixelBlock>&                 source_blocks,
            vector<PixelBlock>&                 output_blocks) const
        {
            assert(pb.m_width >= BlockSplittingThreshold);
            assert(pb.m_height >= BlockSplittingThreshold);

            AABB2i f_half = pb.m_surface, s_half = pb.m_surface;

            if (pb.m_main_axis == PixelBlock::Axis::HORIZONTAL_X)
            {
                // Split horizontaly.
                f_half.max.x = pb.m_splitting_point;
                s_half.min.x = pb.m_splitting_point + 1;
            }
            else
            {
                // Split verticaly.
                f_half.max.y = pb.m_splitting_point;
                s_half.min.y = pb.m_splitting_point + 1;
            }

            assert(f_half.is_valid());
            assert(s_half.is_valid());

            PixelBlock f_block(f_half), s_block(s_half);

            assert(f_block.m_width >= BlockMinAllowedSize && f_block.m_height >= BlockMinAllowedSize);
            assert(s_block.m_width >= BlockMinAllowedSize && s_block.m_height >= BlockMinAllowedSize);

            f_block.m_spp = s_block.m_spp = pb.m_spp;

            output_blocks.push_back(f_block);
            output_blocks.push_back(s_block);
        }

        static float compute_pixel_error(const float* main_pixel, const float* accumulated_pixel)
        {
            // Weight.
            const float main_weight = *main_pixel++;
            const float main_rcp_weight = main_weight == 0.0f ? 0.0f : 1.0f / main_weight;
            const float second_weight = *accumulated_pixel++;
            const float second_rcp_weight = second_weight == 0.0f ? 0.0f : 1.0f / second_weight;

            // Get color.
            Color4f main_color(main_pixel[0], main_pixel[1], main_pixel[2], main_pixel[3]);
            main_color = main_color * main_rcp_weight;
            main_pixel += 4;

            Color4f second_color(accumulated_pixel[0], accumulated_pixel[1], accumulated_pixel[2], accumulated_pixel[3]);
            second_color = second_color * second_rcp_weight;
            accumulated_pixel += 4;

            main_color.unpremultiply();
            main_color.rgb() = fast_linear_rgb_to_srgb(main_color.rgb());
            main_color = saturate(main_color);
            main_color.premultiply();

            second_color.unpremultiply();
            second_color.rgb() = fast_linear_rgb_to_srgb(second_color.rgb());
            second_color = saturate(second_color);
            second_color.premultiply();

            return (
                abs(main_color.r - second_color.r)
                + abs(main_color.g - second_color.g)
                + abs(main_color.b - second_color.b)
                ) / fast_sqrt(main_color.r + main_color.g + main_color.b);
        }

        static Color4f colorize_samples(const float value)
        {
            static const Color4f Black(0.0f, 0.0f, 0.0f, 1.0f);
            static const Color4f White(1.0f, 1.0f, 1.0f, 1.0f);
            return lerp(Black, White, saturate(value));
        }

        static Color4f colorize_variation(const float value)
        {
            return Color4f(0.0f, value, 0.0f, 1.0f);
        }

        // Compute a color from a given integer.
        template <typename T>
        static Color3f integer_to_color(const T i)
        {
            const uint32 u = static_cast<uint32>(i);    // keep the low 32 bits

            const uint32 x = hash_uint32(u);
            const uint32 y = hash_uint32(u + 1);
            const uint32 z = hash_uint32(u + 2);

            return Color3f(
                static_cast<float>(x) * (1.0f / 4294967295.0f),
                static_cast<float>(y) * (1.0f / 4294967295.0f),
                static_cast<float>(z) * (1.0f / 4294967295.0f));
        }
    };
}


//
// AdaptiveTileRendererFactory class implementation.
//

AdaptiveTileRendererFactory::AdaptiveTileRendererFactory(
    const Frame&                        frame,
    ISampleRendererFactory*             sample_renderer_factory,
    IShadingResultFrameBufferFactory*   framebuffer_factory,
    const ParamArray&                   params)
  : m_frame(frame)
  , m_sample_renderer_factory(sample_renderer_factory)
  , m_framebuffer_factory(framebuffer_factory)
  , m_params(params)
{
}

void AdaptiveTileRendererFactory::release()
{
    delete this;
}

ITileRenderer* AdaptiveTileRendererFactory::create(
    const size_t    thread_index)
{
    return
        new AdaptiveTileRenderer(
            m_frame,
            m_sample_renderer_factory,
            m_framebuffer_factory,
            m_params,
            thread_index);
}

Dictionary AdaptiveTileRendererFactory::get_params_metadata()
{
    Dictionary metadata = PixelRendererBaseFactory::get_params_metadata();

    metadata.dictionaries().insert(
        "min_samples",
        Dictionary()
            .insert("type", "int")
            .insert("default", "8")
            .insert("label", "Min Samples")
            .insert("help", "Minimum number of anti-aliasing samples"));

    metadata.dictionaries().insert(
        "max_samples",
        Dictionary()
            .insert("type", "int")
            .insert("default", "256")
            .insert("label", "Max Samples")
            .insert("help", "Maximum number of anti-aliasing samples"));

    metadata.dictionaries().insert(
        "precision",
        Dictionary()
            .insert("type", "float")
            .insert("default", "0.003")
            .insert("label", "Precision")
            .insert("help", "Precision factor, the lower it is, the more it will likely sample a pixel"));

    return metadata;
}

}   // namespace renderer
