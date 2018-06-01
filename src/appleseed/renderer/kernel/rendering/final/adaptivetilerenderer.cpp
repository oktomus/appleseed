
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
#include <list>
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
    //
    // Usefull function block.
    //

    // Compute a color from a given integer.
    template <typename T>
    inline Color3f integer_to_color(const T i)
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


    //
    // Rendering block.
    //

    const size_t BlockMinAllowedSize = 8;
    const size_t BlockSplittingThreshold = 16;
    const size_t SplittingFactor = 50;

    class RenderingBlock
    {
      public:
        RenderingBlock(
            const AABB2i&       surface)
          : m_surface(surface)
          , m_width(surface.max.x - surface.min.x + 1)
          , m_height(surface.max.y - surface.min.y + 1)
          , m_spp(0)
          , m_converged(false)
          , m_block_error(0)
        {
            m_area = m_width * m_height;
        }

        bool has_converged() const
        {
            return m_converged;
        }

        void mark_as_converged()
        {
            m_converged = true;
        }

        size_t get_area() const
        {
            return m_area;
        }

        size_t get_width() const
        {
            return m_width;
        }

        size_t get_height() const
        {
            return m_height;
        }

        const AABB2i& get_bb() const
        {
            return m_surface;
        }

        float get_error() const
        {
            return m_block_error;
        }

        size_t get_spp() const
        {
            return m_spp;
        }

        size_t sample(
            const IAbortSwitch&                         abort_switch,
            const size_t                                batch_size,
            const size_t                                samples_so_far,
            ShadingResultFrameBuffer*                   framebuffer,
            ShadingResultFrameBuffer*                   second_framebuffer,
            const int                                   tile_origin_x,
            const int                                   tile_origin_y,
            const Frame&                                frame,
            const size_t                                frame_width,
            const size_t                                pass_hash,
            const SamplingContext::Mode                 sampling_mode,
            const size_t                                aov_count,
            AOVAccumulatorContainer&                    aov_accumulators,
            const auto_release_ptr<ISampleRenderer>&    sample_renderer,
            PixelRendererBase*                          pixel_renderer)
        {
            size_t invalid_samples = 0;

            // Loop over block pixels.
            for (int y = m_surface.min.y; y <= m_surface.max.y; ++y)
            {
                for (int x = m_surface.min.x; x <= m_surface.max.x; ++x)
                {
                    // Cancel any work done on this tile if rendering is aborted.
                    if (abort_switch.is_aborted())
                        return 0;

                    // Retrieve the coordinates of the pixel in the padded tile.
                    const Vector2i pt(x, y);

                    // Skip pixels outside the intersection of the padded tile and the crop window.
                    if (!m_surface.contains(pt))
                        continue;

                    const Vector2i pi(tile_origin_x + pt.x, tile_origin_y + pt.y);

#ifdef DEBUG_BREAK_AT_PIXEL

                    // Break in the debugger when this pixel is reached.
                    if (pi == DEBUG_BREAK_AT_PIXEL)
                        BREAKPOINT();

#endif

                    // Render this pixel.
                    {
                        const size_t pixel_index = pi.y * frame_width + pi.x;
                        const size_t instance = hash_uint32(static_cast<uint32>(pass_hash + pixel_index * samples_so_far));

                        SamplingContext::RNGType rng(pass_hash, instance);
                        SamplingContext sampling_context(
                            rng,
                            sampling_mode,
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
                            sample_renderer->render_sample(
                                child_sampling_context,
                                pixel_context,
                                sample_position,
                                aov_accumulators,
                                shading_result);

                            // Ignore invalid samples.
                            if (!shading_result.is_valid())
                            {
                                invalid_samples++;
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
                    }
                }
            }

            m_spp += batch_size;

            return invalid_samples;
        }

        float evaluate_error(
            ShadingResultFrameBuffer*           framebuffer,
            ShadingResultFrameBuffer*           second_framebuffer)
        {
            float error = 0;

            // Loop over block pixels.
            for (int x = m_surface.min.x; x <= m_surface.max.x; ++x)
            {
                for (int y = m_surface.min.y; y <= m_surface.max.y; ++y)
                {
                    if (x < 0 || x >= framebuffer->get_width() ||
                        y < 0 || y >= framebuffer->get_height())
                        continue;

                    const float* main_ptr = framebuffer->pixel(x, y);
                    const float* second_ptr = second_framebuffer->pixel(x, y);

                    // Weight.
                    const float main_weight = *main_ptr++;
                    const float main_rcp_weight = main_weight == 0.0f ? 0.0f : 1.0f / main_weight;
                    const float second_weight = *second_ptr++;
                    const float second_rcp_weight = second_weight == 0.0f ? 0.0f : 1.0f / second_weight;

                    // Get color.
                    Color4f main_color(main_ptr[0], main_ptr[1], main_ptr[2], main_ptr[3]);
                    main_color = main_color * main_rcp_weight;
                    main_ptr += 4;

                    Color4f second_color(second_ptr[0], second_ptr[1], second_ptr[2], second_ptr[3]);
                    second_color = second_color * second_rcp_weight;
                    second_ptr += 4;

                    main_color.unpremultiply();
                    main_color.rgb() = fast_linear_rgb_to_srgb(main_color.rgb());
                    main_color = saturate(main_color);
                    main_color.premultiply();

                    second_color.unpremultiply();
                    second_color.rgb() = fast_linear_rgb_to_srgb(second_color.rgb());
                    second_color = saturate(second_color);
                    second_color.premultiply();

                    error += (
                            abs(main_color.r - second_color.r)
                            + abs(main_color.g - second_color.g)
                            + abs(main_color.b - second_color.b)
                            ) / fast_sqrt(main_color.r + main_color.g + main_color.b);
                }
            }

            error *= (1.0f / m_area);
            m_block_error = error;
            return error;
        }

        static list<RenderingBlock>::iterator split_in_place(list<RenderingBlock>& blocks, list<RenderingBlock>::iterator it)
        {
            const RenderingBlock& block = *it;

            assert(block.m_width >= BlockSplittingThreshold);
            assert(block.m_height >= BlockSplittingThreshold);

            AABB2i f_half = block.m_surface, s_half = block.m_surface;

            if (block.m_width >= block.m_height)
            {
                // Split horizontaly.
                size_t splitting_point = block.m_surface.min.x + static_cast<int>(block.m_width * 0.5f - 0.5f);
                f_half.max.x = splitting_point;
                s_half.min.x = splitting_point + 1;
            }
            else
            {
                // Split verticaly.
                size_t splitting_point = block.m_surface.min.y + static_cast<int>(block.m_height * 0.5f - 0.5f);
                f_half.max.y = splitting_point;
                s_half.min.y = splitting_point + 1;
            }

            RenderingBlock f_block(f_half), s_block(s_half);

            assert(f_block.get_width() >= BlockMinAllowedSize && f_block.get_height() >= BlockMinAllowedSize);
            assert(s_block.get_width() >= BlockMinAllowedSize && s_block.get_height() >= BlockMinAllowedSize);

            f_block.m_spp = s_block.m_spp = block.m_spp;

            it = blocks.erase(it);
            blocks.push_front(f_block);
            blocks.push_front(s_block);

            return it;
        }

      private:
        AABB2i    m_surface;
        size_t    m_area;
        size_t    m_width;
        size_t    m_height;
        size_t    m_spp;
        bool      m_converged;
        float     m_block_error;
    };


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
        {
            compute_tile_margins(frame, thread_index == 0);
            compute_pixel_ordering(frame);

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
                m_params.m_error_threshold * SplittingFactor,
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
            list<RenderingBlock>  rendering_blocks;
            const size_t            frame_width = frame.image().properties().m_canvas_width;
            const size_t            aov_count = frame.aov_images().size();

            // Retrieve frame properties.
            const CanvasProperties& frame_properties = frame.image().properties();
            assert(tile_x < frame_properties.m_tile_count_x);
            assert(tile_y < frame_properties.m_tile_count_y);

            // Retrieve tile properties.
            Tile& tile = frame.image().tile(tile_x, tile_y);
            TileStack aov_tiles = frame.aov_images().tiles(tile_x, tile_y);
            const int tile_origin_x = static_cast<int>(frame_properties.m_tile_width * tile_x);
            const int tile_origin_y = static_cast<int>(frame_properties.m_tile_height * tile_y);

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

            rendering_blocks.emplace_back(padded_tile_bbox);

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

            // Pixel begin.
            for (size_t i = 0, e = m_pixel_ordering.size(); i < e; ++i)
            {
                // Retrieve the coordinates of the pixel in the padded tile.
                const Vector2i pt(m_pixel_ordering[i].x, m_pixel_ordering[i].y);

                // Skip pixels outside the intersection of the padded tile and the crop window.
                if (!padded_tile_bbox.contains(pt))
                continue;

                const Vector2i pi(tile_origin_x + pt.x, tile_origin_y + pt.y);

                on_pixel_begin(pi, pt, tile_bbox, m_aov_accumulators);
            }

            // Pixel rendering.
            size_t samples_so_far = 0;

            bool all_converged = false;

            while (!all_converged)
            {
                const int remaining_samples = m_params.m_max_samples - samples_so_far;

                if (remaining_samples < 1)
                    break;

                // Each batch contains 'min' samples.
                const size_t batch_size = min(static_cast<int>(m_params.m_min_samples), remaining_samples);

                const size_t samples_now = samples_so_far;
                const size_t samples_then = samples_now + batch_size;

                all_converged = true;

                // For each block.
                auto rb_it = rendering_blocks.begin();
                while (rb_it != rendering_blocks.end())
                {
                    // Cancel any work done on this tile if rendering is aborted.
                    if (abort_switch.is_aborted())
                        return;

                    RenderingBlock& rb = *rb_it;
                    const AABB2i& rb_aabb = rb.get_bb();

                    // Continue if block is already converged.
                    if (rb.has_converged())
                        continue;

                    all_converged = false;

                    // Draw samples.
                    const size_t invalid = rb.sample(
                        abort_switch,
                        batch_size,
                        samples_now,
                        framebuffer,
                        second_framebuffer,
                        tile_origin_x,
                        tile_origin_y,
                        frame,
                        frame_width,
                        pass_hash,
                        m_params.m_sampling_mode,
                        aov_count,
                        m_aov_accumulators,
                        m_sample_renderer,
                        this);

                    //if (invalid > 0) signal_invalid_sample();

                    // Evaluate error metric.
                    const float rb_error = rb.evaluate_error(framebuffer, second_framebuffer);

                    // Take a decision.
                    if (rb_error <= m_params.m_error_threshold)
                    {
                        rb.mark_as_converged();
                        for (int y = rb_aabb.min.y; y <= rb_aabb.max.y; ++y)
                        {
                            for (int x = rb_aabb.min.x; x <= rb_aabb.max.x; ++x)
                            {
                                // Retrieve the coordinates of the pixel in the padded tile.
                                const Vector2i pt(x, y);
                                const Vector2i pi(tile_origin_x + pt.x, tile_origin_y + pt.y);
                                on_pixel_end(pi, pt, tile_bbox, m_aov_accumulators);
                            }
                        }
                        rb_it++;
                    }
                    else if (remaining_samples - batch_size >= 1
                            && rb_error <= m_params.m_error_threshold * SplittingFactor
                            && rb.get_width() >= BlockSplittingThreshold
                            && rb.get_height() >= BlockSplittingThreshold)
                        rb_it = RenderingBlock::split_in_place(rendering_blocks, rb_it);
                    else
                        rb_it++;
                }

                samples_so_far = samples_then;
            }

            // Pixels end.
            // For each block.
            size_t rb_index = 0;
            for(auto rb_it = rendering_blocks.begin(); rb_it != rendering_blocks.end(); ++rb_it, ++rb_index)
            {
                RenderingBlock& rb = *rb_it;
                AABB2i rb_aabb = rb.get_bb();
                rb_aabb.min.x = max(rb_aabb.min.x, tile_bbox.min.x);
                rb_aabb.max.x = min(rb_aabb.max.x, tile_bbox.max.x);
                rb_aabb.min.y = max(rb_aabb.min.y, tile_bbox.min.y);
                rb_aabb.max.y = min(rb_aabb.max.y, tile_bbox.max.y);
                const size_t rb_pixel_count = rb_aabb.volume();

                // Update statistics.
                m_total_samples += rb.get_spp() * rb_pixel_count;
                m_spp.insert(rb.get_spp());
                m_total_saved_samples += (m_params.m_max_samples - rb.get_spp()) * rb_pixel_count;
                m_saved_samples.insert(m_params.m_max_samples - rb.get_spp());
                m_block_error.insert(rb.get_error());

                for (int y = rb_aabb.min.y; y <= rb_aabb.max.y; ++y)
                {
                    for (int x = rb_aabb.min.x; x <= rb_aabb.max.x; ++x)
                    {
                        // Retrieve the coordinates of the pixel in the padded tile.
                        const Vector2i pt(x, y);

                        // Store diagnostics values in the diagnostics tile.
                        if (are_diagnostics_enabled() && tile_bbox.contains(pt))
                        {
                            Color<float, 5> values;

                            values[0] = rb.get_error();
                            values[1] =
                                m_params.m_min_samples == m_params.m_max_samples
                                ? 1.0f
                                : fit(
                                        static_cast<float>(rb.get_spp()),
                                        static_cast<float>(m_params.m_min_samples),
                                        static_cast<float>(m_params.m_max_samples),
                                        0.0f, 1.0f);

                            Color3f rb_color = integer_to_color(pass_hash + rb_index);
                            values[2] = rb_color[0];
                            values[3] = rb_color[1];
                            values[4] = rb_color[2];

                            m_diagnostics->set_pixel(pt.x, pt.y, values);
                        }

                        if (!rb.has_converged())
                        {
                            const Vector2i pi(tile_origin_x + pt.x, tile_origin_y + pt.y);
                            on_pixel_end(pi, pt, tile_bbox, m_aov_accumulators);
                        }
                    }
                }
                rb_index++;
            }

            // Update statistics.
            m_max_samples += pixel_count * m_params.m_max_samples;

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
            stats.insert("total samples", m_total_samples);
            stats.insert("samples/pixel", m_spp);
            stats.insert("total samples saved", m_total_saved_samples);
            stats.insert("samples saved/pixel", m_saved_samples);
            stats.insert("max samples", m_max_samples);
            stats.insert("block error", m_block_error, 4);

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
        vector<Vector<int16, 2>>            m_pixel_ordering;

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

        void compute_pixel_ordering(const Frame& frame)
        {
            // Compute the dimensions in pixels of the padded tile.
            const CanvasProperties& properties = frame.image().properties();
            const size_t padded_tile_width = properties.m_tile_width + 2 * m_margin_width;
            const size_t padded_tile_height = properties.m_tile_height + 2 * m_margin_height;
            const size_t pixel_count = padded_tile_width * padded_tile_height;

            // Generate the pixel ordering inside the padded tile.
            vector<size_t> ordering;
            ordering.reserve(pixel_count);
            hilbert_ordering(ordering, padded_tile_width, padded_tile_height);
            assert(ordering.size() == pixel_count);

            // Convert the pixel ordering to a 2D representation.
            m_pixel_ordering.resize(pixel_count);
            for (size_t i = 0; i < pixel_count; ++i)
            {
                const size_t x = ordering[i] % padded_tile_width;
                const size_t y = ordering[i] / padded_tile_width;
                assert(x < padded_tile_width);
                assert(y < padded_tile_height);
                m_pixel_ordering[i].x = static_cast<int16>(x) - m_margin_width;
                m_pixel_ordering[i].y = static_cast<int16>(y) - m_margin_height;
            }
        }

      private:
        struct Parameters
        {
            const SamplingContext::Mode     m_sampling_mode;
            const size_t                    m_min_samples;
            const size_t                    m_max_samples;
            const float                     m_error_threshold;

            explicit Parameters(const ParamArray& params)
              : m_sampling_mode(get_sampling_context_mode(params))
              , m_min_samples(params.get_required<size_t>("min_samples", 16))
              , m_max_samples(params.get_required<size_t>("max_samples", 256))
              , m_error_threshold(params.get_optional<float>("precision", 2.0f))
            {
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
        size_t                                  m_max_samples;
        Population<float>                       m_block_error;
        unique_ptr<Tile>                        m_diagnostics;

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
