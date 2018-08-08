
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
#include "adaptivemediantilerenderer.h"

// appleseed.renderer headers.
#include "renderer/global/globallogger.h"
#include "renderer/global/globaltypes.h"
#include "renderer/kernel/aov/aovaccumulator.h"
#include "renderer/kernel/aov/imagestack.h"
#include "renderer/kernel/aov/tilestack.h"
#include "renderer/kernel/rendering/ipixelrenderer.h"
#include "renderer/kernel/rendering/isamplerenderer.h"
#include "renderer/kernel/rendering/ishadingresultframebufferfactory.h"
#include "renderer/kernel/rendering/pixelcontext.h"
#include "renderer/kernel/rendering/pixelrendererbase.h"
#include "renderer/kernel/rendering/shadingresultframebuffer.h"
#include "renderer/kernel/shading/shadingresult.h"
#include "renderer/modeling/aov/aov.h"
#include "renderer/modeling/frame/frame.h"
#include "renderer/utility/settingsparsing.h"

// appleseed.foundation headers.
#include "foundation/image/canvasproperties.h"
#include "foundation/image/color.h"
#include "foundation/image/image.h"
#include "foundation/image/tile.h"
#include "foundation/math/aabb.h"
#include "foundation/math/filter.h"
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
#include <deque>
#include <memory>
#include <string>
#include <vector>

using namespace foundation;
using namespace std;

namespace renderer
{

namespace
{

    const size_t VarianceWindowSize = 3;

    //
    // AdaptivePixel.
    //

    struct AdaptivePixel
    {
        Vector<int16, 2>    m_position;
        bool                m_terminated;
    };

    //
    // Adaptive median tile renderer.
    //

    class AdaptiveMedianTileRenderer
      : public ITileRenderer
    {
      public:
        AdaptiveMedianTileRenderer(
            const Frame&                        frame,
            ISampleRendererFactory*             sample_renderer_factory,
            IShadingResultFrameBufferFactory*   framebuffer_factory,
            const ParamArray&                   params,
            const size_t                        thread_index)
          : m_aov_accumulators(frame)
          , m_framebuffer_factory(framebuffer_factory)
          , m_params(params)
          , m_invalid_sample_count(0)
          , m_sample_aov_tile(nullptr)
          , m_variation_aov_tile(nullptr)
          , m_sample_renderer(sample_renderer_factory->create(thread_index))
          , m_total_pixel(0)
          , m_total_pixel_converged(0)
        {
            compute_tile_margins(frame, thread_index == 0);
            compute_pixel_ordering(frame);

            m_sample_aov_index = frame.aovs().get_index("pixel_sample_count");
            m_variation_aov_index = frame.aovs().get_index("pixel_variation");
        }

        void release() override
        {
            delete this;
        }

        void print_settings() const override
        {
            RENDERER_LOG_INFO(
                "adaptive tile renderer settings:\n"
                "  batch size                    %s\n"
                "  min samples                   %s\n"
                "  max samples                   %s\n"
                "  noise threshold               %f",
                pretty_uint(m_params.m_batch_size).c_str(),
                pretty_uint(m_params.m_min_samples).c_str(),
                pretty_uint(m_params.m_max_samples).c_str(),
                m_params.m_noise_threshold);

            m_sample_renderer->print_settings();
        }

        void render_tile(
            const Frame&                        frame,
            const size_t                        tile_x,
            const size_t                        tile_y,
            const size_t                        pass_hash,
            IAbortSwitch&                       abort_switch) override
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
            on_tile_begin(frame, tile_x, tile_y, tile, aov_tiles);

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
            // Accumulation buffer.
            // If rendering multiple passes, the permanent buffer factory will return
            // the same buffer so we must create a new one.
            ShadingResultFrameBuffer* second_framebuffer = new ShadingResultFrameBuffer(
                tile.get_width(),
                tile.get_height(),
                frame.aov_images().size(),
                tile_bbox,
                frame.get_filter());

            if (m_params.m_passes > 1)
                second_framebuffer->copy_from(*framebuffer);
            else
                second_framebuffer->clear();

            assert(second_framebuffer);

            const size_t pixel_count = tile_bbox.volume();

            // First uniform batch.
            for (size_t i = 0, e = m_pixel_ordering.size(); i < e; ++i)
            {
                // Cancel any work done on this tile if rendering is aborted.
                if (abort_switch.is_aborted())
                    return;

                AdaptivePixel& pixel = m_pixel_ordering[i];
                pixel.m_terminated = false;

                if (m_params.m_min_samples == 0) continue;

                // Retrieve the coordinates of the pixel in the padded tile.
                const Vector2i pt(pixel.m_position.x, pixel.m_position.y);

                // Skip pixels outside the intersection of the padded tile and the crop window.
                if (!padded_tile_bbox.contains(pt))
                    continue;

                const Vector2i pi(tile_origin_x + pt.x, tile_origin_y + pt.y);

                const size_t pixel_index = pi.y * frame_width + pi.x;
                const size_t instance = hash_uint32(static_cast<uint32>(pass_hash + pixel_index));

                sample_pixel(
                    frame,
                    pi,
                    pt,
                    framebuffer,
                    second_framebuffer,
                    pass_hash,
                    instance,
                    m_params.m_min_samples,
                    aov_count);
            }

            // Noise buffer.
            Tile noise_buffer(
                tile.get_width(),
                tile.get_height(),
                1,
                PixelFormatFloat);

            size_t spp = m_params.m_min_samples;
            size_t tile_converged_pixels = 0;

            // We stop to sample padded pixels when some of the pixels have converged.
            // Padded pixels are outside the noise tile and we can't compute their noise.
            size_t padded_pixels_stop_threshold = static_cast<size_t>(static_cast<float>(pixel_count) * 0.5f);

            // Adaptive rendering.
            while (true)
            {
                bool all_pixels_terminated = true;

                const size_t batch_size =
                    m_params.m_max_samples == 0 ? m_params.m_batch_size
                    : min(m_params.m_batch_size, m_params.m_max_samples - spp);

                const size_t batch_instance =
                    spp * frame_width * frame.image().properties().m_canvas_height;

                float window_error_threshold = 0.0f;

                // A Batch.
                for (size_t i = 0, e = m_pixel_ordering.size(); i < e; ++i)
                {
                    // Cancel any work done on this tile if rendering is aborted.
                    if (abort_switch.is_aborted())
                        return;

                    AdaptivePixel& pixel = m_pixel_ordering[i];

                    if (pixel.m_terminated)
                        continue;

                    // Retrieve the coordinates of the pixel in the padded tile.
                    const Vector2i pt(pixel.m_position.x, pixel.m_position.y);

                    const bool padded_pixel = !tile_bbox.contains(pt);

                    // Skip pixels outside the intersection of the padded tile and the crop window.
                    if (!padded_tile_bbox.contains(pt))
                        continue;

                    // We only check for end if we started adaptive sampling.
                    if (spp > m_params.m_min_samples)
                    {
                        if (m_params.m_max_samples != 0 && spp >= m_params.m_max_samples)
                        {
                            // Pixel is terminated if it has reached max samples
                            pixel.m_terminated = true;
                        }
                        else if (padded_pixel)
                        {
                            // Pixel is terminated if it's on the margin and some of the pixels have
                            // already converged.
                            pixel.m_terminated = tile_converged_pixels >= padded_pixels_stop_threshold;
                        }
                        else
                        {
                            // Pixel is terminated if it has reached noise threshold.
                            window_error_threshold = compute_window_variance(
                                pt,
                                noise_buffer,
                                tile_bbox);

                            pixel.m_terminated =
                                window_error_threshold <= m_params.m_noise_threshold;

                            if (pixel.m_terminated) ++tile_converged_pixels;
                        }
                    }

                    if (pixel.m_terminated && !padded_pixel)
                    {
                        // Fill AOVs and stats.
                        m_spp.insert(spp);

                        if ((m_sample_aov_tile != nullptr || m_variation_aov_tile != nullptr)
                            && tile_bbox.contains(pt))
                        {
                            Color3f value(0.0f);

                            if (m_sample_aov_tile != nullptr)
                            {
                                m_sample_aov_tile->get_pixel(pt.x, pt.y, value);
                                value[0] += static_cast<float>(spp);
                                m_sample_aov_tile->set_pixel(pt.x, pt.y, value);
                            }

                            if (m_variation_aov_tile != nullptr)
                            {
                                m_sample_aov_tile->get_pixel(pt.x, pt.y, value);
                                value[0] += window_error_threshold;
                                m_variation_aov_tile->set_pixel(pt.x, pt.y, value);
                            }
                        }

                        continue;
                    }

                    const Vector2i pi(tile_origin_x + pt.x, tile_origin_y + pt.y);

                    const size_t pixel_index = pi.y * frame_width + pi.x;
                    const size_t instance =
                        hash_uint32(static_cast<uint32>(pass_hash + pixel_index + batch_instance));

                    sample_pixel(
                        frame,
                        pi,
                        pt,
                        framebuffer,
                        second_framebuffer,
                        pass_hash,
                        instance,
                        batch_size,
                        aov_count);

                    if (!padded_pixel)
                    {
                        // Compute variance.
                        float pixel_variance = FilteredTile::compute_weighted_pixel_variance(
                            framebuffer->pixel(pt.x, pt.y),
                            second_framebuffer->pixel(pt.x, pt.y));

                        noise_buffer.set_pixel(
                            pt.x,
                            pt.y,
                            &pixel_variance);

                        all_pixels_terminated = false;
                    }
                }

                if (all_pixels_terminated) break;

                spp += batch_size;
            }

            m_total_pixel += pixel_count;
            m_total_pixel_converged += tile_converged_pixels;

            // Develop the framebuffer to the tile.
            framebuffer->develop_to_tile(tile, aov_tiles);

            // Release the framebuffer.
            m_framebuffer_factory->destroy(framebuffer);

            // Delete the accumulation buffer.
            delete second_framebuffer;

            // Inform the AOV accumulators that we are done rendering a tile.
            m_aov_accumulators.on_tile_end(frame, tile_x, tile_y);

            // Inform the pixel renderer that we are done rendering the tile.
            on_tile_end(frame, tile_x, tile_y, tile, aov_tiles);
        }

        StatisticsVector get_statistics() const override
        {
            Statistics stats;

            // How many samples per pixel were made.
            stats.insert("samples/pixel", m_spp);
            // Converged pixels over total pixels.
            stats.insert_percent("convergence rate", m_total_pixel_converged, m_total_pixel);

            StatisticsVector vec;
            vec.insert("adaptive tile renderer statistics", stats);
            vec.merge(m_sample_renderer->get_statistics());

            return vec;
        }

      private:
        struct Parameters
        {
            const SamplingContext::Mode     m_sampling_mode;
            const size_t                    m_batch_size;
            const size_t                    m_min_samples;
            const size_t                    m_max_samples;
            const float                     m_noise_threshold;
            const size_t                    m_passes;

            explicit Parameters(const ParamArray& params)
              : m_sampling_mode(get_sampling_context_mode(params))
              , m_batch_size(params.get_required<size_t>("batch_size", 32))
              , m_min_samples(params.get_required<size_t>("min_samples", 0))
              , m_max_samples(params.get_required<size_t>("max_samples", 256))
              , m_noise_threshold(params.get_required<float>("noise_threshold", 5.0f))
              , m_passes(params.get_optional<size_t>("passes", 1))
            {
            }
        };

        AOVAccumulatorContainer                 m_aov_accumulators;
        IShadingResultFrameBufferFactory*       m_framebuffer_factory;
        int                                     m_margin_width;
        int                                     m_margin_height;
        const Parameters                        m_params;
        size_t                                  m_sample_aov_index;
        size_t                                  m_variation_aov_index;
        size_t                                  m_invalid_sample_count;
        Tile*                                   m_sample_aov_tile;
        Tile*                                   m_variation_aov_tile;
        auto_release_ptr<ISampleRenderer>       m_sample_renderer;
        vector<AdaptivePixel>                   m_pixel_ordering;

        // Members used for statistics.
        Population<uint64>                      m_spp;
        size_t                                  m_total_pixel;
        size_t                                  m_total_pixel_converged;

        void compute_tile_margins(
            const Frame&                        frame,
            const bool                          primary)
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
                AdaptivePixel& pixel = m_pixel_ordering[i];
                pixel.m_position.x = static_cast<int16>(x) - m_margin_width;
                pixel.m_position.y = static_cast<int16>(y) - m_margin_height;
            }
        }

        void on_tile_begin(
            const Frame&                        frame,
            const size_t                        tile_x,
            const size_t                        tile_y,
            Tile&                               tile,
            TileStack&                          aov_tiles)
        {
            if (m_sample_aov_index != ~size_t(0))
                m_sample_aov_tile = &frame.aovs().get_by_index(m_sample_aov_index)->get_image().tile(tile_x, tile_y);
            if (m_variation_aov_index != ~size_t(0))
                m_variation_aov_tile = &frame.aovs().get_by_index(m_variation_aov_index)->get_image().tile(tile_x, tile_y);
        }

        void on_tile_end(
            const Frame&                        frame,
            const size_t                        tile_x,
            const size_t                        tile_y,
            Tile&                               tile,
            TileStack&                          aov_tiles)
        {
            m_sample_aov_tile = nullptr;
            m_variation_aov_tile = nullptr;
        }

        void on_pixel_begin(
            const Frame&                        frame,
            const Vector2i&                     pi,
            const Vector2i&                     pt)
        {
            m_invalid_sample_count = 0;
            m_aov_accumulators.on_pixel_begin(pi);
        }

        void on_pixel_end(
            const Frame&                        frame,
            const Vector2i&                     pi,
            const Vector2i&                     pt)
        {
            static const size_t MaxWarningsPerThread = 2;

            m_aov_accumulators.on_pixel_end(pi);

            // Warns the user for bad pixels.
            if (m_invalid_sample_count > 0)
            {
                // We can't store the number of error per pixel because of adaptive rendering.
                if (m_invalid_sample_count <= MaxWarningsPerThread)
                {
                    RENDERER_LOG_WARNING(
                        "%s sample%s at pixel (%d, %d) had nan, negative or infinite components and %s ignored.",
                        pretty_uint(m_invalid_sample_count).c_str(),
                        m_invalid_sample_count > 1 ? "s" : "",
                        pi.x, pi.y,
                        m_invalid_sample_count > 1 ? "were" : "was");
                }
                else if (m_invalid_sample_count == MaxWarningsPerThread + 1)
                {
                    RENDERER_LOG_WARNING("more invalid samples found, omitting warning messages for brevity.");
                }
            }
        }

        void sample_pixel(
            const Frame&                        frame,
            const Vector2i&                     pi,
            const Vector2i&                     pt,
            ShadingResultFrameBuffer*           framebuffer,
            ShadingResultFrameBuffer*           second_framebuffer,
            const size_t                        pass_hash,
            const size_t                        instance,
            const size_t                        batch_size,
            const size_t                        aov_count)
        {
            on_pixel_begin(frame, pi, pt);

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
                    ++m_invalid_sample_count;
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

            on_pixel_end(frame, pi, pt);
        }

        float compute_window_variance(
            const Vector2i&                     pt,
            const Tile&                         noise_buffer,
            const AABB2i&                       tile_bbox) const
        {
            // Simply ignore out of bound pixels.
            // At this point, we know that pt is in the tile.

            // Create the window.
            AABB2i window;
            window.min.x = pt.x - VarianceWindowSize;
            window.max.x = pt.x + VarianceWindowSize;
            window.min.y = pt.y - VarianceWindowSize;
            window.max.y = pt.y + VarianceWindowSize;
            window = AABB2i::intersect(window, tile_bbox);

            if (!window.is_valid())
            {
                return 0.0f;
            }

            // We take the maximum of the window.
            float result = std::numeric_limits<float>::lowest();

            for (size_t y = window.min.y; y <= window.max.y; ++y)
            {
                const float* ptr = reinterpret_cast<const float*>(noise_buffer.pixel(window.min.x, y));

                for (size_t x = window.min.x; x <= window.max.x; ++x)
                {
                    result = max(result, *ptr++);
                }
            }

            return result;
        }

        static Color4f colorize_samples(
            const float                         value)
        {
            static const Color4f Blue(0.0f, 0.0f, 1.0f, 1.0f);
            static const Color4f Orange(1.0f, 0.6f, 0.0f, 1.0f);
            return lerp(Blue, Orange, saturate(value));
        }
    };
}


//
// AdaptiveMedianTileRendererFactory class implementation.
//

AdaptiveMedianTileRendererFactory::AdaptiveMedianTileRendererFactory(
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

void AdaptiveMedianTileRendererFactory::release()
{
    delete this;
}

ITileRenderer* AdaptiveMedianTileRendererFactory::create(
    const size_t    thread_index)
{
    return
        new AdaptiveMedianTileRenderer(
            m_frame,
            m_sample_renderer_factory,
            m_framebuffer_factory,
            m_params,
            thread_index);
}

Dictionary AdaptiveMedianTileRendererFactory::get_params_metadata()
{
    Dictionary metadata;

    metadata.dictionaries().insert(
        "batch_size",
        Dictionary()
            .insert("type", "int")
            .insert("default", "8")
            .insert("label", "Min Samples")
            .insert("help", "How many samples to render before each noise calculation."));

    metadata.dictionaries().insert(
        "min_samples",
        Dictionary()
            .insert("type", "int")
            .insert("default", "0")
            .insert("label", "Batch Size")
            .insert("help", "How many uniform samples to render before starting adaptive sampling."));

    metadata.dictionaries().insert(
        "max_samples",
        Dictionary()
            .insert("type", "int")
            .insert("default", "256")
            .insert("label", "Max Samples")
            .insert("help", "Maximum number of anti-aliasing samples (0 for unlimited)"));

    metadata.dictionaries().insert(
        "noise_threshold",
        Dictionary()
            .insert("type", "float")
            .insert("default", "0.005")
            .insert("label", "Noise Threshold")
            .insert("help", "Rendering stop threshold"));

    return metadata;
}

}   // namespace renderer
