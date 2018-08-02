
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
#include "invalidsamplesaov.h"

// appleseed.renderer headers.
#include "renderer/kernel/aov/aovaccumulator.h"
#include "renderer/kernel/aov/imagestack.h"
#include "renderer/kernel/rendering/pixelcontext.h"
#include "renderer/kernel/shading/shadingpoint.h"
#include "renderer/kernel/shading/shadingresult.h"
#include "renderer/modeling/aov/aov.h"
#include "renderer/modeling/frame/frame.h"
#include "renderer/utility/bbox.h"

// appleseed.foundation headers.
#include "foundation/math/aabb.h"
#include "foundation/image/color.h"
#include "foundation/image/image.h"
#include "foundation/image/tile.h"
#include "foundation/utility/api/apistring.h"
#include "foundation/utility/api/specializedapiarrays.h"
#include "foundation/utility/containers/dictionary.h"

// Standard headers.
#include <cstddef>
#include <string>

using namespace foundation;
using namespace std;

namespace renderer
{

namespace
{

    //
    // Invalid Samples AOV accumulator.
    //

    const float NoState = 0.0f;
    const float InvalidSample = 1.0f;
    const float ValidSample = 2.0f;

    class InvalidSamplesAOVAccumulator
      : public UnfilteredAOVAccumulator
    {
      public:
        explicit InvalidSamplesAOVAccumulator(Image& image)
          : UnfilteredAOVAccumulator(image)
          , m_invalid_sample_count(0)
        {
        }

        void on_tile_begin(
            const Frame&                frame,
            const size_t                tile_x,
            const size_t                tile_y,
            const size_t                max_spp)
        {
            UnfilteredAOVAccumulator::on_tile_begin(frame, tile_x, tile_y, max_spp);
            m_crop_window = frame.get_crop_window();
        }

        void on_pixel_begin(const Vector2i& pi) override
        {
            m_invalid_sample_count = 0;
        }

        void on_pixel_end(const Vector2i& pi) override
        {
            // Store a hint corresponding to the sample state in the tile.
            if (!outside_tile(pi) && m_crop_window.contains(pi))
            {
                const Vector2i pt(pi.x - m_tile_bbox.min.x, pi.y - m_tile_bbox.min.y);

                Color3f color(0.0f);
                color[0] = m_invalid_sample_count > 0 ? InvalidSample : ValidSample;

                m_tile->set_pixel(pt.x, pt.y, color);
            }
        }

        void write(
            const PixelContext&         pixel_context,
            const ShadingPoint&         shading_point,
            const ShadingComponents&    shading_components,
            const AOVComponents&        aov_components,
            ShadingResult&              shading_result) override
        {
            // Detect invalid samples.
            if (!outside_tile(pixel_context.get_pixel_coords()))
            {
                if (!shading_result.is_valid())
                    m_invalid_sample_count++;
            }
        }

      private:
        size_t m_invalid_sample_count;
        AABB2i m_crop_window;
    };


    //
    // Invalid Sample AOV.
    //

    const char* Invalid_Samples_Model = "invalid_samples_aov";

    class InvalidSamplesAOV
      : public UnfilteredAOV
    {
      public:
        explicit InvalidSamplesAOV(const ParamArray& params)
          : UnfilteredAOV("invalid_samples", params)
        {
        }

        void release() override
        {
            delete this;
        }

        const char* get_model() const override
        {
            return Invalid_Samples_Model;
        }

        void post_process_image(
            const Frame&    frame,
            const AABB2u&   crop_window) override
        {
            const Image& beauty = frame.image();
            const CanvasProperties& props = beauty.properties();

            for (size_t tile_y = 0; tile_y < props.m_tile_count_y; ++tile_y)
            {
                const int tile_origin_y = static_cast<int>(props.m_tile_height * tile_y);
                for (size_t tile_x = 0; tile_x < props.m_tile_count_x; ++tile_x)
                {
                    const int tile_origin_x = static_cast<int>(props.m_tile_width * tile_x);

                    const Tile& tile = beauty.tile(tile_x, tile_y);
                    Tile& aov_tile = m_image->tile(tile_x, tile_y);

                    // Compute the image space bounding box of the pixels to render.
                    AABB2i tile_bbox = compute_tile_space_bbox(
                        tile,
                        tile_origin_x,
                        tile_origin_y,
                        crop_window);

                    if (!tile_bbox.is_valid())
                        continue;

                    for (size_t y = tile_bbox.min.y; y <= tile_bbox.max.y; ++y)
                    {
                        for (size_t x = tile_bbox.min.x; x <= tile_bbox.max.x; ++x)
                        {
                            Color3f color;
                            aov_tile.get_pixel(x, y, color);

                            if (color[0] == NoState)
                            {
                                color = Color3f(0.0f, 1.0f, 1.0f);
                            }
                            else if (color[0] == InvalidSample)
                            {
                                color = Color3f(1.0f, 0.0f, 1.0f);
                            }
                            else if (color[0] == ValidSample)
                            {
                                Color4f beauty_color;
                                tile.get_pixel(x, y, beauty_color);
                                color.set(0.2f * luminance(beauty_color.rgb()));     // 20% of luminance
                            }

                            aov_tile.set_pixel(x, y, color);
                        }
                    }
                }
            }
        }

      protected:
        auto_release_ptr<AOVAccumulator> create_accumulator() const override
        {
            return auto_release_ptr<AOVAccumulator>(
                new InvalidSamplesAOVAccumulator(get_image()));
        }
    };
}


//
// InvalidSamplesAOVFactory class implementation.
//

void InvalidSamplesAOVFactory::release()
{
    delete this;
}

const char* InvalidSamplesAOVFactory::get_model() const
{
    return Invalid_Samples_Model;
}

Dictionary InvalidSamplesAOVFactory::get_model_metadata() const
{
    return
        Dictionary()
            .insert("name", Invalid_Samples_Model)
            .insert("label", "Invalid Samples");
}

DictionaryArray InvalidSamplesAOVFactory::get_input_metadata() const
{
    DictionaryArray metadata;
    return metadata;
}

auto_release_ptr<AOV> InvalidSamplesAOVFactory::create(
    const ParamArray&   params) const
{
    return auto_release_ptr<AOV>(new InvalidSamplesAOV(params));
}

}   // namespace renderer

