
//
// This source file is part of appleseed.
// Visit https://appleseedhq.net/ for additional information and resources.
//
// This software is released under the MIT license.
//
// Copyright (c) 2010-2013 Francois Beaune, Jupiter Jazz Limited
// Copyright (c) 2014-2018 Francois Beaune, The appleseedhq Organization
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
#include "filteredtile.h"

// appleseed.foundation headers.
#include "foundation/image/color.h"
#include "foundation/image/colorspace.h"
#include "foundation/image/pixel.h"
#include "foundation/math/scalar.h"
#include "foundation/math/vector.h"
#include "foundation/platform/atomic.h"

using namespace std;

namespace foundation
{

//
// We use the discrete-to-continuous mapping described in:
//
//   Physically Based Rendering, first edition, p. 295
//
// Other references:
//
//   Physically Based Rendering, first edition, pp. 350-352 and pp. 371-377
//
//   A Pixel Is Not A Little Square, Technical Memo 6, Alvy Ray Smith
//   http://alvyray.com/Memos/CG/Microsoft/6_pixel.pdf
//

FilteredTile::FilteredTile(
    const size_t        width,
    const size_t        height,
    const size_t        channel_count,
    const Filter2f&     filter)
  : Tile(width, height, channel_count + 1, PixelFormatFloat)
  , m_crop_window(Vector2u(0, 0), Vector2u(width - 1, height - 1))
  , m_filter(filter)
{
}

FilteredTile::FilteredTile(
    const size_t        width,
    const size_t        height,
    const size_t        channel_count,
    const AABB2u&       crop_window,
    const Filter2f&     filter)
  : Tile(width, height, channel_count + 1, PixelFormatFloat)
  , m_crop_window(crop_window)
  , m_filter(filter)
{
}

void FilteredTile::clear()
{
    float* ptr = reinterpret_cast<float*>(pixel(0));

    for (size_t i = 0, e = m_pixel_count * m_channel_count; i < e; ++i)
        ptr[i] = 0.0f;
}

void FilteredTile::add(
    const float         x,
    const float         y,
    const float*        values)
{
    // Convert (x, y) from continuous image space to discrete image space.
    const float dx = x - 0.5f;
    const float dy = y - 0.5f;

    // Find the pixels affected by this sample.
    AABB2i footprint;
    footprint.min.x = truncate<int>(fast_ceil(dx - m_filter.get_xradius()));
    footprint.min.y = truncate<int>(fast_ceil(dy - m_filter.get_yradius()));
    footprint.max.x = truncate<int>(fast_floor(dx + m_filter.get_xradius()));
    footprint.max.y = truncate<int>(fast_floor(dy + m_filter.get_yradius()));

    // Don't affect pixels outside the crop window.
    footprint = AABB2i::intersect(footprint, m_crop_window);

    // Bail out if the point does not fall inside the crop window.
    // Only check the x coordinate; the y coordinate is checked in the loop below.
    if (footprint.min.x > footprint.max.x)
        return;

    for (int ry = footprint.min.y; ry <= footprint.max.y; ++ry)
    {
        float* APPLESEED_RESTRICT ptr = reinterpret_cast<float*>(pixel(footprint.min.x, ry));

        for (int rx = footprint.min.x; rx <= footprint.max.x; ++rx)
        {
            const float weight = m_filter.evaluate(rx - dx, ry - dy);
            *ptr++ += weight;

            for (size_t i = 0, e = m_channel_count - 1; i < e; ++i)
                *ptr++ += values[i] * weight;
        }
    }
}

void FilteredTile::atomic_add(
    const float         x,
    const float         y,
    const float*        values)
{
    // Convert (x, y) from continuous image space to discrete image space.
    const float dx = x - 0.5f;
    const float dy = y - 0.5f;

    // Find the pixels affected by this sample.
    AABB2i footprint;
    footprint.min.x = truncate<int>(fast_ceil(dx - m_filter.get_xradius()));
    footprint.min.y = truncate<int>(fast_ceil(dy - m_filter.get_yradius()));
    footprint.max.x = truncate<int>(fast_floor(dx + m_filter.get_xradius()));
    footprint.max.y = truncate<int>(fast_floor(dy + m_filter.get_yradius()));

    // Don't affect pixels outside the crop window.
    footprint = AABB2i::intersect(footprint, m_crop_window);

    // Bail out if the point does not fall inside the crop window.
    // Only check the x coordinate; the y coordinate is checked in the loop below.
    if (footprint.min.x > footprint.max.x)
        return;

    for (int ry = footprint.min.y; ry <= footprint.max.y; ++ry)
    {
        float* APPLESEED_RESTRICT ptr = reinterpret_cast<float*>(pixel(footprint.min.x, ry));

        for (int rx = footprint.min.x; rx <= footprint.max.x; ++rx)
        {
            const float weight = m_filter.evaluate(rx - dx, ry - dy);
            foundation::atomic_add(ptr++, weight);

            for (size_t i = 0, e = m_channel_count - 1; i < e; ++i)
                foundation::atomic_add(ptr++, values[i] * weight);
        }
    }
}

float FilteredTile::compute_weighted_pixel_variance(
    const float*        main,
    const float*        second,
    const bool          convert_to_srgb)
{
    // Get weights.
    const float main_weight = *main++;
    const float main_rcp_weight = main_weight == 0.0f ? 0.0f : 1.0f / main_weight;
    const float second_weight = *second++;
    const float second_rcp_weight = second_weight == 0.0f ? 0.0f : 1.0f / second_weight;

    // Get colors and assign weights.
    Color4f main_color(abs(main[0]), abs(main[1]), abs(main[2]), abs(main[3]));
    main_color *= main_rcp_weight;

    Color4f second_color(abs(second[0]), abs(second[1]), abs(second[2]), abs(second[3]));
    second_color *= second_rcp_weight;

    if (convert_to_srgb)
    {
        main_color.unpremultiply();
        main_color.rgb() = fast_linear_rgb_to_srgb(main_color.rgb());
        main_color = saturate(main_color);
        main_color.premultiply();

        second_color.unpremultiply();
        second_color.rgb() = fast_linear_rgb_to_srgb(second_color.rgb());
        second_color = saturate(second_color);
        second_color.premultiply();
    }

    // Compute variance.
    return (
        abs(main_color.r - second_color.r)
        + abs(main_color.g - second_color.g)
        + abs(main_color.b - second_color.b)
       ) / fast_sqrt(main_color.r + main_color.g + main_color.b);
}

void FilteredTile::compute_tile_variance(
    const AABB2u&       bb,
    const AABB2u&       image,
    FilteredTile*       main,
    FilteredTile*       second,
    float*              block_error,
    float*              max_abs_error,
    const bool          convert_to_srgb)
{
    double error = 0.0;

    assert(main->get_crop_window() == second->get_crop_window());
    assert(main->get_crop_window().contains(bb.min));
    assert(main->get_crop_window().contains(bb.max));
    assert(bb.min.y >= 0 && bb.max.y <= main->get_height());
    assert(bb.min.x >= 0 && bb.max.x <= main->get_width());

    float min_perror = std::numeric_limits<float>::max(),
          max_perror = std::numeric_limits<float>::lowest();

    size_t pixel_count = bb.volume();
    size_t img_pixel_count = image.volume();

    // Loop over block pixels.
    for (int y = bb.min.y; y <= bb.max.y; ++y)
    {
        for (int x = bb.min.x; x <= bb.max.x; ++x)
        {
            const float* main_ptr = main->pixel(x, y);
            const float* second_ptr = second->pixel(x, y);

            const float perror = compute_weighted_pixel_variance(main_ptr, second_ptr, convert_to_srgb);

            if (perror > max_perror)
                max_perror = perror;
            else if (perror < min_perror)
                min_perror = perror;

            error += perror;
        }
    }

    // Compute scale factor as the block area over the image area.
    const float scale_factor = fast_sqrt(static_cast<double>(pixel_count) / static_cast<double>(img_pixel_count))
        / static_cast<double>(pixel_count);

    *block_error = error * scale_factor;
    *max_abs_error = max(abs(error * scale_factor - max_perror), abs(error * scale_factor - min_perror));
}

}   // namespace foundation
