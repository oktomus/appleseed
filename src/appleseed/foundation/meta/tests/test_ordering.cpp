
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

// appleseed.foundation headers.
#include "foundation/image/color.h"
#include "foundation/image/genericimagefilewriter.h"
#include "foundation/image/image.h"
#include "foundation/image/pixel.h"
#include "foundation/math/ordering.h"
#include "foundation/utility/test.h"

// Standard headers.
#include <vector>

using namespace foundation;
using namespace std;

TEST_SUITE(Foundation_Math_Ordering)
{
    void render_hilbert_ordering(
        const size_t    width,
        const size_t    height,
        const char*     filename)
    {
        Image image(width, height, width, height, 3, PixelFormatFloat);

        const size_t pixel_count = width * height;

        // Generate the pixel ordering.
        vector<size_t> ordering;
        ordering.reserve(pixel_count);
        hilbert_ordering(ordering, width, height);
        assert(ordering.size() == pixel_count);

        // Convert the pixel ordering to a 2D representation.
        for (size_t i = 0; i < pixel_count; ++i)
        {
            const size_t x = ordering[i] % width;
            const size_t y = ordering[i] / width;
            assert(x < width);
            assert(y < height);

            Color3f color(i);
            image.set_pixel(x, y, color);
        }

        GenericImageFileWriter writer;
        writer.write(filename, image);
    }

    TEST_CASE(RenderingSquareHilbertOrdering)
    {
        render_hilbert_ordering(16, 16, "unit tests/outputs/test_ordering_hilbert_16_16.exr");
        render_hilbert_ordering(8, 16, "unit tests/outputs/test_ordering_hilbert_8_16.exr");
        render_hilbert_ordering(7, 19, "unit tests/outputs/test_ordering_hilbert_7_19.exr");
    }
}

