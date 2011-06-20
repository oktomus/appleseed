
//
// This source file is part of appleseed.
// Visit http://appleseedhq.net/ for additional information and resources.
//
// This software is released under the MIT license.
//
// Copyright (c) 2010-2011 Francois Beaune
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
#include "environmentedf.h"

// appleseed.renderer headers.
#include "renderer/modeling/input/inputarray.h"
#include "renderer/modeling/input/source.h"

using namespace foundation;

namespace renderer
{

//
// EnvironmentEDF class implementation.
//

namespace
{
    const UniqueID g_class_uid = new_guid();
}

EnvironmentEDF::EnvironmentEDF(
    const char*         name,
    const ParamArray&   params)
  : ConnectableEntity(g_class_uid, params)
{
    set_name(name);
}

void EnvironmentEDF::on_frame_begin(const Project& project)
{
}

void EnvironmentEDF::on_frame_end(const Project& project)
{
}

void EnvironmentEDF::check_uniform(const char* input_name) const
{
    if (!m_inputs.source(input_name)->is_uniform())
    {
        RENDERER_LOG_ERROR(
            "the \"%s\" input of a \"%s\" must be bound to a scalar or a color",
            input_name,
            get_model());
    }
}

}   // namespace renderer
