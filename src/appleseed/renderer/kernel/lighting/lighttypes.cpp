
//
// This source file is part of appleseed.
// Visit https://appleseedhq.net/ for additional information and resources.
//
// This software is released under the MIT license.
//
// Copyright (c) 2018 Esteban Tovagliari, The appleseedhq Organization
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
#include "lighttypes.h"

// appleseed.renderer headers.
#include "renderer/kernel/intersection/intersector.h"
#include "renderer/kernel/lighting/lightsample.h"

// appleseed.foundation headers.
#include "foundation/math/sampling/mappings.h"

using namespace foundation;

namespace renderer
{

//
// EmittingShape class implementation.
//

EmittingShape::EmittingShape(
    const ShapeType         shape_type,
    const AssemblyInstance* assembly_instance,
    const size_t            object_instance_index,
    const size_t            primitive_index)
{
    m_assembly_instance_and_type.set(
        assembly_instance,
        static_cast<foundation::uint16>(shape_type));

    m_object_instance_index = object_instance_index;
    m_primitive_index = primitive_index;
}

void EmittingShape::sample_uniform(
    const Vector2f&         s,
    const float             shape_prob,
    LightSample&            light_sample) const
{
    // Store a pointer to the emitting shape.
    light_sample.m_shape = this;

    const auto shape_type = get_shape_type();

    if (shape_type == TriangleShape)
    {
        // Uniformly sample the surface of the shape.
        const Vector3d bary = sample_triangle_uniform(Vector2d(s));

        // Set the barycentric coordinates.
        light_sample.m_bary[0] = static_cast<float>(bary[0]);
        light_sample.m_bary[1] = static_cast<float>(bary[1]);

        // Compute the world space position of the sample.
        light_sample.m_point =
              bary[0] * m_v0
            + bary[1] * m_v1
            + bary[2] * m_v2;

        // Compute the world space shading normal at the position of the sample.
        light_sample.m_shading_normal =
              bary[0] * m_n0
            + bary[1] * m_n1
            + bary[2] * m_n2;
        light_sample.m_shading_normal = normalize(light_sample.m_shading_normal);

        // Set the world space geometric normal.
        light_sample.m_geometric_normal = m_geometric_normal;
    }
    else if (shape_type == SphereShape)
    {
        // todo: implement me...
        // ...
    }
    else if (shape_type == RectShape)
    {
        // Set the barycentric coordinates.
        light_sample.m_bary[0] = static_cast<float>(s[0]);
        light_sample.m_bary[1] = static_cast<float>(s[1]);

        // Compute the world space position of the sample.
        light_sample.m_point =
              m_v0
            + static_cast<double>(s[0]) * m_v1
            + static_cast<double>(s[1]) * m_v2;

        // Compute the world space shading normal at the position of the sample.
        light_sample.m_shading_normal = m_geometric_normal;

        // Set the world space geometric normal.
        light_sample.m_geometric_normal = m_geometric_normal;
    }

    // Compute the probability density of this sample.
    light_sample.m_probability = shape_prob * get_rcp_area();
}

void EmittingShape::sample_solid_angle(
    const ShadingPoint&     shading_point,
    const Vector2f&         s,
    const float             shape_prob,
    LightSample&            light_sample) const
{
    // todo: implement me...
    sample_uniform(s, shape_prob, light_sample);
}

void EmittingShape::make_shading_point(
    ShadingPoint&           shading_point,
    const Vector3d&         point,
    const Vector3d&         direction,
    const Vector2f&         bary,
    const Intersector&      intersector) const
{
    assert(get_shape_type() == TriangleShape);

    intersector.make_surface_shading_point(
        shading_point,
        ShadingRay(
            point,
            direction,
            0.0,
            0.0,
            ShadingRay::Time(),
            VisibilityFlags::CameraRay, 0),
        ShadingPoint::PrimitiveTriangle,    // note: we assume light samples are always on shapes (and not on curves)
        bary,
        get_assembly_instance(),
        get_assembly_instance()->transform_sequence().get_earliest_transform(),
        get_object_instance_index(),
        get_primitive_index(),
        m_shape_support_plane);
}

}       // namespace renderer
