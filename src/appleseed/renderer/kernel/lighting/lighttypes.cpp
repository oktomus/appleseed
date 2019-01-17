
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
#include "foundation/math/basis.h"
#include "foundation/math/sampling/mappings.h"
#include "foundation/math/intersection/raytrianglemt.h"

using namespace foundation;

namespace renderer
{

//
// EmittingShape class implementation.
//

EmittingShape EmittingShape::create_triangle_shape(
    const AssemblyInstance*     assembly_instance,
    const size_t                object_instance_index,
    const size_t                primitive_index,
    const Material*             material,
    const Vector3d&             v0,
    const Vector3d&             v1,
    const Vector3d&             v2,
    const Vector3d&             n0,
    const Vector3d&             n1,
    const Vector3d&             n2,
    const Vector3d&             geometric_normal)
{
    EmittingShape shape(
        TriangleShape,
        assembly_instance,
        object_instance_index,
        primitive_index,
        material);

    shape.m_v0 = v0;
    shape.m_v1 = v1;
    shape.m_v2 = v2;
    shape.m_n0 = n0;
    shape.m_n1 = n1;
    shape.m_n2 = n2;
    shape.m_geometric_normal = geometric_normal;
    return shape;
}

EmittingShape EmittingShape::create_sphere_shape(
    const AssemblyInstance*     assembly_instance,
    const size_t                object_instance_index,
    const Material*             material,
    const Vector3d&             center,
    const double                radius)
{
    EmittingShape shape(
        SphereShape,
        assembly_instance,
        object_instance_index,
        0,
        material);

    shape.m_v0 = center;
    shape.m_v1 = Vector3d(radius);

    shape.m_area = static_cast<float>(FourPi<double>() * square(radius));

    if (shape.m_area != 0.0f)
        shape.m_rcp_area = 1.0f / shape.m_area;

    return shape;
}

EmittingShape EmittingShape::create_rect_shape(
    const AssemblyInstance*     assembly_instance,
    const size_t                object_instance_index,
    const Material*             material,
    const Vector3d&             p,
    const Vector3d&             x,
    const Vector3d&             y,
    const Vector3d&             n)
{
    EmittingShape shape(
        RectShape,
        assembly_instance,
        object_instance_index,
        0,
        material);

    shape.m_v0 = p;
    shape.m_v1 = x;
    shape.m_v2 = y;
    shape.m_geometric_normal = n;
    return shape;
}

EmittingShape::EmittingShape(
    const ShapeType         shape_type,
    const AssemblyInstance* assembly_instance,
    const size_t            object_instance_index,
    const size_t            primitive_index,
    const Material*         material)
{
    m_assembly_instance_and_type.set(
        assembly_instance,
        static_cast<foundation::uint16>(shape_type));

    m_object_instance_index = object_instance_index;
    m_primitive_index = primitive_index;
    m_material = material;
    m_shape_prob = 0.0f;
    m_average_radiance = 1.0f;
}

void EmittingShape::sample_uniform(
    const Vector2f&         s,
    const float             shape_prob,
    LightSample&            light_sample) const
{
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
        // Set the barycentric coordinates.
        light_sample.m_bary = s;

        Vector3d p(sample_sphere_uniform(s));

        // Set the world space shading and geometric normal.
        light_sample.m_shading_normal = p;
        light_sample.m_geometric_normal = p;

        p *= m_v1.x; // Scale p by the radius
        p += m_v0;   // Translate by the sphere center.

        // Compute the world space position of the sample.
        light_sample.m_point = p;
    }
    else if (shape_type == RectShape)
    {
        // Set the barycentric coordinates.
        light_sample.m_bary = s;
        // todo: set P, N, Ng, ... here
    }
    else
    {
        assert(false);
    }

    // Store a pointer to the emitting shape.
    light_sample.m_shape = this;

    // Compute the probability density of this sample.
    light_sample.m_probability = shape_prob * get_rcp_area();
}

float EmittingShape::evaluate_pdf_uniform() const
{
    return get_shape_prob() * get_rcp_area();
}

namespace
{
    // Compute the cartesian coordinates of the given spherical point.
    // Y and Z are swapped because of the projection frame.
    // https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
    Vector3d spherical_to_cartesian(const double theta, const double phi)
    {
        const double sintheta   = sin(theta);
        const double costheta   = cos(theta);
        const double sinphi     = sin(phi);
        const double cosphi     = cos(phi);

        return Vector3d(
            sintheta * cosphi,
            costheta,
            sintheta * sinphi);
    }
}

void EmittingShape::sample_solid_angle(
    const ShadingPoint&     shading_point,
    const Vector2f&         s,
    const float             shape_prob,
    LightSample&            light_sample) const
{
    const auto shape_type = get_shape_type();

    if (shape_type == TriangleShape)
    {
        const Vector3d o = shading_point.get_point();
        const Vector3d A = normalize(m_v0 - o);
        const Vector3d B = normalize(m_v1 - o);
        const Vector3d C = normalize(m_v2 - o);

        double solid_angle;
        const Vector3d d = sample_spherical_triangle_uniform(A, B, C, Vector2d(s), &solid_angle);

        // Project the point on the triangle.
        TriangleMT<double>::RayType ray(o, d);
        TriangleMT<double> triangle(m_v0, m_v1, m_v2);

        double t, u, v;
        if (triangle.intersect(ray, t, u, v))
        {
            light_sample.m_point = o + t * d;
            light_sample.m_bary[0] = static_cast<float>(u);
            light_sample.m_bary[1] = static_cast<float>(v);

            // Compute the probability.
            const double cos_theta = std::abs(dot(m_geometric_normal, d));
            const double rcp_solid_angle = 1.0 / solid_angle;

            const double pdf = rcp_solid_angle * cos_theta / square(t);
            light_sample.m_probability = shape_prob * static_cast<float>(pdf);
        }
        else
        {
            assert(false);
            light_sample.m_probability = 0.0f;
        }
    }
    else if (shape_type == SphereShape)
    {
        // Source:
        // https://schuttejoe.github.io/post/arealightsampling/
        const Vector3d& center  = m_v0;
        const Vector3d& origin  = shading_point.get_point();
        const double    radius  = m_v1[0];

        Vector3d        w               = center - origin;
        const double    dist_to_center  = norm(w);

        // Normalize center to origin vector.
        w *= 1.0 / dist_to_center;

        // Create a orthogonal frame that simplifies the projection.
        const Basis3d frame(w);
        const Vector3d& u = frame.get_tangent_u();
        const Vector3d& v = frame.get_tangent_v();

        // Compute the matrix groing from local to world space.
        Matrix3d world;
        world[0] = u[0];
        world[1] = u[1];
        world[2] = u[2];
        world[3] = w[0];
        world[4] = w[1];
        world[5] = w[2];
        world[6] = v[0];
        world[7] = v[1];
        world[8] = v[2];

        // Compute local space sample position.
        const double q = sqrt(1.0 - square(radius / dist_to_center));
        const double    theta   = acos(1.0 - static_cast<double>(s[0]) + static_cast<double>(s[0]) * q);
        const double    phi     = TwoPi<double>() * static_cast<double>(s[1]);
        const Vector3d  local   = spherical_to_cartesian(theta, phi);

        // Compute world space sample position.
        {
            const Vector3d nwp = local * world;
            const Vector3d x = origin - center;

            const double b = 2.0 * dot(nwp, x);
            const double c = dot(x, x) - radius * radius;

            double t;

            const double root = b * b - 4.0 * c;
            if(root < 0.0)
            {
                // Project x onto v.
                const Vector3d projected_x = (dot(x, nwp) / dot(nwp, nwp)) * nwp;
                t = norm(projected_x);
            }
            else if(root == 0.0)
            {
                t = -0.5 * b;
            }
            else
            {
                const double q = (b > 0.0) ? -0.5 * (b + sqrt(root)) : -0.5 * (b - sqrt(root));
                const double t0 = q;
                const double t1 = c / q;
                t = min(t0, t1);
            }

            light_sample.m_point = origin + t * nwp;
        }

        // Compute the normal at the sample.
        light_sample.m_shading_normal = normalize(light_sample.m_point - m_v0);
        light_sample.m_geometric_normal = light_sample.m_shading_normal;

        // Compute the probability.
        const float pdf = 1.0f / (TwoPi<float>() * (1.0f - static_cast<float>(q)));
        light_sample.m_probability = shape_prob * pdf;
    }
    else if (shape_type == RectShape)
    {
        // todo: implement me...
        sample_uniform(s, shape_prob, light_sample);
    }
    else
    {
        assert(false);
    }

    // Store a pointer to the emitting shape.
    light_sample.m_shape = this;
}

float EmittingShape::evaluate_pdf_solid_angle(
    const Vector3d&         p,
    const Vector3d&         l) const
{
    const auto shape_type = get_shape_type();

    const float shape_probability = get_shape_prob();

    if (shape_type == TriangleShape)
    {
        const Vector3d A = normalize(m_v0 - p);
        const Vector3d B = normalize(m_v1 - p);
        const Vector3d C = normalize(m_v2 - p);
        const double area = compute_spherical_triangle_area(A, B, C);

        Vector3d d = l - p;
        const double d_norm = norm(d);
        d /= d_norm;

        const double cos_theta = std::abs(dot(m_geometric_normal, d));
        const double rcp_solid_angle = 1.0 / area;

        const double pdf = rcp_solid_angle * cos_theta / square(d_norm);
        return shape_probability * static_cast<float>(pdf);
    }
    else if (shape_type == SphereShape)
    {
        // todo: implement me...
        return evaluate_pdf_uniform();
    }
    else if (shape_type == RectShape)
    {
        // todo: implement me...
        return evaluate_pdf_uniform();
    }
    else
    {
        assert(false);
        return -1.0f;
    }
}

void EmittingShape::make_shading_point(
    ShadingPoint&           shading_point,
    const Vector3d&         point,
    const Vector3d&         direction,
    const Vector2f&         bary,
    const Intersector&      intersector) const
{
    const ShadingRay ray(
        point,
        direction,
        0.0,
        0.0,
        ShadingRay::Time(),
        VisibilityFlags::CameraRay, 0);

    const auto shape_type = get_shape_type();

    if (shape_type == TriangleShape)
    {
        intersector.make_triangle_shading_point(
            shading_point,
            ray,
            bary,
            get_assembly_instance(),
            get_assembly_instance()->transform_sequence().get_earliest_transform(),
            get_object_instance_index(),
            get_primitive_index(),
            m_shape_support_plane);
    }
    else if (shape_type == SphereShape)
    {
        // todo: compute P, N, Ng, uv, ...
        // todo: pass them to the shading point.
        assert(false);

        intersector.make_procedural_surface_shading_point(
            shading_point,
            ray,
            bary,
            get_assembly_instance(),
            get_assembly_instance()->transform_sequence().get_earliest_transform(),
            get_object_instance_index(),
            get_primitive_index());
    }
    else if (shape_type == RectShape)
    {
        // todo: compute P, N, Ng, uv, ...
        // todo: pass them to the shading point.
        assert(false);

        intersector.make_procedural_surface_shading_point(
            shading_point,
            ray,
            bary,
            get_assembly_instance(),
            get_assembly_instance()->transform_sequence().get_earliest_transform(),
            get_object_instance_index(),
            get_primitive_index());
    }
}

void EmittingShape::estimate_average_radiance()
{
    // todo:
    /*
    if (constant EDF)
        return EDF->radiance();

    // Varying EDF or OSL emission case.
    for i = 0..N:
    {
        s = random2d()
        make_shading_point(shading_point, p, d, s, intersector);
        radiance += eval EDF or ShaderGroup
    }

    radiance /= N;
    return radiance;
    */

    m_average_radiance = 1.0f;
}

}       // namespace renderer
