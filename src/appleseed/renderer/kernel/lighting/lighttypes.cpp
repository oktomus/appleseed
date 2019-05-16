
// #define DO_UNIFORM

//
// This source file is part of appleseed.
// Visit https://appleseedhq.net/ for additional information and resources.
//
// This software is released under the MIT license.
//
// Copyright (c) 2019 Esteban Tovagliari, The appleseedhq Organization
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
#include "renderer/kernel/shading/shadingpoint.h"

// appleseed.foundation headers.
#include "foundation/math/basis.h"
#include "foundation/math/distance.h"
#include "foundation/math/fp.h"
#include "foundation/math/intersection/rayparallelogram.h"
#include "foundation/math/intersection/raysphere.h"
#include "foundation/math/intersection/raytrianglemt.h"
#include "foundation/math/sampling/mappings.h"


// wip
#include "foundation/math/sampling/PSCMaps.h"

using namespace foundation;

namespace renderer
{

//
// EmittingShape class implementation.
//
// References:
//
//   [1] Monte Carlo Techniques for Direct Lighting Calculations.
//       http://www.cs.virginia.edu/~jdl/bib/globillum/mis/shirley96.pdf
//
//   [2] Stratified Sampling of Spherical Triangles.
//       https://www.graphics.cornell.edu/pubs/1995/Arv95c.pdf
//
//   [3] An Area-Preserving Parametrization for Spherical Rectangles.
//       https://www.arnoldrenderer.com/research/egsr2013_spherical_rectangle.pdf
//
//   [4] Solid Angle of Conical Surfaces, Polyhedral Cones, and Intersecting Spherical Caps.
//       https://arxiv.org/ftp/arxiv/papers/1205/1205.1396.pdf
//

namespace
{
    template <typename Shape>
    double signed_plane_distance(const Shape& shape, const Vector3d& p)
    {
        return dot(p, shape.m_geometric_normal) + shape.m_plane_dist;
    }
}

EmittingShape EmittingShape::create_triangle_shape(
    const AssemblyInstance*     assembly_instance,
    const size_t                object_instance_index,
    const size_t                primitive_index,
    const Material*             material,
    const double                area,
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

    shape.m_geom.m_triangle.m_v0 = v0;
    shape.m_geom.m_triangle.m_v1 = v1;
    shape.m_geom.m_triangle.m_v2 = v2;
    shape.m_geom.m_triangle.m_n0 = n0;
    shape.m_geom.m_triangle.m_n1 = n1;
    shape.m_geom.m_triangle.m_n2 = n2;
    shape.m_geom.m_triangle.m_geometric_normal = geometric_normal;
    shape.m_geom.m_triangle.m_plane_dist = -dot(v0, geometric_normal);

    shape.m_bbox.invalidate();
    shape.m_bbox.insert(v0);
    shape.m_bbox.insert(v1);
    shape.m_bbox.insert(v2);

    shape.m_centroid = (v0 + v1 + v2) * (1.0 / 3.0);

    shape.m_area = static_cast<float>(area);

    if (shape.m_area != 0.0f)
        shape.m_rcp_area = 1.0f / shape.m_area;
    else
        shape.m_rcp_area = FP<float>().snan();


    return shape;
}

EmittingShape EmittingShape::create_rectangle_shape(
    const AssemblyInstance*     assembly_instance,
    const size_t                object_instance_index,
    const Material*             material,
    const double                area,
    const Vector3d&             o,
    const Vector3d&             x,
    const Vector3d&             y,
    const Vector3d&             n)
{
    EmittingShape shape(
        RectangleShape,
        assembly_instance,
        object_instance_index,
        0,
        material);

    shape.m_geom.m_rectangle.m_origin = o;
    shape.m_geom.m_rectangle.m_x = x;
    shape.m_geom.m_rectangle.m_y = y;
    shape.m_geom.m_rectangle.m_width = norm(x);
    shape.m_geom.m_rectangle.m_height = norm(y);
    shape.m_geom.m_rectangle.m_geometric_normal = n;
    shape.m_geom.m_rectangle.m_plane_dist = -dot(o, n);

    shape.m_area = static_cast<float>(area);

    if (shape.m_area != 0.0f)
        shape.m_rcp_area = 1.0f / shape.m_area;
    else
        shape.m_rcp_area = FP<float>().snan();

    return shape;
}

EmittingShape EmittingShape::create_sphere_shape(
    const AssemblyInstance*     assembly_instance,
    const size_t                object_instance_index,
    const Material*             material,
    const double                area,
    const Vector3d&             center,
    const double                radius)
{
    EmittingShape shape(
        SphereShape,
        assembly_instance,
        object_instance_index,
        0,
        material);

    shape.m_geom.m_sphere.m_center = center;
    shape.m_geom.m_sphere.m_radius = radius;

    shape.m_area = static_cast<float>(area);

    if (shape.m_area != 0.0f)
        shape.m_rcp_area = 1.0f / shape.m_area;
    else
        shape.m_rcp_area = FP<float>().snan();

    return shape;
}

EmittingShape EmittingShape::create_disk_shape(
    const AssemblyInstance*     assembly_instance,
    const size_t                object_instance_index,
    const Material*             material,
    const double                area,
    const Vector3d&             c,
    const double                r,
    const Vector3d&             n,
    const Vector3d&             x,
    const Vector3d&             y)
{
    EmittingShape shape(
        DiskShape,
        assembly_instance,
        object_instance_index,
        0,
        material);

    shape.m_geom.m_disk.m_center = c;
    shape.m_geom.m_disk.m_radius = r;
    shape.m_geom.m_disk.m_geometric_normal = n;
    shape.m_geom.m_disk.m_x = x;
    shape.m_geom.m_disk.m_y = y;

    shape.m_area = static_cast<float>(area);

    if (shape.m_area != 0.0f)
        shape.m_rcp_area = 1.0f / shape.m_area;
    else
        shape.m_rcp_area = FP<float>().snan();

    return shape;
}

#include <iostream>
EmittingShape::EmittingShape(
    const ShapeType         shape_type,
    const AssemblyInstance* assembly_instance,
    const size_t            object_instance_index,
    const size_t            primitive_index,
    const Material*         material)
{
#ifdef DO_UNIFORM
    std::cout << "DOING UNIFORM SAMPLING\n";
#else
    std::cout << "DOING SOLID ANGLE SAMPLING\n";
#endif

    m_assembly_instance_and_type.set(
        assembly_instance,
        static_cast<foundation::uint16>(shape_type));

    m_object_instance_index = object_instance_index;
    m_primitive_index = primitive_index;
    m_material = material;
    m_shape_prob = 0.0f;
    m_average_flux = 1.0f;
}

void EmittingShape::sample_uniform(
    const Vector2f&         s,
    const float             shape_prob,
    LightSample&            light_sample) const
{
    // Store a pointer to the emitting shape.
    light_sample.m_shape = this;

    if (shape_prob != m_shape_prob)
    {
        std::cout << "probs are different: " << shape_prob << " != " << m_shape_prob << "\n";
    }

    const auto shape_type = get_shape_type();

    if (shape_type == TriangleShape)
    {
        // Uniformly sample the surface of the shape.
        const Vector3d bary = sample_triangle_uniform(Vector2d(s));

        // Set the parametric coordinates.
        light_sample.m_param_coords[0] = static_cast<float>(bary[0]);
        light_sample.m_param_coords[1] = static_cast<float>(bary[1]);

        // Compute the world space position of the sample.
        light_sample.m_point =
              bary[0] * m_geom.m_triangle.m_v0
            + bary[1] * m_geom.m_triangle.m_v1
            + bary[2] * m_geom.m_triangle.m_v2;

        // Compute the world space shading normal at the position of the sample.
        light_sample.m_shading_normal =
              bary[0] * m_geom.m_triangle.m_n0
            + bary[1] * m_geom.m_triangle.m_n1
            + bary[2] * m_geom.m_triangle.m_n2;
        light_sample.m_shading_normal = normalize(light_sample.m_shading_normal);

        // Set the world space geometric normal.
        light_sample.m_geometric_normal = m_geom.m_triangle.m_geometric_normal;
    }
    else if (shape_type == RectangleShape)
    {
        // Set the parametric coordinates.
        light_sample.m_param_coords = s;

        light_sample.m_point =
            m_geom.m_rectangle.m_origin +
            static_cast<double>(s[0]) * m_geom.m_rectangle.m_x +
            static_cast<double>(s[1]) * m_geom.m_rectangle.m_y;

        // Set the world space shading and geometric normals.
        light_sample.m_shading_normal = m_geom.m_rectangle.m_geometric_normal;
        light_sample.m_geometric_normal = m_geom.m_rectangle.m_geometric_normal;
    }
    else if (shape_type == SphereShape)
    {
        // Set the parametric coordinates.
        light_sample.m_param_coords = s;

        Vector3d n(sample_sphere_uniform(s));

        // Set the world space shading and geometric normals.
        light_sample.m_shading_normal = n;
        light_sample.m_geometric_normal = n;

        // Compute the world space position of the sample.
        light_sample.m_point = m_geom.m_sphere.m_center + n * m_geom.m_sphere.m_radius;
    }
    else if (shape_type == DiskShape)
    {
        const Vector2f param_coords = sample_disk_uniform(s);

        // Compute the world space position of the sample.
        Vector3d p =
            m_geom.m_disk.m_center +
            static_cast<double>(param_coords[0]) * m_geom.m_disk.m_x +
            static_cast<double>(param_coords[1]) * m_geom.m_disk.m_y;

        light_sample.m_point = p;

        // Set the parametric coordinates.
        light_sample.m_param_coords = param_coords;

        // Set the world space shading and geometric normals.
        light_sample.m_shading_normal = m_geom.m_disk.m_geometric_normal;
        light_sample.m_geometric_normal = m_geom.m_disk.m_geometric_normal;
    }
    else
    {
        assert(false && "Unknown emitter shape type");
    }

    // Compute the probability density of this sample.
    light_sample.m_probability = shape_prob * get_rcp_area();
}

float EmittingShape::evaluate_pdf_uniform() const
{
    return m_shape_prob * get_rcp_area();
}

namespace {
    /*!This structure carries intermediate results that only need to be computed
        once per spherical cap for sampling proportional to solid angle.*/
    struct SphericalCap {
        /*! These three vectors define an orthonormal, positively oriented frame
            in which the normal points towards the center of the spherical cap and
            the other two directions are arbitrary.*/
        Vector3f tangent, bitangent, normal;
        /*! The minimal dot product between normal and a point inside the spherical
            cap. This is an efficient way to express the opening angle of the cap.*/
        float minimalDot;
        /*! The solid angle of the spherical cap in steradians.*/
        float solidAngle;
    };

    /*! Reciprocal of the square root.*/
    inline float rsqrt(float x) {
        return 1.0f / sqrt(x);
    }

    /*! A multiply-add operation a*b+c for scalars.*/
    inline float mad(float a, float b, float c) {
        return a * b + c;
    }

#define M_PI2_FLOAT 6.28318530717958647692f

    /*!	Prepares all intermediate values to sample a spherical cap proportional to
    solid angle. The sphere center is given relative to the surface point for
    which samples are taken.*/
    inline void prepareSphericalCapSampling(SphericalCap& cap, Vector3f sphereCenter, float sphereRadius) {
        float invCenterDistance = rsqrt(dot(sphereCenter, sphereCenter));
        // Construct a coordinate frame where z is aligned with the vector to the 
        // sphere center
        cap.normal = invCenterDistance * sphereCenter;
        cap.tangent = normalize(cross(cap.normal, Vector3f(0.0f, 1.0f, 0.0f)));
        cap.bitangent = cross(cap.normal, cap.tangent);
        // Compute the radius of the circle that bounds the spherical cap
        float maximalRadius = sphereRadius * invCenterDistance;
        cap.minimalDot = sqrt(saturate(mad(-maximalRadius, maximalRadius, 1.0f)));
        cap.solidAngle = mad(-cap.minimalDot, M_PI2_FLOAT, M_PI2_FLOAT);
    }
    /*! Maps independent, uniform random numbers from 0 to 1 to world space samples
    in the given spherical cap. Samples are distributed in proportion to solid
    angle.
    \param cap The output of prepareProjectedSphericalCapSampling().
    \return The sampled direction in world space.*/
    inline Vector3f sampleSphericalCap(const SphericalCap& cap, Vector2f randomNumbers) {
        Vector3f local;
        local.z = lerp(cap.minimalDot, 1.0f, randomNumbers.x);
        // Complete to a point on the sphere
        float radius = sqrt(saturate(mad(-local.z, local.z, 1.0f)));
        local.x = radius * cos(M_PI2_FLOAT*randomNumbers.y);
        local.y = radius * sin(M_PI2_FLOAT*randomNumbers.y);
        // Now turn that into a world space sample
        return local.x*cap.tangent + local.y*cap.bitangent + local.z*cap.normal;
        //return local;
    }

    float evaluate_sphere_pdf_solid_angle(
        const Vector3f&     sphere_center,
        const float         sphere_radius,
        const Vector3f&     surface_point,
        const Vector3f&     surface_normal,
        const Vector3f&     light_point,
        const float         shape_prob,
        const float         area,
        const SphericalCap& cap)
    {
        int algo = 6;

        float pdf, cosine, rcp_solid_angle, cos_theta;

        switch (algo)
        {
        case 0:
            // brighter on the edge of planes
            return (1.0f / cap.solidAngle);
        case 1:
            // not working
            // complete mess -> render fails
            return (1.0f / cap.solidAngle) * dot(-surface_normal, normalize(light_point - surface_point));
        case 2:
            //From the realtime PCS paper 
            // Dull image, but seems brighter on the edges
            cosine = dot(surface_normal, normalize(light_point - surface_point));
            return cosine * cap.solidAngle;
        case 3:
            // pretty good, but brighter on the center of the planes
            cosine = dot(surface_normal, normalize(light_point - surface_point));
            rcp_solid_angle = 1.0f / cap.solidAngle;

            pdf = rcp_solid_angle * cosine;
            return (1.0f / area) * shape_prob * pdf;
        case 4:
            // complete mess, like algo 1
            cos_theta = -dot(surface_normal, normalize(light_point - surface_point));
            rcp_solid_angle = 1.0f / cap.solidAngle;

            pdf = rcp_solid_angle * cos_theta / square_distance(light_point, surface_point);
            return area * static_cast<float>(pdf);
        case 5:
            // pretty good, but a bit too dark
            cosine = dot(surface_normal, normalize(light_point - surface_point));
            rcp_solid_angle = 1.0f / cap.solidAngle;

            pdf = rcp_solid_angle * cosine / sqrt(square_distance(light_point, surface_point));
            return (1.0f / area) * shape_prob * pdf;
        case 6:
            cosine = dot(surface_normal, normalize(light_point - surface_point));
            rcp_solid_angle = 1.0f / cap.solidAngle;

            pdf = rcp_solid_angle * cosine / square_distance(light_point, surface_point);
            return pdf;
        default:
            return 1.0f;
        }
    }
}

bool EmittingShape::sample_solid_angle(
    const ShadingPoint&     shading_point,
    const Vector2f&         s,
    const float             shape_prob,
    LightSample&            light_sample) const
{
#ifdef DO_UNIFORM
    sample_uniform(s, shape_prob, light_sample);
    return true;
#endif
    // Store a pointer to the emitting shape.
    light_sample.m_shape = this;

    const auto shape_type = get_shape_type();

    if (shape_type == TriangleShape)
    {
        assert(false);
        sample_uniform(s, shape_prob, light_sample);
        return true;
    }
    else if (shape_type == RectangleShape)
    {
        assert(false);
        sample_uniform(s, shape_prob, light_sample);
        return true;
    }
    else if (shape_type == SphereShape)
    {
        const Vector3f sphere_center(m_geom.m_sphere.m_center);

        const float    sphere_radius = static_cast<float>(m_geom.m_sphere.m_radius);
        const Vector3f surface_point(shading_point.get_point());
        const Vector3f surface_normal(shading_point.get_shading_normal());

        SphericalCap cap;
        prepareSphericalCapSampling(cap, sphere_center - surface_point, sphere_radius);
        Vector3f sampled_dir = sampleSphericalCap(cap, s);

        const Ray3f r(surface_point, sampled_dir);
        float t;
        if (!intersect_sphere(r, sphere_center, sphere_radius, t))
        {
            std::cout << "Not intersecting. wtf.\n";
            return false;
        }

        const Vector3f sampled_point = r.point_at(t);
        //const Vector3f p = sphere_center + sampledDirection;
        const Vector3f n = normalize(sampled_point - sphere_center);
        assert(is_normalized(n));
        light_sample.m_param_coords = s;

        light_sample.m_shading_normal = Vector3d(n);
        light_sample.m_geometric_normal = Vector3d(n);

        // Compute the world space position of the sample.
        light_sample.m_point = Vector3d(sampled_point);
        light_sample.m_probability = evaluate_sphere_pdf_solid_angle(
            sphere_center,
            sphere_radius,
            surface_point,
            surface_normal,
            sampled_point,
            m_shape_prob,
            m_area,
            cap);

        return true;
    }
    else if (shape_type == DiskShape)
    {
        assert(false);
        sample_uniform(s, shape_prob, light_sample);
        return true;
    }
    else
    {
        assert(false && "Unknown emitter shape type");
    }

    return false;
}

float EmittingShape::evaluate_pdf_solid_angle(
    const ShadingPoint&     shading_point,
    const Vector3d&         light_point) const
{
#ifdef DO_UNIFORM
    return evaluate_pdf_uniform();
#endif

    const auto shape_type = get_shape_type();

    if (shape_type == TriangleShape)
    {
        assert(false);
        return evaluate_pdf_uniform();
    }
    else if (shape_type == RectangleShape)
    {
        assert(false);
        return evaluate_pdf_uniform();
    }
    else if (shape_type == SphereShape)
    {
        const Vector3f sphere_center(m_geom.m_sphere.m_center);

        const float    sphere_radius = static_cast<float>(m_geom.m_sphere.m_radius);
        const Vector3f surface_point(shading_point.get_point());
        const Vector3f surface_normal(shading_point.get_shading_normal());
        const Vector3f sampled_point(light_point);

        SphericalCap cap;
        prepareSphericalCapSampling(cap, sphere_center - surface_point, sphere_radius);

        return evaluate_sphere_pdf_solid_angle(
            sphere_center,
            sphere_radius,
            surface_point,
            surface_normal,
            sampled_point,
            m_shape_prob,
            m_area,
            cap);
    }
    else if (shape_type == DiskShape)
    {
        assert(false);
        return evaluate_pdf_uniform();
    }
    else
    {
        assert(false && "Unknown emitter shape type");
        return -1.0f;
    }
}

void EmittingShape::make_shading_point(
    ShadingPoint&           shading_point,
    const Vector3d&         point,
    const Vector3d&         direction,
    const Vector2f&         param_coords,
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
            param_coords,
            get_assembly_instance(),
            get_assembly_instance()->transform_sequence().get_earliest_transform(),
            get_object_instance_index(),
            get_primitive_index(),
            m_shape_support_plane);
    }
    else if (shape_type == RectangleShape)
    {
        const Vector3d p =
            m_geom.m_rectangle.m_origin +
            static_cast<double>(param_coords[0]) * m_geom.m_rectangle.m_x +
            static_cast<double>(param_coords[1]) * m_geom.m_rectangle.m_y;

        intersector.make_procedural_surface_shading_point(
            shading_point,
            ray,
            param_coords,
            get_assembly_instance(),
            get_assembly_instance()->transform_sequence().get_earliest_transform(),
            get_object_instance_index(),
            get_primitive_index(),
            p,
            m_geom.m_rectangle.m_geometric_normal,
            m_geom.m_rectangle.m_x,
            cross(m_geom.m_rectangle.m_x, m_geom.m_rectangle.m_geometric_normal));
    }
    else if (shape_type == SphereShape)
    {
        const double theta = static_cast<double>(param_coords[0]);
        const double phi = static_cast<double>(param_coords[1]);

        const Vector3d n = Vector3d::make_unit_vector(theta, phi);
        const Vector3d p = m_geom.m_sphere.m_center + m_geom.m_sphere.m_radius * n;

        const Vector3d dpdu(-TwoPi<double>() * n.y, TwoPi<double>() + n.x, 0.0);
        const Vector3d dpdv = cross(dpdu, n);

        intersector.make_procedural_surface_shading_point(
            shading_point,
            ray,
            param_coords,
            get_assembly_instance(),
            get_assembly_instance()->transform_sequence().get_earliest_transform(),
            get_object_instance_index(),
            get_primitive_index(),
            p,
            n,
            dpdu,
            dpdv);
    }
    else if (shape_type == DiskShape)
    {
        const Vector3d p =
            m_geom.m_disk.m_center +
            static_cast<double>(param_coords[0]) * m_geom.m_disk.m_x +
            static_cast<double>(param_coords[1]) * m_geom.m_disk.m_y;

        intersector.make_procedural_surface_shading_point(
            shading_point,
            ray,
            param_coords,
            get_assembly_instance(),
            get_assembly_instance()->transform_sequence().get_earliest_transform(),
            get_object_instance_index(),
            get_primitive_index(),
            p,
            m_geom.m_disk.m_geometric_normal,
            m_geom.m_disk.m_x,
            cross(m_geom.m_disk.m_x, m_geom.m_disk.m_geometric_normal));
    }
    else
    {
        assert(false && "Unknown emitter shape type");
    }
}

void EmittingShape::estimate_flux()
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

    m_average_flux = 1.0f;
    m_max_flux = 1.0f;
}

}   // namespace renderer
