
#define DO_UNIFORM
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
    }
    else if (shape_type == RectangleShape)
    {
        assert(false);
        sample_uniform(s, shape_prob, light_sample);
    }
    else if (shape_type == SphereShape)
    {
        // todo: do uniform sampling if the surface point is inside the sphere.
        // wip
        /*

        // c: sphere center.
        const Vector3d& c = m_geom.m_sphere.m_center;

        // r: sphere radius.
        const float r = m_geom.m_sphere.m_radius;

        // x: the point being illuminated.
        const Vector3d& x = shading_point.get_point();

        // Build a basis (w, v, u) with w facing in the direction of the sphere.
        const Vector3d w = normalize(c - x);
        */

        /*
        const Vector3d& c = m_geom.m_sphere.m_center;
        const Vector3d& x = shading_point.get_point();
        const double    r = m_geom.m_sphere.m_radius;

        const double dist_to_center = norm(c - x);
        const Vector3d w = normalize(c - x);

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
        const double q = sqrt(1.0 - square(r / dist_to_center));
        const double    theta = acos(1.0 - static_cast<double>(s[0]) + static_cast<double>(s[0]) * q);
        const double    phi = TwoPi<double>() * static_cast<double>(s[1]);
        const Vector3d  local = Vector3d::make_unit_vector(theta, phi);

        // Compute world space sample position.
        const Vector3d nwp = local * world;
        const Vector3d xp = x - c;

        const double b = 2.0 * dot(nwp, xp);
        const double cd = dot(xp, xp) - r * r;

        double t;

        const double root = b * b - 4.0 * cd;
        if (root < 0.0)
        {
            // Project x onto v.
            const Vector3d projected_x = (dot(xp, nwp) / dot(nwp, nwp)) * nwp;
            t = norm(projected_x);
        }
        else if (root == 0.0)
        {
            t = -0.5 * b;
        }
        else
        {
            const double q = (b > 0.0) ? -0.5 * (b + sqrt(root)) : -0.5 * (b - sqrt(root));
            const double t0 = q;
            const double t1 = cd / q;
            t = min(t0, t1);
        }

        light_sample.m_point = x + t * nwp;

        // Compute the normal at the sample.
        light_sample.m_shading_normal = normalize(
            light_sample.m_point - m_geom.m_sphere.m_center);
        light_sample.m_geometric_normal = light_sample.m_shading_normal;
        light_sample.m_param_coords[0] = static_cast<float>(theta);
        light_sample.m_param_coords[1] = static_cast<float>(phi);

        // Compute the probability.
        const float pdf = 1.0f / (TwoPi<float>() * (1.0f - static_cast<float>(q)));
        light_sample.m_probability = shape_prob * pdf;
        */
        /*
        const Vector3d& sphere_center = m_geom.m_sphere.m_center;
        const double    sphere_radius = m_geom.m_sphere.m_radius;
        const Vector3d& surface_point = shading_point.get_point();
        const Vector3d& surface_normal = shading_point.get_shading_normal();

        // Create a coordinate system.
        //const double dist_to_center = norm(sphere_center - surface_point);
        //const Vector3d wc = normalize(sphere_center - surface_point);
        const Vector3d basisZ = surface_normal;
        const Vector3d basisY = normalize(cross(basisZ, (sphere_center - surface_point)));
        const Vector3d basisX = cross(basisY, basisZ);
        -
        // Project a direction.
        const double s0 = static_cast<float>(s[0]);
        const double s1 = static_cast<float>(s[1]);
        const Vector3d wdir = basisX * s0 + basisY * s1 + basisZ * sqrt(1.0 - square(s0) - square(s1));

        // Compute aperture parameters.
        const Vector3d apterure_center = normalize(sphere_center - surface_point);
        const double aperture_alpha = std::asin(sphere_radius / norm(sphere_center - surface_point));
        const double apeture_beta = std::asin(dot(basisZ, apterure_center - surface_point));

        // Create the map.
        PSCM::PSCMaps<double> map;
        map.initialize(aperture_alpha, apeture_beta, true);
        const double area = map.get_area();
        double outx, outy;
        map.eval_map(s[0], s[1], outx, outy);
        const Vector3d sample_dir_wc = outx * basisX + outy * basisY + sqrt(1.0 - outx * outx - outy * outy) * basisZ;

        const Ray3d r(surface_point, sample_dir_wc);
        double t;
        const bool intersects = intersect_sphere(r, sphere_center, sphere_radius, t);
        assert(intersects);
        const Vector3d p = r.point_at(t);
        const Vector3d n = p - sphere_center;
        assert(is_normalized(n));
        light_sample.m_param_coords = s;

        // Set the world space shading and geometric normals.
        light_sample.m_shading_normal = n;
        light_sample.m_geometric_normal = n;

        // Compute the world space position of the sample.
        light_sample.m_point = p;
        light_sample.m_probability = shape_prob * (1.0f / static_cast<float>(area));
        */
        const Vector3f sphere_center(
            static_cast<float>(m_geom.m_sphere.m_center.x),
            static_cast<float>(m_geom.m_sphere.m_center.y),
            static_cast<float>(m_geom.m_sphere.m_center.x));

        const float    sphere_radius = static_cast<float>(m_geom.m_sphere.m_radius);
        const Vector3f surface_point(
            static_cast<float>(shading_point.get_point().x),
            static_cast<float>(shading_point.get_point().y),
            static_cast<float>(shading_point.get_point().z));
        const Vector3f surface_normal(
            static_cast<float>(shading_point.get_shading_normal().x),
            static_cast<float>(shading_point.get_shading_normal().y),
            static_cast<float>(shading_point.get_shading_normal().z));

        SphericalCap cap;
        prepareSphericalCapSampling(cap, sphere_center - surface_point, sphere_radius);
        Vector3f sampledDirection = sampleSphericalCap(cap, s);

        // Set the world space shading and geometric normals.
        //const Ray3f r(surface_point, sampledDirection);
        //float t;
        //if (intersect_sphere(r, sphere_center, sphere_radius, t))
        {
            //const Vector3f p = r.point_at(t);
            const Vector3f p = sphere_center + sampledDirection;
            const Vector3f n = sampledDirection;
            assert(is_normalized(n));
            light_sample.m_param_coords = s;

            light_sample.m_shading_normal.x = static_cast<double>(n.x);
            light_sample.m_shading_normal.y = static_cast<double>(n.y);
            light_sample.m_shading_normal.z = static_cast<double>(n.z);
            light_sample.m_geometric_normal = light_sample.m_shading_normal;

            // Compute the world space position of the sample.
            light_sample.m_point.x = static_cast<double>(p.x);
            light_sample.m_point.y = static_cast<double>(p.y);
            light_sample.m_point.z = static_cast<double>(p.z);
            //light_sample.m_probability = shape_prob * cosine * cap.solidAngle;
            const double cosine = static_cast<double>(dot(surface_normal, sampledDirection));

            //if (cosine <= 0.0f)
                //return false;
            const double rcp_solid_angle = 1.0 / static_cast<double>(cap.solidAngle);
            const double pdf = rcp_solid_angle * cosine;//static_cast<double>(cap.solidAngle) * cosine;
            light_sample.m_probability = shape_prob * static_cast<float>(pdf);
            return true;
            //finalColor += (cosine*cap.solidAngle)*(brdf*pSphereLight->mSurfaceRadiance);
        }
    }
    else if (shape_type == DiskShape)
    {
        assert(false);
        sample_uniform(s, shape_prob, light_sample);
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

    const float shape_probability = m_shape_prob;

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
        const Vector3f sphere_center(
            static_cast<float>(m_geom.m_sphere.m_center.x),
            static_cast<float>(m_geom.m_sphere.m_center.y),
            static_cast<float>(m_geom.m_sphere.m_center.x));

        const float    sphere_radius = static_cast<float>(m_geom.m_sphere.m_radius);
        const Vector3f surface_point(
            static_cast<float>(shading_point.get_point().x),
            static_cast<float>(shading_point.get_point().y),
            static_cast<float>(shading_point.get_point().z));
        const Vector3f surface_normal(
            static_cast<float>(shading_point.get_shading_normal().x),
            static_cast<float>(shading_point.get_shading_normal().y),
            static_cast<float>(shading_point.get_shading_normal().z));

        SphericalCap cap;
        Vector3d d = shading_point.get_point() - light_point;
        const double d_norm = norm(d);
        d /= d_norm;
        prepareSphericalCapSampling(cap, sphere_center - surface_point, sphere_radius);
        const Vector3d p = light_point;
        assert(is_normalized(n));
        const double cosine = dot(shading_point.get_shading_normal(), d);
        const double rcp_solid_angle = 1.0 / static_cast<double>(cap.solidAngle);

        const double pdf = rcp_solid_angle * cosine;//static_cast<double>(cap.solidAngle) * cos_theta;
        return shape_probability * static_cast<float>(pdf);


        const Ray3d r(shading_point.get_point(), d);
        double t;
        if (intersect_sphere_unit_direction(
            r,
            m_geom.m_sphere.m_center,
            m_geom.m_sphere.m_radius,
            t))
        {
        }

        return 0.0f;

        // hey, compute uvs.
        /*
        const Vector3d dir = normalize(light_point - surface_point);
        const Ray3d r(surface_point, dir);

        double t;

        bool intersects = intersect_sphere_unit_direction(
            r,
            m_geom.m_sphere.m_center,
            m_geom.m_sphere.m_radius,
            t);

        assert(intersects);

        if (intersects)
        {
            const Vector3d p(r.point_at(t) / sphere_radius);
            const double s0 = atan2(-p.z, p.x) * RcpTwoPi<double>();
            const double s1 = 1.0 - (acos(p.y) * RcpPi<double>());

            const Vector3d basisZ = surface_normal;
            const Vector3d basisY = normalize(cross(basisZ, (sphere_center - surface_point)));
            const Vector3d basisX = cross(basisY, basisZ);

            const Vector3d wdir = basisX * s0 + basisY * s1 + basisZ * sqrt(1.0 - square(s0) - square(s1));

            // Compute aperture parameters.
            const Vector3d apterure_center = normalize(sphere_center - surface_point);
            const double aperture_alpha = std::asin(sphere_radius / norm(sphere_center - surface_point));
            const double apeture_beta = std::asin(dot(basisZ, apterure_center - surface_point));

            // Create the map.
            PSCM::PSCMaps<double> map;
            map.initialize(aperture_alpha, apeture_beta, true);
            const double area = map.get_area();
            return (1.0f / static_cast<float>(area));
        }

        std::cout << "not intersecting, lol\n";
        */

        return -1.0f;

        // wip
        assert(false);
        return 0.0f;
        /*
        const Vector3d& center = m_geom.m_sphere.m_center;
        const double radius_sqr = square(m_geom.m_sphere.m_radius);

        const double dist_to_center = square_distance(surface_point, center);
        if (dist_to_center <= radius_sqr)
            return shape_probability * FourPi<float>();

        const float sin_theta_sqr = static_cast<float>(radius_sqr / dist_to_center);
        const float cos_theta = sqrt(max(0.0f, 1.0f - sin_theta_sqr));

        return shape_probability * TwoPi<float>() * (1.0f - cos_theta);
        */
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
