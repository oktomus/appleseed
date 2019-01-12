
//
// This source file is part of appleseed.
// Visit https://appleseedhq.net/ for additional information and resources.
//
// This software is released under the MIT license.
//
// Copyright (c) 2018 Francois Beaune, The appleseedhq Organization
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
#include "lightpathstream.h"

// appleseed.renderer headers.
#include "renderer/kernel/lighting/lighttypes.h"
#include "renderer/kernel/lighting/pathvertex.h"
#include "renderer/kernel/rendering/pixelcontext.h"
#include "renderer/kernel/shading/shadingpoint.h"
#include "renderer/modeling/camera/camera.h"
#include "renderer/modeling/color/colorspace.h"
#include "renderer/modeling/light/light.h"
#include "renderer/modeling/scene/assembly.h"
#include "renderer/modeling/scene/assemblyinstance.h"
#include "renderer/modeling/scene/containers.h"
#include "renderer/modeling/scene/objectinstance.h"

// appleseed.foundation headers.
#include "foundation/utility/memory.h"
#include "foundation/utility/otherwise.h"

// Standard headers.
#include <cassert>

using namespace foundation;

namespace renderer
{

void LightPathStream::clear()
{
    clear_release_memory(m_events);
    clear_release_memory(m_hit_reflector_data);
    clear_release_memory(m_hit_emitter_data);
    clear_release_memory(m_sampled_emitter_data);

    clear_release_memory(m_paths);
    clear_release_memory(m_vertices);
}

void LightPathStream::begin_path(
    const PixelContext&     pixel_context,
    const Camera*           camera,
    const Vector3d&         camera_vertex_position)
{
    assert(m_events.empty());
    assert(m_hit_reflector_data.empty());
    assert(m_hit_emitter_data.empty());
    assert(m_sampled_emitter_data.empty());
    assert(camera != nullptr);

    m_camera = camera;
    m_camera_vertex_position = Vector3f(camera_vertex_position);
    m_pixel_coords = pixel_context.get_pixel_coords();
    m_sample_position = Vector2f(pixel_context.get_sample_position());
}

void LightPathStream::hit_reflector(const PathVertex& vertex)
{
    assert(m_hit_reflector_data.size() < 256);

    Event event;
    event.m_type = EventType::HitReflector;
    event.m_data_index = static_cast<uint8>(m_hit_reflector_data.size());
    m_events.push_back(event);

    HitReflectorData data;
    data.m_entity = &vertex.m_shading_point->get_object_instance();
    data.m_vertex_position = Vector3f(vertex.get_point());
    data.m_path_throughput = vertex.m_throughput.to_rgb(g_std_lighting_conditions);
    m_hit_reflector_data.push_back(data);
}

void LightPathStream::hit_emitter(
    const PathVertex&       vertex,
    const Spectrum&         emitted_radiance)
{
    assert(m_hit_emitter_data.size() < 256);

    Event event;
    event.m_type = EventType::HitEmitter;
    event.m_data_index = static_cast<uint8>(m_hit_emitter_data.size());
    m_events.push_back(event);

    HitEmitterData data;
    data.m_entity = &vertex.m_shading_point->get_object_instance();
    data.m_vertex_position = Vector3f(vertex.get_point());
    data.m_path_throughput = vertex.m_throughput.to_rgb(g_std_lighting_conditions);
    data.m_emitted_radiance = emitted_radiance.to_rgb(g_std_lighting_conditions);
    m_hit_emitter_data.push_back(data);
}

void LightPathStream::sampled_emitting_shape(
    const EmittingShape*    shape,
    const Vector3d&         emission_position,
    const Spectrum&         material_value,
    const Spectrum&         emitted_radiance)
{
    assert(shape != nullptr);

    Event event;
    event.m_type = EventType::SampledEmitter;
    event.m_data_index = static_cast<uint8>(m_sampled_emitter_data.size());
    m_events.push_back(event);

    SampledEmitterData data;
    data.m_entity =
        shape->get_assembly_instance()->get_assembly().object_instances().get_by_index(
            shape->get_object_instance_index());
    data.m_vertex_position = Vector3f(emission_position);
    data.m_material_value = material_value.to_rgb(g_std_lighting_conditions);
    data.m_emitted_radiance = emitted_radiance.to_rgb(g_std_lighting_conditions);
    m_sampled_emitter_data.push_back(data);
}

void LightPathStream::sampled_non_physical_light(
    const Light*            light,
    const Vector3d&         emission_position,
    const Spectrum&         material_value,
    const Spectrum&         emitted_radiance)
{
    assert(light != nullptr);

    Event event;
    event.m_type = EventType::SampledEmitter;
    event.m_data_index = static_cast<uint8>(m_sampled_emitter_data.size());
    m_events.push_back(event);

    SampledEmitterData data;
    data.m_entity = light;
    data.m_vertex_position = Vector3f(emission_position);
    data.m_material_value = material_value.to_rgb(g_std_lighting_conditions);
    data.m_emitted_radiance = emitted_radiance.to_rgb(g_std_lighting_conditions);
    m_sampled_emitter_data.push_back(data);
}

void LightPathStream::end_path()
{
    // Ignore paths that fall outside of the supported range.
    if (m_pixel_coords.x >= 0 &&
        m_pixel_coords.y >= 0 &&
        m_pixel_coords.x < 65536 &&
        m_pixel_coords.y < 65536)
    {
        for (size_t i = 0, e = m_events.size(); i < e; ++i)
        {
            switch (m_events[i].m_type)
            {
              case EventType::HitReflector:
                break;

              case EventType::HitEmitter:
                create_path_from_hit_emitter(i);
                break;

              case EventType::SampledEmitter:
                create_path_from_sampled_emitter(i);
                break;

              assert_otherwise;
            }
        }
    }

    clear_keep_memory(m_events);
    clear_keep_memory(m_hit_reflector_data);
    clear_keep_memory(m_hit_emitter_data);
    clear_keep_memory(m_sampled_emitter_data);
}

void LightPathStream::create_path_from_hit_emitter(const size_t emitter_event_index)
{
    const auto& hit_emitter_event = m_events[emitter_event_index];
    const auto& hit_emitter_data = m_hit_emitter_data[hit_emitter_event.m_data_index];

    // Create path.
    StoredPath stored_path;
    stored_path.m_pixel_coords = Vector2u16(m_pixel_coords);
    stored_path.m_sample_position = m_sample_position;
    stored_path.m_vertex_begin_index = static_cast<uint32>(m_vertices.size());

    // Emitter vertex.
    StoredPathVertex emitter_vertex;
    emitter_vertex.m_entity = hit_emitter_data.m_entity;
    emitter_vertex.m_position = hit_emitter_data.m_vertex_position;
    emitter_vertex.m_radiance = hit_emitter_data.m_emitted_radiance;
    m_vertices.push_back(emitter_vertex);

    Color3f current_radiance = hit_emitter_data.m_emitted_radiance;
    Color3f prev_throughput = hit_emitter_data.m_path_throughput;

    // Walk back the list of events and create path vertices.
    for (size_t i = emitter_event_index; i > 0; --i)
    {
        const auto event_index = i - 1;
        const auto& event = m_events[event_index];
        if (event.m_type == EventType::HitReflector ||
            event.m_type == EventType::HitEmitter)
        {
            const auto& event_data = get_reflector_data(event_index);

            // Reflector vertex.
            StoredPathVertex reflector_vertex;
            reflector_vertex.m_entity = event_data.m_entity;
            reflector_vertex.m_position = event_data.m_vertex_position;
            reflector_vertex.m_radiance = current_radiance;
            m_vertices.push_back(reflector_vertex);

            // Update current radiance.
            const auto throughput = event_data.m_path_throughput;
            current_radiance *= prev_throughput;
            current_radiance /= throughput;
            prev_throughput = throughput;
        }
    }

    // Camera vertex.
    StoredPathVertex camera_vertex;
    camera_vertex.m_entity = m_camera;
    camera_vertex.m_position = m_camera_vertex_position;
    camera_vertex.m_radiance = current_radiance;
    m_vertices.push_back(camera_vertex);

    // Store path.
    stored_path.m_vertex_end_index = static_cast<uint32>(m_vertices.size());
    m_paths.push_back(stored_path);
}

void LightPathStream::create_path_from_sampled_emitter(const size_t emitter_event_index)
{
    const auto& sampled_emitter_event = m_events[emitter_event_index];
    const auto& sampled_emitter_data = m_sampled_emitter_data[sampled_emitter_event.m_data_index];

    // Create path.
    StoredPath stored_path;
    stored_path.m_pixel_coords = Vector2u16(m_pixel_coords);
    stored_path.m_sample_position = m_sample_position;
    stored_path.m_vertex_begin_index = static_cast<uint32>(m_vertices.size());

    // Emitter vertex.
    StoredPathVertex emitter_vertex;
    emitter_vertex.m_entity = sampled_emitter_data.m_entity;
    emitter_vertex.m_position = sampled_emitter_data.m_vertex_position;
    emitter_vertex.m_radiance = sampled_emitter_data.m_emitted_radiance;
    m_vertices.push_back(emitter_vertex);

    // Find the last event that is not a SampledEmitter one.
    assert(emitter_event_index > 0);
    size_t last_scattering_event_index = emitter_event_index - 1;
    while (m_events[last_scattering_event_index].m_type == EventType::SampledEmitter)
        --last_scattering_event_index;

    Color3f current_radiance = sampled_emitter_data.m_emitted_radiance;
    Color3f prev_throughput =
        sampled_emitter_data.m_material_value *
        get_reflector_data(last_scattering_event_index).m_path_throughput;

    // Walk back the list of events and create path vertices.
    for (size_t i = last_scattering_event_index + 1; i > 0; --i)
    {
        const auto event_index = i - 1;
        const auto& event = m_events[event_index];
        if (event.m_type == EventType::HitReflector ||
            event.m_type == EventType::HitEmitter)
        {
            const auto& event_data = get_reflector_data(event_index);

            // Reflector vertex.
            StoredPathVertex reflector_vertex;
            reflector_vertex.m_entity = event_data.m_entity;
            reflector_vertex.m_position = event_data.m_vertex_position;
            reflector_vertex.m_radiance = current_radiance;
            m_vertices.push_back(reflector_vertex);

            // Update current radiance.
            const auto throughput = event_data.m_path_throughput;
            current_radiance *= prev_throughput;
            current_radiance /= throughput;
            prev_throughput = throughput;
        }
    }

    // Camera vertex.
    StoredPathVertex camera_vertex;
    camera_vertex.m_entity = m_camera;
    camera_vertex.m_position = m_camera_vertex_position;
    camera_vertex.m_radiance = current_radiance;
    m_vertices.push_back(camera_vertex);

    // Store path.
    stored_path.m_vertex_end_index = static_cast<uint32>(m_vertices.size());
    m_paths.push_back(stored_path);
}

const LightPathStream::HitReflectorData& LightPathStream::get_reflector_data(const size_t event_index) const
{
    const auto& event = m_events[event_index];

    assert(
        event.m_type == EventType::HitReflector ||
        event.m_type == EventType::HitEmitter);

    return
        event.m_type == EventType::HitReflector
            ? m_hit_reflector_data[event.m_data_index]
            : m_hit_emitter_data[event.m_data_index];
}

}   // namespace renderer
