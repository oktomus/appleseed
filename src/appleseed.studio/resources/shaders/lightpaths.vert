//
// This source file is part of appleseed.
// Visit https://appleseedhq.net/ for additional information and resources.
//
// This software is released under the MIT license.
//
// Copyright (c) 2019 Gray Olson, The appleseedhq Organization
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

#version 410

const float AA_BUFFER_SIZE = 1.0;

layout(location = 0) in vec3 v_previous;
layout(location = 1) in vec3 v_position;
layout(location = 2) in vec3 v_next;
layout(location = 3) in float v_luminance;
layout(location = 4) in int v_direction_start_end; // flag defining which part of a light path is currently drawn.
layout(location = 5) in vec3 v_color;
layout(location = 6) in vec3 v_surface_normal;

uniform mat4 u_proj;
uniform mat4 u_view;
uniform vec2 u_res;  // resolution of the frame.
uniform float u_max_luminance;
uniform float u_max_thickness;
uniform float u_min_thickness;

uniform int u_first_selected;
uniform int u_last_selected;

flat out vec4 f_color;
out float f_aa_norm;
flat out float f_thickness;
flat out float f_total_thickness;
flat out float f_aspect_expansion_len;

const float CLIPPING_PREVENTION_FACTOR = 0.05;

//
// Reference:
// - https://mattdesl.svbtle.com/drawing-lines-is-hard#screenspace-projected-lines_2
//

#define DRAW_START_PATH 1 // Drawing the start of the light path
#define DRAW_MIDDLE_PATH 2 // Drawing a point in the middle of the light path
#define DRAW_END_PATH 3 // Drawing the end of the light path

int get_drawing_mode()
{
    if ((v_direction_start_end & 2) == 2)
    {
        //starting point uses (next - current)
        return DRAW_START_PATH;
    }
    else if ((v_direction_start_end & 4) == 4)
    {
        //ending point uses (current - previous)
        return DRAW_END_PATH;
    }
    else
    {
        // middle point uses (next - current)
        return DRAW_MIDDLE_PATH;
    }
}

// Each point on the light path is duplicated to render a real line.
// The duplicated vertex of each light path point is flagged.
bool is_second_point()
{
    return ((v_direction_start_end & 1) == 1);
}

void main() {
    // Aspect ratio correction is applied on
    // screen points (only on the X axis).
    float aspect = u_res.x / u_res.y;
    vec2 aspect_correction = vec2(aspect, 1.0);

    // Project points.
    // The currently drawn point is offset
    // from the surface to ensure path
    // doesn't go through it.
    mat4 vp = u_proj * u_view;
    vec4 curr_proj = vp * vec4(v_position + v_surface_normal * CLIPPING_PREVENTION_FACTOR, 1.0);
    vec4 prev_proj = vp * vec4(v_previous, 1.0);
    vec4 next_proj = vp * vec4(v_next, 1.0);

    // Project points in screenspace.
    vec2 curr_screen = curr_proj.xy / curr_proj.w;
    vec2 prev_screen = prev_proj.xy / prev_proj.w;
    vec2 next_screen = next_proj.xy / next_proj.w;

    // Apply aspect ratio correction.
    curr_screen *= aspect_correction;
    prev_screen *= aspect_correction;
    next_screen *= aspect_correction;

    int drawing_mode = get_drawing_mode();
    bool is_second_point = is_second_point();

    // Compute current line directionection.
    // Depending on which part of the path we
    // are drawing (start, middle or end), we
    // compute it differently.
    vec2 line_direction =
        drawing_mode == DRAW_START_PATH ? normalize(next_screen - curr_screen) :
        drawing_mode == DRAW_END_PATH ? normalize(curr_screen - prev_screen) :
        normalize(next_screen - curr_screen);

    // Compute the normal of the line.
    vec2 line_normal = vec2(-line_direction.y, line_direction.x);

    vec4 normal_clip = vp * vec4(v_surface_normal, 0.0);
    normal_clip.xy *= aspect_correction;
    normal_clip = normalize(normal_clip);
    vec2 tang_clip = vec2(-normal_clip.y, normal_clip.x);

    float tdp = dot(tang_clip, line_normal);
    vec2 expansion = line_normal;
    if (tdp > 0.05)
        expansion = tang_clip / tdp;

    vec2 norm_exp = normalize(expansion);
    vec2 res_exp_line_direction = vec2(norm_exp.x * u_res.x, norm_exp.y * u_res.y);
    f_aspect_expansion_len = length(res_exp_line_direction);

    f_thickness = (max(max(min(v_luminance / u_max_luminance, 1.0), 0.0) * u_max_thickness, u_min_thickness) / 2.0) / f_aspect_expansion_len;

    f_total_thickness = f_thickness + AA_BUFFER_SIZE / f_aspect_expansion_len;

    expansion *= f_total_thickness;

    // Reverse expansion line_directionection for the duplicated point.
    if (is_second_point) expansion *= -1.0;

    gl_Position = vec4((curr_screen + expansion) / aspect_correction, curr_proj.z / curr_proj.w, 1.0);
    f_aa_norm = is_second_point ? 1.0 : -1.0;

    bool is_selected = gl_VertexID >= u_first_selected && gl_VertexID < u_last_selected;
    float a = is_selected ? 1.0 : 0.05;
    f_color = vec4(v_color, a);
}
