
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
#include "frame.h"

// appleseed.renderer headers.
#include "renderer/global/globallogger.h"
#include "renderer/kernel/aov/aovsettings.h"
#include "renderer/kernel/aov/imagestack.h"
#include "renderer/kernel/denoising/denoiser.h"
#include "renderer/kernel/rendering/permanentshadingresultframebufferfactory.h"
#include "renderer/kernel/rendering/shadingresultframebuffer.h"
#include "renderer/modeling/aov/aov.h"
#include "renderer/modeling/aov/aovfactoryregistrar.h"
#include "renderer/modeling/aov/denoiseraov.h"
#include "renderer/modeling/aov/iaovfactory.h"
#include "renderer/modeling/postprocessingstage/postprocessingstage.h"
#include "renderer/utility/bbox.h"
#include "renderer/utility/filesystem.h"
#include "renderer/utility/paramarray.h"

// appleseed.foundation headers.
#include "foundation/image/color.h"
#include "foundation/image/genericimagefilewriter.h"
#include "foundation/image/genericprogressiveimagefilereader.h"
#include "foundation/image/image.h"
#include "foundation/image/imageattributes.h"
#include "foundation/image/pixel.h"
#include "foundation/image/tile.h"
#include "foundation/math/scalar.h"
#include "foundation/platform/defaulttimers.h"
#include "foundation/platform/path.h"
#include "foundation/utility/containers/dictionary.h"
#include "foundation/utility/api/specializedapiarrays.h"
#include "foundation/utility/iostreamop.h"
#include "foundation/utility/job/iabortswitch.h"
#include "foundation/utility/stopwatch.h"
#include "foundation/utility/string.h"

// Boost headers.
#include "boost/filesystem.hpp"

// BCD headers.
#include "bcd/DeepImage.h"
#include "bcd/ImageIO.h"

// Standard headers.
#include <algorithm>
#include <exception>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

using namespace bcd;
using namespace foundation;
using namespace std;
namespace bf = boost::filesystem;
namespace bsys = boost::system;

namespace renderer
{

//
// Frame class implementation.
//

namespace
{
    const UniqueID g_class_uid = new_guid();
}

UniqueID Frame::get_class_uid()
{
    return g_class_uid;
}

struct Frame::Impl
{
    // Parameters.
    size_t                          m_frame_width;
    size_t                          m_frame_height;
    size_t                          m_tile_width;
    size_t                          m_tile_height;
    string                          m_filter_name;
    float                           m_filter_radius;
    unique_ptr<Filter2f>            m_filter;
    AABB2u                          m_crop_window;
    ParamArray                      m_render_info;
    DenoisingMode                   m_denoising_mode;
    bool                            m_checkpoint;
    string                          m_checkpoint_path;

    // When resuming a render, first pass index should be
    // the number of the resumed render's pass + 1.
    // Otherwise it's 0.
    size_t                          m_initial_pass;

    // Child entities.
    AOVContainer                    m_aovs;
    AOVContainer                    m_internal_aovs;
    PostProcessingStageContainer    m_post_processing_stages;

    // Images.
    unique_ptr<Image>               m_image;
    unique_ptr<ImageStack>          m_aov_images;
    DenoiserAOV*                    m_denoiser_aov;
};

Frame::Frame(
    const char*         name,
    const ParamArray&   params,
    const AOVContainer& aovs)
  : Entity(g_class_uid, params)
  , impl(new Impl())
{
    set_name(name);

    extract_parameters();

    // Create the underlying image.
    impl->m_image.reset(
        new Image(
            impl->m_frame_width,
            impl->m_frame_height,
            impl->m_tile_width,
            impl->m_tile_height,
            4,
            PixelFormatFloat));

    // Retrieve the image properties.
    m_props = impl->m_image->properties();

    // Create the image stack for AOVs.
    impl->m_aov_images.reset(
        new ImageStack(
            impl->m_frame_width,
            impl->m_frame_height,
            impl->m_tile_width,
            impl->m_tile_height));

    if (aovs.size() > MaxAOVCount)
    {
        RENDERER_LOG_WARNING(
            "could not create all aovs, keeping the first (" FMT_SIZE_T ") aovs.",
            MaxAOVCount);
    }

    // Copy and add AOVs.
    const AOVFactoryRegistrar aov_registrar;
    for (size_t i = 0, e = min(aovs.size(), MaxAOVCount); i < e; ++i)
    {
        const AOV* original_aov = aovs.get_by_index(i);
        const IAOVFactory* aov_factory = aov_registrar.lookup(original_aov->get_model());
        assert(aov_factory);

        auto_release_ptr<AOV> aov = aov_factory->create(original_aov->get_parameters());
        aov->create_image(
            impl->m_frame_width,
            impl->m_frame_height,
            impl->m_tile_width,
            impl->m_tile_height,
            aov_images());
        impl->m_aovs.insert(aov);
    }

    // Create internal AOVs.
    if (impl->m_denoising_mode != DenoisingMode::Off)
    {
        auto_release_ptr<DenoiserAOV> aov = DenoiserAOVFactory::create();
        impl->m_denoiser_aov = aov.get();

        aov->create_image(
            impl->m_frame_width,
            impl->m_frame_height,
            impl->m_tile_width,
            impl->m_tile_height,
            aov_images());

        impl->m_internal_aovs.insert(auto_release_ptr<AOV>(aov));
    }
    else
        impl->m_denoiser_aov = nullptr;

    impl->m_initial_pass = 0;
}

Frame::~Frame()
{
    delete impl;
}

void Frame::release()
{
    delete this;
}

void Frame::print_settings()
{
    const char* camera_name = get_active_camera_name();

    RENDERER_LOG_INFO(
        "frame \"%s\" settings:\n"
        "  camera                        \"%s\"\n"
        "  resolution                    %s x %s\n"
        "  tile size                     %s x %s\n"
        "  filter                        %s\n"
        "  filter size                   %f\n"
        "  crop window                   (%s, %s)-(%s, %s)\n"
        "  denoising mode                %s\n"
        "  checkpoint                    %s",
        get_path().c_str(),
        camera_name ? camera_name : "none",
        pretty_uint(impl->m_frame_width).c_str(),
        pretty_uint(impl->m_frame_height).c_str(),
        pretty_uint(impl->m_tile_width).c_str(),
        pretty_uint(impl->m_tile_height).c_str(),
        impl->m_filter_name.c_str(),
        impl->m_filter_radius,
        pretty_uint(impl->m_crop_window.min[0]).c_str(),
        pretty_uint(impl->m_crop_window.min[1]).c_str(),
        pretty_uint(impl->m_crop_window.max[0]).c_str(),
        pretty_uint(impl->m_crop_window.max[1]).c_str(),
        impl->m_denoising_mode == DenoisingMode::Off ? "off" :
        impl->m_denoising_mode == DenoisingMode::WriteOutputs ? "write outputs" : "denoise",
        impl->m_checkpoint ? impl->m_checkpoint_path.c_str() : "off");
}

AOVContainer& Frame::aovs() const
{
    return impl->m_aovs;
}

PostProcessingStageContainer& Frame::post_processing_stages() const
{
    return impl->m_post_processing_stages;
}

const char* Frame::get_active_camera_name() const
{
    if (m_params.strings().exist("camera"))
        return m_params.strings().get("camera");

    return nullptr;
}

Image& Frame::image() const
{
    return *impl->m_image.get();
}

void Frame::clear_main_and_aov_images()
{
    impl->m_image->clear(Color4f(0.0));

    for (size_t i = 0, e = aovs().size(); i < e; ++i)
        aovs().get_by_index(i)->clear_image();

    for (size_t i = 0, e = internal_aovs().size(); i < e; ++i)
        internal_aovs().get_by_index(i)->clear_image();
}

ImageStack& Frame::aov_images() const
{
    return *impl->m_aov_images;
}

const Filter2f& Frame::get_filter() const
{
    return *impl->m_filter.get();
}

size_t Frame::get_initial_pass() const
{
    return impl->m_initial_pass;
}

void Frame::reset_crop_window()
{
    impl->m_crop_window =
        AABB2u(
            Vector2u(0, 0),
            Vector2u(impl->m_frame_width - 1, impl->m_frame_height - 1));

    m_params.strings().remove("crop_window");
}

bool Frame::has_crop_window() const
{
    return
        impl->m_crop_window.min.x > 0 ||
        impl->m_crop_window.min.y > 0 ||
        impl->m_crop_window.max.x < impl->m_frame_width - 1 ||
        impl->m_crop_window.max.y < impl->m_frame_height - 1;
}

void Frame::set_crop_window(const AABB2u& crop_window)
{
    impl->m_crop_window = crop_window;

    m_params.insert("crop_window", crop_window);
}

const AABB2u& Frame::get_crop_window() const
{
    return impl->m_crop_window;
}

void Frame::collect_asset_paths(StringArray& paths) const
{
    for (const AOV& aov : aovs())
        aov.collect_asset_paths(paths);

    for (const PostProcessingStage& stage : post_processing_stages())
        stage.collect_asset_paths(paths);
}

void Frame::update_asset_paths(const StringDictionary& mappings)
{
    for (AOV& aov : aovs())
        aov.update_asset_paths(mappings);

    for (PostProcessingStage& stage : post_processing_stages())
        stage.update_asset_paths(mappings);
}

bool Frame::on_frame_begin(
    const Project&                                  project,
    const BaseGroup*                                parent,
    OnFrameBeginRecorder&                           recorder,
    IAbortSwitch*                                   abort_switch)
{
    for (AOV& aov : aovs())
    {
        if (is_aborted(abort_switch))
            return false;

        if (!aov.on_frame_begin(project, parent, recorder, abort_switch))
            return false;
    }

    for (PostProcessingStage& stage : post_processing_stages())
    {
        if (is_aborted(abort_switch))
            return false;

        if (!stage.on_frame_begin(project, parent, recorder, abort_switch))
            return false;
    }

    return true;
}

void Frame::post_process_aov_images() const
{
    for (size_t i = 0, e = aovs().size(); i < e; ++i)
        aovs().get_by_index(i)->post_process_image(*this, get_crop_window());

    for (size_t i = 0, e = internal_aovs().size(); i < e; ++i)
        internal_aovs().get_by_index(i)->post_process_image(*this, get_crop_window());
}

ParamArray& Frame::render_info()
{
    return impl->m_render_info;
}

Frame::DenoisingMode Frame::get_denoising_mode() const
{
    return impl->m_denoising_mode;
}

void Frame::denoise(
    const size_t                                thread_count,
    IAbortSwitch*                               abort_switch) const
{
    DenoiserOptions options;

    const bool skip_denoised = m_params.get_optional<bool>("skip_denoised", true);
    options.m_marked_pixels_skipping_probability = skip_denoised ? 1.0f : 0.0f;

    options.m_use_random_pixel_order = m_params.get_optional<bool>("random_pixel_order", true);

    options.m_prefilter_spikes = m_params.get_optional<bool>("prefilter_spikes", true);

    options.m_prefilter_threshold_stddev_factor =
        m_params.get_optional<float>(
            "spike_threshold",
            options.m_prefilter_threshold_stddev_factor);

    options.m_prefilter_threshold_stddev_factor =
        m_params.get_optional<float>(
            "spike_threshold",
            options.m_prefilter_threshold_stddev_factor);

    options.m_histogram_patch_distance_threshold =
        m_params.get_optional<float>(
            "patch_distance_threshold",
            options.m_histogram_patch_distance_threshold);

    options.m_num_scales =
        m_params.get_optional<size_t>(
            "denoise_scales",
            options.m_num_scales);

    options.m_num_cores = thread_count;

    options.m_mark_invalid_pixels =
        m_params.get_optional<bool>("mark_invalid_pixels", false);

    assert(impl->m_denoiser_aov);

    impl->m_denoiser_aov->fill_empty_samples();

    Deepimf num_samples;
    impl->m_denoiser_aov->extract_num_samples_image(num_samples);

    Deepimf covariances;
    impl->m_denoiser_aov->compute_covariances_image(covariances);

    RENDERER_LOG_INFO("denoising beauty image...");
    denoise_beauty_image(
        image(),
        num_samples,
        impl->m_denoiser_aov->histograms_image(),
        covariances,
        options,
        abort_switch);

    for (size_t i = 0, e = aovs().size(); i < e; ++i)
    {
        const AOV* aov = aovs().get_by_index(i);

        if (aov->has_color_data())
        {
            RENDERER_LOG_INFO("denoising %s aov...", aov->get_name());
            denoise_aov_image(
                aov->get_image(),
                num_samples,
                impl->m_denoiser_aov->histograms_image(),
                covariances,
                options,
                abort_switch);
        }
    }
}

namespace
{

    typedef vector<tuple<string, CanvasProperties, ImageAttributes>> CheckpointProperties;

    //
    // Interface used to save the rendering buffer in checkpoints.
    //

    class ShadingBufferCanvas
      : public ICanvas
    {
      public:
        ShadingBufferCanvas(
            const Frame&                                frame,
            PermanentShadingResultFrameBufferFactory*   buffer_factory)
          : m_frame(frame)
          , m_buffer_factory(buffer_factory)
          , m_props(
                m_frame.image().properties().m_canvas_width,
                m_frame.image().properties().m_canvas_height,
                m_frame.image().properties().m_tile_width,
                m_frame.image().properties().m_tile_height,
                (1 + m_frame.aov_images().size()) * 4 + 1,
                PixelFormatFloat)
        {
            assert(buffer_factory);
        }

        const CanvasProperties& properties() const override
        {
            return m_props;
        }

        Tile& tile(
            const size_t            tile_x,
            const size_t            tile_y) override
        {
            ShadingResultFrameBuffer* tile_buffer = m_buffer_factory->create(
                m_frame,
                tile_x,
                tile_y,
                get_tile_bbox(tile_x, tile_y));

            assert(tile_buffer);
            return *tile_buffer;
        }

        const Tile& tile(
            const size_t            tile_x,
            const size_t            tile_y) const override
        {
            const ShadingResultFrameBuffer* tile_buffer = m_buffer_factory->create(
                m_frame,
                tile_x,
                tile_y,
                get_tile_bbox(tile_x, tile_y));

            assert(tile_buffer);
            return *tile_buffer;
        }

      private:
        const Frame&                                m_frame;
        PermanentShadingResultFrameBufferFactory*   m_buffer_factory;
        const CanvasProperties                      m_props;

        AABB2i get_tile_bbox(
            const size_t            tile_x,
            const size_t            tile_y) const
        {
            const Image& image = m_frame.image();
            const CanvasProperties& props = image.properties();

            // Compute the image space bounding box of the tile.
            const int tile_origin_x = static_cast<int>(
                props.m_tile_width * tile_x);
            const int tile_origin_y = static_cast<int>(
                props.m_tile_height * tile_y);

            const Tile& frame_tile = image.tile(tile_x, tile_y);

            // Compute the tile space bounding box of the pixels to render.
            return compute_tile_space_bbox(
                frame_tile,
                tile_origin_x,
                tile_origin_y,
                m_frame.get_crop_window());
        }

    };

    void get_denoiser_checkpoint_paths(
        const string&                   checkpoint_path,
        string&                         hist_path,
        string&                         cov_path,
        string&                         sum_path)
    {
        const bf::path boost_file_path(checkpoint_path);
        const bf::path directory = boost_file_path.parent_path();
        const string base_file_name = boost_file_path.stem().string() + ".denoiser";
        const string extension = boost_file_path.extension().string();

        const string hist_file_name = base_file_name + ".hist" + extension;
        hist_path = (directory / hist_file_name).string();

        const string cov_file_name = base_file_name + ".cov" + extension;
        cov_path = (directory / cov_file_name).string();

        const string sum_file_name = base_file_name + ".sum" + extension;
        sum_path = (directory / sum_file_name).string();
    }

    bool is_checkpoint_compatible(
        const string&                   checkpoint_path,
        const Frame&                    frame,
        const CheckpointProperties&     checkpoint_props)
    {
        const Image& frame_image = frame.image();
        const CanvasProperties& frame_props = frame_image.properties();
        const CanvasProperties& beauty_props = get<1>(checkpoint_props[0]);
        const ImageAttributes& exr_attributes = get<2>(checkpoint_props[0]);
        const string initial_layer_name = get<0>(checkpoint_props[0]);

        // Check for atttributes.
        if (!exr_attributes.exist("appleseed:LastPass"))
        {
            RENDERER_LOG_ERROR("incorrect checkpoint: some attributes are missing.");
            return false;
        }

        // Check for beauty layer.
        if (initial_layer_name != "beauty")
        {
            RENDERER_LOG_ERROR("incorrect checkpoint: beauty layer is missing.");
            return false;
        }

        // Check for weight layer.
        const string second_layer_name = get<0>(checkpoint_props[1]);
        if (second_layer_name != "appleseed:RenderingBuffer")
        {
            RENDERER_LOG_ERROR("incorrect checkpoint: rendering buffer layer is missing.");
            return false;
        }

        // Check that weight layer has correct amount of channel
        const size_t expect_channel_count = (1 + frame.aov_images().size()) * 4 + 1;
        if (get<1>(checkpoint_props[1]).m_channel_count != expect_channel_count)
        {
            RENDERER_LOG_ERROR("incorrect checkpoint: the shading buffer doesn't contain the correct number of channels.");
            return false;
        }

        // Check canvas properties.
        if (frame_props.m_canvas_width != beauty_props.m_canvas_width ||
            frame_props.m_canvas_height != beauty_props.m_canvas_height ||
            frame_props.m_tile_width != beauty_props.m_tile_width ||
            frame_props.m_tile_height != beauty_props.m_tile_height ||
            frame_props.m_channel_count != beauty_props.m_channel_count ||
            frame_props.m_pixel_format != beauty_props.m_pixel_format)
        {
            RENDERER_LOG_ERROR("incorrect checkpoint: the beauty layer properties doesn't match the renderer properties.");
            return false;
        }

        // Check if aovs are here.
        if (checkpoint_props.size() < frame.aovs().size() + 2)
        {
            RENDERER_LOG_ERROR("incorrect checkpoint: some aovs are missing.");
            return false;
        }

        // Check if denoising is enabled and pass exists.
        if (frame.get_denoising_mode() != Frame::DenoisingMode::Off)
        {
            string hist_file_path, cov_file_path, sum_file_path;
            get_denoiser_checkpoint_paths(
                checkpoint_path,
                hist_file_path,
                cov_file_path,
                sum_file_path);

            if (!bf::exists(bf::path(hist_file_path.c_str())) ||
                !bf::exists(bf::path(cov_file_path.c_str())) ||
                !bf::exists(bf::path(sum_file_path.c_str())))
            {
                RENDERER_LOG_ERROR("cannot load denoiser's checkpoint from disk because one or several files are missing.");
                return false;
            }
        }

        return true;
    }

    bool load_denoiser_checkpoint(
        const string&                   checkpoint_path,
        DenoiserAOV*                    denoiser_aov)
    {
        // todo: reload denoiser checkpoint from the same file.
        Deepimf& histograms_image = denoiser_aov->histograms_image();
        Deepimf& covariance_image = denoiser_aov->covariance_image();
        Deepimf& sum_image = denoiser_aov->sum_image();

        string hist_file_path, cov_file_path, sum_file_path;
        get_denoiser_checkpoint_paths(checkpoint_path, hist_file_path, cov_file_path, sum_file_path);

        // Load histograms.
        bool result = ImageIO::loadMultiChannelsEXR(histograms_image, hist_file_path.c_str());
        // Load covariance accumulator.
        result = result && ImageIO::loadMultiChannelsEXR(covariance_image, cov_file_path.c_str());
        // Load sum accumulator.
        result = result && ImageIO::loadMultiChannelsEXR(sum_image, sum_file_path.c_str());

        if (!result)
            RENDERER_LOG_ERROR("could not load denoiser checkpoint.");

        return result;
    }

    void save_denoiser_checkpoint(
        const string&                   checkpoint_path,
        const DenoiserAOV*              denoiser_aov)
    {
        // todo: save denoiser checkpoint in the same file.
        const Deepimf& histograms_image = denoiser_aov->histograms_image();
        const Deepimf& covariance_image = denoiser_aov->covariance_image();
        const Deepimf& sum_image = denoiser_aov->sum_image();

        string hist_file_path, cov_file_path, sum_file_path;
        get_denoiser_checkpoint_paths(checkpoint_path, hist_file_path, cov_file_path, sum_file_path);

        // Add histograms layer.
        bool result = ImageIO::writeMultiChannelsEXR(histograms_image, hist_file_path.c_str());
        // Add covariances layer.
        result = result && ImageIO::writeMultiChannelsEXR(covariance_image, cov_file_path.c_str());
        // Add sum layer.
        result = result && ImageIO::writeMultiChannelsEXR(sum_image, sum_file_path.c_str());

        if (!result)
            RENDERER_LOG_ERROR("could not save denoiser checkpoint.");
    }

}

bool Frame::load_checkpoint(PermanentShadingResultFrameBufferFactory* buffer_factory)
{
    if  (!impl->m_checkpoint)
        return true;

    bf::path bf_path(impl->m_checkpoint_path.c_str());

    // Check the extension, only .exr is allowed.
    string extension = lower_case(bf_path.extension().string());
    if (extension != ".exr")
    {
        RENDERER_LOG_ERROR("checkpoint file must be an \".exr\" file");
        return false;
    }

    // Check if the file exists.
    if (!bf::exists(bf_path))
    {
        RENDERER_LOG_WARNING("no checkpoint found, starting a new render");
        return true;
    }

    // Open the file.
    GenericProgressiveImageFileReader reader;
    reader.open(impl->m_checkpoint_path.c_str());

    // First, read layers name and properties.
    CheckpointProperties checkpoint_props;

    size_t layer_index = 0;
    while (reader.choose_subimage(layer_index))
    {
        CanvasProperties layer_canvas_props;
        ImageAttributes layer_attributes;

        reader.read_canvas_properties(layer_canvas_props);
        reader.read_image_attributes(layer_attributes);

        const string layer_name =
            layer_attributes.exist("name")
                ? layer_attributes.get<string>("name")
                : "undefined";

        checkpoint_props.push_back(
            make_tuple(
                layer_name,
                layer_canvas_props,
                layer_attributes));

        ++layer_index;
    }

    // Check checkpoint's compatibility.
    if (!is_checkpoint_compatible(impl->m_checkpoint_path, *this, checkpoint_props))
    {
        reader.close();
        return false;
    }

    const size_t start_pass = get<2>(checkpoint_props[0]).get<size_t>("appleseed:LastPass") + 1;

    // Interface the shading buffer in a canvas.
    ShadingBufferCanvas shading_canvas(*this, buffer_factory);

    // Load tiles from the checkpoint.
    const CanvasProperties& beauty_props = get<1>(checkpoint_props[0]);

    for (size_t tile_y = 0; tile_y < beauty_props.m_tile_count_y; ++tile_y)
    {
        for (size_t tile_x = 0; tile_x < beauty_props.m_tile_count_x; ++tile_x)
        {
            // Read shading buffer.
            reader.choose_subimage(1);
            Tile& shading_tile = shading_canvas.tile(tile_x, tile_y);
            reader.read_tile(tile_x, tile_y, &shading_tile);

            // No need to read beauty and filtered AOVs because they
            // are in the shading buffer.

            // Read Unfiltered AOV layers.
            vector<unique_ptr<Tile>> aov_tiles;
            vector<const float*> aov_ptrs;
            for (size_t i = 0; i < aovs().size(); ++i)
            {
                UnfilteredAOV* aov = dynamic_cast<UnfilteredAOV*>(
                    aovs().get_by_index(i));

                if (aov == nullptr)
                    continue;

                const string aov_name = aov->get_name();

                // Search layer index in the file.
                size_t subimage_index(~0);
                for (size_t s = 0; s < checkpoint_props.size(); ++s)
                {
                    if (get<0>(checkpoint_props[s]) == aov_name)
                    {
                        subimage_index = s;
                        break;
                    }
                }

                assert(subimage_index != size_t(~0));

                Image& aov_image = aov->get_image();
                Tile& aov_tile = aov_image.tile(tile_x, tile_y);
                reader.choose_subimage(subimage_index);
                reader.read_tile(tile_x, tile_y, &aov_tile);
            }
        }
    }

    reader.close();

    // Load internal AOVs (from external files).
    for (size_t i = 0, e = internal_aovs().size(); i < e; ++i)
    {
        AOV* aov = internal_aovs().get_by_index(i);

        DenoiserAOV* denoiser_aov = dynamic_cast<DenoiserAOV*>(aov);

        // Save denoiser checkpoint.
        if (denoiser_aov != nullptr)
        {
            if (!load_denoiser_checkpoint(impl->m_checkpoint_path, denoiser_aov))
                return false;
        }
    }

    RENDERER_LOG_INFO("opened checkpoint resuming at pass %s",
        pretty_uint(start_pass).c_str());

    impl->m_initial_pass = start_pass;

    return true;
}

void Frame::save_checkpoint(
    PermanentShadingResultFrameBufferFactory*   buffer_factory,
    const size_t                                pass) const
{
    if (!impl->m_checkpoint)
        return;

    create_parent_directories(impl->m_checkpoint_path.c_str());

    GenericImageFileWriter writer(impl->m_checkpoint_path.c_str());

    ImageAttributes image_attributes = ImageAttributes::create_default_attributes();
    image_attributes.insert("appleseed:LastPass", pass);

    // Create layers.
    // Buffer containing pixels' weight.
    ShadingBufferCanvas weight_canvas(*this, buffer_factory);

    // Create channel names.
    const size_t shading_channel_count = weight_canvas.properties().m_channel_count;
    vector<string> shading_channel_names; // for shading buffer
    vector<const char *> shading_channel_names_cr;

    static const string channel_name_prefix = "channel_";
    for (size_t i = 0; i < shading_channel_count; ++i)
    {
        shading_channel_names.push_back(channel_name_prefix + pad_left(to_string(i + 1), '0', 4));
        shading_channel_names_cr.push_back(shading_channel_names[i].c_str());
    }

    assert(shading_channel_names.size() == shading_channel_count);

    // Add layers in the file.
    writer.append_image(&image());
    image_attributes.insert("image_name", "beauty");
    writer.set_image_attributes(image_attributes);

    writer.append_image(&weight_canvas);
    image_attributes.insert("image_name", "appleseed:RenderingBuffer");
    writer.set_image_channels(shading_channel_count, shading_channel_names_cr.data());
    writer.set_image_attributes(image_attributes);

    // Add AOV layers.
    for (size_t i = 0, e = aovs().size(); i < e; ++i)
    {
        const AOV* aov = aovs().get_by_index(i);

        const char* aov_name = aov->get_name();
        const Image& aov_image = aov->get_image();
        writer.append_image(&aov_image);
        image_attributes.insert("image_name", aov_name);
        writer.set_image_channels(aov->get_channel_count(), aov->get_channel_names());
        writer.set_image_attributes(image_attributes);
    }

    // Write the file.
    writer.write();

    // Add internal AOVs layers (in external files).
    for (size_t i = 0, e = internal_aovs().size(); i < e; ++i)
    {
        const AOV* aov = internal_aovs().get_by_index(i);

        const DenoiserAOV* denoiser_aov = dynamic_cast<const DenoiserAOV*>(aov);

        // Save denoiser checkpoint.
        if (denoiser_aov != nullptr)
        {
            save_denoiser_checkpoint(impl->m_checkpoint_path, denoiser_aov);
        }
    }

    RENDERER_LOG_INFO("wrote checkpoint for pass %s.", pretty_uint(pass + 1).c_str());
}

void Frame::backup_checkpoint() const
{
    if (!impl->m_checkpoint)
        return;

    const bf::path boost_file_path(impl->m_checkpoint_path);
    const bf::path directory = boost_file_path.parent_path();
    const string base_file_name = boost_file_path.stem().string();
    const string extension = boost_file_path.extension().string();
    const string backup_file_name = base_file_name + ".checkpoint_backup" + extension;

    assert(bf::exists(boost_file_path));

    // Rename last checkpoint.
    bf::rename(impl->m_checkpoint_path, backup_file_name.c_str());

    RENDERER_LOG_INFO(
        "moved checkpoint to %s, you can use it to render the scene with a higher number of passes.",
        backup_file_name.c_str());
}

namespace
{

    void add_chromaticities(ImageAttributes& image_attributes)
    {
        // Scene-linear sRGB / Rec. 709 chromaticities.
        image_attributes.insert("white_xy_chromaticity", Vector2f(0.3127f, 0.3290f));
        image_attributes.insert("red_xy_chromaticity", Vector2f(0.64f, 0.33f));
        image_attributes.insert("green_xy_chromaticity", Vector2f(0.30f, 0.60f));
        image_attributes.insert("blue_xy_chromaticity",  Vector2f(0.15f, 0.06f));
    }

    void write_exr_image(
        const bf::path&                 file_path,
        const Image&                    image,
        ImageAttributes&                image_attributes,
        const AOV*                      aov)
    {
        create_parent_directories(file_path);

        const std::string filename = file_path.string();

        GenericImageFileWriter writer(filename.c_str());

        writer.append_image(&image);

        if (aov)
        {
            // If the AOV has color data, assume we can save it as half floats.
            if (aov->has_color_data())
                writer.set_image_output_format(PixelFormatHalf);

            writer.set_image_channels(aov->get_channel_count(), aov->get_channel_names());
        }

        image_attributes.insert("color_space", "linear");
        writer.set_image_attributes(image_attributes);

        writer.write();
    }

    void transform_to_srgb(Tile& tile)
    {
        assert(tile.get_channel_count() == 4);
        assert(tile.get_pixel_format() == PixelFormatHalf);

        typedef Color<half, 4> Color4h;

        Color4h* pixel_ptr = reinterpret_cast<Color4h*>(tile.pixel(0));
        Color4h* pixel_end = pixel_ptr + tile.get_pixel_count();

        for (; pixel_ptr < pixel_end; ++pixel_ptr)
        {
            // Load the pixel color.
            Color4f color(*pixel_ptr);

            // Apply color space conversion and clamping.
            color.unpremultiply();
            color.rgb() = fast_linear_rgb_to_srgb(color.rgb());
            color = saturate(color);
            color.premultiply();

            // Store the pixel color.
            *pixel_ptr = color;
        }
    }

    void transform_to_srgb(Image& image)
    {
        const CanvasProperties& image_props = image.properties();

        for (size_t ty = 0; ty < image_props.m_tile_count_y; ++ty)
        {
            for (size_t tx = 0; tx < image_props.m_tile_count_x; ++tx)
                transform_to_srgb(image.tile(tx, ty));
        }
    }

    void write_png_image(
        const bf::path&                 file_path,
        const Image&                    image,
        ImageAttributes&                image_attributes)
    {
        const CanvasProperties& props = image.properties();

        Image transformed_image(image);

        if (props.m_channel_count == 4)
            transform_to_srgb(transformed_image);

        create_parent_directories(file_path);

        const std::string filename = file_path.string();

        GenericImageFileWriter writer(filename.c_str());

        writer.append_image(&transformed_image);

        writer.set_image_output_format(PixelFormat::PixelFormatUInt8);

        image_attributes.insert("color_space", "sRGB");
        writer.set_image_attributes(image_attributes);

        writer.write();
    }

    bool write_image(
        const char*                     file_path,
        const Image&                    image,
        const AOV*                      aov = nullptr)
    {
        assert(file_path);

        Stopwatch<DefaultWallclockTimer> stopwatch;
        stopwatch.start();

        bf::path bf_file_path(file_path);
        string extension = lower_case(bf_file_path.extension().string());

        if (!has_extension(bf_file_path))
        {
            extension = ".exr";
            bf_file_path.replace_extension(extension);
        }

        ImageAttributes image_attributes = ImageAttributes::create_default_attributes();
        add_chromaticities(image_attributes);

        try
        {
            if (extension == ".exr")
            {
                write_exr_image(
                    bf_file_path,
                    image,
                    image_attributes,
                    aov);
            }
            else if (extension == ".png")
            {
                write_png_image(
                    bf_file_path,
                    image,
                    image_attributes);
            }
            else
            {
                RENDERER_LOG_ERROR(
                    "failed to write image file %s: unsupported image format.",
                    bf_file_path.string().c_str());

                return false;
            }
        }
        catch (const exception& e)
        {
            RENDERER_LOG_ERROR(
                "failed to write image file %s: %s.",
                bf_file_path.string().c_str(),
                e.what());

            return false;
        }

        stopwatch.measure();

        RENDERER_LOG_INFO(
            "wrote image file %s in %s.",
            bf_file_path.string().c_str(),
            pretty_time(stopwatch.get_seconds()).c_str());

        return true;
    }
}

bool Frame::write_main_image(const char* file_path) const
{
    assert(file_path);

    // Convert main image to half floats.
    const Image& image = *impl->m_image;
    const CanvasProperties& props = image.properties();
    const Image half_image(image, props.m_tile_width, props.m_tile_height, PixelFormatHalf);

    // Write main image.
    if (!write_image(file_path, half_image))
        return false;

    // Write BCD histograms and covariances if enabled.
    if (impl->m_denoising_mode == DenoisingMode::WriteOutputs)
    {
        bf::path boost_file_path(file_path);
        boost_file_path.replace_extension(".exr");

        if (!impl->m_denoiser_aov->write_images(boost_file_path.string().c_str()))
            return false;
    }

    return true;
}

bool Frame::write_aov_images(const char* file_path) const
{
    assert(file_path);

    bf::path boost_file_path(file_path);
    const bf::path file_path_ext = boost_file_path.extension();

    if (file_path_ext != ".exr")
    {
        if (!aovs().empty() && has_extension(file_path_ext))
        {
            RENDERER_LOG_WARNING(
                "aovs cannot be saved to %s files; saving them to exr files instead.",
                file_path_ext.string().substr(1).c_str());
        }

        boost_file_path.replace_extension(".exr");
    }

    const bf::path directory = boost_file_path.parent_path();
    const string base_file_name = boost_file_path.stem().string();

    bool success = true;

    for (size_t i = 0, e = aovs().size(); i < e; ++i)
    {
        const AOV* aov = aovs().get_by_index(i);

        // Compute AOV image file path.
        const string aov_name = aov->get_name();
        const string safe_aov_name = make_safe_filename(aov_name);
        const string aov_file_name = base_file_name + "." + safe_aov_name + ".exr";
        const string aov_file_path = (directory / aov_file_name).string();

        // Write AOV image.
        if (!write_image(aov_file_path.c_str(), aov->get_image(), aov))
            success = false;
    }

    return success;
}

bool Frame::write_main_and_aov_images() const
{
    bool success = true;

    // Write main image.
    {
        const string filepath = get_parameters().get_optional<string>("output_filename");
        if (!filepath.empty())
        {
            if (!write_main_image(filepath.c_str()))
                success = false;
        }
    }

    // Write AOV images.
    for (size_t i = 0, e = aovs().size(); i < e; ++i)
    {
        const AOV* aov = aovs().get_by_index(i);
        bf::path filepath = aov->get_parameters().get_optional<string>("output_filename");
        if (!filepath.empty())
        {
            const bf::path filepath_ext = filepath.extension();
            if (filepath_ext != ".exr")
            {
                bf::path new_filepath(filepath);
                new_filepath.replace_extension(".exr");

                if (has_extension(filepath_ext))
                {
                    RENDERER_LOG_WARNING(
                        "aov \"%s\" cannot be saved to %s file; saving it to \"%s\" instead.",
                        aov->get_path().c_str(),
                        filepath_ext.string().substr(1).c_str(),
                        new_filepath.string().c_str());
                }

                filepath = new_filepath;
            }

            if (!write_image(filepath.string().c_str(), aov->get_image(), aov))
                success = false;
        }
    }

    return success;
}

void Frame::write_main_and_aov_images_to_multipart_exr(const char* file_path) const
{
    Stopwatch<DefaultWallclockTimer> stopwatch;
    stopwatch.start();

    ImageAttributes image_attributes = ImageAttributes::create_default_attributes();
    add_chromaticities(image_attributes);
    image_attributes.insert("color_space", "linear");

    std::vector<Image> images;

    create_parent_directories(file_path);

    GenericImageFileWriter writer(file_path);

    // Always save the main image as half floats.
    {
        const Image& image = *impl->m_image;
        const CanvasProperties& props = image.properties();
        images.emplace_back(image, props.m_tile_width, props.m_tile_height, PixelFormatHalf);

        image_attributes.insert("image_name", "beauty");

        writer.append_image(&(images.back()));
        writer.set_image_attributes(image_attributes);
    }

    for (size_t i = 0, e = impl->m_aovs.size(); i < e; ++i)
    {
        const AOV* aov = impl->m_aovs.get_by_index(i);
        const string aov_name = aov->get_name();
        const Image& image = aov->get_image();

        if (aov->has_color_data())
        {
            // If the AOV has color data, assume we can save it as half floats.
            const CanvasProperties& props = image.properties();
            images.emplace_back(image, props.m_tile_width, props.m_tile_height, PixelFormatHalf);
            writer.append_image(&(images.back()));
        }
        else
            writer.append_image(&image);

        image_attributes.insert("image_name", aov_name.c_str());

        writer.set_image_channels(aov->get_channel_count(), aov->get_channel_names());
        writer.set_image_attributes(image_attributes);
    }

    writer.write();

    RENDERER_LOG_INFO(
        "wrote multipart exr image file %s in %s.",
        file_path,
        pretty_time(stopwatch.get_seconds()).c_str());
}

bool Frame::archive(
    const char*                                 directory,
    char**                                      output_path) const
{
    assert(directory);

    // Construct the name of the image file.
    const string filename =
        "autosave." + get_time_stamp_string() + ".exr";

    // Construct the path to the image file.
    const string file_path = (bf::path(directory) / filename).string();

    // Return the path to the image file.
    if (output_path)
        *output_path = duplicate_string(file_path.c_str());

    return write_image(file_path.c_str(), *impl->m_image);
}

void Frame::extract_parameters()
{
    // Retrieve frame resolution parameter.
    {
        const Vector2i DefaultResolution(512, 512);
        Vector2i resolution = m_params.get_required<Vector2i>("resolution", DefaultResolution);
        if (resolution[0] < 1 || resolution[1] < 1)
        {
            RENDERER_LOG_ERROR(
                "invalid value \"%d %d\" for parameter \"%s\", using default value \"%d %d\".",
                resolution[0],
                resolution[1],
                "resolution",
                DefaultResolution[0],
                DefaultResolution[1]);
            resolution = DefaultResolution;
        }
        impl->m_frame_width = static_cast<size_t>(resolution[0]);
        impl->m_frame_height = static_cast<size_t>(resolution[1]);
    }

    // Retrieve tile size parameter.
    {
        const Vector2i DefaultTileSize(64, 64);
        Vector2i tile_size = m_params.get_optional<Vector2i>("tile_size", DefaultTileSize);
        if (tile_size[0] < 1 || tile_size[1] < 1)
        {
            RENDERER_LOG_ERROR(
                "invalid value \"%d %d\" for parameter \"%s\", using default value \"%d %d\".",
                tile_size[0],
                tile_size[1],
                "tile_size",
                DefaultTileSize[0],
                DefaultTileSize[1]);
            tile_size = DefaultTileSize;
        }
        impl->m_tile_width = static_cast<size_t>(tile_size[0]);
        impl->m_tile_height = static_cast<size_t>(tile_size[1]);
    }

    // Retrieve reconstruction filter parameters.
    {
        const char* DefaultFilterName = "blackman-harris";

        impl->m_filter_name = m_params.get_optional<string>("filter", DefaultFilterName);
        impl->m_filter_radius = m_params.get_optional<float>("filter_size", 1.5f);

        if (impl->m_filter_name == "box")
            impl->m_filter.reset(new BoxFilter2<float>(impl->m_filter_radius, impl->m_filter_radius));
        else if (impl->m_filter_name == "triangle")
            impl->m_filter.reset(new TriangleFilter2<float>(impl->m_filter_radius, impl->m_filter_radius));
        else if (impl->m_filter_name == "gaussian")
            impl->m_filter.reset(new FastGaussianFilter2<float>(impl->m_filter_radius, impl->m_filter_radius, 8.0f));
        else if (impl->m_filter_name == "mitchell")
            impl->m_filter.reset(new MitchellFilter2<float>(impl->m_filter_radius, impl->m_filter_radius, 1.0f/3, 1.0f/3));
        else if (impl->m_filter_name == "bspline")
            impl->m_filter.reset(new MitchellFilter2<float>(impl->m_filter_radius, impl->m_filter_radius, 1.0f, 0.0f));
        else if (impl->m_filter_name == "catmull")
            impl->m_filter.reset(new MitchellFilter2<float>(impl->m_filter_radius, impl->m_filter_radius, 0.0f, 0.5f));
        else if (impl->m_filter_name == "lanczos")
            impl->m_filter.reset(new LanczosFilter2<float>(impl->m_filter_radius, impl->m_filter_radius, 3.0f));
        else if (impl->m_filter_name == "blackman-harris")
            impl->m_filter.reset(new FastBlackmanHarrisFilter2<float>(impl->m_filter_radius, impl->m_filter_radius));
        else
        {
            RENDERER_LOG_ERROR(
                "invalid value \"%s\" for parameter \"%s\", using default value \"%s\".",
                impl->m_filter_name.c_str(),
                "filter",
                DefaultFilterName);
            impl->m_filter_name = DefaultFilterName;
            impl->m_filter.reset(new FastBlackmanHarrisFilter2<float>(impl->m_filter_radius, impl->m_filter_radius));
        }
    }

    // Retrieve crop window parameter.
    const AABB2u default_crop_window(
        Vector2u(0, 0),
        Vector2u(impl->m_frame_width - 1, impl->m_frame_height - 1));
    impl->m_crop_window = m_params.get_optional<AABB2u>("crop_window", default_crop_window);

    // Retrieve denoiser parameters.
    {
        const string denoise_mode = m_params.get_optional<string>("denoiser", "off");

        if (denoise_mode == "off")
            impl->m_denoising_mode = DenoisingMode::Off;
        else if (denoise_mode == "on")
            impl->m_denoising_mode = DenoisingMode::Denoise;
        else if (denoise_mode == "write_outputs")
            impl->m_denoising_mode = DenoisingMode::WriteOutputs;
        else
        {
            RENDERER_LOG_ERROR(
                "invalid value \"%s\" for parameter \"%s\", using default value \"%s\".",
                denoise_mode.c_str(),
                "denoiser",
                "off");
            impl->m_denoising_mode = DenoisingMode::Off;
        }
    }

    // Retrieve checkpoint parameters.
    {

        impl->m_checkpoint = m_params.get_optional<bool>("checkpoint", false);
        impl->m_checkpoint_path = m_params.get_optional<string>("checkpoint_path", "");
    }
}

AOVContainer& Frame::internal_aovs() const
{
    return impl->m_internal_aovs;
}


//
// FrameFactory class implementation.
//

DictionaryArray FrameFactory::get_input_metadata()
{
    DictionaryArray metadata;

    metadata.push_back(
        Dictionary()
            .insert("name", "camera")
            .insert("label", "Camera")
            .insert("type", "entity")
            .insert("entity_types",
                Dictionary().insert("camera", "Camera"))
            .insert("use", "optional"));

    metadata.push_back(
        Dictionary()
            .insert("name", "resolution")
            .insert("label", "Resolution")
            .insert("type", "text")
            .insert("use", "required"));

    metadata.push_back(
        Dictionary()
            .insert("name", "crop_window")
            .insert("label", "Crop Window")
            .insert("type", "text")
            .insert("use", "optional"));

    metadata.push_back(
        Dictionary()
            .insert("name", "tile_size")
            .insert("label", "Tile Size")
            .insert("type", "text")
            .insert("use", "required"));

    metadata.push_back(
        Dictionary()
            .insert("name", "filter")
            .insert("label", "Filter")
            .insert("type", "enumeration")
            .insert("items",
                Dictionary()
                    .insert("Box", "box")
                    .insert("Triangle", "triangle")
                    .insert("Gaussian", "gaussian")
                    .insert("Mitchell-Netravali", "mitchell")
                    .insert("Cubic B-spline", "bspline")
                    .insert("Catmull-Rom Spline", "catmull")
                    .insert("Lanczos", "lanczos")
                    .insert("Blackman-Harris", "blackman-harris"))
            .insert("use", "optional")
            .insert("default", "blackman-harris"));

    metadata.push_back(
        Dictionary()
            .insert("name", "filter_size")
            .insert("label", "Filter Size")
            .insert("type", "numeric")
            .insert("min",
                Dictionary()
                    .insert("value", "0.5")
                    .insert("type", "hard"))
            .insert("max",
                Dictionary()
                    .insert("value", "4.0")
                    .insert("type", "soft"))
            .insert("use", "optional")
            .insert("default", "1.5"));

    metadata.push_back(
        Dictionary()
            .insert("name", "denoiser")
            .insert("label", "Denoiser")
            .insert("type", "enumeration")
            .insert("items",
                Dictionary()
                    .insert("Off", "off")
                    .insert("On", "on")
                    .insert("Write Outputs", "write_outputs"))
            .insert("use", "required")
            .insert("default", "off")
            .insert("on_change", "rebuild_form"));

    metadata.push_back(
        Dictionary()
            .insert("name", "skip_denoised")
            .insert("label", "Skip Denoised Pixels")
            .insert("type", "boolean")
            .insert("use", "optional")
            .insert("default", "true")
            .insert("visible_if",
                Dictionary()
                    .insert("denoiser", "on")));

    metadata.push_back(
        Dictionary()
            .insert("name", "random_pixel_order")
            .insert("label", "Random Pixel Order")
            .insert("type", "boolean")
            .insert("use", "optional")
            .insert("default", "true")
            .insert("visible_if",
                Dictionary()
                    .insert("denoiser", "on")));

    metadata.push_back(
        Dictionary()
            .insert("name", "prefilter_spikes")
            .insert("label", "Prefilter Spikes")
            .insert("type", "boolean")
            .insert("use", "optional")
            .insert("default", "true")
            .insert("visible_if",
                Dictionary()
                    .insert("denoiser", "on")));

    metadata.push_back(
        Dictionary()
            .insert("name", "spike_threshold")
            .insert("label", "Spike Threshold")
            .insert("type", "numeric")
            .insert("min",
                Dictionary()
                    .insert("value", "0.1")
                    .insert("type", "hard"))
            .insert("max",
                Dictionary()
                    .insert("value", "4.0")
                    .insert("type", "hard"))
            .insert("use", "optional")
            .insert("default", "2.0")
            .insert("visible_if",
                Dictionary()
                    .insert("denoiser", "on")));

    metadata.push_back(
        Dictionary()
            .insert("name", "patch_distance_threshold")
            .insert("label", "Patch Distance")
            .insert("type", "numeric")
            .insert("min",
                Dictionary()
                    .insert("value", "0.5")
                    .insert("type", "hard"))
            .insert("max",
                Dictionary()
                    .insert("value", "3.0")
                    .insert("type", "hard"))
            .insert("use", "optional")
            .insert("default", "1.0")
            .insert("visible_if",
                Dictionary()
                    .insert("denoiser", "on")));

    metadata.push_back(
        Dictionary()
            .insert("name", "denoise_scales")
            .insert("label", "Denoise Scales")
            .insert("type", "integer")
            .insert("min",
                Dictionary()
                    .insert("value", "1")
                    .insert("type", "hard"))
            .insert("max",
                Dictionary()
                    .insert("value", "10")
                    .insert("type", "hard"))
            .insert("use", "optional")
            .insert("default", "3")
            .insert("visible_if",
                Dictionary()
                    .insert("denoiser", "on")));

    metadata.push_back(
        Dictionary()
            .insert("name", "mark_invalid_pixels")
            .insert("label", "Mark Invalid pixels")
            .insert("type", "boolean")
            .insert("use", "optional")
            .insert("default", "false")
            .insert("visible_if",
                Dictionary()
                    .insert("denoiser", "on")));

    return metadata;
}

auto_release_ptr<Frame> FrameFactory::create(
    const char*         name,
    const ParamArray&   params)
{
    return
        auto_release_ptr<Frame>(
            new Frame(name, params, AOVContainer()));
}

auto_release_ptr<Frame> FrameFactory::create(
    const char*         name,
    const ParamArray&   params,
    const AOVContainer& aovs)
{
    return
        auto_release_ptr<Frame>(
            new Frame(name, params, aovs));
}

}   // namespace renderer
