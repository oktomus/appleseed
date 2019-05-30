
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
#include "lightpathsviewportmanager.h"

// appleseed.studio headers.
#include "mainwindow/rendering/lightpathspickinghandler.h"
#include "mainwindow/rendering/lightpathslayer.h"
#include "mainwindow/rendering/viewporttab.h"
#include "utility/settingskeys.h"

// appleseed.qtcommon headers.
#include "utility/miscellaneous.h"
#include "widgets/mousecoordinatestracker.h"

// appleseed.renderer headers.
#include "renderer/api/frame.h"
#include "renderer/api/lighting.h"
#include "renderer/api/project.h"

// appleseed.foundation headers.
#include "foundation/image/canvasproperties.h"
#include "foundation/image/image.h"
#include "foundation/math/aabb.h"
#include "foundation/math/vector.h"

// Qt headers.
#include <QAction>
#include <QApplication>
#include <QDir>
#include <QEvent>
#include <QFileInfo>
#include <QGridLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QMenu>
#include <QScrollArea>
#include <QSize>
#include <QString>
#include <Qt>
#include <QToolBar>
#include <QToolButton>

// Standard headers.
#include <cassert>

using namespace appleseed::qtcommon;
using namespace foundation;
using namespace renderer;

namespace appleseed {
namespace studio {

//
// LightPathsViewportManager class implementation.
//

LightPathsViewportManager::LightPathsViewportManager(
    ViewportTab*                    viewport_tab,
    Project*                        project,
    ParamArray&                     settings,
    const MouseCoordinatesTracker&  mouse_tracker)
  : m_enabled(false)
  , m_picking_enabled(true)
  , m_paths_display_active(false)
  , m_project(project)
  , m_settings(settings)
  , m_viewport_tab(viewport_tab)
  , m_viewport_widget(m_viewport_tab->get_viewport_widget())
  , m_mouse_tracker(mouse_tracker)
{
    LightPathsLayer* light_paths_layer = m_viewport_widget->get_light_paths_layer();

    connect(
        light_paths_layer, SIGNAL(signal_light_path_selection_changed(const int, const int)),
        SLOT(slot_light_path_selection_changed(const int, const int)));
    connect(
        this, SIGNAL(signal_display_next_light_path()),
        light_paths_layer, SLOT(slot_display_next_light_path()));
    connect(
        this, SIGNAL(signal_display_previous_light_path()),
        light_paths_layer, SLOT(slot_display_previous_light_path()));

    create_toolbar();

    recreate_handlers();

    m_viewport_widget->installEventFilter(this);
}

void LightPathsViewportManager::reset(renderer::Project* project)
{
    set_display_enabled(false);
    m_project = project;
}

void LightPathsViewportManager::slot_base_layer_changed(const ViewportWidget::BaseLayer layer)
{
    if (layer == ViewportWidget::BaseLayer::FinalRender)
        set_picking_enabled(true);
    else
        set_picking_enabled(false);

    refresh_backface_culling_button();
}

void LightPathsViewportManager::set_enabled(const bool enabled)
{
    m_enabled = enabled;

    if (enabled)
    {
        refresh_toolbar();
        m_toolbar->show();
    }
    else
    {
        set_display_enabled(false);
        m_toolbar->hide();
    }

    m_toolbar->setDisabled(!enabled);

    m_screen_space_paths_picking_handler->set_enabled(enabled && m_picking_enabled);

    emit signal_should_display(m_enabled && m_paths_display_active);
}

void LightPathsViewportManager::set_display_enabled(const bool enabled)
{
    m_paths_display_active = enabled;

    emit signal_should_display(m_enabled && m_paths_display_active);
}

QToolBar* LightPathsViewportManager::toolbar() const
{
    return m_toolbar;
}

void LightPathsViewportManager::clear_light_paths()
{
    m_viewport_widget->get_light_paths_layer()->clear_light_paths_selection();
    m_project->get_light_path_recorder().clear();
    m_viewport_widget->update();
}

void LightPathsViewportManager::slot_light_paths_display_toggled(const bool active)
{
    set_enabled(active);
}

void LightPathsViewportManager::slot_rectangle_selection(const QRect& rect)
{
    if (!m_picking_enabled || !m_enabled) return;

    m_screen_space_paths_picking_handler->pick(
        AABB2i(
            Vector2i(rect.x(), rect.y()),
            Vector2i(rect.x() + rect.width() - 1, rect.y() + rect.height() - 1)));
}

void LightPathsViewportManager::slot_light_path_selection_changed(
    const int       selected_light_path_index,
    const int       total_light_paths)
{
    if (total_light_paths > 0)
    {
        set_display_enabled(true);
        m_prev_path_button->setEnabled(selected_light_path_index > -1);
        m_next_path_button->setEnabled(selected_light_path_index < total_light_paths - 1);
    }
    else
    {
        set_display_enabled(false);
        m_prev_path_button->setEnabled(false);
        m_next_path_button->setEnabled(false);
    }
}

void LightPathsViewportManager::slot_save_light_paths()
{
    QString filepath =
        get_save_filename(
            m_viewport_tab,
            "Save Light Paths As...",
            "Light Paths Files (*.aspaths);;All Files (*.*)",
            m_settings,
            SETTINGS_FILE_DIALOG_LIGHT_PATHS);

    if (filepath.isEmpty())
        return;

    if (QFileInfo(filepath).suffix().isEmpty())
        filepath += ".aspaths";

    filepath = QDir::toNativeSeparators(filepath);

    // Write light paths to disk.
    m_project->get_light_path_recorder().write(filepath.toUtf8().constData());
}

void LightPathsViewportManager::slot_camera_changed()
{
    if (!m_enabled) return;
}

void LightPathsViewportManager::set_picking_enabled(const bool enabled)
{
    m_picking_enabled = enabled;
    m_screen_space_paths_picking_handler->set_enabled(enabled);
}

void LightPathsViewportManager::clear_selection() const
{
    m_viewport_widget->get_light_paths_layer()->clear_light_paths_selection();
    m_viewport_widget->update();
}

void LightPathsViewportManager::create_toolbar()
{
    LightPathsLayer* light_paths_layer = m_viewport_tab->get_viewport_widget()->get_light_paths_layer();

    // Create the render toolbar.
    m_toolbar = new QToolBar();
    m_toolbar->setObjectName("render_toolbar");
    m_toolbar->setIconSize(QSize(18, 18));

    // Save Light Paths button.
    m_save_light_paths_button = new QToolButton();
    m_save_light_paths_button->setIcon(load_icons("lightpathstab_save_light_paths"));
    connect(
        m_save_light_paths_button , SIGNAL(clicked()),
        SLOT(slot_save_light_paths()));
    m_toolbar->addWidget(m_save_light_paths_button);

    m_toolbar->addSeparator();

    // Previous Light Path button.
    m_prev_path_button = new QToolButton();
    m_prev_path_button->setIcon(load_icons("lightpathstab_prev_light_path"));
    m_prev_path_button->setToolTip("Display previous light path");
    m_prev_path_button->setEnabled(false);
    connect(
        m_prev_path_button, SIGNAL(clicked()),
        light_paths_layer, SLOT(slot_display_previous_light_path()));
    m_toolbar->addWidget(m_prev_path_button);

    // Next Light Path button.
    m_next_path_button = new QToolButton();
    m_next_path_button->setIcon(load_icons("lightpathstab_next_light_path"));
    m_next_path_button->setToolTip("Display next light path");
    m_next_path_button->setEnabled(false);
    connect(
        m_next_path_button, SIGNAL(clicked()),
        light_paths_layer, SLOT(slot_display_next_light_path()));
    m_toolbar->addWidget(m_next_path_button);

    m_toolbar->addSeparator();

    // Toggle Backface Culling button.
    QToolButton* backface_culling_button = new QToolButton();
    backface_culling_button->setIcon(load_icons("lightpathstab_toggle_backface_culling"));
    backface_culling_button->setToolTip("Show/hide backfacing surfaces");
    backface_culling_button->setCheckable(true);
    backface_culling_button->setChecked(false);
    connect(
        backface_culling_button, SIGNAL(toggled(bool)),
        light_paths_layer, SLOT(slot_toggle_backface_culling(bool)));
    m_backface_culling_action = m_toolbar->addWidget(backface_culling_button);
    m_backface_culling_action->setVisible(false);

    // Synchronize Camera button.
    QToolButton* sync_camera_button = new QToolButton();
    sync_camera_button->setIcon(load_icons("lightpathstab_synchronize_camera"));
    sync_camera_button->setToolTip("Synchronize the rendering camera with this camera");
    connect(
        sync_camera_button, SIGNAL(clicked()),
        light_paths_layer, SLOT(slot_synchronize_camera()));
    m_toolbar->addWidget(sync_camera_button);

    // Add stretchy spacer.
    // This places interactive widgets on the left and info on the right.
    QWidget* spacer = new QWidget();
    spacer->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    m_toolbar->addWidget(spacer);

    // Create a label to display various information such as mouse coordinates, etc.
    m_info_label = new QLabel();
    m_info_label->setObjectName("info_label");
    m_toolbar->addWidget(m_info_label);

    m_toolbar->setDisabled(true);
    m_toolbar->hide();
}

void LightPathsViewportManager::refresh_toolbar() const
{
    const auto light_path_count = m_project->get_light_path_recorder().get_light_path_count();

    m_save_light_paths_button->setToolTip(
        QString("Save %1 Light Path%2...")
            .arg(QString::fromStdString(pretty_uint(light_path_count)))
            .arg(light_path_count > 1 ? "s" : ""));
}

void LightPathsViewportManager::refresh_backface_culling_button() const
{
    if (m_viewport_widget->get_active_layer() == ViewportWidget::BaseLayer::OpenGL)
    { 
        m_backface_culling_action->setVisible(true);
    }
    else
    {
        m_backface_culling_action->setVisible(false);
    }
}

void LightPathsViewportManager::recreate_handlers()
{
    // The screen-space paths picking handler is used to pick paths from the render widget.
    m_screen_space_paths_picking_handler.reset(
        new LightPathsPickingHandler(
            m_viewport_widget,
            m_mouse_tracker,
            *m_project));
    m_screen_space_paths_picking_handler->set_enabled(false);

    // Camera handler.
    m_viewport_widget->setMouseTracking(true);
}

bool LightPathsViewportManager::eventFilter(QObject* object, QEvent* event)
{
    if (m_enabled)
    {
        if (event->type() == QEvent::KeyRelease)
        {
            const QKeyEvent* key_event = static_cast<QKeyEvent*>(event);

            if (!(key_event->modifiers() & (Qt::AltModifier | Qt::ShiftModifier | Qt::ControlModifier)))
            {
                const int key = key_event->key();

                if (key == Qt::Key_Escape)
                    clear_selection();
                else if (key == Qt::Key_Left)
                    emit signal_display_previous_light_path();
                else if (key == Qt::Key_Right)
                    emit signal_display_next_light_path();
            }
        }
    }

    return QObject::eventFilter(object, event);
}

}   // namespace studio
}   // namespace appleseed
