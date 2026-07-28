use crate::app::VoiceHarmApp;
use crate::drawing::draw_piano;
use crate::ui_canvas::render_canvas;
use imgui::*;
use std::sync::atomic::Ordering;

/// Called each frame inside `imgui_context.frame()`.
/// Builds the entire UI: header bar, piano sidebar, and spectrogram canvas.
pub(crate) fn build_ui(app: &mut VoiceHarmApp, ui: &Ui, f0: Option<f32>) {
    // --- style tweaks for dark look ---
    let _pad = ui.push_style_var(StyleVar::WindowPadding([0.0, 0.0]));
    let _bg = ui.push_style_color(StyleColor::WindowBg, [0.047, 0.071, 0.098, 1.0]);
    let _border = ui.push_style_color(StyleColor::Border, [0.169, 0.243, 0.302, 1.0]);

    let viewport = ui.io().display_size;
    ui.window("##main")
        .position([0.0, 0.0], Condition::Always)
        .size(viewport, Condition::Always)
        .title_bar(false)
        .resizable(false)
        .movable(false)
        .bring_to_front_on_focus(false)
        .scrollable(false)
        .mouse_inputs(false)
        .build(|| {
            let content = ui.content_region_avail();
            let cx1 = ui.cursor_screen_pos();

            // --- header ---
            let header_h = 34.0;
            let header_rect = [cx1[0], cx1[1], cx1[0] + content[0], cx1[1] + header_h];
            let draw = ui.get_window_draw_list();

            // Header background
            draw.add_rect(
                [header_rect[0], header_rect[1]],
                [header_rect[2], header_rect[3]],
                [20.0 / 255., 30.0 / 255., 40.0 / 255., 1.0],
            )
            .filled(true)
            .build();
            // Header bottom line
            draw.add_line(
                [header_rect[0], header_rect[3]],
                [header_rect[2], header_rect[3]],
                [43.0 / 255., 62.0 / 255., 77.0 / 255., 1.0],
            )
            .build();

            // Header widgets
            ui.set_cursor_pos([header_rect[0] + 12.0, header_rect[1] + 8.0]);

            ui.text_colored([0.92, 0.95, 0.96, 1.0], "Voice Harmonics");

            ui.same_line();
            let live = !app.audio_failed.load(Ordering::Relaxed);
            if live {
                ui.text_colored([0.188, 0.804, 0.722, 1.0], "  ● LIVE");
            } else {
                ui.text_colored([0.878, 0.455, 0.361, 1.0], "  ● AUDIO ERROR");
            }

            ui.same_line();
            ui.separator();
            ui.same_line();

            if let Some(f) = f0 {
                ui.text_colored([0.71, 0.882, 0.863, 1.0], format!("F0  {f:.1} Hz"));
            } else {
                ui.text_colored([0.51, 0.51, 0.51, 1.0], "F0  --");
            }

            ui.same_line();
            ui.separator();
            ui.same_line();

            // Drone toggle button
            let _drone = if app.drone_on {
                let s1 = ui.push_style_color(StyleColor::Button, [0.086, 0.506, 0.435, 1.0]);
                let s2 = ui.push_style_color(StyleColor::ButtonHovered, [0.11, 0.62, 0.53, 1.0]);
                let s3 = ui.push_style_color(StyleColor::ButtonActive, [0.07, 0.42, 0.36, 1.0]);
                Some((s1, s2, s3))
            } else {
                None
            };
            if ui.button("Drone") {
                app.drone_on = !app.drone_on;
                app.drone_state
                    .enabled
                    .store(app.drone_on, Ordering::Relaxed);
            }
            drop(_drone);

            ui.same_line();
            ui.set_next_item_width(72.0);
            let mut vol = app.drone_vol;
            if ui.slider("##vol", 0.0, 1.0, &mut vol) {
                app.drone_vol = vol;
                app.drone_state
                    .amplitude
                    .store(vol.to_bits(), Ordering::Relaxed);
            }

            // --- piano sidebar ---
            let piano_w = 104.0;
            let sidebar_top = header_rect[3];
            let sidebar_bot = cx1[1] + content[1];
            let piano_rect = [cx1[0], sidebar_top, cx1[0] + piano_w, sidebar_bot];

            // Piano background
            draw.add_rect(
                [piano_rect[0], piano_rect[1]],
                [piano_rect[2], piano_rect[3]],
                [20.0 / 255., 30.0 / 255., 40.0 / 255., 1.0],
            )
            .filled(true)
            .build();
            // Piano right border
            draw.add_line(
                [piano_rect[2], piano_rect[1]],
                [piano_rect[2], piano_rect[3]],
                [43.0 / 255., 62.0 / 255., 77.0 / 255., 1.0],
            )
            .build();

            let piano_inner = [
                piano_rect[0] + 7.,
                piano_rect[1] + 6.,
                piano_rect[2] - 7.,
                piano_rect[3] - 6.,
            ];
            draw_piano(&draw, piano_inner);

            // --- canvas ---
            let canvas_rect = [piano_rect[2], sidebar_top, cx1[0] + content[0], sidebar_bot];
            render_canvas(app, ui, &draw, canvas_rect, f0);
        });
}
