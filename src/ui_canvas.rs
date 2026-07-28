use crate::analysis::freq_to_y;
use crate::app::VoiceHarmApp;
use crate::config::*;
use crate::ui_overlay::{draw_cursor, draw_harmonics, draw_waveform_area};
use imgui::{DrawListMut, Ui};

/// `canvas` = [x1, y1, x2, y2] — the region right of piano, below header
pub(crate) fn render_canvas(
    app: &mut VoiceHarmApp,
    ui: &Ui,
    draw: &DrawListMut,
    canvas: [f32; 4],
    f0: Option<f32>,
) {
    // Canvas background
    draw.add_rect(
        [canvas[0], canvas[1]],
        [canvas[2], canvas[3]],
        [7.0 / 255., 11.0 / 255., 16.0 / 255., 1.0],
    )
    .filled(true)
    .build();

    let plot_rect = [canvas[0], canvas[1], canvas[2], canvas[3] - 100.0];
    let spec_rect = [
        canvas[0] + 54.0,
        plot_rect[1],
        canvas[2],
        plot_rect[3],
    ];

    draw_waterfall(app, draw, spec_rect);
    draw_axes(app, draw, canvas, spec_rect);
    draw_waveform_area(app, draw, canvas, plot_rect, f0);
    draw_harmonics(app, draw, spec_rect, f0);
    draw_cursor(app, ui, draw, canvas, spec_rect);
}

fn draw_waterfall(app: &mut VoiceHarmApp, draw: &DrawListMut, rect: [f32; 4]) {
    let Some(tid) = app.waterfall.texture_id else {
        return;
    };
    let wf = &app.waterfall;

    // No UV flip needed: glium row 0 = bottom = high freq, and imgui's
    // default mapping (uv_min → display top-left, OpenGL (0,0) = bottom-left)
    // naturally places high freq at the display top.

    if !wf.filled {
        let fraction = wf.pos as f32 / SPEC_ROWS as f32;
        if fraction > 0.0 {
            // Data at texture columns 0..pos, newest column = pos-1.
            // Display on right side of widget (newest=rightmost).
            let data_left = rect[2] - (rect[2] - rect[0]) * fraction;
            draw.add_image(tid, [data_left, rect[1]], [rect[2], rect[3]])
                .uv_min([0.0, 0.0])
                .uv_max([fraction, 1.0])
                .build();
        }
    } else if wf.pos == 0 {
        draw.add_image(tid, [rect[0], rect[1]], [rect[2], rect[3]])
            .uv_min([0.0, 0.0])
            .uv_max([1.0, 1.0])
            .build();
    } else {
        // After wrap: columns 0..pos-1 = newer, columns pos..799 = older.
        // Split point in UV = pos / SPEC_ROWS.
        // Older data (UV [fraction, 1.0]) → left side of display.
        // Newer data (UV [0.0, fraction]) → right side of display.
        let fraction = wf.pos as f32 / SPEC_ROWS as f32;
        let split_x = rect[2] - (rect[2] - rect[0]) * fraction;
        draw.add_image(tid, [rect[0], rect[1]], [split_x, rect[3]])
            .uv_min([fraction, 0.0])
            .uv_max([1.0, 1.0])
            .build();
        draw.add_image(tid, [split_x, rect[1]], [rect[2], rect[3]])
            .uv_min([0.0, 0.0])
            .uv_max([fraction, 1.0])
            .build();
    }

    draw.add_rect(
        [rect[0], rect[1]],
        [rect[2], rect[3]],
        [42.0 / 255., 61.0 / 255., 76.0 / 255., 1.0],
    )
    .build();
}

fn draw_axes(
    app: &VoiceHarmApp,
    draw: &DrawListMut,
    canvas: [f32; 4],
    spec: [f32; 4],
) {
    // Octave labels (C2–C7)
    for octave in 2..=7 {
        let frequency = 16.3516 * 2.0_f32.powi(octave);
        if !(FREQ_MIN..=FREQ_MAX).contains(&frequency) {
            continue;
        }
        let y = freq_to_y(frequency, &spec);
        draw.add_text(
            [canvas[0] + 4., y - 7.],
            [125.0 / 255., 176.0 / 255., 180.0 / 255., 150.0 / 255.],
            &format!("C{octave}"),
        );
        draw.add_line(
            [spec[0], y],
            [spec[2], y],
            [68.0 / 255., 127.0 / 255., 134.0 / 255., 35.0 / 255.],
        )
        .build();
    }

    // Frequency labels
    let labels = [50., 100., 200., 500., 1000., 2000., 3000., 4000.];
    for &frequency in &labels {
        if !(FREQ_MIN..=FREQ_MAX).contains(&frequency) {
            continue;
        }
        let y = freq_to_y(frequency, &spec);
        let label = if frequency >= 1000. {
            format!("{}k", (frequency / 1000.) as u32)
        } else {
            format!("{frequency}")
        };
        // right-align: approximate char width ~7px
        let text_w = label.len() as f32 * 7.;
        draw.add_text(
            [spec[0] - 5. - text_w, y - 7.],
            [149.0 / 255., 165.0 / 255., 178.0 / 255., 1.0],
            &label,
        );
        draw.add_line(
            [spec[0], y],
            [spec[2], y],
            [114.0 / 255., 145.0 / 255., 162.0 / 255., 40.0 / 255.],
        )
        .build();
    }

    // Time axis
    for second in 0..=4 {
        let x = spec[0] + second as f32 / 4. * (spec[2] - spec[0]);
        draw.add_line(
            [x, spec[1]],
            [x, spec[3]],
            [114.0 / 255., 145.0 / 255., 162.0 / 255., 32.0 / 255.],
        )
        .build();
        let label = format!(
            "-{:.1}s",
            (4 - second) as f32 * app.visible_history_seconds() / 4.0
        );
        let text_w = label.len() as f32 * 5.;
        draw.add_text(
            [x - text_w * 0.5, spec[3] + 4.],
            [128.0 / 255., 151.0 / 255., 166.0 / 255., 1.0],
            &label,
        );
    }
}
