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

    // Dim octave grid lines drawn *behind* the waterfall
    draw_grid(app, draw, canvas, spec_rect);

    draw_waterfall(app, draw, spec_rect);
    draw_axes(app, ui, draw, canvas, spec_rect);
    draw_waveform_area(app, ui, draw, canvas, plot_rect, f0);
    draw_harmonics(app, ui, draw, spec_rect, f0);
    draw_cursor(app, ui, draw, canvas, spec_rect);
}

/// Dim horizontal lines at octave boundaries behind the spectrogram.
fn draw_grid(_app: &VoiceHarmApp, draw: &DrawListMut, canvas: [f32; 4], spec: [f32; 4]) {
    for octave in 2..=7 {
        let frequency = 16.3516 * 2.0_f32.powi(octave);
        if !(FREQ_MIN..=FREQ_MAX).contains(&frequency) {
            continue;
        }
        let y = freq_to_y(frequency, &spec);
        draw.add_line(
            [canvas[0] + 54., y],
            [spec[2], y],
            [68.0 / 255., 127.0 / 255., 134.0 / 255., 12.0 / 255.],
        )
        .build();
    }
}

fn draw_waterfall(app: &mut VoiceHarmApp, draw: &DrawListMut, rect: [f32; 4]) {
    let Some(tid) = app.waterfall.texture_id else {
        return;
    };
    let wf = &app.waterfall;

    if !wf.filled {
        let fraction = wf.pos as f32 / SPEC_ROWS as f32;
        if fraction > 0.0 {
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

    // 2 px border: outer + inner offset rects
    let border_col = [42.0 / 255., 61.0 / 255., 76.0 / 255., 1.0];
    draw.add_rect(
        [rect[0] - 1., rect[1] - 1.],
        [rect[2] + 1., rect[3] + 1.],
        border_col,
    )
    .build();
    draw.add_rect([rect[0], rect[1]], [rect[2], rect[3]], border_col).build();
}

fn draw_axes(
    app: &VoiceHarmApp,
    ui: &Ui,
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
            [68.0 / 255., 127.0 / 255., 134.0 / 255., 55.0 / 255.],
        )
        .build();
    }

    // Frequency labels — use calc_text_size for accurate right-alignment
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
        let text_w = ui.calc_text_size(&label)[0];
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
        let text_w = ui.calc_text_size(&label)[0];
        draw.add_text(
            [x - text_w * 0.5, spec[3] + 4.],
            [128.0 / 255., 151.0 / 255., 166.0 / 255., 1.0],
            &label,
        );
    }
}
