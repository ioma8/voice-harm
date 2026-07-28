use crate::analysis::{find_peaks, freq_to_note, freq_to_y, label_harmonics, magnitude_dbfs};
use crate::app::VoiceHarmApp;
use crate::config::*;
use crate::drawing::draw_waveform;
use imgui::{DrawListMut, Ui};

pub(crate) fn draw_harmonics(
    app: &VoiceHarmApp,
    draw: &DrawListMut,
    spec: [f32; 4],
    f0: Option<f32>,
) {
    let Some(frequency) = f0 else { return };
    let peaks = find_peaks(&app.current_mags);
    for &(_bin, harmonic, _magnitude) in &label_harmonics(&peaks, &app.freqs, frequency) {
        let y = freq_to_y(
            (harmonic as f32 * frequency).clamp(FREQ_MIN, FREQ_MAX),
            &spec,
        );
        let tag = format!("{harmonic}");
        // tag background pill
        let tag_left = spec[0] - 16.;
        draw.add_rect(
            [tag_left, y - 7.],
            [tag_left + 16., y + 7.],
            [25.0 / 255., 65.0 / 255., 73.0 / 255., 1.0],
        )
        .filled(true)
        .build();
        draw.add_text(
            [tag_left + 2., y - 7.],
            [190.0 / 255., 238.0 / 255., 227.0 / 255., 1.0],
            &tag,
        );
        // tick into spec area
        draw.add_line(
            [spec[0], y],
            [spec[0] + 7., y],
            [85.0 / 255., 210.0 / 255., 192.0 / 255., 1.0],
        )
        .build();
    }
}

pub(crate) fn draw_cursor(
    app: &VoiceHarmApp,
    ui: &Ui,
    draw: &DrawListMut,
    canvas: [f32; 4],
    spec: [f32; 4],
) {
    let data_start = if app.waterfall.filled {
        spec[0]
    } else {
        spec[2] - (spec[2] - spec[0]) * app.waterfall.pos as f32 / SPEC_ROWS as f32
    };

    let pointer = ui.io().mouse_pos;
    if pointer[0] < spec[0] || pointer[0] > spec[2] || pointer[1] < spec[1] || pointer[1] > spec[3] {
        return;
    }
    if pointer[0] < data_start {
        return;
    }

    let x = ((pointer[0] - spec[0]) / (spec[2] - spec[0])).clamp(0., 1.);
    let y = ((pointer[1] - spec[1]) / (spec[3] - spec[1])).clamp(0., 1.);
    let frequency = FREQ_MIN * (FREQ_MAX / FREQ_MIN).powf(1. - y);
    let bin = ((1. - y) * (NUM_BINS - 1) as f32).round() as usize;
    let db = magnitude_dbfs(app.current_mags.get(bin).copied().unwrap_or(0.));
    let seconds = (1. - x) * app.visible_history_seconds();

    // Crosshair
    let gray = [200.0 / 255., 200.0 / 255., 200.0 / 255., 70.0 / 255.];
    draw.add_line([pointer[0], spec[1]], [pointer[0], spec[3]], gray).build();
    draw.add_line([spec[0], pointer[1]], [spec[2], pointer[1]], gray).build();

    // Tooltip
    let (note, octave, _) = freq_to_note(frequency);
    let formatted = if frequency >= 1000. {
        format!("{:.1}kHz", frequency / 1000.)
    } else {
        format!("{frequency:.1}Hz")
    };
    let info = format!("{note}{octave} ({formatted})\n{db:.1} dBFS\n-{seconds:.1}s");

    // Measure text size (approximately — ~7px/char wide, ~15px/line tall)
    let lines: Vec<&str> = info.lines().collect();
    let text_w = lines.iter().map(|l| l.len()).max().unwrap_or(0) as f32 * 7.0 + 10.0;
    let text_h = lines.len() as f32 * 15.0 + 10.0;

    let mut pos = [pointer[0] + 14., pointer[1] - text_h - 6.];
    pos[0] = pos[0].clamp(canvas[0] + 2., canvas[2] - text_w - 2.);
    pos[1] = pos[1].clamp(spec[1] + 2., spec[3] - text_h - 2.);

    draw.add_rect(
        [pos[0], pos[1]],
        [pos[0] + text_w, pos[1] + text_h],
        [0.0, 0.0, 0.0, 210.0 / 255.],
    )
    .filled(true)
    .build();
    draw.add_rect(
        [pos[0], pos[1]],
        [pos[0] + text_w, pos[1] + text_h],
        [200.0 / 255., 200.0 / 255., 200.0 / 255., 100.0 / 255.],
    )
    .build();
    for (i, line) in lines.iter().enumerate() {
        draw.add_text(
            [pos[0] + 5., pos[1] + 5. + i as f32 * 15.],
            [1.0, 1.0, 1.0, 1.0],
            line,
        );
    }
}

pub(crate) fn draw_waveform_area(
    app: &mut VoiceHarmApp,
    draw: &DrawListMut,
    canvas: [f32; 4],
    plot: [f32; 4],
    f0: Option<f32>,
) {
    let rect = [
        canvas[0] + 4.,
        plot[3] + 24.,
        canvas[2] - 4.,
        canvas[3] - 20.,
    ];
    app.waveform_render.clear();
    app.waveform_render
        .extend(app.waveform_samples.iter().copied());
    draw_waveform(draw, rect, &app.waveform_render, &mut app.waveform_points);

    draw.add_text(
        [rect[0] + 4., rect[1] + 3.],
        [128.0 / 255., 177.0 / 255., 181.0 / 255., 1.0],
        "INPUT LEVEL",
    );

    let readout = f0.map_or_else(
        || "--    listening…".into(),
        |frequency| {
            let (note, octave, cents) = crate::analysis::freq_to_note(frequency);
            format!("{note}{octave} {cents:+.0} ct    {frequency:.0} Hz")
        },
    );
    draw.add_rect(
        [canvas[0], canvas[3] - 18.],
        [canvas[2], canvas[3]],
        [15.0 / 255., 23.0 / 255., 31.0 / 255., 1.0],
    )
    .filled(true)
    .build();
    draw.add_text(
        [canvas[0] + 8., canvas[3] - 15.],
        [181.0 / 255., 225.0 / 255., 220.0 / 255., 1.0],
        &readout,
    );
    let tag = "Voice Harmonics Analyzer";
    draw.add_text(
        [canvas[2] - 8. - tag.len() as f32 * 7., canvas[3] - 15.],
        [109.0 / 255., 137.0 / 255., 151.0 / 255., 1.0],
        tag,
    );
}
