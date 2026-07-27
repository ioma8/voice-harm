use crate::analysis::{find_peaks, freq_to_note, freq_to_y, label_harmonics, magnitude_dbfs};
use crate::app::VoiceHarmApp;
use crate::config::*;
use crate::drawing::draw_waveform;

pub(crate) fn draw_harmonics(
    app: &VoiceHarmApp,
    painter: &egui::Painter,
    spec: egui::Rect,
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
        let tag_rect =
            egui::Rect::from_min_size(egui::pos2(spec.left() - 16., y - 7.), egui::vec2(16., 14.));
        painter.rect_filled(tag_rect, 3., egui::Color32::from_rgb(25, 65, 73));
        painter.text(
            tag_rect.center(),
            egui::Align2::CENTER_CENTER,
            tag,
            egui::FontId::proportional(10.),
            egui::Color32::from_rgb(190, 238, 227),
        );
        painter.line_segment(
            [egui::pos2(spec.left(), y), egui::pos2(spec.left() + 7., y)],
            egui::Stroke::new(1., egui::Color32::from_rgb(85, 210, 192)),
        );
    }
}

pub(crate) fn draw_cursor(
    app: &VoiceHarmApp,
    ui: &egui::Ui,
    painter: &egui::Painter,
    canvas: egui::Rect,
    spec: egui::Rect,
) {
    let data_start = if app.waterfall.filled {
        spec.left()
    } else {
        spec.right() - spec.width() * app.waterfall.pos as f32 / SPEC_ROWS as f32
    };
    let Some(pointer) = ui.ctx().pointer_hover_pos() else {
        return;
    };
    if !spec.contains(pointer) || pointer.x < data_start {
        return;
    }
    let x = ((pointer.x - spec.left()) / spec.width()).clamp(0., 1.);
    let y = ((pointer.y - spec.top()) / spec.height()).clamp(0., 1.);
    let frequency = FREQ_MIN * (FREQ_MAX / FREQ_MIN).powf(1. - y);
    let bin = ((1. - y) * (NUM_BINS - 1) as f32).round() as usize;
    let db = magnitude_dbfs(app.current_mags.get(bin).copied().unwrap_or(0.));
    let seconds = (1. - x) * app.visible_history_seconds();
    let stroke = egui::Color32::from_rgba_premultiplied(200, 200, 200, 70);
    painter.line_segment(
        [
            egui::pos2(pointer.x, spec.top()),
            egui::pos2(pointer.x, spec.bottom()),
        ],
        egui::Stroke::new(1., stroke),
    );
    painter.line_segment(
        [
            egui::pos2(spec.left(), pointer.y),
            egui::pos2(spec.right(), pointer.y),
        ],
        egui::Stroke::new(1., stroke),
    );
    let (note, octave, _) = freq_to_note(frequency);
    let formatted = if frequency >= 1000. {
        format!("{:.1}kHz", frequency / 1000.)
    } else {
        format!("{frequency:.1}Hz")
    };
    let info = format!("{note}{octave} ({formatted})\n{db:.1} dBFS\n-{seconds:.1}s");
    let galley = painter.layout_no_wrap(info, egui::FontId::monospace(13.), egui::Color32::WHITE);
    let padding = 5.;
    let size = egui::vec2(
        galley.size().x + padding * 2.,
        galley.size().y + padding * 2.,
    );
    let mut position = egui::pos2(pointer.x + 14., pointer.y - size.y - 6.);
    position.x = position
        .x
        .clamp(canvas.left() + 2., canvas.right() - size.x - 2.);
    position.y = position
        .y
        .clamp(spec.top() + 2., spec.bottom() - size.y - 2.);
    let rect = egui::Rect::from_min_size(position, size);
    painter.rect_filled(
        rect,
        3.,
        egui::Color32::from_rgba_premultiplied(0, 0, 0, 210),
    );
    painter.rect_stroke(
        rect,
        3.,
        egui::Stroke::new(
            1.,
            egui::Color32::from_rgba_premultiplied(200, 200, 200, 100),
        ),
        egui::StrokeKind::Outside,
    );
    painter.galley(
        egui::pos2(position.x + padding, position.y + padding),
        galley,
        egui::Color32::WHITE,
    );
}

pub(crate) fn draw_waveform_area(
    app: &mut VoiceHarmApp,
    painter: &egui::Painter,
    canvas: egui::Rect,
    plot: egui::Rect,
    f0: Option<f32>,
) {
    let rect = egui::Rect::from_min_max(
        egui::pos2(canvas.left() + 4., plot.bottom() + 24.),
        egui::pos2(canvas.right() - 4., canvas.bottom() - 20.),
    );
    app.waveform_render.clear();
    app.waveform_render
        .extend(app.waveform_samples.iter().copied());
    draw_waveform(
        painter,
        rect,
        &app.waveform_render,
        &mut app.waveform_points,
    );
    painter.text(
        egui::pos2(rect.left() + 4., rect.top() + 3.),
        egui::Align2::LEFT_TOP,
        "INPUT LEVEL",
        egui::FontId::proportional(10.),
        egui::Color32::from_rgb(128, 177, 181),
    );
    let readout = f0.map_or_else(
        || "--    listening…".into(),
        |frequency| {
            let (note, octave, cents) = crate::analysis::freq_to_note(frequency);
            format!("{note}{octave} {cents:+.0} ct    {frequency:.0} Hz")
        },
    );
    painter.rect_filled(
        egui::Rect::from_min_max(egui::pos2(canvas.left(), canvas.bottom() - 18.), canvas.max),
        0.,
        egui::Color32::from_rgb(15, 23, 31),
    );
    painter.text(
        egui::pos2(canvas.left() + 8., canvas.bottom() - 9.),
        egui::Align2::LEFT_CENTER,
        readout,
        egui::FontId::monospace(10.),
        egui::Color32::from_rgb(181, 225, 220),
    );
    painter.text(
        egui::pos2(canvas.right() - 8., canvas.bottom() - 9.),
        egui::Align2::RIGHT_CENTER,
        "Voice Harmonics Analyzer",
        egui::FontId::proportional(10.),
        egui::Color32::from_rgb(109, 137, 151),
    );
}
