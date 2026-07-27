use crate::analysis::freq_to_y;
use crate::app::VoiceHarmApp;
use crate::config::*;
use crate::ui_overlay::{draw_cursor, draw_harmonics, draw_waveform_area};
use crate::waterfall::Waterfall;

pub(crate) fn render_canvas(app: &mut VoiceHarmApp, ui: &mut egui::Ui, f0: Option<f32>) {
    egui::CentralPanel::default().show(ui, |ui| {
        let painter = ui.painter();
        let canvas = ui.max_rect();
        painter.rect_filled(canvas, 0.0, egui::Color32::from_rgb(7, 11, 16));
        let plot_rect = egui::Rect::from_min_max(
            canvas.min,
            egui::pos2(canvas.right(), canvas.bottom() - 100.0),
        );
        let spec_rect = egui::Rect::from_min_size(
            egui::pos2(canvas.left() + 54.0, plot_rect.top()),
            egui::vec2((plot_rect.width() - 54.0).max(64.0), plot_rect.height()),
        );
        draw_waterfall(&mut app.waterfall, painter, ui.ctx(), spec_rect);
        draw_axes(app, painter, canvas, plot_rect, spec_rect);
        draw_waveform_area(app, painter, canvas, plot_rect, f0);
        draw_harmonics(app, painter, spec_rect, f0);
        draw_cursor(app, ui, painter, canvas, spec_rect);
    });
}

fn draw_waterfall(
    waterfall: &mut Waterfall,
    painter: &egui::Painter,
    ctx: &egui::Context,
    rect: egui::Rect,
) {
    let texture = waterfall.upload(ctx);
    let split = if waterfall.filled { waterfall.pos } else { 0 };
    if !waterfall.filled {
        let fraction = waterfall.pos as f32 / SPEC_ROWS as f32;
        if fraction > 0.0 {
            let data_rect = egui::Rect::from_min_max(
                egui::pos2(rect.right() - rect.width() * fraction, rect.top()),
                rect.max,
            );
            painter.image(
                texture,
                data_rect,
                egui::Rect::from_min_max(egui::pos2(0., 0.), egui::pos2(fraction, 1.)),
                egui::Color32::WHITE,
            );
        }
    } else if split == 0 {
        painter.image(texture, rect, egui::Rect::EVERYTHING, egui::Color32::WHITE);
    } else {
        let first_fraction = (SPEC_ROWS - split) as f32 / SPEC_ROWS as f32;
        let first_rect = egui::Rect::from_min_max(
            rect.min,
            egui::pos2(rect.left() + rect.width() * first_fraction, rect.bottom()),
        );
        painter.image(
            texture,
            first_rect,
            egui::Rect::from_min_max(egui::pos2(first_fraction, 0.), egui::pos2(1., 1.)),
            egui::Color32::WHITE,
        );
        let second_rect =
            egui::Rect::from_min_max(egui::pos2(first_rect.right(), rect.top()), rect.max);
        painter.image(
            texture,
            second_rect,
            egui::Rect::from_min_max(egui::pos2(0., 0.), egui::pos2(first_fraction, 1.)),
            egui::Color32::WHITE,
        );
    }
    painter.rect_stroke(
        rect,
        2.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(42, 61, 76)),
        egui::StrokeKind::Inside,
    );
}

fn draw_axes(
    app: &VoiceHarmApp,
    painter: &egui::Painter,
    canvas: egui::Rect,
    plot: egui::Rect,
    spec: egui::Rect,
) {
    for octave in 2..=7 {
        let frequency = 16.3516 * 2.0_f32.powi(octave);
        if !(FREQ_MIN..=FREQ_MAX).contains(&frequency) {
            continue;
        }
        let y = freq_to_y(frequency, &spec);
        painter.text(
            egui::pos2(canvas.left() + 4.0, y),
            egui::Align2::LEFT_CENTER,
            format!("C{octave}"),
            egui::FontId::proportional(10.0),
            egui::Color32::from_rgba_premultiplied(125, 176, 180, 150),
        );
        painter.line_segment(
            [egui::pos2(spec.left(), y), egui::pos2(spec.right(), y)],
            egui::Stroke::new(
                1.0,
                egui::Color32::from_rgba_premultiplied(68, 127, 134, 35),
            ),
        );
    }
    let labels = [50., 100., 200., 500., 1000., 2000., 3000., 4000.];
    for frequency in labels {
        if !(FREQ_MIN..=FREQ_MAX).contains(&frequency) {
            continue;
        }
        let y = freq_to_y(frequency, &spec);
        let label = if frequency >= 1000. {
            format!("{}k", (frequency / 1000.) as u32)
        } else {
            format!("{frequency}")
        };
        painter.text(
            egui::pos2(spec.left() - 5., y),
            egui::Align2::RIGHT_CENTER,
            label,
            egui::FontId::proportional(11.),
            egui::Color32::from_rgb(149, 165, 178),
        );
        painter.line_segment(
            [egui::pos2(spec.left(), y), egui::pos2(spec.right(), y)],
            egui::Stroke::new(
                1.,
                egui::Color32::from_rgba_premultiplied(114, 145, 162, 40),
            ),
        );
    }
    painter.text(
        egui::pos2(4., plot.top() + 4.),
        egui::Align2::LEFT_TOP,
        "Frequency\n(Hz)",
        egui::FontId::proportional(10.),
        egui::Color32::from_rgb(128, 151, 166),
    );
    for second in 0..=4 {
        let x = spec.left() + second as f32 / 4. * spec.width();
        painter.line_segment(
            [egui::pos2(x, spec.top()), egui::pos2(x, spec.bottom())],
            egui::Stroke::new(
                1.,
                egui::Color32::from_rgba_premultiplied(114, 145, 162, 32),
            ),
        );
        painter.text(
            egui::pos2(x, spec.bottom() + 4.),
            egui::Align2::CENTER_TOP,
            format!(
                "-{:.1}s",
                (4 - second) as f32 * app.visible_history_seconds() / 4.0
            ),
            egui::FontId::proportional(9.),
            egui::Color32::from_rgb(128, 151, 166),
        );
    }
}
