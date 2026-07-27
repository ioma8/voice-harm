pub(crate) fn draw_piano(p: &egui::Painter, rect: egui::Rect) {
    p.rect_filled(rect, 0.0, egui::Color32::from_rgb(232, 233, 229));
    let white_h = rect.height() / 31.0;
    for key in 0..31 {
        let y = rect.bottom() - (key + 1) as f32 * white_h;
        p.line_segment(
            [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
            egui::Stroke::new(1.0, egui::Color32::from_gray(75)),
        );
    }
    // A compact vertical piano: black keys repeat C#, D#, F#, G#, A#.
    for midi in 36..=84 {
        if matches!(midi % 12, 1 | 3 | 6 | 8 | 10) {
            let n = (midi - 36) as f32 / 48.0;
            let y = rect.bottom() - n * rect.height();
            let black = egui::Rect::from_min_size(
                egui::pos2(rect.right() - rect.width() * 0.60, y - white_h * 0.34),
                egui::vec2(rect.width() * 0.60, white_h * 0.68),
            );
            if black.intersects(rect) {
                p.rect_filled(black, 1.0, egui::Color32::from_rgb(24, 25, 26));
            }
        }
    }
    p.rect_stroke(
        rect,
        0.0,
        egui::Stroke::new(1.0, egui::Color32::from_gray(90)),
        egui::StrokeKind::Inside,
    );
}

pub(crate) fn draw_waveform(
    p: &egui::Painter,
    rect: egui::Rect,
    samples: &[f32],
    points: &mut Vec<egui::Pos2>,
) {
    p.rect_filled(rect, 4.0, egui::Color32::from_rgb(16, 23, 31));
    let mid = rect.center().y;
    p.line_segment(
        [egui::pos2(rect.left(), mid), egui::pos2(rect.right(), mid)],
        egui::Stroke::new(1.0, egui::Color32::from_rgb(51, 68, 82)),
    );
    if samples.len() > 1 {
        let width = rect.width().max(1.) as usize;
        points.clear();
        points.reserve(width.saturating_sub(points.capacity()));
        for x in 0..width {
            let i = x * (samples.len() - 1) / width;
            points.push(egui::pos2(
                rect.left() + x as f32,
                mid - samples[i] * rect.height() * 0.42,
            ));
        }
        p.add(egui::Shape::line(
            std::mem::take(points),
            egui::Stroke::new(1.0, egui::Color32::from_rgb(48, 205, 184)),
        ));
    }
    p.rect_stroke(
        rect,
        0.0,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(53, 75, 91)),
        egui::StrokeKind::Inside,
    );
}
