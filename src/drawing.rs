use imgui::DrawListMut;

/// A vertical piano keyboard drawn with the draw list.
/// `rect = [x1, y1, x2, y2]` (y1=top, y2=bottom), covering MIDI 36–84 (C2–C6).
pub(crate) fn draw_piano(draw: &DrawListMut, rect: [f32; 4]) {
    draw.add_rect([rect[0], rect[1]], [rect[2], rect[3]], [232.0 / 255., 233.0 / 255., 229.0 / 255., 1.0])
        .filled(true)
        .build();
    draw.add_rect([rect[0], rect[1]], [rect[2], rect[3]], [90.0 / 255., 90.0 / 255., 90.0 / 255., 1.0])
        .build();

    let white_h = (rect[3] - rect[1]) / 31.0;
    for key in 0..31 {
        let y = rect[3] - (key + 1) as f32 * white_h;
        draw.add_line(
            [rect[0], y],
            [rect[2], y],
            [75.0 / 255., 75.0 / 255., 75.0 / 255., 1.0],
        )
        .build();
    }

    // Black keys for C#, D#, F#, G#, A# pattern
    for midi in 36..=84 {
        if matches!(midi % 12, 1 | 3 | 6 | 8 | 10) {
            let n = (midi - 36) as f32 / 48.0;
            let y = rect[3] - n * (rect[3] - rect[1]);
            let black_left = rect[2] - (rect[2] - rect[0]) * 0.60;
            let black_top = y - white_h * 0.34;
            let black_bottom = y + white_h * 0.34;
            if black_top >= rect[1] && black_bottom <= rect[3] {
                draw.add_rect(
                    [black_left, black_top],
                    [rect[2], black_bottom],
                    [24.0 / 255., 25.0 / 255., 26.0 / 255., 1.0],
                )
                .filled(true)
                .build();
            }
        }
    }
}

/// Waveform line strip. Points buffer is reused for allocation avoidance.
pub(crate) fn draw_waveform(
    draw: &DrawListMut,
    rect: [f32; 4],
    samples: &[f32],
    points: &mut Vec<[f32; 2]>,
) {
    draw.add_rect([rect[0], rect[1]], [rect[2], rect[3]], [16.0 / 255., 23.0 / 255., 31.0 / 255., 1.0])
        .filled(true)
        .build();
    draw.add_rect([rect[0], rect[1]], [rect[2], rect[3]], [53.0 / 255., 75.0 / 255., 91.0 / 255., 1.0])
        .build();

    let mid = (rect[1] + rect[3]) * 0.5;
    draw.add_line(
        [rect[0], mid],
        [rect[2], mid],
        [51.0 / 255., 68.0 / 255., 82.0 / 255., 1.0],
    )
    .build();

    if samples.len() > 1 {
        let width = ((rect[2] - rect[0]).max(1.0) as usize).min(2048);
        points.clear();
        if points.capacity() < width {
            points.reserve(width - points.capacity());
        }
        for x in 0..width {
            let i = x * (samples.len() - 1) / width;
            let y = mid - samples[i] * (rect[3] - rect[1]) * 0.42;
            points.push([rect[0] + x as f32, y]);
        }
        draw.add_polyline(points.clone(), [48.0 / 255., 205.0 / 255., 184.0 / 255., 1.0])
            .thickness(1.0)
            .build();
    }
}
