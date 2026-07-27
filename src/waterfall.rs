use crate::analysis::{magnitude_dbfs, spec_color};
use crate::config::*;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// waterfall
// ---------------------------------------------------------------------------

pub(crate) struct Waterfall {
    pub(crate) pos: usize,
    pub(crate) filled: bool,
    pub(crate) peak_db: f32,
    pub(crate) tex: Option<egui::TextureHandle>,
    pub(crate) pending_columns: VecDeque<(usize, Vec<egui::Color32>)>,
}

impl Waterfall {
    pub(crate) fn new() -> Self {
        Self {
            pos: 0,
            filled: false,
            peak_db: -60.0,
            tex: None,
            pending_columns: VecDeque::new(),
        }
    }
    pub(crate) fn push(&mut self, m: &[f32]) {
        let frame_peak = m.iter().copied().fold(1e-12_f32, f32::max);
        let frame_peak_db = magnitude_dbfs(frame_peak);
        self.peak_db = frame_peak_db.max(self.peak_db - 0.15);
        let floor_db = self.peak_db - 60.0;
        let pixels = m
            .iter()
            .rev()
            .map(|&mag| {
                let db = magnitude_dbfs(mag);
                spec_color(((db - floor_db) / 60.0).clamp(0.0, 1.0))
            })
            .collect();
        let column = self.pos;
        self.pos = (self.pos + 1) % SPEC_ROWS;
        if self.pos == 0 {
            self.filled = true;
        }
        self.pending_columns.push_back((column, pixels));
    }

    pub(crate) fn reset(&mut self) {
        self.pos = 0;
        self.filled = false;
        self.peak_db = -60.0;
        self.pending_columns.clear();
    }

    pub(crate) fn upload(&mut self, ctx: &egui::Context) -> egui::TextureId {
        let tex = self.tex.get_or_insert_with(|| {
            ctx.load_texture(
                "waterfall",
                egui::ColorImage::filled([SPEC_ROWS, NUM_BINS], egui::Color32::BLACK),
                egui::TextureOptions::NEAREST,
            )
        });
        while let Some((column, pixels)) = self.pending_columns.pop_front() {
            tex.set_partial(
                [column, 0],
                egui::ColorImage::new([1, NUM_BINS], pixels),
                egui::TextureOptions::NEAREST,
            );
        }
        tex.id()
    }
}
