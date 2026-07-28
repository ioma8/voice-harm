use crate::analysis::{magnitude_dbfs, spec_color};
use crate::config::*;
use glium::texture::{ClientFormat, MipmapsOption, RawImage2d, Texture2d};
use glium::uniforms::{MagnifySamplerFilter, MinifySamplerFilter, SamplerBehavior};
use glium::Display;
use glutin::surface::WindowSurface;
use std::borrow::Cow;
use std::rc::Rc;

// ---------------------------------------------------------------------------
// waterfall — GPU texture ring buffer
// ---------------------------------------------------------------------------

pub(crate) struct Waterfall {
    pub(crate) pos: usize,
    pub(crate) filled: bool,
    pub(crate) peak_db: f32,
    /// RGBA pixel buffer, row-major. In glium, row 0 = *bottom* of the texture.
    /// We store row 0 = highest freq, row NUM_BINS-1 = lowest freq, and flip
    /// UVs at draw time (uv_min=[0,1], uv_max=[1,0]).
    buffer: Vec<u8>,
    texture: Option<Rc<Texture2d>>,
    pub(crate) texture_id: Option<imgui::TextureId>,
    dirty: bool,
}

impl Waterfall {
    pub(crate) fn new() -> Self {
        Self {
            pos: 0,
            filled: false,
            peak_db: -60.0,
            buffer: vec![0; SPEC_ROWS * NUM_BINS * 4],
            texture: None,
            texture_id: None,
            dirty: false,
        }
    }

    pub(crate) fn push(&mut self, m: &[f32]) {
        let frame_peak = m.iter().copied().fold(1e-12_f32, f32::max);
        let frame_peak_db = magnitude_dbfs(frame_peak);
        self.peak_db = frame_peak_db.max(self.peak_db - 0.15);
        let floor_db = self.peak_db - 60.0;

        let col = self.pos;
        // m[0] = lowest freq bin → highest row index (NUM_BINS-1)
        // m[NUM_BINS-1] = highest freq bin → row 0 (bottom of glium texture)
        for (bin, &mag) in m.iter().enumerate() {
            let db = magnitude_dbfs(mag);
            let color = spec_color(((db - floor_db) / 60.0).clamp(0.0, 1.0));
            let row = NUM_BINS - 1 - bin;
            let idx = (row * SPEC_ROWS + col) * 4;
            // Deref to ImColor32Fields for u8 field access
            let fields = &*color;
            self.buffer[idx] = fields.r;
            self.buffer[idx + 1] = fields.g;
            self.buffer[idx + 2] = fields.b;
            self.buffer[idx + 3] = fields.a;
        }

        self.pos = (self.pos + 1) % SPEC_ROWS;
        if self.pos == 0 {
            self.filled = true;
        }
        self.dirty = true;
    }

    pub(crate) fn reset(&mut self) {
        self.pos = 0;
        self.filled = false;
        self.peak_db = -60.0;
        self.buffer.fill(0);
        self.dirty = true;
    }

    /// Upload dirty pixels to the GPU texture. Call once per frame (before
    /// the imgui frame) if `dirty` is true.
    pub(crate) fn upload(
        &mut self,
        display: &Display<WindowSurface>,
        renderer: &mut imgui_glium_renderer::Renderer,
    ) {
        if !self.dirty {
            return;
        }
        self.dirty = false;

        let raw = RawImage2d {
            data: Cow::Borrowed(self.buffer.as_slice()),
            width: SPEC_ROWS as u32,
            height: NUM_BINS as u32,
            format: ClientFormat::U8U8U8U8,
        };

        if let Some(ref tex) = self.texture {
            tex.write(
                glium::Rect {
                    left: 0,
                    bottom: 0,
                    width: SPEC_ROWS as u32,
                    height: NUM_BINS as u32,
                },
                raw,
            );
        } else {
            let tex = Texture2d::with_format(
                display,
                raw,
                glium::texture::UncompressedFloatFormat::U8U8U8U8,
                MipmapsOption::NoMipmap,
            )
            .expect("create waterfall texture");
            let rc = Rc::new(tex);
            let id = renderer.textures().insert(imgui_glium_renderer::Texture {
                texture: Rc::clone(&rc),
                sampler: SamplerBehavior {
                    magnify_filter: MagnifySamplerFilter::Nearest,
                    minify_filter: MinifySamplerFilter::Nearest,
                    ..Default::default()
                },
            });
            self.texture = Some(rc);
            self.texture_id = Some(id);
        }
    }
}
