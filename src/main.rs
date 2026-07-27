use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::Mutex;
use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};
use std::f32::consts::PI;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// constants
// ---------------------------------------------------------------------------

const FFT_SIZE: usize = 16384;
const NUM_BINS: usize = 1600;
const SPEC_ROWS: usize = 800;
const FREQ_MIN: f32 = 40.0;
const FREQ_MAX: f32 = 4000.0;
const MAX_HARMONIC: u32 = 16;

// ---------------------------------------------------------------------------
// pre-computed FFT setup
// ---------------------------------------------------------------------------

struct BinMap {
    idx: usize,
    frac: f32,
}

struct FftSetup {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    bin_map: Vec<BinMap>,
    sample_rate: f32,
    bin_res: f32,
    in_buf: Mutex<Vec<f32>>,
    out_buf: Mutex<Vec<Complex<f32>>>,
}

impl FftSetup {
    fn new(sample_rate: f32) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let fft_clone = Arc::clone(&fft);

        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (FFT_SIZE - 1) as f32).cos()))
            .collect();

        let ratio = (FREQ_MAX / FREQ_MIN).powf(1.0 / (NUM_BINS - 1) as f32);
        let bin_res = sample_rate / FFT_SIZE as f32;
        let bin_map: Vec<_> = (0..NUM_BINS)
            .map(|i| {
                let f = FREQ_MIN * ratio.powi(i as i32);
                let exact = f / bin_res;
                let idx = (exact as usize).min(FFT_SIZE / 2 - 1);
                BinMap { idx, frac: exact - idx as f32 }
            })
            .collect();

        Self {
            fft: fft_clone,
            window,
            bin_map,
            sample_rate,
            bin_res,
            in_buf: Mutex::new(fft.make_input_vec()),
            out_buf: Mutex::new(fft.make_output_vec()),
        }
    }

    fn process_frame(&self, samples: &[f32], mags_out: &mut [f32]) {
        let mut in_buf = self.in_buf.lock();
        let mut out_buf = self.out_buf.lock();

        for (dst, (&s, &w)) in in_buf.iter_mut().zip(samples.iter().zip(self.window.iter())) {
            *dst = s * w;
        }
        let _ = self.fft.process(&mut *in_buf, &mut *out_buf);

        for (out, bm) in mags_out.iter_mut().zip(self.bin_map.iter()) {
            let c0 = out_buf[bm.idx];
            let mag0 = (c0.re * c0.re + c0.im * c0.im).sqrt();
            if bm.frac > 0.0 && bm.idx + 1 < out_buf.len() {
                let c1 = out_buf[bm.idx + 1];
                let mag1 = (c1.re * c1.re + c1.im * c1.im).sqrt();
                *out = mag0 * (1.0 - bm.frac) + mag1 * bm.frac;
            } else {
                *out = mag0;
            }
        }
    }

    /// Interpolated magnitude at an arbitrary frequency (Hz).
    fn magnitude_at_freq(&self, freq: f32, mags: &[f32]) -> f32 {
        let exact = freq / self.bin_res;
        let idx = (exact as usize).min(FFT_SIZE / 2 - 1);
        if idx >= mags.len() - 1 {
            return *mags.last().unwrap_or(&0.0);
        }
        let frac = exact - idx as f32;
        if frac > 0.0 {
            mags[idx] * (1.0 - frac) + mags[idx + 1] * frac
        } else {
            mags[idx]
        }
    }

    /// Magnitudes for harmonics 2..=count at the given fundamental.
    fn harmonic_mags(&self, f0: f32, mags: &[f32], count: u32) -> Vec<f32> {
        (2..=count).map(|n| self.magnitude_at_freq(f0 * n as f32, mags)).collect()
    }
}

// ---------------------------------------------------------------------------
// professional spectrogram colour ramp
// ---------------------------------------------------------------------------

fn spec_color(t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.003 {
        return egui::Color32::BLACK;
    }
    let stops: [(f32, (u8, u8, u8)); 7] = [
        (0.00, (0, 0, 0)),
        (0.08, (2, 2, 35)),
        (0.20, (22, 4, 80)),
        (0.35, (100, 6, 28)),
        (0.50, (175, 45, 12)),
        (0.70, (230, 175, 10)),
        (1.00, (255, 250, 235)),
    ];
    for i in 0..stops.len() - 1 {
        let (t0, c0) = stops[i];
        let (t1, c1) = stops[i + 1];
        if t >= t0 && t <= t1 {
            let u = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            return egui::Color32::from_rgb(
                (c0.0 as f32 + (c1.0 as f32 - c0.0 as f32) * u) as u8,
                (c0.1 as f32 + (c1.1 as f32 - c0.1 as f32) * u) as u8,
                (c0.2 as f32 + (c1.2 as f32 - c0.2 as f32) * u) as u8,
            );
        }
    }
    egui::Color32::WHITE
}

fn draw_color_bar(painter: &egui::Painter, rect: egui::Rect) {
    let h = rect.height();
    if h < 2.0 { return; }
    for i in 0..20 {
        let t = i as f32 / 19.0;
        let y0 = rect.bottom() - (i as f32 / 20.0) * h;
        let y1 = rect.bottom() - ((i + 1) as f32 / 20.0) * h;
        let band = egui::Rect::from_min_max(
            egui::pos2(rect.left(), y1),
            egui::pos2(rect.right(), y0),
        );
        painter.rect_filled(band, 0.0, spec_color(t));
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn estimate_fundamental(mags: &[f32], freqs: &[f32]) -> Option<f32> {
    let start = freqs.iter().position(|&f| f >= 60.0)?;
    let end = freqs.iter().position(|&f| f > 300.0).unwrap_or(freqs.len());
    if end <= start { return None; }
    let max_idx = (start..end).max_by(|&a, &b| {
        mags[a].partial_cmp(&mags[b]).unwrap_or(std::cmp::Ordering::Equal)
    })?;
    (mags[max_idx] > 1e-6).then(|| freqs[max_idx])
}

fn freq_to_y(freq: f32, rect: &egui::Rect) -> f32 {
    let n = (freq.ln() - FREQ_MIN.ln()) / (FREQ_MAX.ln() - FREQ_MIN.ln());
    rect.top() + (1.0 - n.clamp(0.0, 1.0)) * rect.height()
}

fn find_peaks(mags: &[f32]) -> Vec<(usize, f32)> {
    let mut peaks = Vec::new();
    for i in 1..mags.len().saturating_sub(1) {
        let m = mags[i];
        if m > mags[i - 1] && m >= mags[i + 1] && m > 1e-6 {
            peaks.push((i, m));
        }
    }
    peaks
}

fn label_harmonics<'a>(
    peaks: &[(usize, f32)],
    freqs: &[f32],
    f0: f32,
) -> Vec<(usize, u32, f32)> {
    let tol = 0.06; // 6 % frequency tolerance
    peaks
        .iter()
        .filter_map(|&(bin, mag)| {
            let freq = freqs[bin];
            let n = (freq / f0).round() as u32;
            if n < 2 || n > MAX_HARMONIC {
                return None;
            }
            let hf = n as f32 * f0;
            if (freq - hf).abs() / hf < tol {
                Some((bin, n, mag))
            } else {
                None
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// drone state (shared with output thread)
// ---------------------------------------------------------------------------

struct DroneState {
    enabled: bool,
    frequency: f32,
    amplitude: f32,
    phase: f32,
}

impl DroneState {
    fn new() -> Self {
        Self { enabled: false, frequency: 110.0, amplitude: 0.3, phase: 0.0 }
    }
}

// ---------------------------------------------------------------------------
// scrolling waterfall
// ---------------------------------------------------------------------------

struct Waterfall {
    buf: Vec<f32>,
    pos: usize,
    filled: bool,
    running_max: f32,
    tex: Option<egui::TextureHandle>,
}

impl Waterfall {
    fn new() -> Self {
        Self {
            buf: vec![0.0; SPEC_ROWS * NUM_BINS],
            pos: 0,
            filled: false,
            running_max: 1e-6,
            tex: None,
        }
    }

    fn push_frame(&mut self, mags: &[f32]) {
        let base = self.pos * NUM_BINS;
        for (dst, &src) in self.buf[base..base + NUM_BINS].iter_mut().zip(mags) {
            *dst = src;
        }
        self.pos = (self.pos + 1) % SPEC_ROWS;
        if self.pos == 0 { self.filled = true; }
    }

    fn build_image(&mut self, peek_mags: &[f32]) -> egui::ColorImage {
        let mut peak = self.running_max;
        for &v in &self.buf { if v > peak { peak = v; } }
        for &v in peek_mags { if v > peak { peak = v; } }
        self.running_max = peak * 0.999 + self.running_max * 0.001;
        if self.running_max < 1e-6 { self.running_max = 1e-6; }

        let max_db = if self.running_max > 0.0 { 20.0 * self.running_max.log10() } else { -60.0 };
        let floor_db = max_db - 60.0;

        let mut pixels = Vec::with_capacity(SPEC_ROWS * NUM_BINS);

        // Row 0 (top) = oldest, bottom row = newest (t = 0)
        if self.filled {
            for i in 0..SPEC_ROWS {
                let row = (self.pos + i) % SPEC_ROWS;
                let base = row * NUM_BINS;
                for bin in 0..NUM_BINS {
                    let mag = self.buf[base + bin];
                    let db = if mag > 0.0 { 20.0 * mag.log10() } else { -100.0 };
                    let norm = ((db - floor_db) / (max_db - floor_db)).clamp(0.0, 1.0);
                    pixels.push(spec_color(norm));
                }
            }
        } else {
            for _ in self.pos..SPEC_ROWS {
                for _ in 0..NUM_BINS {
                    pixels.push(egui::Color32::BLACK);
                }
            }
            for i in 0..self.pos {
                let base = i * NUM_BINS;
                for bin in 0..NUM_BINS {
                    let mag = self.buf[base + bin];
                    let db = if mag > 0.0 { 20.0 * mag.log10() } else { -100.0 };
                    let norm = ((db - floor_db) / (max_db - floor_db)).clamp(0.0, 1.0);
                    pixels.push(spec_color(norm));
                }
            }
        }
        egui::ColorImage::new([NUM_BINS, SPEC_ROWS], pixels)
    }
}

// ---------------------------------------------------------------------------
// application
// ---------------------------------------------------------------------------

struct VoiceHarmApp {
    audio_buf: Arc<Mutex<Vec<f32>>>,
    fft_setup: FftSetup,
    waterfall: Waterfall,
    freqs: Vec<f32>,
    current_mags: Vec<f32>,
    drone_state: Arc<Mutex<DroneState>>,
    drone_on: bool,
    drone_vol: f32,
}

impl VoiceHarmApp {
    fn new(sample_rate: f32, audio_buf: Arc<Mutex<Vec<f32>>>, drone: Arc<Mutex<DroneState>>) -> Self {
        let ratio = (FREQ_MAX / FREQ_MIN).powf(1.0 / (NUM_BINS - 1) as f32);
        let freqs: Vec<_> = (0..NUM_BINS).map(|i| FREQ_MIN * ratio.powi(i as i32)).collect();
        Self {
            audio_buf,
            fft_setup: FftSetup::new(sample_rate),
            waterfall: Waterfall::new(),
            freqs,
            current_mags: vec![0.0; NUM_BINS],
            drone_state: drone,
            drone_on: false,
            drone_vol: 0.3,
        }
    }
}

impl eframe::App for VoiceHarmApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // ── pull samples & run FFT ──
        let samples = {
            let buf = self.audio_buf.lock();
            if buf.len() >= FFT_SIZE {
                Some(buf[buf.len() - FFT_SIZE..].to_vec())
            } else {
                None
            }
        };
        if let Some(s) = samples {
            {
                let mut buf = self.audio_buf.lock();
                let len = buf.len();
                if len > FFT_SIZE * 3 { buf.drain(0..len - FFT_SIZE * 2); }
            }
            self.fft_setup.process_frame(&s, &mut self.current_mags);
            self.waterfall.push_frame(&self.current_mags);
        }

        // ── layout ──
        let rect = ui.max_rect();
        let painter = ui.painter_at(rect);
        let canvas = rect.shrink2(egui::vec2(10.0, 4.0));
        let top_bar_h = 22.0;

        // top bar area
        let top_rect = egui::Rect::from_min_size(
            canvas.left_top(),
            egui::vec2(canvas.width(), top_bar_h),
        );
        let content_rect = egui::Rect::from_min_size(
            egui::pos2(canvas.left(), canvas.top() + top_bar_h),
            egui::vec2(canvas.width(), (canvas.height() - top_bar_h).max(64.0)),
        );
        painter.rect_filled(content_rect, 0.0, egui::Color32::BLACK);
        painter.rect_filled(top_rect, 0.0, egui::Color32::from_rgb(8, 8, 20));

        // ── top bar: F0 + drone controls ──
        let f0_opt = estimate_fundamental(&self.current_mags, &self.freqs);
        if let Some(f0) = f0_opt {
            painter.text(
                egui::pos2(top_rect.left() + 4.0, top_rect.center().y),
                egui::Align2::LEFT_CENTER,
                format!("F0: {:.1} Hz", f0),
                egui::FontId::monospace(14.0),
                egui::Color32::WHITE,
            );

            // sync drone frequency
            self.drone_state.lock().frequency = f0;
        } else {
            painter.text(
                egui::pos2(top_rect.left() + 4.0, top_rect.center().y),
                egui::Align2::LEFT_CENTER,
                "F0: --",
                egui::FontId::monospace(14.0),
                egui::Color32::from_gray(80),
            );
        }

        // drone toggle button
        let drone_btn_rect = egui::Rect::from_min_size(
            egui::pos2(top_rect.right() - 200.0, top_rect.top() + 2.0),
            egui::vec2(60.0, top_bar_h - 4.0),
        );
        let drone_col = if self.drone_on {
            egui::Color32::from_rgb(30, 180, 30)
        } else {
            egui::Color32::from_rgb(60, 60, 60)
        };
        painter.rect_filled(drone_btn_rect, 3.0, drone_col);
        painter.text(
            drone_btn_rect.center(),
            egui::Align2::CENTER_CENTER,
            "Drone",
            egui::FontId::proportional(12.0),
            egui::Color32::WHITE,
        );
        let drone_clicked = ui.interact(drone_btn_rect, ui.next_auto_id(), egui::Sense::click())
            .clicked();
        if drone_clicked {
            self.drone_on = !self.drone_on;
            self.drone_state.lock().enabled = self.drone_on;
        }

        // drone volume slider
        let vol_rect = egui::Rect::from_min_size(
            egui::pos2(drone_btn_rect.right() + 8.0, top_rect.top() + 4.0),
            egui::vec2(120.0, top_bar_h - 8.0),
        );
        let vol_response = ui.interact(vol_rect, ui.next_auto_id(), egui::Sense::drag());
        if vol_response.hovered() || vol_response.dragged() {
            if let Some(mp) = ui.input(|i| i.pointer.interact_pos()) {
                let clamped = ((mp.x - vol_rect.left()) / vol_rect.width()).clamp(0.0, 1.0);
                self.drone_vol = clamped;
                self.drone_state.lock().amplitude = clamped;
            }
        }
        // draw slider
        let fill_w = vol_rect.width() * self.drone_vol;
        if fill_w > 0.0 {
            let fill = egui::Rect::from_min_size(vol_rect.left_top(), egui::vec2(fill_w, vol_rect.height()));
            painter.rect_filled(fill, 2.0, egui::Color32::from_rgb(80, 140, 200));
        }
        painter.rect_stroke(vol_rect, 2.0, egui::Stroke::new(1.0, egui::Color32::from_gray(80)), egui::StrokeKind::Inside);
        painter.text(
            egui::pos2(vol_rect.right() + 4.0, vol_rect.center().y),
            egui::Align2::LEFT_CENTER,
            format!("{:.0}%", self.drone_vol * 100.0),
            egui::FontId::proportional(10.0),
            egui::Color32::GRAY,
        );

        // ── spectrogram + sidebar layout ──
        let label_w = 44.0;
        let profile_w = 55.0;
        let cbar_w = 16.0;
        let gap = 6.0;

        let aw = content_rect.width();
        let spec_w = (aw - label_w - profile_w - cbar_w - gap * 3.0).max(64.0);
        let x0 = content_rect.left() + label_w;

        let spec_rect = egui::Rect::from_min_size(
            egui::pos2(x0, content_rect.top()),
            egui::vec2(spec_w, content_rect.height()),
        );

        // ── build & upload texture ──
        let image = self.waterfall.build_image(&self.current_mags);
        let tex = self.waterfall.tex.get_or_insert_with(|| {
            ui.ctx().load_texture("wf", image.clone(), egui::TextureOptions::NEAREST)
        });
        tex.set(image, egui::TextureOptions::NEAREST);

        painter.image(
            tex.id(),
            spec_rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );

        // ── harmonic number labels on peaks ──
        if let Some(f0) = f0_opt {
            let peaks = find_peaks(&self.current_mags);
            let labels = label_harmonics(&peaks, &self.freqs, f0);
            for &(_bin, n, _mag) in &labels {
                let freq = (n as f32 * f0).max(FREQ_MIN).min(FREQ_MAX);
                let y = freq_to_y(freq, &spec_rect);
                // pill-shaped tag at the left edge of the spectrogram
                let tag = format!("{}", n);
                let tag_w = 16.0;
                let tag_h = 14.0;
                let tag_rect = egui::Rect::from_min_size(
                    egui::pos2(spec_rect.left() - tag_w, y - tag_h / 2.0),
                    egui::vec2(tag_w, tag_h),
                );
                painter.rect_filled(tag_rect, 2.0, egui::Color32::from_rgb(60, 60, 80));
                painter.text(
                    tag_rect.center(),
                    egui::Align2::CENTER_CENTER,
                    &tag,
                    egui::FontId::proportional(10.0),
                    egui::Color32::from_rgb(220, 220, 200),
                );
                // faint horizontal line across the spectrogram
                painter.line_segment(
                    [egui::pos2(spec_rect.left(), y), egui::pos2(spec_rect.right(), y)],
                    egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(255, 255, 200, 25)),
                );
            }
        }

        // ── harmonic profile sidebar ──
        let profile_rect = egui::Rect::from_min_size(
            egui::pos2(spec_rect.right() + gap, content_rect.top()),
            egui::vec2(profile_w, content_rect.height()),
        );
        painter.rect_filled(profile_rect, 0.0, egui::Color32::from_rgb(5, 5, 15));

        if let Some(f0) = f0_opt {
            let h_mags = self.fft_setup.harmonic_mags(f0, &self.current_mags, MAX_HARMONIC);
            let h_max = h_mags.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(1e-6).max(1e-6);

            let title_h = 14.0;
            let bar_area = profile_rect.height() - title_h;
            let n_harm = (MAX_HARMONIC - 1) as usize; // 2..=16 = 15
            let bar_h = bar_area / n_harm as f32;

            // "H" title
            painter.text(
                egui::pos2(profile_rect.left() + 2.0, profile_rect.top()),
                egui::Align2::LEFT_TOP,
                "H",
                egui::FontId::proportional(10.0),
                egui::Color32::DARK_GRAY,
            );

            for hi in 0..n_harm {
                let n = hi + 2;
                let y = profile_rect.top() + title_h + hi as f32 * bar_h;
                let mag = h_mags[hi];

                // harmonic number label
                painter.text(
                    egui::pos2(profile_rect.left() + 2.0, y + 1.0),
                    egui::Align2::LEFT_TOP,
                    format!("{}", n),
                    egui::FontId::proportional(10.0),
                    egui::Color32::from_gray(140),
                );

                // magnitude bar
                let frac = (mag / h_max).clamp(0.0, 1.0);
                if frac > 0.01 {
                    let bar_max_w = (profile_rect.width() - 18.0).max(1.0);
                    let bar_w = frac * bar_max_w;
                    let bar_rect = egui::Rect::from_min_size(
                        egui::pos2(profile_rect.left() + 18.0, y + 2.0),
                        egui::vec2(bar_w, (bar_h - 3.0).max(1.0)),
                    );
                    // colour: warm tone relative to strength
                    let c = egui::Color32::from_rgb(
                        (180.0 + frac * 75.0) as u8,
                        (80.0 + frac * 120.0) as u8,
                        (40.0 + frac * 40.0) as u8,
                    );
                    painter.rect_filled(bar_rect, 1.0, c);
                }
            }
        } else {
            painter.text(
                profile_rect.center(),
                egui::Align2::CENTER_CENTER,
                "no F0",
                egui::FontId::proportional(10.0),
                egui::Color32::from_gray(50),
            );
        }

        // ── colour bar ──
        let cbar_rect = egui::Rect::from_min_size(
            egui::pos2(profile_rect.right() + gap, content_rect.top()),
            egui::vec2(cbar_w, content_rect.height()),
        );
        draw_color_bar(&painter, cbar_rect);
        painter.text(
            egui::pos2(cbar_rect.right() + 2.0, cbar_rect.top()),
            egui::Align2::LEFT_TOP, "0", egui::FontId::proportional(9.0), egui::Color32::GRAY,
        );
        painter.text(
            egui::pos2(cbar_rect.right() + 2.0, cbar_rect.bottom()),
            egui::Align2::LEFT_BOTTOM, "-60", egui::FontId::proportional(9.0), egui::Color32::GRAY,
        );

        // ── frequency labels ──
        let label_freqs: [f32; 8] = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0];
        for &f in &label_freqs {
            if f < FREQ_MIN || f > FREQ_MAX { continue; }
            let x = spec_rect.left() + (f.ln() - FREQ_MIN.ln()) / (FREQ_MAX.ln() - FREQ_MIN.ln()) * spec_rect.width();
            let label = if f >= 1000.0 { format!("{}k", (f / 1000.0) as u32) } else { format!("{f}") };
            painter.text(
                egui::pos2(x, content_rect.bottom() + 8.0),
                egui::Align2::CENTER_TOP, &label, egui::FontId::proportional(11.0), egui::Color32::GRAY,
            );
            painter.line_segment(
                [egui::pos2(x, content_rect.top()), egui::pos2(x, content_rect.bottom())],
                egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(80, 80, 80, 18)),
            );
        }

        // ── now label ──
        painter.text(
            egui::pos2(spec_rect.left(), content_rect.bottom() + 8.0),
            egui::Align2::LEFT_TOP, "now", egui::FontId::proportional(10.0), egui::Color32::from_gray(100),
        );
        painter.text(
            egui::pos2(spec_rect.left(), content_rect.bottom() + 20.0),
            egui::Align2::LEFT_TOP, "Frequency", egui::FontId::proportional(9.0), egui::Color32::DARK_GRAY,
        );

        // ── cursor overlay ──
        if let Some(mp) = ui.input(|i| i.pointer.hover_pos()) {
            if spec_rect.contains(mp) {
                let xn = ((mp.x - spec_rect.left()) / spec_rect.width()).clamp(0.0, 1.0);
                let yn = ((mp.y - content_rect.top()) / content_rect.height()).clamp(0.0, 1.0);

                let freq = FREQ_MIN * (FREQ_MAX / FREQ_MIN).powf(xn);
                let bin = (xn * (NUM_BINS - 1) as f32).round() as usize;
                let mag = self.current_mags.get(bin).copied().unwrap_or(0.0);
                let db = if mag > 0.0 { 20.0 * mag.log10() } else { -100.0 };
                let t_sec = (1.0 - yn) * (SPEC_ROWS as f32 / 60.0);

                let cross_col = egui::Color32::from_rgba_premultiplied(200, 200, 200, 70);
                painter.line_segment(
                    [egui::pos2(mp.x, content_rect.top()), egui::pos2(mp.x, content_rect.bottom())],
                    egui::Stroke::new(1.0, cross_col),
                );
                painter.line_segment(
                    [egui::pos2(spec_rect.left(), mp.y), egui::pos2(spec_rect.right(), mp.y)],
                    egui::Stroke::new(1.0, cross_col),
                );

                let freq_str = if freq >= 1000.0 { format!("{:.1} kHz", freq / 1000.0) } else { format!("{:.1} Hz", freq) };
                let info = format!("{freq_str}\n{db:.1} dB\n-{t_sec:.1}s");
                let font = egui::FontId::monospace(13.0);
                let galley = painter.layout_no_wrap(info, font, egui::Color32::WHITE);
                let pad = 5.0;
                let sz = egui::vec2(galley.size().x + pad * 2.0, galley.size().y + pad * 2.0);
                let mut bp = egui::pos2(mp.x + 14.0, mp.y - sz.y - 6.0);
                bp.x = bp.x.clamp(canvas.left() + 2.0, canvas.right() - sz.x - 2.0);
                bp.y = bp.y.clamp(canvas.top() + 2.0, canvas.bottom() - sz.y - 2.0);
                let box_r = egui::Rect::from_min_size(bp, sz);
                painter.rect_filled(box_r, 3.0, egui::Color32::from_rgba_premultiplied(0, 0, 0, 210));
                painter.rect_stroke(box_r, 3.0, egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(200, 200, 200, 100)), egui::StrokeKind::Outside);
                painter.galley(egui::pos2(bp.x + pad, bp.y + pad), galley, egui::Color32::WHITE);
            }
        }

        ui.ctx().request_repaint();
    }
}

// ---------------------------------------------------------------------------
// audio capture — ring buffer
// ---------------------------------------------------------------------------

fn run_audio(audio_buf: Arc<Mutex<Vec<f32>>>) -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let device = host.default_input_device().ok_or("no microphone found")?;
    let config = device.default_input_config()?;

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => {
            let b = audio_buf.clone();
            device.build_input_stream::<f32, _, _>(
                config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mut buf = b.lock();
                    buf.extend_from_slice(data);
                    let len = buf.len();
                    if len > FFT_SIZE * 3 { buf.drain(0..len - FFT_SIZE * 2); }
                },
                |err| eprintln!("audio err: {err}"),
                None,
            )?
        }
        cpal::SampleFormat::I16 => {
            let b = audio_buf.clone();
            device.build_input_stream::<i16, _, _>(
                config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mut buf = b.lock();
                    buf.extend(data.iter().map(|&s| s as f32 / i16::MAX as f32));
                    let len = buf.len();
                    if len > FFT_SIZE * 3 { buf.drain(0..len - FFT_SIZE * 2); }
                },
                |err| eprintln!("audio err: {err}"),
                None,
            )?
        }
        cpal::SampleFormat::I32 => {
            let b = audio_buf.clone();
            device.build_input_stream::<i32, _, _>(
                config.into(),
                move |data: &[i32], _: &cpal::InputCallbackInfo| {
                    let mut buf = b.lock();
                    buf.extend(data.iter().map(|&s| s as f32 / i32::MAX as f32));
                    let len = buf.len();
                    if len > FFT_SIZE * 3 { buf.drain(0..len - FFT_SIZE * 2); }
                },
                |err| eprintln!("audio err: {err}"),
                None,
            )?
        }
        cpal::SampleFormat::U16 => {
            let b = audio_buf.clone();
            device.build_input_stream::<u16, _, _>(
                config.into(),
                move |data: &[u16], _: &cpal::InputCallbackInfo| {
                    let mut buf = b.lock();
                    buf.extend(data.iter().map(|&s| (s as f32 - 32768.0) / 32768.0));
                    let len = buf.len();
                    if len > FFT_SIZE * 3 { buf.drain(0..len - FFT_SIZE * 2); }
                },
                |err| eprintln!("audio err: {err}"),
                None,
            )?
        }
        _ => return Err(cpal::Error::new(cpal::ErrorKind::InvalidInput).into()),
    };

    stream.play()?;
    loop { std::thread::sleep(std::time::Duration::from_secs(1)); }
}

// ---------------------------------------------------------------------------
// drone output — continuous sine wave
// ---------------------------------------------------------------------------

fn run_drone(drone: Arc<Mutex<DroneState>>) -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let device = host.default_output_device().ok_or("no output device")?;
    let config = device.default_output_config()?;
    let sample_rate = config.sample_rate() as f32;

    let stream = device.build_output_stream::<f32, _, _>(
        config.into(),
        move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let mut d = drone.lock();
            if d.enabled && d.frequency > 0.0 {
                for sample in data.iter_mut() {
                    *sample = (d.phase * 2.0 * PI).sin() * d.amplitude;
                    d.phase = (d.phase + d.frequency / sample_rate) % 1.0;
                }
            } else {
                for sample in data.iter_mut() {
                    *sample = 0.0;
                }
            }
        },
        |err| eprintln!("drone err: {err}"),
        None,
    )?;

    stream.play()?;
    loop { std::thread::sleep(std::time::Duration::from_secs(1)); }
}

// ---------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------

fn main() -> Result<(), eframe::Error> {
    let sample_rate = cpal::default_host()
        .default_input_device()
        .and_then(|d| d.default_input_config().ok())
        .map(|c| c.sample_rate() as f32)
        .unwrap_or(44100.0);

    let audio_buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let drone_state: Arc<Mutex<DroneState>> = Arc::new(Mutex::new(DroneState::new()));

    let ab = audio_buf.clone();
    std::thread::spawn(move || {
        if let Err(e) = run_audio(ab) {
            eprintln!("[voice-harm] audio thread: {e}");
        }
    });

    let ds = drone_state.clone();
    std::thread::spawn(move || {
        if let Err(e) = run_drone(ds) {
            eprintln!("[voice-harm] drone thread: {e}");
        }
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1800.0, 860.0])
            .with_title("Voice Harmonics Analyzer"),
        ..Default::default()
    };

    eframe::run_native(
        "Voice Harmonics Analyzer",
        options,
        Box::new(move |_cc| Ok(Box::new(VoiceHarmApp::new(sample_rate, audio_buf, drone_state)))),
    )
}
