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
}

// ---------------------------------------------------------------------------
// professional spectrogram colour ramp
// black → very dark blue → purple → red → orange → yellow → warm white
// ---------------------------------------------------------------------------

fn spec_color(t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.003 {
        return egui::Color32::BLACK;
    }

    let stops: [(f32, (u8, u8, u8)); 7] = [
        (0.00, (0, 0, 0)),
        (0.08, (2, 2, 35)),     // near-black navy
        (0.20, (22, 4, 80)),    // dark purple
        (0.35, (100, 6, 28)),   // deep red
        (0.50, (175, 45, 12)),  // orange-red
        (0.70, (230, 175, 10)), // golden yellow
        (1.00, (255, 250, 235)),// warm white
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

/// Draw a tiny colour-bar swatch at the given rect.
fn draw_color_bar(painter: &egui::Painter, rect: egui::Rect) {
    let h = rect.height();
    if h < 2.0 { return; }
    // sample 20 stops
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
// fundamental estimation (strongest low peak)
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

// ---------------------------------------------------------------------------
// scrolling waterfall state  (time on Y, newest at top)
// ---------------------------------------------------------------------------

struct Waterfall {
    buf: Vec<f32>,           // flat: [row * NUM_BINS + bin], row 0 = oldest
    pos: usize,              // next write row index
    filled: bool,            // wrapped at least once
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
        // adaptive dB reference
        let mut peak = self.running_max;
        for &v in &self.buf { if v > peak { peak = v; } }
        // also include the latest magnitudes for instant response
        for &v in peek_mags { if v > peak { peak = v; } }
        self.running_max = peak * 0.999 + self.running_max * 0.001;
        if self.running_max < 1e-6 { self.running_max = 1e-6; }

        let max_db = if self.running_max > 0.0 { 20.0 * self.running_max.log10() } else { -60.0 };
        let floor_db = max_db - 60.0;

        let mut pixels = Vec::with_capacity(SPEC_ROWS * NUM_BINS);

        // Image layout: width = NUM_BINS (freq), height = SPEC_ROWS (time)
        // Row 0 (top) = oldest frame, bottom row = newest frame (t=0)

        if self.filled {
            // pos is the oldest frame (next to be overwritten).
            // iterate pos → pos-1 (oldest → newest) so newest lands at bottom.
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
            // empty rows at top, then written rows 0..pos (oldest → newest) at bottom
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
}

impl VoiceHarmApp {
    fn new(sample_rate: f32, audio_buf: Arc<Mutex<Vec<f32>>>) -> Self {
        let ratio = (FREQ_MAX / FREQ_MIN).powf(1.0 / (NUM_BINS - 1) as f32);
        let freqs: Vec<_> = (0..NUM_BINS).map(|i| FREQ_MIN * ratio.powi(i as i32)).collect();

        Self {
            audio_buf,
            fft_setup: FftSetup::new(sample_rate),
            waterfall: Waterfall::new(),
            freqs,
            current_mags: vec![0.0; NUM_BINS],
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
            {   // trim the ring buffer
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
        painter.rect_filled(canvas, 0.0, egui::Color32::BLACK);

        // left margin for freq labels, right margin for colour bar
        let label_w = 44.0;
        let cbar_w = 16.0;
        let gap = 6.0;

        let spec_rect = egui::Rect::from_min_size(
            egui::pos2(canvas.left() + label_w, canvas.top()),
            egui::vec2(
                (canvas.width() - label_w - cbar_w - gap).max(64.0),
                canvas.height(),
            ),
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

        // ── colour bar (right) ──
        let cbar_rect = egui::Rect::from_min_size(
            egui::pos2(spec_rect.right() + gap, spec_rect.top()),
            egui::vec2(cbar_w, spec_rect.height()),
        );
        draw_color_bar(&painter, cbar_rect);
        // dB labels next to colour bar
        painter.text(
            egui::pos2(cbar_rect.right() + 2.0, cbar_rect.top()),
            egui::Align2::LEFT_TOP,
            "0",
            egui::FontId::proportional(9.0),
            egui::Color32::GRAY,
        );
        painter.text(
            egui::pos2(cbar_rect.right() + 2.0, cbar_rect.bottom()),
            egui::Align2::LEFT_BOTTOM,
            "-60",
            egui::FontId::proportional(9.0),
            egui::Color32::GRAY,
        );

        // ── frequency labels (bottom axis) ──
        let label_freqs: [f32; 8] = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0];
        for &f in &label_freqs {
            if f < FREQ_MIN || f > FREQ_MAX { continue; }
            let x_norm = (f.ln() - FREQ_MIN.ln()) / (FREQ_MAX.ln() - FREQ_MIN.ln());
            let x = spec_rect.left() + x_norm * spec_rect.width();
            let label = if f >= 1000.0 {
                format!("{}k", (f / 1000.0) as u32)
            } else {
                format!("{f}")
            };
            painter.text(
                egui::pos2(x, spec_rect.bottom() + 10.0),
                egui::Align2::CENTER_TOP,
                label,
                egui::FontId::proportional(11.0),
                egui::Color32::GRAY,
            );
            // vertical grid line
            painter.line_segment(
                [egui::pos2(x, spec_rect.top()), egui::pos2(x, spec_rect.bottom())],
                egui::Stroke::new(1.0, egui::Color32::from_rgba_premultiplied(80, 80, 80, 20)),
            );
        }

        // ── time axis (now at bottom) ──
        painter.text(
            egui::pos2(spec_rect.left(), spec_rect.bottom() + 10.0),
            egui::Align2::LEFT_TOP,
            "now",
            egui::FontId::proportional(10.0),
            egui::Color32::from_gray(100),
        );

        // ── F0 ──
        if let Some(f0) = estimate_fundamental(&self.current_mags, &self.freqs) {
            painter.text(
                egui::pos2(canvas.left() + 2.0, canvas.top() + 2.0),
                egui::Align2::LEFT_TOP,
                format!("F0: {:.1} Hz", f0),
                egui::FontId::monospace(14.0),
                egui::Color32::WHITE,
            );
        }

        // ── freq range header ──
        painter.text(
            egui::pos2(spec_rect.left(), spec_rect.bottom() + 22.0),
            egui::Align2::LEFT_TOP,
            "Frequency",
            egui::FontId::proportional(9.0),
            egui::Color32::DARK_GRAY,
        );

        // ── cursor overlay (frequency readout) ──
        if let Some(mp) = ui.input(|i| i.pointer.hover_pos()) {
            if spec_rect.contains(mp) {
                let xn = ((mp.x - spec_rect.left()) / spec_rect.width()).clamp(0.0, 1.0);
                let yn = ((mp.y - spec_rect.top()) / spec_rect.height()).clamp(0.0, 1.0);

                let freq = FREQ_MIN * (FREQ_MAX / FREQ_MIN).powf(xn);
                let bin = (xn * (NUM_BINS - 1) as f32).round() as usize;
                let mag = self.current_mags.get(bin).copied().unwrap_or(0.0);
                let db = if mag > 0.0 { 20.0 * mag.log10() } else { -100.0 };
                let t_sec = (1.0 - yn) * (SPEC_ROWS as f32 / 60.0);

                let cross_col = egui::Color32::from_rgba_premultiplied(200, 200, 200, 70);
                painter.line_segment(
                    [egui::pos2(mp.x, spec_rect.top()), egui::pos2(mp.x, spec_rect.bottom())],
                    egui::Stroke::new(1.0, cross_col),
                );
                painter.line_segment(
                    [egui::pos2(spec_rect.left(), mp.y), egui::pos2(spec_rect.right(), mp.y)],
                    egui::Stroke::new(1.0, cross_col),
                );

                let freq_str = if freq >= 1000.0 {
                    format!("{:.1} kHz", freq / 1000.0)
                } else {
                    format!("{:.1} Hz", freq)
                };
                let info = format!("{}\n{:.1} dB\n-{:.1}s", freq_str, db, t_sec);
                let font = egui::FontId::monospace(13.0);
                let col_txt = egui::Color32::WHITE;
                let col_bg = egui::Color32::from_rgba_premultiplied(0, 0, 0, 210);
                let col_border = egui::Color32::from_rgba_premultiplied(200, 200, 200, 100);

                let galley = painter.layout_no_wrap(info, font, col_txt);
                let pad = 5.0;
                let sz = egui::vec2(galley.size().x + pad * 2.0, galley.size().y + pad * 2.0);

                let mut bp = egui::pos2(mp.x + 14.0, mp.y - sz.y - 6.0);
                bp.x = bp.x.clamp(canvas.left() + 2.0, canvas.right() - sz.x - 2.0);
                bp.y = bp.y.clamp(canvas.top() + 2.0, canvas.bottom() - sz.y - 2.0);

                let box_r = egui::Rect::from_min_size(bp, sz);
                painter.rect_filled(box_r, 3.0, col_bg);
                painter.rect_stroke(box_r, 3.0, egui::Stroke::new(1.0, col_border), egui::StrokeKind::Outside);
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
                    if len > FFT_SIZE * 4 { buf.drain(0..len - FFT_SIZE * 3); }
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
                    if len > FFT_SIZE * 4 { buf.drain(0..len - FFT_SIZE * 3); }
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
                    if len > FFT_SIZE * 4 { buf.drain(0..len - FFT_SIZE * 3); }
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
                    if len > FFT_SIZE * 4 { buf.drain(0..len - FFT_SIZE * 3); }
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
// entry point
// ---------------------------------------------------------------------------

fn main() -> Result<(), eframe::Error> {
    let sample_rate = cpal::default_host()
        .default_input_device()
        .and_then(|d| d.default_input_config().ok())
        .map(|c| c.sample_rate() as f32)
        .unwrap_or(44100.0);

    let audio_buf: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    let ab = audio_buf.clone();
    std::thread::spawn(move || {
        if let Err(e) = run_audio(ab) {
            eprintln!("[voice-harm] audio thread: {e}");
        }
    });

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1800.0, 840.0])
            .with_title("Voice Harmonics Analyzer"),
        ..Default::default()
    };

    eframe::run_native(
        "Voice Harmonics Analyzer",
        options,
        Box::new(move |_cc| Ok(Box::new(VoiceHarmApp::new(sample_rate, audio_buf)))),
    )
}
