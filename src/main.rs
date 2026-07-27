use cpal::Sample;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use parking_lot::Mutex;
use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex};
use rtrb::{Consumer, Producer, RingBuffer};
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

// Keep the analysis window below 400 ms at common input rates so F0 feedback
// stays responsive. The log display is denser than the FFT bins and uses
// interpolation for a smooth visual scale.
const FFT_SIZE: usize = 16384;
const NUM_BINS: usize = 2400;
const SPEC_ROWS: usize = 800;
const FFT_HOP_SIZE: usize = 1024;
const AUDIO_RING_SAMPLES: usize = 131_072;
const WAVEFORM_SAMPLES: usize = 8_192;
const MAX_AUDIO_SAMPLES_PER_FRAME: usize = 16_384;
const MAX_AUDIO_BACKLOG: usize = AUDIO_RING_SAMPLES / 2;
const FREQ_MIN: f32 = 40.0;
const FREQ_MAX: f32 = 4000.0;
const MAX_HARMONIC: u32 = 16;

// ---------------------------------------------------------------------------
// FFT setup
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
                BinMap {
                    idx,
                    frac: exact - idx as f32,
                }
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

    fn process_frame(&self, samples: &[f32], out: &mut [f32]) {
        let mut ib = self.in_buf.lock();
        let mut ob = self.out_buf.lock();
        for (d, (&s, &w)) in ib.iter_mut().zip(samples.iter().zip(self.window.iter())) {
            *d = s * w;
        }
        let _ = self.fft.process(&mut ib, &mut ob);
        for (o, bm) in out.iter_mut().zip(self.bin_map.iter()) {
            let c0 = ob[bm.idx];
            let m0 = (c0.re * c0.re + c0.im * c0.im).sqrt();
            *o = if bm.frac > 0.0 && bm.idx + 1 < ob.len() {
                let c1 = ob[bm.idx + 1];
                let m1 = (c1.re * c1.re + c1.im * c1.im).sqrt();
                m0 * (1.0 - bm.frac) + m1 * bm.frac
            } else {
                m0
            };
        }
    }
}

// ---------------------------------------------------------------------------
// colour ramp
// ---------------------------------------------------------------------------

fn spec_color(t: f32) -> egui::Color32 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.003 {
        return egui::Color32::BLACK;
    }
    let s: [(f32, (u8, u8, u8)); 7] = [
        (0.00, (0, 0, 0)),
        (0.08, (2, 2, 35)),
        (0.20, (22, 4, 80)),
        (0.35, (100, 6, 28)),
        (0.50, (175, 45, 12)),
        (0.70, (230, 175, 10)),
        (1.00, (255, 250, 235)),
    ];
    for i in 0..s.len() - 1 {
        let (t0, c0) = s[i];
        let (t1, c1) = s[i + 1];
        if t >= t0 && t <= t1 {
            let u = ((t - t0) / (t1 - t0)).clamp(0., 1.);
            return egui::Color32::from_rgb(
                (f32::from(c0.0) + f32::from(i16::from(c1.0) - i16::from(c0.0)) * u) as u8,
                (f32::from(c0.1) + f32::from(i16::from(c1.1) - i16::from(c0.1)) * u) as u8,
                (f32::from(c0.2) + f32::from(i16::from(c1.2) - i16::from(c0.2)) * u) as u8,
            );
        }
    }
    egui::Color32::WHITE
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn magnitude_dbfs(magnitude: f32) -> f32 {
    // Hann coherent gain is 0.5; a full-scale sine has a one-sided FFT
    // magnitude of N * 0.5 / 2.
    let full_scale_sine = FFT_SIZE as f32 * 0.25;
    20.0 * (magnitude.max(1e-12) / full_scale_sine).log10()
}

fn estimate_f0(mags: &[f32], freqs: &[f32]) -> Option<f32> {
    let start = freqs.iter().position(|&f| f >= 60.0)?;
    let end = freqs.iter().position(|&f| f > 300.0).unwrap_or(freqs.len());
    let (_, f0) = (start..end)
        .filter_map(|candidate| {
            let f0 = freqs[candidate];
            let score = (1..=8)
                .filter_map(|harmonic| {
                    let target = f0 * harmonic as f32;
                    (target <= FREQ_MAX).then(|| {
                        let bin = freqs
                            .partition_point(|&frequency| frequency < target)
                            .min(mags.len() - 1);
                        // Square-root compression prevents one loud whistle harmonic
                        // from dominating the fundamental candidate score.
                        mags[bin].sqrt() / harmonic as f32
                    })
                })
                .sum::<f32>();
            (score > 1e-5).then_some((score, f0))
        })
        .max_by(|(a, _), (b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;
    Some(f0)
}

fn freq_to_y(f: f32, r: &egui::Rect) -> f32 {
    let n = (f.ln() - FREQ_MIN.ln()) / (FREQ_MAX.ln() - FREQ_MIN.ln());
    r.top() + (1. - n.clamp(0., 1.)) * r.height()
}

fn freq_to_note(f: f32) -> (String, i32, f32) {
    if f <= 0.0 {
        return ("--".into(), 0, 0.0);
    }
    let semis = 12.0 * (f / 440.0).log2();
    let midi = 69.0_f32 + semis;
    let rnd = (midi + 0.5).floor() as i32;
    let notes = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    ];
    let ni = ((rnd % 12) + 12) % 12;
    let oct = (rnd / 12) - 1;
    let cents = (midi - rnd as f32) * 100.0;
    (notes[ni as usize].to_string(), oct, cents)
}

fn find_peaks(m: &[f32]) -> Vec<(usize, f32)> {
    let mut v = Vec::new();
    for i in 1..m.len().saturating_sub(1) {
        let x = m[i];
        if x > m[i - 1] && x >= m[i + 1] && x > 1e-6 {
            v.push((i, x));
        }
    }
    v
}

fn label_harmonics(p: &[(usize, f32)], freqs: &[f32], f0: f32) -> Vec<(usize, u32, f32)> {
    p.iter()
        .filter_map(|&(b, m)| {
            let f = freqs[b];
            let n = (f / f0).round() as u32;
            if !(2..=MAX_HARMONIC).contains(&n) {
                return None;
            }
            let hf = n as f32 * f0;
            ((f - hf).abs() / hf < 0.06).then_some((b, n, m))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// drone
// ---------------------------------------------------------------------------

struct DroneState {
    enabled: AtomicBool,
    frequency: AtomicU32,
    amplitude: AtomicU32,
}
impl DroneState {
    fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            frequency: AtomicU32::new(110.0_f32.to_bits()),
            amplitude: AtomicU32::new(0.3_f32.to_bits()),
        }
    }
}

// ---------------------------------------------------------------------------
// waterfall
// ---------------------------------------------------------------------------

struct Waterfall {
    pos: usize,
    filled: bool,
    peak_db: f32,
    tex: Option<egui::TextureHandle>,
    pending_columns: VecDeque<(usize, Vec<egui::Color32>)>,
}

impl Waterfall {
    fn new() -> Self {
        Self {
            pos: 0,
            filled: false,
            peak_db: -60.0,
            tex: None,
            pending_columns: VecDeque::new(),
        }
    }
    fn push(&mut self, m: &[f32]) {
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

    fn reset(&mut self) {
        self.pos = 0;
        self.filled = false;
        self.peak_db = -60.0;
        self.pending_columns.clear();
    }

    fn upload(&mut self, ctx: &egui::Context) -> egui::TextureId {
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

fn draw_piano(p: &egui::Painter, rect: egui::Rect) {
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

fn draw_waveform(
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

// ---------------------------------------------------------------------------
// app
// ---------------------------------------------------------------------------

struct VoiceHarmApp {
    audio_consumer: Consumer<f32>,
    audio_overflowed: Arc<AtomicBool>,
    audio_failed: Arc<AtomicBool>,
    analysis_samples: Vec<f32>,
    analysis_write: usize,
    analysis_len: usize,
    waveform_samples: VecDeque<f32>,
    fft_frame: Vec<f32>,
    samples_since_frame: usize,
    sample_rate: f32,
    fft_setup: FftSetup,
    waterfall: Waterfall,
    freqs: Vec<f32>,
    current_mags: Vec<f32>,
    current_f0: Option<f32>,
    waveform_render: Vec<f32>,
    waveform_points: Vec<egui::Pos2>,
    drone_state: Arc<DroneState>,
    drone_on: bool,
    drone_vol: f32,
}

impl VoiceHarmApp {
    fn new(
        sr: f32,
        audio_consumer: Consumer<f32>,
        audio_overflowed: Arc<AtomicBool>,
        audio_failed: Arc<AtomicBool>,
        drone: Arc<DroneState>,
    ) -> Self {
        let r = (FREQ_MAX / FREQ_MIN).powf(1. / (NUM_BINS - 1) as f32);
        let fs: Vec<_> = (0..NUM_BINS).map(|i| FREQ_MIN * r.powi(i as i32)).collect();
        Self {
            audio_consumer,
            audio_overflowed,
            audio_failed,
            analysis_samples: vec![0.0; FFT_SIZE],
            analysis_write: 0,
            analysis_len: 0,
            waveform_samples: VecDeque::with_capacity(WAVEFORM_SAMPLES),
            fft_frame: vec![0.0; FFT_SIZE],
            samples_since_frame: 0,
            sample_rate: sr,
            fft_setup: FftSetup::new(sr),
            waterfall: Waterfall::new(),
            freqs: fs,
            current_mags: vec![0.; NUM_BINS],
            current_f0: None,
            waveform_render: Vec::with_capacity(WAVEFORM_SAMPLES),
            waveform_points: Vec::new(),
            drone_state: drone,
            drone_on: false,
            drone_vol: 0.3,
        }
    }

    fn history_seconds(&self) -> f32 {
        SPEC_ROWS as f32 * FFT_HOP_SIZE as f32 / self.sample_rate
    }

    fn visible_history_seconds(&self) -> f32 {
        if self.waterfall.filled {
            self.history_seconds()
        } else {
            self.waterfall.pos as f32 * FFT_HOP_SIZE as f32 / self.sample_rate
        }
    }
}

impl eframe::App for VoiceHarmApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(12, 18, 25);
        visuals.window_fill = visuals.panel_fill;
        visuals.faint_bg_color = egui::Color32::from_rgb(23, 34, 45);
        visuals.selection.bg_fill = egui::Color32::from_rgb(27, 124, 133);
        ui.ctx().set_visuals(visuals);
        // A full ring means the visualizer is no longer real-time. Discard the
        // stale queue and restart from the newest samples instead of rendering
        // seconds-old audio.
        if self.audio_overflowed.swap(false, Ordering::Acquire)
            || self.audio_consumer.slots() > MAX_AUDIO_BACKLOG
        {
            while self.audio_consumer.pop().is_ok() {}
            self.analysis_write = 0;
            self.analysis_len = 0;
            self.waveform_samples.clear();
            self.samples_since_frame = 0;
            self.current_mags.fill(0.0);
            self.current_f0 = None;
            self.waterfall.reset();
        }

        let mut samples_processed = 0;
        let mut new_frame = false;
        // Consume a bounded amount per repaint. This prevents one delayed UI
        // frame from running an unbounded FFT backlog.
        while samples_processed < MAX_AUDIO_SAMPLES_PER_FRAME {
            let Ok(sample) = self.audio_consumer.pop() else {
                break;
            };
            samples_processed += 1;
            self.analysis_samples[self.analysis_write] = sample;
            self.analysis_write = (self.analysis_write + 1) % FFT_SIZE;
            self.analysis_len = (self.analysis_len + 1).min(FFT_SIZE);
            if self.waveform_samples.len() == WAVEFORM_SAMPLES {
                self.waveform_samples.pop_front();
            }
            self.waveform_samples.push_back(sample);
            self.samples_since_frame += 1;
            if self.analysis_len == FFT_SIZE && self.samples_since_frame >= FFT_HOP_SIZE {
                let first = FFT_SIZE - self.analysis_write;
                self.fft_frame[..first]
                    .copy_from_slice(&self.analysis_samples[self.analysis_write..]);
                self.fft_frame[first..]
                    .copy_from_slice(&self.analysis_samples[..self.analysis_write]);
                self.fft_setup
                    .process_frame(&self.fft_frame, &mut self.current_mags);
                self.waterfall.push(&self.current_mags);
                self.samples_since_frame = 0;
                new_frame = true;
            }
        }

        if new_frame {
            self.current_f0 = estimate_f0(&self.current_mags, &self.freqs);
            if let Some(f) = self.current_f0 {
                self.drone_state
                    .frequency
                    .store(f.to_bits(), Ordering::Relaxed);
            }
        }
        let f0 = self.current_f0;

        // One quiet control strip; the analysis itself stays the visual focus.
        egui::containers::Panel::top("top")
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(20, 30, 40))
                    .inner_margin(egui::Margin::symmetric(12, 7))
                    .stroke(egui::Stroke::new(1., egui::Color32::from_rgb(43, 62, 77))),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("Voice Harmonics")
                            .size(16.)
                            .strong()
                            .color(egui::Color32::from_rgb(235, 242, 246)),
                    );
                    let audio_live = !self.audio_failed.load(Ordering::Relaxed);
                    ui.colored_label(
                        if audio_live {
                            egui::Color32::from_rgb(48, 205, 184)
                        } else {
                            egui::Color32::from_rgb(224, 116, 92)
                        },
                        if audio_live {
                            "● LIVE"
                        } else {
                            "● AUDIO ERROR"
                        },
                    );
                    ui.separator();
                    if let Some(f) = f0 {
                        ui.colored_label(
                            egui::Color32::from_rgb(181, 225, 220),
                            format!("F0  {f:.1} Hz"),
                        );
                    } else {
                        ui.colored_label(egui::Color32::from_gray(130), "F0  --");
                    }
                    ui.separator();
                    let drone_label = egui::RichText::new("Drone").color(egui::Color32::WHITE);
                    let btn = if self.drone_on {
                        egui::Button::new(drone_label).fill(egui::Color32::from_rgb(22, 129, 111))
                    } else {
                        egui::Button::new(drone_label)
                    };
                    if ui.add(btn).clicked() {
                        self.drone_on = !self.drone_on;
                        self.drone_state
                            .enabled
                            .store(self.drone_on, Ordering::Relaxed);
                    }

                    ui.add_sized(
                        [72.0, 18.0],
                        egui::Slider::new(&mut self.drone_vol, 0.0..=1.0).show_value(false),
                    );
                    self.drone_state
                        .amplitude
                        .store(self.drone_vol.to_bits(), Ordering::Relaxed);
                });
            });

        // Fixed piano rail on the left, matching the frequency orientation.
        egui::containers::Panel::left("piano")
            .resizable(false)
            .default_size(104.0)
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(20, 30, 40))
                    .inner_margin(egui::Margin::symmetric(7, 6))
                    .stroke(egui::Stroke::new(1., egui::Color32::from_rgb(43, 62, 77))),
            )
            .show(ui, |ui| {
                let rect = ui.max_rect().shrink2(egui::vec2(2., 2.));
                draw_piano(ui.painter(), rect);
            });

        // ── central panel: spectrogram canvas ──
        egui::CentralPanel::default().show(ui, |ui| {
            let painter = ui.painter();
            let canvas = ui.max_rect();
            painter.rect_filled(canvas, 0.0, egui::Color32::from_rgb(7, 11, 16));

            // ── frequency + note labels (left axis) ──
            let label_w = 54.0;
            let waveform_h = 100.0;
            let plot_rect = egui::Rect::from_min_max(
                canvas.min,
                egui::pos2(canvas.right(), canvas.bottom() - waveform_h),
            );
            let spec_rect = egui::Rect::from_min_size(
                egui::pos2(canvas.left() + label_w, plot_rect.top()),
                egui::vec2((plot_rect.width() - label_w).max(64.0), plot_rect.height()),
            );

            // Only the newest spectral column is uploaded. Draw the circular
            // texture in two pieces so the display remains chronological.
            let texture_id = self.waterfall.upload(ui.ctx());
            let split = if self.waterfall.filled {
                self.waterfall.pos
            } else {
                0
            };
            if !self.waterfall.filled {
                let fraction = self.waterfall.pos as f32 / SPEC_ROWS as f32;
                if fraction > 0.0 {
                    let data_rect = egui::Rect::from_min_max(
                        egui::pos2(
                            spec_rect.right() - spec_rect.width() * fraction,
                            spec_rect.top(),
                        ),
                        spec_rect.max,
                    );
                    painter.image(
                        texture_id,
                        data_rect,
                        egui::Rect::from_min_max(egui::pos2(0., 0.), egui::pos2(fraction, 1.)),
                        egui::Color32::WHITE,
                    );
                }
            } else if split == 0 {
                painter.image(
                    texture_id,
                    spec_rect,
                    egui::Rect::EVERYTHING,
                    egui::Color32::WHITE,
                );
            } else {
                let first_fraction = (SPEC_ROWS - split) as f32 / SPEC_ROWS as f32;
                let first_rect = egui::Rect::from_min_max(
                    spec_rect.min,
                    egui::pos2(
                        spec_rect.left() + spec_rect.width() * first_fraction,
                        spec_rect.bottom(),
                    ),
                );
                painter.image(
                    texture_id,
                    first_rect,
                    egui::Rect::from_min_max(egui::pos2(first_fraction, 0.), egui::pos2(1., 1.)),
                    egui::Color32::WHITE,
                );
                let second_rect = egui::Rect::from_min_max(
                    egui::pos2(first_rect.right(), spec_rect.top()),
                    spec_rect.max,
                );
                painter.image(
                    texture_id,
                    second_rect,
                    egui::Rect::from_min_max(egui::pos2(0., 0.), egui::pos2(first_fraction, 1.)),
                    egui::Color32::WHITE,
                );
            }
            painter.rect_stroke(
                spec_rect,
                2.0,
                egui::Stroke::new(1.0, egui::Color32::from_rgb(42, 61, 76)),
                egui::StrokeKind::Inside,
            );

            // ── note octave grid lines ──
            for oct in 2..=7 {
                let cf = 16.3516 * 2.0_f32.powi(oct);
                if !(FREQ_MIN..=FREQ_MAX).contains(&cf) {
                    continue;
                }
                let y = freq_to_y(cf, &spec_rect);
                painter.text(
                    egui::pos2(canvas.left() + 4.0, y),
                    egui::Align2::LEFT_CENTER,
                    format!("C{oct}"),
                    egui::FontId::proportional(10.0),
                    egui::Color32::from_rgba_premultiplied(125, 176, 180, 150),
                );
                painter.line_segment(
                    [
                        egui::pos2(spec_rect.left(), y),
                        egui::pos2(spec_rect.right(), y),
                    ],
                    egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgba_premultiplied(68, 127, 134, 35),
                    ),
                );
            }

            // ── horizontal frequency grid ──
            let lf: [f32; 8] = [50., 100., 200., 500., 1000., 2000., 3000., 4000.];
            for &f in &lf {
                if !(FREQ_MIN..=FREQ_MAX).contains(&f) {
                    continue;
                }
                let y = freq_to_y(f, &spec_rect);
                let lb = if f >= 1000. {
                    format!("{}k", (f / 1000.) as u32)
                } else {
                    format!("{f}")
                };
                painter.text(
                    egui::pos2(spec_rect.left() - 5., y),
                    egui::Align2::RIGHT_CENTER,
                    &lb,
                    egui::FontId::proportional(11.),
                    egui::Color32::from_rgb(149, 165, 178),
                );
                painter.line_segment(
                    [
                        egui::pos2(spec_rect.left(), y),
                        egui::pos2(spec_rect.right(), y),
                    ],
                    egui::Stroke::new(
                        1.,
                        egui::Color32::from_rgba_premultiplied(114, 145, 162, 40),
                    ),
                );
            }
            painter.text(
                egui::pos2(4., plot_rect.top() + 4.),
                egui::Align2::LEFT_TOP,
                "Frequency\n(Hz)",
                egui::FontId::proportional(10.),
                egui::Color32::from_rgb(128, 151, 166),
            );
            for sec in 0..=4 {
                let x = spec_rect.left() + sec as f32 / 4. * spec_rect.width();
                painter.line_segment(
                    [
                        egui::pos2(x, spec_rect.top()),
                        egui::pos2(x, spec_rect.bottom()),
                    ],
                    egui::Stroke::new(
                        1.,
                        egui::Color32::from_rgba_premultiplied(114, 145, 162, 32),
                    ),
                );
                painter.text(
                    egui::pos2(x, spec_rect.bottom() + 4.),
                    egui::Align2::CENTER_TOP,
                    format!(
                        "-{:.1}s",
                        (4 - sec) as f32 * self.visible_history_seconds() / 4.0
                    ),
                    egui::FontId::proportional(9.),
                    egui::Color32::from_rgb(128, 151, 166),
                );
            }
            let wave_rect = egui::Rect::from_min_max(
                egui::pos2(canvas.left() + 4., plot_rect.bottom() + 24.),
                egui::pos2(canvas.right() - 4., canvas.bottom() - 20.),
            );
            self.waveform_render.clear();
            self.waveform_render
                .extend(self.waveform_samples.iter().copied());
            draw_waveform(
                painter,
                wave_rect,
                &self.waveform_render,
                &mut self.waveform_points,
            );
            painter.text(
                egui::pos2(wave_rect.left() + 4., wave_rect.top() + 3.),
                egui::Align2::LEFT_TOP,
                "INPUT LEVEL",
                egui::FontId::proportional(10.),
                egui::Color32::from_rgb(128, 177, 181),
            );
            let note_readout = f0.map_or_else(
                || "--    listening…".into(),
                |f| {
                    let (n, o, c) = freq_to_note(f);
                    format!("{n}{o} {c:+.0} ct    {f:.0} Hz")
                },
            );
            painter.rect_filled(
                egui::Rect::from_min_max(
                    egui::pos2(canvas.left(), canvas.bottom() - 18.),
                    canvas.max,
                ),
                0.,
                egui::Color32::from_rgb(15, 23, 31),
            );
            painter.text(
                egui::pos2(canvas.left() + 8., canvas.bottom() - 9.),
                egui::Align2::LEFT_CENTER,
                note_readout,
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

            // ── harmonic number labels ──
            if let Some(f) = f0 {
                let peaks = find_peaks(&self.current_mags);
                for &(_b, n, _m) in &label_harmonics(&peaks, &self.freqs, f) {
                    let ff = (n as f32 * f).clamp(FREQ_MIN, FREQ_MAX);
                    let y = freq_to_y(ff, &spec_rect);
                    let tag = format!("{n}");
                    let tr = egui::Rect::from_min_size(
                        egui::pos2(spec_rect.left() - 16., y - 7.),
                        egui::vec2(16., 14.),
                    );
                    painter.rect_filled(tr, 3., egui::Color32::from_rgb(25, 65, 73));
                    painter.text(
                        tr.center(),
                        egui::Align2::CENTER_CENTER,
                        &tag,
                        egui::FontId::proportional(10.),
                        egui::Color32::from_rgb(190, 238, 227),
                    );
                    // Keep the detected-harmonic marker at the scale, without
                    // drawing a full-width guide that can be mistaken for data.
                    painter.line_segment(
                        [
                            egui::pos2(spec_rect.left(), y),
                            egui::pos2(spec_rect.left() + 7., y),
                        ],
                        egui::Stroke::new(1., egui::Color32::from_rgb(85, 210, 192)),
                    );
                }
            }

            // ── cursor overlay ──
            let data_start_x = if self.waterfall.filled {
                spec_rect.left()
            } else {
                spec_rect.right() - spec_rect.width() * self.waterfall.pos as f32 / SPEC_ROWS as f32
            };
            if let Some(mp) = ui.ctx().pointer_hover_pos()
                && spec_rect.contains(mp)
                && mp.x >= data_start_x
            {
                let xn = ((mp.x - spec_rect.left()) / spec_rect.width()).clamp(0., 1.);
                let yn = ((mp.y - spec_rect.top()) / spec_rect.height()).clamp(0., 1.);
                let freq = FREQ_MIN * (FREQ_MAX / FREQ_MIN).powf(1. - yn);
                let bin = ((1. - yn) * (NUM_BINS - 1) as f32).round() as usize;
                let mag = self.current_mags.get(bin).copied().unwrap_or(0.);
                let db = magnitude_dbfs(mag);
                let t_sec = (1. - xn) * self.visible_history_seconds();

                let cc = egui::Color32::from_rgba_premultiplied(200, 200, 200, 70);
                painter.line_segment(
                    [
                        egui::pos2(mp.x, spec_rect.top()),
                        egui::pos2(mp.x, spec_rect.bottom()),
                    ],
                    egui::Stroke::new(1., cc),
                );
                painter.line_segment(
                    [
                        egui::pos2(spec_rect.left(), mp.y),
                        egui::pos2(spec_rect.right(), mp.y),
                    ],
                    egui::Stroke::new(1., cc),
                );

                let (n, oct, _c) = freq_to_note(freq);
                let fs = if freq >= 1000. {
                    format!("{:.1}kHz", freq / 1000.)
                } else {
                    format!("{freq:.1}Hz")
                };
                let info = format!("{n}{oct} ({fs})\n{db:.1} dBFS\n-{t_sec:.1}s");
                let font = egui::FontId::monospace(13.);
                let g = painter.layout_no_wrap(info, font, egui::Color32::WHITE);
                let pad = 5.;
                let sz = egui::vec2(g.size().x + pad * 2., g.size().y + pad * 2.);
                let mut bp = egui::pos2(mp.x + 14., mp.y - sz.y - 6.);
                bp.x = bp.x.clamp(canvas.left() + 2., canvas.right() - sz.x - 2.);
                bp.y =
                    bp.y.clamp(spec_rect.top() + 2., spec_rect.bottom() - sz.y - 2.);
                let br = egui::Rect::from_min_size(bp, sz);
                painter.rect_filled(br, 3., egui::Color32::from_rgba_premultiplied(0, 0, 0, 210));
                painter.rect_stroke(
                    br,
                    3.,
                    egui::Stroke::new(
                        1.,
                        egui::Color32::from_rgba_premultiplied(200, 200, 200, 100),
                    ),
                    egui::StrokeKind::Outside,
                );
                painter.galley(egui::pos2(bp.x + pad, bp.y + pad), g, egui::Color32::WHITE);
            }
        });

        ui.ctx().request_repaint();
    }
}

// ---------------------------------------------------------------------------
// audio capture thread
// ---------------------------------------------------------------------------

fn push_mono<T>(producer: &mut Producer<f32>, data: &[T], channels: usize, overflowed: &AtomicBool)
where
    T: Sample,
    f32: cpal::FromSample<T>,
{
    for frame in data.chunks(channels.max(1)) {
        let mono = frame
            .iter()
            .copied()
            .map(|sample| sample.to_sample::<f32>())
            .sum::<f32>()
            / frame.len() as f32;
        if producer.push(mono).is_err() {
            // Never block the audio callback. The consumer will discard its
            // stale backlog and restart from fresh input on the next repaint.
            overflowed.store(true, Ordering::Release);
        }
    }
}

fn run_audio(
    mut producer: Producer<f32>,
    overflowed: Arc<AtomicBool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let dev = host.default_input_device().ok_or("no mic")?;
    let cfg = dev.default_input_config()?;
    let channels = usize::from(cfg.channels());
    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => dev.build_input_stream::<f32, _, _>(
            cfg.into(),
            move |d, _| push_mono(&mut producer, d, channels, &overflowed),
            |e| eprintln!("audio: {e}"),
            None,
        )?,
        cpal::SampleFormat::F64 => dev.build_input_stream::<f64, _, _>(
            cfg.into(),
            move |d, _| push_mono(&mut producer, d, channels, &overflowed),
            |e| eprintln!("audio: {e}"),
            None,
        )?,
        cpal::SampleFormat::I8 => dev.build_input_stream::<i8, _, _>(
            cfg.into(),
            move |d, _| push_mono(&mut producer, d, channels, &overflowed),
            |e| eprintln!("audio: {e}"),
            None,
        )?,
        cpal::SampleFormat::I16 => dev.build_input_stream::<i16, _, _>(
            cfg.into(),
            move |d, _| push_mono(&mut producer, d, channels, &overflowed),
            |e| eprintln!("audio: {e}"),
            None,
        )?,
        cpal::SampleFormat::I24 => dev.build_input_stream::<cpal::I24, _, _>(
            cfg.into(),
            move |d, _| push_mono(&mut producer, d, channels, &overflowed),
            |e| eprintln!("audio: {e}"),
            None,
        )?,
        cpal::SampleFormat::I32 => dev.build_input_stream::<i32, _, _>(
            cfg.into(),
            move |d, _| push_mono(&mut producer, d, channels, &overflowed),
            |e| eprintln!("audio: {e}"),
            None,
        )?,
        cpal::SampleFormat::I64 => dev.build_input_stream::<i64, _, _>(
            cfg.into(),
            move |d, _| push_mono(&mut producer, d, channels, &overflowed),
            |e| eprintln!("audio: {e}"),
            None,
        )?,
        cpal::SampleFormat::U8 => dev.build_input_stream::<u8, _, _>(
            cfg.into(),
            move |d, _| push_mono(&mut producer, d, channels, &overflowed),
            |e| eprintln!("audio: {e}"),
            None,
        )?,
        cpal::SampleFormat::U16 => dev.build_input_stream::<u16, _, _>(
            cfg.into(),
            move |d, _| push_mono(&mut producer, d, channels, &overflowed),
            |e| eprintln!("audio: {e}"),
            None,
        )?,
        cpal::SampleFormat::U24 => dev.build_input_stream::<cpal::U24, _, _>(
            cfg.into(),
            move |d, _| push_mono(&mut producer, d, channels, &overflowed),
            |e| eprintln!("audio: {e}"),
            None,
        )?,
        cpal::SampleFormat::U32 => dev.build_input_stream::<u32, _, _>(
            cfg.into(),
            move |d, _| push_mono(&mut producer, d, channels, &overflowed),
            |e| eprintln!("audio: {e}"),
            None,
        )?,
        cpal::SampleFormat::U64 => dev.build_input_stream::<u64, _, _>(
            cfg.into(),
            move |d, _| push_mono(&mut producer, d, channels, &overflowed),
            |e| eprintln!("audio: {e}"),
            None,
        )?,
        _ => return Err(cpal::Error::new(cpal::ErrorKind::InvalidInput).into()),
    };
    stream.play()?;
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

// ---------------------------------------------------------------------------
// drone output thread
// ---------------------------------------------------------------------------

fn write_drone<T>(
    data: &mut [T],
    channels: usize,
    drone: &DroneState,
    phase: &mut f32,
    sample_rate: f32,
) where
    T: Sample + cpal::FromSample<f32>,
{
    let enabled = drone.enabled.load(Ordering::Relaxed);
    let frequency = f32::from_bits(drone.frequency.load(Ordering::Relaxed));
    let amplitude = f32::from_bits(drone.amplitude.load(Ordering::Relaxed));
    for frame in data.chunks_mut(channels.max(1)) {
        let sample = if enabled && frequency > 0.0 {
            let value = (*phase * 2.0 * PI).sin() * amplitude;
            *phase = (*phase + frequency / sample_rate) % 1.0;
            value
        } else {
            0.0
        };
        for output in frame {
            *output = sample.to_sample();
        }
    }
}

fn run_drone(drone: Arc<DroneState>) -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();
    let dev = host.default_output_device().ok_or("no output")?;
    let cfg = dev.default_output_config()?;
    let sr = cfg.sample_rate() as f32;
    let channels = usize::from(cfg.channels());
    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => dev.build_output_stream::<f32, _, _>(
            cfg.into(),
            {
                let mut phase = 0.0;
                move |data, _| write_drone(data, channels, &drone, &mut phase, sr)
            },
            |e| eprintln!("drone: {e}"),
            None,
        )?,
        cpal::SampleFormat::F64 => dev.build_output_stream::<f64, _, _>(
            cfg.into(),
            {
                let mut phase = 0.0;
                move |data, _| write_drone(data, channels, &drone, &mut phase, sr)
            },
            |e| eprintln!("drone: {e}"),
            None,
        )?,
        cpal::SampleFormat::I8 => dev.build_output_stream::<i8, _, _>(
            cfg.into(),
            {
                let mut phase = 0.0;
                move |data, _| write_drone(data, channels, &drone, &mut phase, sr)
            },
            |e| eprintln!("drone: {e}"),
            None,
        )?,
        cpal::SampleFormat::I16 => dev.build_output_stream::<i16, _, _>(
            cfg.into(),
            {
                let mut phase = 0.0;
                move |data, _| write_drone(data, channels, &drone, &mut phase, sr)
            },
            |e| eprintln!("drone: {e}"),
            None,
        )?,
        cpal::SampleFormat::I24 => dev.build_output_stream::<cpal::I24, _, _>(
            cfg.into(),
            {
                let mut phase = 0.0;
                move |data, _| write_drone(data, channels, &drone, &mut phase, sr)
            },
            |e| eprintln!("drone: {e}"),
            None,
        )?,
        cpal::SampleFormat::I32 => dev.build_output_stream::<i32, _, _>(
            cfg.into(),
            {
                let mut phase = 0.0;
                move |data, _| write_drone(data, channels, &drone, &mut phase, sr)
            },
            |e| eprintln!("drone: {e}"),
            None,
        )?,
        cpal::SampleFormat::I64 => dev.build_output_stream::<i64, _, _>(
            cfg.into(),
            {
                let mut phase = 0.0;
                move |data, _| write_drone(data, channels, &drone, &mut phase, sr)
            },
            |e| eprintln!("drone: {e}"),
            None,
        )?,
        cpal::SampleFormat::U8 => dev.build_output_stream::<u8, _, _>(
            cfg.into(),
            {
                let mut phase = 0.0;
                move |data, _| write_drone(data, channels, &drone, &mut phase, sr)
            },
            |e| eprintln!("drone: {e}"),
            None,
        )?,
        cpal::SampleFormat::U16 => dev.build_output_stream::<u16, _, _>(
            cfg.into(),
            {
                let mut phase = 0.0;
                move |data, _| write_drone(data, channels, &drone, &mut phase, sr)
            },
            |e| eprintln!("drone: {e}"),
            None,
        )?,
        cpal::SampleFormat::U24 => dev.build_output_stream::<cpal::U24, _, _>(
            cfg.into(),
            {
                let mut phase = 0.0;
                move |data, _| write_drone(data, channels, &drone, &mut phase, sr)
            },
            |e| eprintln!("drone: {e}"),
            None,
        )?,
        cpal::SampleFormat::U32 => dev.build_output_stream::<u32, _, _>(
            cfg.into(),
            {
                let mut phase = 0.0;
                move |data, _| write_drone(data, channels, &drone, &mut phase, sr)
            },
            |e| eprintln!("drone: {e}"),
            None,
        )?,
        cpal::SampleFormat::U64 => dev.build_output_stream::<u64, _, _>(
            cfg.into(),
            {
                let mut phase = 0.0;
                move |data, _| write_drone(data, channels, &drone, &mut phase, sr)
            },
            |e| eprintln!("drone: {e}"),
            None,
        )?,
        _ => return Err(cpal::Error::new(cpal::ErrorKind::InvalidInput).into()),
    };
    stream.play()?;
    loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() -> Result<(), eframe::Error> {
    let sr = cpal::default_host()
        .default_input_device()
        .and_then(|d| d.default_input_config().ok())
        .map_or(44100., |c| c.sample_rate() as f32);
    let (audio_producer, audio_consumer) = RingBuffer::<f32>::new(AUDIO_RING_SAMPLES);
    let audio_overflowed = Arc::new(AtomicBool::new(false));
    let audio_failed = Arc::new(AtomicBool::new(false));
    let drone = Arc::new(DroneState::new());

    std::thread::spawn({
        let a = audio_producer;
        let overflowed = Arc::clone(&audio_overflowed);
        let failed = Arc::clone(&audio_failed);
        move || {
            if let Err(e) = run_audio(a, overflowed) {
                eprintln!("audio:{e}");
                failed.store(true, Ordering::Release);
            }
        }
    });
    std::thread::spawn({
        let d = drone.clone();
        move || {
            if let Err(e) = run_drone(d) {
                eprintln!("drone:{e}");
            }
        }
    });

    eframe::run_native(
        "Voice Harmonics Analyzer",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1800., 860.]),
            ..Default::default()
        },
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(VoiceHarmApp::new(
                sr,
                audio_consumer,
                audio_overflowed,
                audio_failed,
                drone,
            )))
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_frequencies() -> Vec<f32> {
        let ratio = (FREQ_MAX / FREQ_MIN).powf(1.0 / (NUM_BINS - 1) as f32);
        (0..NUM_BINS)
            .map(|i| FREQ_MIN * ratio.powi(i as i32))
            .collect()
    }

    #[test]
    fn dbfs_is_calibrated_for_a_full_scale_sine() {
        assert!(magnitude_dbfs(FFT_SIZE as f32 * 0.25).abs() < 1e-4);
    }

    #[test]
    fn harmonic_scoring_finds_a_weak_fundamental() {
        let freqs = log_frequencies();
        let mut mags = vec![0.0; NUM_BINS];
        let fundamental = 110.0;
        for harmonic in 1..=8 {
            let target = fundamental * harmonic as f32;
            let index = freqs
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (*a - target)
                        .abs()
                        .partial_cmp(&(*b - target).abs())
                        .unwrap()
                })
                .unwrap()
                .0;
            mags[index] = if harmonic == 1 { 0.02 } else { 1.0 };
        }
        assert!((estimate_f0(&mags, &freqs).unwrap() - fundamental).abs() < 1.0);
    }

    #[test]
    fn stereo_input_is_downmixed_per_audio_frame() {
        let (mut producer, mut consumer) = RingBuffer::new(8);
        let overflowed = AtomicBool::new(false);
        push_mono(&mut producer, &[0.5_f32, -0.5, 1.0, 1.0], 2, &overflowed);
        assert_eq!(consumer.pop().unwrap(), 0.0);
        assert_eq!(consumer.pop().unwrap(), 1.0);
    }

    #[test]
    fn waterfall_peak_releases_after_a_loud_frame() {
        let mut waterfall = Waterfall::new();
        waterfall.push(&vec![1.0; NUM_BINS]);
        let loud_peak = waterfall.peak_db;
        for _ in 0..20 {
            waterfall.push(&vec![0.001; NUM_BINS]);
        }
        assert!(waterfall.peak_db < loud_peak);
    }
}
