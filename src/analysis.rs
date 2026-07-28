use crate::config::*;
use imgui::ImColor32;
use parking_lot::Mutex;
use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex};
use std::f32::consts::PI;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// FFT setup
// ---------------------------------------------------------------------------

struct BinMap {
    idx: usize,
    frac: f32,
}

pub(crate) struct FftSetup {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    bin_map: Vec<BinMap>,
    in_buf: Mutex<Vec<f32>>,
    out_buf: Mutex<Vec<Complex<f32>>>,
}

impl FftSetup {
    pub(crate) fn new(sample_rate: f32) -> Self {
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

    pub(crate) fn process_frame(&self, samples: &[f32], out: &mut [f32]) {
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

fn u8f(v: u8) -> f32 {
    v as f32 / 255.0
}

pub(crate) fn spec_color(t: f32) -> ImColor32 {
    let t = t.clamp(0.0, 1.0);
    if t < 0.003 {
        return ImColor32::BLACK;
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
            return ImColor32::from([
                u8f((f32::from(c0.0) + f32::from(i16::from(c1.0) - i16::from(c0.0)) * u) as u8),
                u8f((f32::from(c0.1) + f32::from(i16::from(c1.1) - i16::from(c0.1)) * u) as u8),
                u8f((f32::from(c0.2) + f32::from(i16::from(c1.2) - i16::from(c0.2)) * u) as u8),
                1.0,
            ]);
        }
    }
    ImColor32::WHITE
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

pub(crate) fn magnitude_dbfs(magnitude: f32) -> f32 {
    // Hann coherent gain is 0.5; a full-scale sine has a one-sided FFT
    // magnitude of N * 0.5 / 2.
    let full_scale_sine = FFT_SIZE as f32 * 0.25;
    20.0 * (magnitude.max(1e-12) / full_scale_sine).log10()
}

pub(crate) fn estimate_f0(mags: &[f32], freqs: &[f32]) -> Option<f32> {
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

/// rect = [x1, y1, x2, y2] (y1=top, y2=bottom)
pub(crate) fn freq_to_y(f: f32, rect: &[f32; 4]) -> f32 {
    let n = (f.ln() - FREQ_MIN.ln()) / (FREQ_MAX.ln() - FREQ_MIN.ln());
    rect[1] + (1. - n.clamp(0., 1.)) * (rect[3] - rect[1])
}

pub(crate) fn freq_to_note(f: f32) -> (String, i32, f32) {
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

pub(crate) fn find_peaks(m: &[f32]) -> Vec<(usize, f32)> {
    let mut v = Vec::new();
    for i in 1..m.len().saturating_sub(1) {
        let x = m[i];
        if x > m[i - 1] && x >= m[i + 1] && x > 1e-6 {
            v.push((i, x));
        }
    }
    v
}

pub(crate) fn label_harmonics(
    p: &[(usize, f32)],
    freqs: &[f32],
    f0: f32,
) -> Vec<(usize, u32, f32)> {
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
