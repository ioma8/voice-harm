use std::sync::atomic::AtomicBool;
use rtrb::RingBuffer;
use crate::analysis::{estimate_f0, magnitude_dbfs};
use crate::audio::push_mono;
use crate::config::*;
use crate::waterfall::Waterfall;
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
