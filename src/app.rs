use crate::analysis::FftSetup;
use crate::config::*;
use crate::drone::DroneState;
use crate::waterfall::Waterfall;
use rtrb::Consumer;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) struct VoiceHarmApp {
    pub(crate) audio_consumer: Consumer<f32>,
    pub(crate) audio_overflowed: Arc<AtomicBool>,
    pub(crate) audio_failed: Arc<AtomicBool>,
    pub(crate) analysis_samples: Vec<f32>,
    pub(crate) analysis_write: usize,
    pub(crate) analysis_len: usize,
    pub(crate) waveform_samples: VecDeque<f32>,
    pub(crate) fft_frame: Vec<f32>,
    pub(crate) samples_since_frame: usize,
    pub(crate) sample_rate: f32,
    pub(crate) fft_setup: FftSetup,
    pub(crate) waterfall: Waterfall,
    pub(crate) freqs: Vec<f32>,
    pub(crate) current_mags: Vec<f32>,
    pub(crate) current_f0: Option<f32>,
    pub(crate) waveform_render: Vec<f32>,
    pub(crate) waveform_points: Vec<[f32; 2]>,
    pub(crate) drone_state: Arc<DroneState>,
    pub(crate) drone_on: bool,
    pub(crate) drone_vol: f32,
}

impl VoiceHarmApp {
    pub(crate) fn new(
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

    pub(crate) fn history_seconds(&self) -> f32 {
        SPEC_ROWS as f32 * FFT_HOP_SIZE as f32 / self.sample_rate
    }

    pub(crate) fn visible_history_seconds(&self) -> f32 {
        if self.waterfall.filled {
            self.history_seconds()
        } else {
            self.waterfall.pos as f32 * FFT_HOP_SIZE as f32 / self.sample_rate
        }
    }

    pub(crate) fn update_audio(&mut self) -> Option<f32> {
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

        let mut processed = 0;
        let mut new_frame = false;
        while processed < MAX_AUDIO_SAMPLES_PER_FRAME {
            let Ok(sample) = self.audio_consumer.pop() else {
                break;
            };
            processed += 1;
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
            self.current_f0 = crate::analysis::estimate_f0(&self.current_mags, &self.freqs);
            if let Some(f) = self.current_f0 {
                self.drone_state
                    .frequency
                    .store(f.to_bits(), Ordering::Relaxed);
            }
        }
        self.current_f0
    }
}
