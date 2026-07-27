mod analysis;
mod app;
mod audio;
mod config;
mod drawing;
mod drone;
mod ui;
mod ui_canvas;
mod ui_overlay;
mod waterfall;

use crate::app::VoiceHarmApp;
use crate::audio::run_audio;
use crate::config::AUDIO_RING_SAMPLES;
use crate::drone::{DroneState, run_drone};
use cpal::traits::{DeviceTrait, HostTrait};
use rtrb::RingBuffer;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
        let overflowed = Arc::clone(&audio_overflowed);
        let failed = Arc::clone(&audio_failed);
        move || {
            if let Err(e) = run_audio(audio_producer, overflowed) {
                eprintln!("audio:{e}");
                failed.store(true, Ordering::Release);
            }
        }
    });
    std::thread::spawn({
        let d = Arc::clone(&drone);
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
    include!("tests.rs");
}
