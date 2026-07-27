# Voice Harmonics Analyzer

Real-time microphone spectral analyzer for **aliquot (overtone) singing** training — shows which harmonics appear as you sing into the microphone.

![screenshot](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-blue)

## Features

- **FFT‑based spectrogram** — 16384‑point Hann‑windowed real FFT, 2.7 Hz/bin resolution at 44.1 kHz
- **2400 log‑spaced frequency bins** (40 Hz – 4 kHz — vocal fundamentals + harmonics)
- **Fixed-hop scrolling waterfall** (800 columns, newest at right; duration follows the active sample rate)
- **Calibrated dBFS colormap** — black → navy → purple → red → orange → yellow → white
- **Cursor readout** — hover anywhere for frequency (Hz/kHz), dBFS level, and time offset
- **F0 detection** — estimated fundamental shown top‑left
- **Stereo/multichannel-safe capture** — channels are downmixed before analysis

## Dependencies

| Crate | Role |
|---|---|
| `cpal` | Microphone capture (cross‑platform) |
| `realfft` / `rustfft` | Hann‑windowed real FFT |
| `eframe` / `egui` | Native GUI window |
| `parking_lot` | Fast mutex |
| `rtrb` | Lock-free single-producer/single-consumer audio handoff |

## Build & Run

```sh
cargo run
```

A window opens. Sing into your microphone — the spectrogram scrolls left-to-right showing your vocal harmonics.

## Controls

- **Mouse over the graph** — shows frequency, dBFS, and time offset at cursor
- **F0** (top-left) — harmonic-scoring estimate for 60–300 Hz fundamentals
- **Drone** — play the detected F0; the volume slider adjusts its level
