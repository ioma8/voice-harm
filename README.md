# Voice Harmonics Analyzer

Real-time microphone spectral analyzer for **aliquot (overtone) singing** training — shows which harmonics appear as you sing into the microphone.

![screenshot](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-blue)

## Features

- **FFT‑based spectrogram** — 16384‑point Hann‑windowed real FFT, 2.7 Hz/bin resolution
- **1600 log‑spaced frequency bins** (40 Hz – 4 kHz — vocal fundamentals + harmonics)
- **Scrolling waterfall** (800 rows, newest at bottom, ~13 s history)
- **Professional dBFS colormap** — black → navy → purple → red → orange → yellow → white
- **Cursor readout** — hover anywhere for frequency (Hz/kHz), dB level, and time offset
- **F0 detection** — estimated fundamental shown top‑left
- **dBFS colour bar** on the right edge

## Dependencies

| Crate | Role |
|---|---|
| `cpal` | Microphone capture (cross‑platform) |
| `realfft` / `rustfft` | Hann‑windowed real FFT |
| `eframe` / `egui` | Native GUI window |
| `parking_lot` | Fast mutex |

## Build & Run

```sh
cargo run
```

A window opens. Sing into your microphone — the spectrogram scrolls upward showing your vocal harmonics.

## Controls

- **Mouse over the graph** — shows frequency, dB, and time offset at cursor
- **F0** (top‑left) — detected fundamental frequency (60–300 Hz peak)
- **Colour bar** (right) — 0 to −60 dB reference
