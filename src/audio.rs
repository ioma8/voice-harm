use cpal::Sample;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rtrb::Producer;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

// ---------------------------------------------------------------------------
// audio capture thread
// ---------------------------------------------------------------------------

pub(crate) fn push_mono<T>(
    producer: &mut Producer<f32>,
    data: &[T],
    channels: usize,
    overflowed: &AtomicBool,
) where
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

pub(crate) fn run_audio(
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
        std::thread::sleep(Duration::from_secs(1));
    }
}
