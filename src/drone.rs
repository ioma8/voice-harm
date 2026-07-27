use cpal::Sample;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::f32::consts::PI;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicBool, AtomicU32};

// ---------------------------------------------------------------------------
// drone
// ---------------------------------------------------------------------------

pub(crate) struct DroneState {
    pub(crate) enabled: AtomicBool,
    pub(crate) frequency: AtomicU32,
    pub(crate) amplitude: AtomicU32,
}
impl DroneState {
    pub(crate) fn new() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            frequency: AtomicU32::new(110.0_f32.to_bits()),
            amplitude: AtomicU32::new(0.3_f32.to_bits()),
        }
    }
}
// ---------------------------------------------------------------------------
// drone output thread
// ---------------------------------------------------------------------------

pub(crate) fn write_drone<T>(
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

pub(crate) fn run_drone(drone: Arc<DroneState>) -> Result<(), Box<dyn std::error::Error>> {
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
