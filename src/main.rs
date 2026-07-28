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
use glium::Surface;
use glutin::{
    config::ConfigTemplateBuilder,
    context::ContextAttributesBuilder,
    display::GetGlDisplay,
    prelude::*,
    surface::{SurfaceAttributesBuilder, WindowSurface},
};
use winit::raw_window_handle::HasWindowHandle;
use imgui_winit_support::winit::{
    dpi::LogicalSize,
    event::Event,
    event_loop::EventLoop,
    window::{Window, WindowAttributes},
};
use rtrb::RingBuffer;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

fn main() {
    // --- audio setup (unchanged) ---
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

    // --- window + OpenGL ---
    let (event_loop, window, display) = create_window();
    let (mut winit_platform, mut imgui_context, _font_small, font_large) = imgui_init(&window);

    // --- app state ---
    let mut app = VoiceHarmApp::new(sr, audio_consumer, audio_overflowed, audio_failed, drone, font_large);
    let mut renderer = imgui_glium_renderer::Renderer::new(&mut imgui_context, &display)
        .expect("Failed to initialize renderer");

    let mut last_frame = Instant::now();

    #[allow(deprecated)]
    let _ = event_loop.run(move |event, window_target| {
        winit_platform.handle_event(imgui_context.io_mut(), &window, &event);

        match event {
            Event::NewEvents(_) => {
                let now = Instant::now();
                imgui_context.io_mut().update_delta_time(now - last_frame);
                last_frame = now;
            }
            Event::AboutToWait => {
                winit_platform
                    .prepare_frame(imgui_context.io_mut(), &window)
                    .expect("Failed to prepare frame");
                window.request_redraw();
            }
            Event::WindowEvent {
                event: winit::event::WindowEvent::RedrawRequested,
                ..
            } => {
                // Process audio and upload waterfall data BEFORE the imgui frame
                let f0 = app.update_audio();
                app.waterfall.upload(&display, &mut renderer);

                let ui = imgui_context.frame();

                // Build UI (no audio processing inside)
                crate::ui::build_ui(&mut app, &ui, f0);

                // Render
                let mut target = display.draw();
                target.clear_color_srgb(0.047, 0.071, 0.098, 1.0);
                winit_platform.prepare_render(&ui, &window);
                let draw_data = imgui_context.render();
                renderer
                    .render(&mut target, draw_data)
                    .expect("Rendering failed");
                target.finish().expect("Failed to swap buffers");
            }
            Event::WindowEvent {
                event: winit::event::WindowEvent::Resized(new_size),
                ..
            } => {
                if new_size.width > 0 && new_size.height > 0 {
                    display.resize((new_size.width, new_size.height));
                }
            }
            Event::WindowEvent {
                event: winit::event::WindowEvent::CloseRequested,
                ..
            } => window_target.exit(),
            _ => {}
        }
    });
}

fn create_window() -> (EventLoop<()>, Window, glium::Display<WindowSurface>) {
    let event_loop = EventLoop::new().expect("Failed to create EventLoop");

    let window_attributes = WindowAttributes::default()
        .with_title("Voice Harmonics Analyzer")
        .with_inner_size(LogicalSize::new(1800.0, 860.0));

    let (window, cfg) = glutin_winit::DisplayBuilder::new()
        .with_window_attributes(Some(window_attributes.clone()))
        .build(&event_loop, ConfigTemplateBuilder::new(), |mut configs| {
            configs.next().unwrap()
        })
        .expect("Failed to create OpenGL window");
    let window = window.unwrap();

    let context_attribs =
        ContextAttributesBuilder::new().build(Some(window.window_handle().unwrap().as_raw()));
    let context = unsafe {
        cfg.display()
            .create_context(&cfg, &context_attribs)
            .expect("Failed to create OpenGL context")
    };

    let size = window.inner_size();
    let surface_attribs = SurfaceAttributesBuilder::<WindowSurface>::new().build(
        window.window_handle().unwrap().as_raw(),
        NonZeroU32::new(size.width).unwrap(),
        NonZeroU32::new(size.height).unwrap(),
    );
    let surface = unsafe {
        cfg.display()
            .create_window_surface(&cfg, &surface_attribs)
            .expect("Failed to create OpenGL surface")
    };

    let context = context
        .make_current(&surface)
        .expect("Failed to make OpenGL context current");

    let display = glium::Display::from_context_surface(context, surface)
        .expect("Failed to create glium Display");

    (event_loop, window, display)
}

fn imgui_init(window: &Window) -> (imgui_winit_support::WinitPlatform, imgui::Context, imgui::FontId, imgui::FontId) {
    let mut imgui_context = imgui::Context::create();
    imgui_context.set_ini_filename(None);

    let mut winit_platform = imgui_winit_support::WinitPlatform::new(&mut imgui_context);
    let dpi_mode = imgui_winit_support::HiDpiMode::Default;
    winit_platform.attach_window(imgui_context.io_mut(), window, dpi_mode);

    let _font_small = imgui_context
        .fonts()
        .add_font(&[imgui::FontSource::DefaultFontData { config: None }]);
    let font_large = imgui_context
        .fonts()
        .add_font(&[imgui::FontSource::DefaultFontData {
            config: Some(imgui::FontConfig {
                size_pixels: 16.0,
                ..Default::default()
            }),
        }]);

    // Dark style + polish
    let style = imgui_context.style_mut();
    style.use_dark_colors();
    style.window_rounding = 4.0;
    style.frame_rounding = 3.0;
    style.grab_rounding = 3.0;
    style.scrollbar_rounding = 3.0;
    style.frame_border_size = 0.0;

    (winit_platform, imgui_context, _font_small, font_large)
}

#[cfg(test)]
mod tests {
    include!("tests.rs");
}
