use crate::app::VoiceHarmApp;
use crate::drawing::draw_piano;
use crate::ui_canvas::render_canvas;
use std::sync::atomic::Ordering;

impl eframe::App for VoiceHarmApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::from_rgb(12, 18, 25);
        visuals.window_fill = visuals.panel_fill;
        visuals.faint_bg_color = egui::Color32::from_rgb(23, 34, 45);
        visuals.selection.bg_fill = egui::Color32::from_rgb(27, 124, 133);
        ui.ctx().set_visuals(visuals);
        let f0 = self.update_audio();
        render_header(self, ui, f0);
        render_piano(ui);
        render_canvas(self, ui, f0);
        ui.ctx().request_repaint();
    }
}

fn render_header(app: &mut VoiceHarmApp, ui: &mut egui::Ui, f0: Option<f32>) {
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
                let live = !app.audio_failed.load(Ordering::Relaxed);
                ui.colored_label(
                    if live {
                        egui::Color32::from_rgb(48, 205, 184)
                    } else {
                        egui::Color32::from_rgb(224, 116, 92)
                    },
                    if live { "● LIVE" } else { "● AUDIO ERROR" },
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
                let label = egui::RichText::new("Drone").color(egui::Color32::WHITE);
                let button = if app.drone_on {
                    egui::Button::new(label).fill(egui::Color32::from_rgb(22, 129, 111))
                } else {
                    egui::Button::new(label)
                };
                if ui.add(button).clicked() {
                    app.drone_on = !app.drone_on;
                    app.drone_state
                        .enabled
                        .store(app.drone_on, Ordering::Relaxed);
                }
                ui.add_sized(
                    [72.0, 18.0],
                    egui::Slider::new(&mut app.drone_vol, 0.0..=1.0).show_value(false),
                );
                app.drone_state
                    .amplitude
                    .store(app.drone_vol.to_bits(), Ordering::Relaxed);
            });
        });
}

fn render_piano(ui: &mut egui::Ui) {
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
}
