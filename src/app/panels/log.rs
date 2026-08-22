//! Application log view.

use egui::{Color32, RichText};
use tracing::Level;

use super::PanelContext;
use crate::config::LogLevel;

pub struct LogPanel {
    follow: bool,
}

impl Default for LogPanel {
    fn default() -> Self {
        Self { follow: true }
    }
}

impl LogPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        ui.horizontal(|ui| {
            ui.label("Level");
            egui::ComboBox::from_id_salt("log_level")
                .selected_text(ctx.config.ui.log_level.label())
                .show_ui(ui, |ui| {
                    for level in LogLevel::ALL {
                        ui.selectable_value(&mut ctx.config.ui.log_level, level, level.label());
                    }
                });
            ui.checkbox(&mut self.follow, "Follow");
            if ui.button("Clear").clicked() {
                ctx.log.clear();
            }
        });

        ui.separator();

        let filter = ctx.config.ui.log_level;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(self.follow)
            .show(ui, |ui| {
                ctx.log.with_records(|records| {
                    for record in records.iter().filter(|r| filter.includes(r.level)) {
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing.x = 6.0;
                            ui.label(
                                RichText::new(record.at.format("%H:%M:%S%.3f").to_string())
                                    .monospace()
                                    .weak(),
                            );
                            ui.label(
                                RichText::new(level_label(record.level))
                                    .monospace()
                                    .color(level_color(record.level)),
                            );
                            ui.label(RichText::new(&record.target).monospace().weak());
                            ui.label(&record.message);
                        });
                    }
                });
            });
    }
}

fn level_label(level: Level) -> &'static str {
    match level {
        Level::ERROR => "ERROR",
        Level::WARN => "WARN ",
        Level::INFO => "INFO ",
        Level::DEBUG => "DEBUG",
        Level::TRACE => "TRACE",
    }
}

fn level_color(level: Level) -> Color32 {
    match level {
        Level::ERROR => Color32::from_rgb(240, 100, 100),
        Level::WARN => Color32::from_rgb(230, 180, 80),
        Level::INFO => Color32::from_rgb(120, 190, 240),
        Level::DEBUG => Color32::from_rgb(150, 150, 150),
        Level::TRACE => Color32::from_rgb(120, 120, 120),
    }
}
