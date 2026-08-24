//! Guided calibration against the headset.

use egui::{Color32, RichText};

use crate::vr::{LinkState, Role};

use super::{PanelContext, not_yet_implemented};

#[derive(Default)]
pub struct CalibrationPanel;

impl CalibrationPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        self.link(ui, ctx);

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(4.0);

        not_yet_implemented(
            ui,
            "M3",
            &[
                "Correspondence recording during a calibration walk",
                "Per-camera coverage map and residual reporting",
                "3D view of the solved camera positions",
            ],
        );
    }

    fn link(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        ui.horizontal(|ui| {
            ui.strong("SteamVR");

            let mut enabled = ctx.config.vr.enabled;
            if ui.checkbox(&mut enabled, "Connect").changed() {
                ctx.config.vr.enabled = enabled;
                ctx.dirty = true;
                if enabled {
                    ctx.vr.start(&ctx.config.vr, ctx.supervisor);
                } else {
                    ctx.vr.stop();
                }
            }
        });

        let Some(channel) = ctx.vr.channel() else {
            ui.label(RichText::new("Not connecting. Camera setup does not need SteamVR.").weak());
            return;
        };

        let stats = channel.stats();

        ui.horizontal(|ui| {
            let (colour, text) = match stats.state {
                LinkState::Connected => (
                    Color32::from_rgb(120, 200, 120),
                    format!("connected, {:.0} Hz", stats.measured_hz),
                ),
                LinkState::Searching if stats.installed => (
                    Color32::from_rgb(220, 190, 110),
                    "SteamVR is installed but not running".to_owned(),
                ),
                LinkState::Searching => (
                    Color32::from_rgb(200, 120, 120),
                    "no SteamVR runtime found on this machine".to_owned(),
                ),
                LinkState::Stopped => (Color32::GRAY, "stopped".to_owned()),
            };
            ui.colored_label(colour, text);
        });

        if let Some(runtime) = &stats.runtime {
            ui.label(RichText::new(runtime.display().to_string()).weak().small());
        }
        if stats.state != LinkState::Connected
            && let Some(error) = &stats.last_error
        {
            ui.label(RichText::new(error).weak().small());
        }

        let Some(snapshot) = channel.latest() else {
            return;
        };

        ui.add_space(6.0);
        egui::Grid::new("vr-devices")
            .num_columns(4)
            .striped(true)
            .show(ui, |ui| {
                ui.strong("Device");
                ui.strong("Position");
                ui.strong("Tracking");
                ui.strong("Model");
                ui.end_row();

                for device in &snapshot.devices {
                    ui.label(device.role.label());

                    let p = device.pose.translation.vector;
                    ui.label(format!("{:.2}, {:.2}, {:.2}", p.x, p.y, p.z));

                    if device.tracking {
                        ui.colored_label(Color32::from_rgb(120, 200, 120), "ok");
                    } else {
                        ui.colored_label(Color32::from_rgb(220, 190, 110), "lost");
                    }

                    ui.label(RichText::new(&device.model).weak());
                    ui.end_row();
                }
            });

        // The calibration cannot separate a rig offset from a shift of every
        // camera unless the device turned during the walk, so it is worth
        // saying which devices are present before the recording starts.
        let missing: Vec<&str> = Role::RIGS
            .iter()
            .filter(|role| snapshot.device(**role).is_none())
            .map(|role| role.label())
            .collect();
        if !missing.is_empty() {
            ui.add_space(4.0);
            ui.label(
                RichText::new(format!("not reporting: {}", missing.join(", ")))
                    .weak()
                    .small(),
            );
        }
    }
}
