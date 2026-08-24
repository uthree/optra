//! The panels making up the main window.

pub mod calibration;
pub mod cameras;
pub mod log;
pub mod models;
pub mod output;
pub mod tracking;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Panel {
    Cameras,
    Models,
    Calibration,
    Tracking,
    Output,
    Log,
}

impl Panel {
    pub const ALL: [Panel; 6] = [
        Panel::Cameras,
        Panel::Models,
        Panel::Calibration,
        Panel::Tracking,
        Panel::Output,
        Panel::Log,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Panel::Cameras => "Cameras",
            Panel::Models => "Models",
            Panel::Calibration => "Calibration",
            Panel::Tracking => "Tracking",
            Panel::Output => "Output",
            Panel::Log => "Log",
        }
    }

    /// One-line description shown under the panel heading.
    pub fn description(self) -> &'static str {
        match self {
            Panel::Cameras => {
                "Select capture devices and check that every camera streams reliably."
            }
            Panel::Models => "Download pose models and assign them to cameras.",
            Panel::Calibration => "Solve camera intrinsics and extrinsics against the headset.",
            Panel::Tracking => "Watch the reconstructed skeleton and per-joint quality.",
            Panel::Output => "Configure the tracker output backend.",
            Panel::Log => "Application log.",
        }
    }
}

/// Everything a panel is allowed to touch.
pub struct PanelContext<'a> {
    pub config: &'a mut crate::config::Config,
    pub log: &'a crate::logging::LogBuffer,
    pub supervisor: &'a mut crate::worker::Supervisor,
    pub capture: &'a mut crate::capture::CaptureManager,
    pub pipeline: &'a mut crate::pipeline::Pipeline,
    pub vr: &'a mut crate::vr::VrLink,
    pub recorder: &'a mut crate::calib::Recorder,
    /// The room profile in force, if one has been solved or loaded.
    pub room: &'a mut Option<crate::calib::RoomCalibration>,
    /// Set by a panel when it changed the config, so the shell can save it.
    pub dirty: bool,
}

/// Placeholder body for panels whose milestone has not landed yet.
pub(crate) fn not_yet_implemented(ui: &mut egui::Ui, milestone: &str, items: &[&str]) {
    ui.label(format!("Arrives in milestone {milestone}."));
    ui.add_space(8.0);
    for item in items {
        ui.label(format!("\u{2022} {item}"));
    }
}
