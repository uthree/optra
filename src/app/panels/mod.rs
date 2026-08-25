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

    /// Whether the shell should put this panel's body in a scroll area.
    ///
    /// Almost every panel wants one: they are lists and tables that grow with
    /// the number of cameras, models or joints, and a window shorter than the
    /// content silently truncates it otherwise.
    ///
    /// The exceptions manage their own, because they have a header that has to
    /// stay put while the body underneath moves — a filter that scrolls off
    /// screen is a filter the user cannot reach. Scrolling those from here as
    /// well would nest one scroll area inside another, and the wheel would then
    /// belong to whichever the pointer happened to be over.
    pub fn scrolls(self) -> bool {
        !matches!(self, Panel::Cameras | Panel::Models | Panel::Log)
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
    pub fusion: &'a mut crate::fusion::stage::Fusion,
    /// Why fusion is not running, when the shell could not start it.
    pub fusion_problem: Option<&'a str>,
    /// The output stage, which sends what fusion reconstructs.
    pub sender: &'a mut crate::output::stage::Output,
    /// Why the output stage is not running, when the shell could not start it.
    pub output_problem: Option<&'a str>,
    /// Set by a panel when it changed the config, so the shell can save it.
    pub dirty: bool,
}
