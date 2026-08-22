//! Guided calibration against the headset.

use super::{PanelContext, not_yet_implemented};

#[derive(Default)]
pub struct CalibrationPanel;

impl CalibrationPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, _ctx: &mut PanelContext<'_>) {
        not_yet_implemented(
            ui,
            "M3",
            &[
                "OpenVR headset and controller pose source",
                "Correspondence recording during a calibration walk",
                "DLT resection plus Levenberg-Marquardt bundle refinement",
                "Per-camera coverage map and residual reporting",
            ],
        );
    }
}
