//! Tracker output backend configuration.

use super::{PanelContext, not_yet_implemented};

#[derive(Default)]
pub struct OutputPanel;

impl OutputPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, _ctx: &mut PanelContext<'_>) {
        not_yet_implemented(
            ui,
            "M5",
            &[
                "VRChat OSC Trackers backend",
                "SteamVR virtual tracker backend via VMT",
                "Tracker roles, offsets and prediction horizon",
            ],
        );
    }
}
