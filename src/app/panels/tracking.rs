//! Live skeleton view and per-joint quality.

use super::{PanelContext, not_yet_implemented};

#[derive(Default)]
pub struct TrackingPanel;

impl TrackingPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, _ctx: &mut PanelContext<'_>) {
        not_yet_implemented(
            ui,
            "M4",
            &[
                "Fusion clock with per-camera temporal alignment",
                "Angular-weighted triangulation with RANSAC",
                "Bone-length constrained skeleton fit and filtering",
                "3D viewport with camera frusta and per-joint residuals",
            ],
        );
    }
}
