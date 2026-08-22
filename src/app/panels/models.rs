//! Model catalogue, per-camera assignment and benchmarking.

use super::{PanelContext, not_yet_implemented};

#[derive(Default)]
pub struct ModelsPanel;

impl ModelsPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, _ctx: &mut PanelContext<'_>) {
        not_yet_implemented(
            ui,
            "M2",
            &[
                "Model manifest with license gate and SHA-256 verification",
                "Architecture adapters: yolox, rtdetr, simcc, heatmap, movenet",
                "Per-camera model assignment and inference stride",
                "Benchmark and runtime model swap",
            ],
        );
    }
}
