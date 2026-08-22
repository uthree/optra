//! Model catalogue, per-camera assignment and execution provider.

use std::collections::HashMap;
use std::sync::Arc;

use egui::{Color32, ProgressBar, RichText};
use parking_lot::Mutex;

use super::PanelContext;
use crate::infer::ProviderChoice;
use crate::models::manifest::{Manifest, ModelKind, ModelSpec};
use crate::models::store::{self, Stage};

#[derive(Default)]
pub struct ModelsPanel {
    catalogue: Vec<ModelSpec>,
    /// Set once the catalogue has been read, so it is not re-read every frame.
    loaded: bool,
    load_error: Option<String>,
    /// Progress of the installs that are running or have finished.
    installs: HashMap<String, Arc<Mutex<Stage>>>,
}

impl ModelsPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        self.ensure_catalogue();

        if let Some(error) = &self.load_error {
            ui.label(RichText::new(error).color(Color32::from_rgb(240, 100, 100)));
        }

        self.settings(ui, ctx);
        ui.add_space(8.0);
        ui.separator();

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.catalogue_ui(ui, ctx);
            });
    }

    fn ensure_catalogue(&mut self) {
        if self.loaded {
            return;
        }
        self.loaded = true;
        match Manifest::load() {
            Ok(models) => self.catalogue = models,
            Err(err) => {
                tracing::error!("failed to load the model catalogue: {err:#}");
                self.load_error = Some(format!("{err:#}"));
            }
        }
    }

    fn settings(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("Execution provider");
            egui::ComboBox::from_id_salt("provider")
                .selected_text(ctx.config.inference.provider.label())
                .show_ui(ui, |ui| {
                    for provider in ProviderChoice::ALL {
                        changed |= ui
                            .selectable_value(
                                &mut ctx.config.inference.provider,
                                provider,
                                provider.label(),
                            )
                            .changed();
                    }
                });

            ui.separator();
            ui.label("Detect every");
            changed |= ui
                .add(
                    egui::DragValue::new(&mut ctx.config.inference.detect_every)
                        .range(1..=30)
                        .suffix(" frames"),
                )
                .on_hover_text(
                    "How often the detector runs. Between runs the box comes from the \
                     previous keypoints, which is enough for one slowly moving person.",
                )
                .changed();
        });

        ui.horizontal(|ui| {
            changed |= self.model_picker(
                ui,
                "Detector",
                ModelKind::Detector,
                &mut ctx.config.inference.detector_model,
            );
            changed |= self.model_picker(
                ui,
                "Pose",
                ModelKind::Pose2d,
                &mut ctx.config.inference.pose_model,
            );
        });

        if changed {
            ctx.dirty = true;
            // The stage picks new settings up on its next tick; a camera keeps
            // running its current model until the replacement has loaded.
            ctx.pipeline
                .configure(ctx.config.inference.clone(), &ctx.config.cameras);
        }
    }

    /// A dropdown of installed models of one kind.
    fn model_picker(
        &self,
        ui: &mut egui::Ui,
        label: &str,
        kind: ModelKind,
        selected: &mut String,
    ) -> bool {
        let mut changed = false;
        ui.label(label);

        let current = self
            .catalogue
            .iter()
            .find(|spec| &spec.id == selected)
            .map(|spec| spec.name.clone())
            .unwrap_or_else(|| selected.clone());

        egui::ComboBox::from_id_salt(("model", label))
            .selected_text(current)
            .width(280.0)
            .show_ui(ui, |ui| {
                for spec in self.catalogue.iter().filter(|spec| spec.kind == kind) {
                    let installed = store::is_installed(spec);
                    let text = if installed {
                        RichText::new(&spec.name)
                    } else {
                        RichText::new(format!("{} (not installed)", spec.name)).weak()
                    };
                    changed |= ui
                        .selectable_value(selected, spec.id.clone(), text)
                        .changed();
                }
            });
        changed
    }

    fn catalogue_ui(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        for index in 0..self.catalogue.len() {
            let spec = self.catalogue[index].clone();
            let installed = store::is_installed(&spec);
            let stage = self
                .installs
                .get(&spec.id)
                .map(|state| state.lock().clone());

            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&spec.name).strong());
                    ui.label(RichText::new(spec.kind.label()).weak().small());
                    ui.label(
                        RichText::new(&spec.license)
                            .small()
                            .color(Color32::from_rgb(120, 190, 240)),
                    )
                    .on_hover_text(&spec.license_url);
                    if let Some(zoo) = &spec.zoo {
                        ui.label(RichText::new(format!("zoo {zoo}")).weak().small());
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if installed {
                            if ui.button("Remove").clicked() {
                                if let Err(err) = store::remove(&spec) {
                                    tracing::warn!("{err:#}");
                                } else {
                                    self.installs.remove(&spec.id);
                                }
                            }
                            ui.label(
                                RichText::new("installed").color(Color32::from_rgb(120, 210, 140)),
                            );
                        } else {
                            let busy = matches!(
                                stage,
                                Some(
                                    Stage::Downloading { .. }
                                        | Stage::Verifying
                                        | Stage::Extracting
                                )
                            );
                            if ui
                                .add_enabled(!busy, egui::Button::new("Install"))
                                .clicked()
                            {
                                self.start_install(&spec);
                            }
                            if let Some(size) = spec.source.size() {
                                ui.label(RichText::new(store::human_bytes(size)).weak());
                            }
                        }
                    });
                });

                if let Some(notes) = &spec.notes {
                    ui.label(RichText::new(notes).weak().small());
                }

                if let Some(stage) = stage
                    && !installed
                {
                    match &stage {
                        Stage::Failed(err) => {
                            ui.label(
                                RichText::new(format!("failed: {err}"))
                                    .color(Color32::from_rgb(240, 100, 100))
                                    .small(),
                            );
                        }
                        other => {
                            let bar = ProgressBar::new(other.fraction().unwrap_or(0.0))
                                .text(other.label());
                            ui.add(bar);
                            ui.ctx().request_repaint();
                        }
                    }
                }
            });
        }

        let _ = ctx;
    }

    /// Installs on a worker thread so a slow download cannot freeze the UI.
    fn start_install(&mut self, spec: &ModelSpec) {
        let state = Arc::new(Mutex::new(Stage::Downloading {
            received: 0,
            total: spec.source.size(),
        }));
        self.installs.insert(spec.id.clone(), state.clone());

        let spec = spec.clone();
        let id = spec.id.clone();
        let spawned = std::thread::Builder::new()
            .name(format!("install:{id}"))
            .spawn(move || {
                let result = store::install(&spec, &mut |stage| {
                    *state.lock() = stage;
                });
                if let Err(err) = result {
                    tracing::error!("failed to install {}: {err:#}", spec.id);
                    *state.lock() = Stage::Failed(format!("{err:#}"));
                }
            });

        if let Err(err) = spawned {
            tracing::error!("failed to start the installer for {id}: {err}");
        }
    }
}
