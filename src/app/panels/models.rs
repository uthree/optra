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
    /// The form for registering an ONNX file the user already has.
    local: LocalForm,
}

/// What the user has typed into the local-model form so far.
struct LocalForm {
    kind: ModelKind,
    path: String,
    name: String,
    input_name: String,
    width: u32,
    height: u32,
    /// Keypoint layout, for a pose model.
    layout: String,
    /// The outcome of the last attempt, and whether it was a success.
    message: Option<(String, bool)>,
}

impl Default for LocalForm {
    fn default() -> Self {
        Self {
            kind: ModelKind::Pose2d,
            path: String::new(),
            name: String::new(),
            input_name: "input".to_owned(),
            // The RTMPose export size, which is what a converted pose
            // checkpoint most likely is.
            width: 192,
            height: 256,
            layout: "halpe26".to_owned(),
            message: None,
        }
    }
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
                self.local_form(ui);
                ui.add_space(8.0);
                self.catalogue_ui(ui, ctx);
            });
    }

    /// Registers an ONNX file the user already has as a catalogue entry.
    ///
    /// The manifest has supported a local source from the start; what was
    /// missing was a way to write the entry without hand-editing a TOML file
    /// whose field names live in another repository's documentation. The form
    /// asks only for what a template cannot guess — the file, a name, the
    /// input size, and the keypoint layout — and the conventions of the
    /// architecture fill in the rest. An unusual export edits the entry this
    /// writes, which is a far shorter road than starting from a blank file.
    fn local_form(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Register a local ONNX file")
            .id_salt("local_model")
            .show(ui, |ui| {
                let form = &mut self.local;

                ui.horizontal(|ui| {
                    for kind in [ModelKind::Detector, ModelKind::Pose2d] {
                        if ui
                            .selectable_value(&mut form.kind, kind, kind.label())
                            .clicked()
                        {
                            // The size is a property of the checkpoint, but
                            // each kind has a size its exports overwhelmingly
                            // use, and a stale one from the other kind is
                            // wrong for certain rather than probably right.
                            (form.width, form.height) = match kind {
                                ModelKind::Detector => (640, 640),
                                ModelKind::Pose2d => (192, 256),
                            };
                        }
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("ONNX file");
                    ui.add(
                        egui::TextEdit::singleline(&mut form.path)
                            .hint_text("C:\\path\\to\\model.onnx")
                            .desired_width(360.0),
                    );
                });

                ui.horizontal(|ui| {
                    ui.label("Name");
                    ui.add(
                        egui::TextEdit::singleline(&mut form.name)
                            .hint_text("shown in the pickers; blank uses the file name")
                            .desired_width(240.0),
                    );
                });

                ui.horizontal(|ui| {
                    ui.label("Input tensor");
                    ui.add(egui::TextEdit::singleline(&mut form.input_name).desired_width(100.0));
                    ui.separator();
                    ui.label("Input size");
                    ui.add(egui::DragValue::new(&mut form.width).range(32..=2048));
                    ui.label("\u{00d7}");
                    ui.add(egui::DragValue::new(&mut form.height).range(32..=2048));
                    if form.kind == ModelKind::Pose2d {
                        ui.separator();
                        ui.label("Keypoints");
                        egui::ComboBox::from_id_salt("local_layout")
                            .selected_text(form.layout.clone())
                            .show_ui(ui, |ui| {
                                for name in crate::models::keypoints::names() {
                                    ui.selectable_value(&mut form.layout, name.to_owned(), name);
                                }
                            });
                    }
                });

                if ui.button("Register").clicked() {
                    self.register_local();
                }

                if let Some((message, good)) = &self.local.message {
                    let colour = if *good {
                        Color32::from_rgb(120, 210, 140)
                    } else {
                        Color32::from_rgb(240, 100, 100)
                    };
                    ui.label(RichText::new(message).color(colour));
                }
            });
    }

    fn register_local(&mut self) {
        let form = &mut self.local;
        form.message = Some(match Self::registered(form, &self.catalogue) {
            Ok(name) => {
                // Reloaded so the new entry appears in the catalogue and the
                // pickers without a restart.
                self.loaded = false;
                (format!("registered {name}"), true)
            }
            Err(error) => (format!("{error:#}"), false),
        });
    }

    /// Validates the form and writes the entry. Returns the registered name.
    fn registered(form: &mut LocalForm, catalogue: &[ModelSpec]) -> anyhow::Result<String> {
        let path = form.path.trim().to_owned();
        let file = std::path::Path::new(&path);
        if !file.is_file() {
            anyhow::bail!("there is no file at {path}");
        }

        // The name falls back to the file stem, which is what the user calls
        // the file already.
        let name = match form.name.trim() {
            "" => file
                .file_stem()
                .map(|stem| stem.to_string_lossy().into_owned())
                .unwrap_or_default(),
            name => name.to_owned(),
        };
        let id = ModelSpec::slug(&name);
        if id.is_empty() {
            anyhow::bail!("the name needs at least one letter or digit in it");
        }

        // Refused against the whole catalogue, not just the user manifest. A
        // user entry with a builtin id silently replaces the builtin at load,
        // which is a feature for someone editing the file on purpose and a
        // trap from a form.
        if catalogue.iter().any(|spec| spec.id == id) {
            anyhow::bail!("the catalogue already has a model called {id}; pick another name");
        }

        let layout = (form.kind == ModelKind::Pose2d).then(|| form.layout.clone());
        let spec = ModelSpec::local(
            form.kind,
            &path,
            &name,
            form.input_name.trim(),
            form.width,
            form.height,
            layout,
        );
        Manifest::register(spec)?;

        form.path.clear();
        form.name.clear();
        Ok(name)
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
