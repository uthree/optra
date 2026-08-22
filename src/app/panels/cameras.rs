//! Capture device selection and live preview.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use egui::{Color32, RichText, TextureHandle, TextureOptions};

use super::PanelContext;
use crate::capture::source::ControlInfo;
use crate::capture::{CameraCommand, CameraState};
use crate::config::{CameraConfig, ControlName, LensKind, Rotation, SourceConfig};

/// How long a value the user just set keeps priority over what the device
/// reports, so that a dragged slider does not snap back while the camera thread
/// is still reading the property back.
const PENDING_GRACE: Duration = Duration::from_millis(800);

#[derive(Default)]
pub struct CamerasPanel {
    /// Devices found by the last scan.
    detected: Vec<DetectedDevice>,
    detect_error: Option<String>,
    /// Preview textures, keyed by camera id, with the sequence number they show.
    previews: HashMap<String, Preview>,
    /// Set when the configuration changed while capture is running.
    needs_restart: bool,
    /// Property values the user changed but the device has not confirmed yet.
    pending: HashMap<(String, ControlName), (i64, Instant)>,
}

struct DetectedDevice {
    name: String,
    path: String,
}

struct Preview {
    texture: TextureHandle,
    seq: u64,
}

impl CamerasPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        self.toolbar(ui, ctx);
        ui.add_space(4.0);
        self.device_picker(ui, ctx);
        ui.add_space(8.0);

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.camera_list(ui, ctx);
                ui.add_space(12.0);
                self.previews(ui, ctx);
            });
    }

    fn toolbar(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        ui.horizontal(|ui| {
            let running = ctx.capture.is_running();

            if ui
                .button(if running {
                    "Restart capture"
                } else {
                    "Start capture"
                })
                .clicked()
            {
                ctx.capture.start(&ctx.config.cameras, ctx.supervisor);
                self.needs_restart = false;
            }
            if ui.add_enabled(running, egui::Button::new("Stop")).clicked() {
                ctx.capture.stop();
                self.previews.clear();
                self.needs_restart = false;
            }

            let enabled = ctx.config.cameras.iter().filter(|c| c.enabled).count();
            ui.separator();
            ui.label(format!(
                "{enabled} enabled / {} configured",
                ctx.config.cameras.len()
            ));

            if self.needs_restart && running {
                ui.separator();
                ui.label(
                    RichText::new("configuration changed \u{2014} restart capture to apply")
                        .color(Color32::from_rgb(230, 180, 80)),
                );
            }
        });
    }

    fn device_picker(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        ui.horizontal(|ui| {
            if ui.button("Detect devices").clicked() {
                self.detect();
            }
            if ui.button("Add synthetic camera").clicked() {
                let seat = ctx
                    .config
                    .cameras
                    .iter()
                    .filter(|c| c.source.is_synthetic())
                    .count() as u32;
                let id = ctx.config.fresh_camera_id();
                ctx.config.cameras.push(CameraConfig {
                    label: format!("Synthetic {}", seat + 1),
                    id,
                    source: SourceConfig::Synthetic { seat },
                    width: 960,
                    height: 540,
                    fps: 30,
                    ..CameraConfig::default()
                });
                ctx.dirty = true;
                self.needs_restart = true;
            }
        });

        if let Some(error) = &self.detect_error {
            ui.label(RichText::new(error).color(Color32::from_rgb(240, 100, 100)));
        }

        if self.detected.is_empty() {
            return;
        }

        ui.group(|ui| {
            ui.label(RichText::new("Detected devices").strong());
            for device in &self.detected {
                ui.horizontal(|ui| {
                    let already = ctx.config.cameras.iter().any(|camera| {
                        matches!(&camera.source,
                            SourceConfig::Webcam { device_path, .. } if device_path == &device.path)
                    });

                    if ui
                        .add_enabled(!already, egui::Button::new("Add"))
                        .on_disabled_hover_text("Already configured")
                        .clicked()
                    {
                        let id = ctx.config.fresh_camera_id();
                        ctx.config.cameras.push(CameraConfig {
                            label: device.name.clone(),
                            id,
                            source: SourceConfig::Webcam {
                                device_path: device.path.clone(),
                                device_name: device.name.clone(),
                            },
                            ..CameraConfig::default()
                        });
                        ctx.dirty = true;
                        self.needs_restart = true;
                    }
                    ui.label(&device.name);
                    ui.label(RichText::new(&device.path).weak().small());
                });
            }
        });
    }

    fn camera_list(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        let mut remove = None;

        for index in 0..ctx.config.cameras.len() {
            let id = ctx.config.cameras[index].id.clone();
            let stats = ctx.capture.channel(&id).map(|channel| channel.stats());

            ui.group(|ui| {
                ui.horizontal(|ui| {
                    let camera = &mut ctx.config.cameras[index];
                    if ui.checkbox(&mut camera.enabled, "").changed() {
                        ctx.dirty = true;
                        self.needs_restart = true;
                    }
                    if ui.text_edit_singleline(&mut camera.label).changed() {
                        ctx.dirty = true;
                    }
                    ui.label(RichText::new(&camera.id).weak().small());

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Remove").clicked() {
                            remove = Some(index);
                        }
                        if let Some(stats) = &stats {
                            ui.label(state_text(stats.state));
                        }
                    });
                });

                let camera = &mut ctx.config.cameras[index];
                ui.label(RichText::new(source_summary(&camera.source)).weak().small());

                ui.horizontal(|ui| {
                    let mut changed = false;
                    ui.label("Resolution");
                    changed |= ui
                        .add(egui::DragValue::new(&mut camera.width).range(64..=4096))
                        .changed();
                    ui.label("x");
                    changed |= ui
                        .add(egui::DragValue::new(&mut camera.height).range(64..=4096))
                        .changed();
                    ui.label("FPS");
                    changed |= ui
                        .add(egui::DragValue::new(&mut camera.fps).range(1..=240))
                        .changed();

                    ui.label("Lens");
                    egui::ComboBox::from_id_salt(("lens", index))
                        .selected_text(camera.lens.label())
                        .show_ui(ui, |ui| {
                            for lens in LensKind::ALL {
                                changed |= ui
                                    .selectable_value(&mut camera.lens, lens, lens.label())
                                    .changed();
                            }
                        });

                    ui.label("Rotation");
                    egui::ComboBox::from_id_salt(("rotation", index))
                        .selected_text(camera.rotation.label())
                        .show_ui(ui, |ui| {
                            for rotation in Rotation::ALL {
                                changed |= ui
                                    .selectable_value(
                                        &mut camera.rotation,
                                        rotation,
                                        rotation.label(),
                                    )
                                    .changed();
                            }
                        });

                    if changed {
                        ctx.dirty = true;
                        self.needs_restart = true;
                    }
                });

                if let Some(stats) = stats {
                    let requested = camera.fps as f32;
                    let measured = stats.measured_fps;
                    // A camera that cannot sustain the requested rate is the
                    // usual symptom of USB bandwidth trouble, so it is called
                    // out rather than left for the user to compare by eye.
                    let starved = stats.state == CameraState::Running
                        && measured > 0.0
                        && measured < requested * 0.8;

                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 12.0;
                        let fps = RichText::new(format!("{measured:.1} / {requested:.0} fps"));
                        ui.label(if starved {
                            fps.color(Color32::from_rgb(230, 180, 80))
                        } else {
                            fps
                        });
                        ui.label(format!("{} frames", stats.captured));
                        ui.label(format!("{} missed", stats.overwritten));
                        ui.label(format!("decode {:.1} ms", stats.decode_ms));
                        if let Some(format) = &stats.negotiated {
                            ui.label(RichText::new(format.to_string()).weak());
                        }
                    });

                    if let Some(error) = &stats.last_error {
                        ui.label(
                            RichText::new(error)
                                .color(Color32::from_rgb(240, 100, 100))
                                .small(),
                        );
                    }

                    if !stats.controls.is_empty() {
                        self.device_controls(ui, ctx, index, &stats.controls);
                    }
                }
            });
        }

        if let Some(index) = remove {
            let removed = ctx.config.cameras.remove(index);
            self.previews.remove(&removed.id);
            ctx.dirty = true;
            self.needs_restart = true;
        }
    }

    /// Sliders for the device's own properties.
    ///
    /// Exposure is the one that matters: left on automatic, a camera in a dimly
    /// lit room stretches its shutter until it can only deliver half the frames
    /// it promised, and smears anything that moves.
    fn device_controls(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &mut PanelContext<'_>,
        index: usize,
        controls: &[ControlInfo],
    ) {
        let id = ctx.config.cameras[index].id.clone();
        let requested_fps = ctx.config.cameras[index].fps;

        egui::CollapsingHeader::new("Device properties")
            .id_salt(("controls", index))
            .show(ui, |ui| {
                for info in controls {
                    let key = (id.clone(), info.name);
                    ui.horizontal(|ui| {
                        ui.add_sized([140.0, 18.0], egui::Label::new(info.name.label()));

                        let mut auto = info.auto;
                        if info.auto_supported {
                            if ui.checkbox(&mut auto, "Auto").changed() {
                                let value = self.displayed(&key, info);
                                self.apply(ctx, index, info.name, value, auto);
                            }
                        } else {
                            ui.add_space(48.0);
                        }

                        let mut value = self.displayed(&key, info);
                        let enabled = !info.auto && info.manual_supported;
                        let slider = ui.add_enabled(
                            enabled,
                            egui::Slider::new(&mut value, info.min..=info.max)
                                .step_by(info.step as f64)
                                .show_value(true),
                        );
                        if slider.changed() {
                            self.pending.insert(key.clone(), (value, Instant::now()));
                            self.apply(ctx, index, info.name, value, false);
                        }

                        if let Some(text) = info.name.describe_value(value) {
                            ui.label(RichText::new(text).weak());
                        }

                        if info.name == ControlName::Exposure
                            && ui
                                .button("Fit frame rate")
                                .on_hover_text(
                                    "Pin the shutter just short of one frame period, so the \
                                     camera can reach the frame rate it advertises",
                                )
                                .clicked()
                        {
                            let target = ControlName::exposure_for_fps(requested_fps)
                                .clamp(info.min, info.max);
                            self.pending.insert(key.clone(), (target, Instant::now()));
                            self.apply(ctx, index, info.name, target, false);
                        }

                        if ui.button("Default").clicked() {
                            self.pending
                                .insert(key.clone(), (info.default, Instant::now()));
                            self.apply(ctx, index, info.name, info.default, info.auto_supported);
                        }
                    });
                }
            });
    }

    /// The value to show: what the user just set, until the camera thread has
    /// read the property back from the device.
    fn displayed(&self, key: &(String, ControlName), info: &ControlInfo) -> i64 {
        match self.pending.get(key) {
            Some((value, set_at)) if set_at.elapsed() < PENDING_GRACE => *value,
            _ => info.value,
        }
    }

    /// Sends a property change to the camera and records it in the config so it
    /// is reapplied the next time the camera opens.
    fn apply(
        &mut self,
        ctx: &mut PanelContext<'_>,
        index: usize,
        name: ControlName,
        value: i64,
        auto: bool,
    ) {
        ctx.config.cameras[index].set_control(name, auto, value);
        ctx.dirty = true;

        let id = &ctx.config.cameras[index].id;
        if let Some(channel) = ctx.capture.channel(id) {
            channel.send(CameraCommand::SetControl { name, value, auto });
        }
    }

    fn previews(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        if !ctx.capture.is_running() {
            return;
        }

        let available = ui.available_width();
        let columns = if ctx.capture.channels().len() > 1 {
            2
        } else {
            1
        };
        let width = ((available - 12.0 * columns as f32) / columns as f32).max(160.0);

        egui::Grid::new("previews")
            .num_columns(columns)
            .show(ui, |ui| {
                for (index, channel) in ctx.capture.channels().iter().enumerate() {
                    let id = channel.config.id.clone();

                    if let Some(frame) = channel.peek() {
                        let entry = self.previews.get_mut(&id);
                        let stale = entry
                            .as_ref()
                            .is_none_or(|preview| preview.seq != frame.seq);

                        if stale {
                            let image = egui::ColorImage::from_rgb(
                                [frame.width as usize, frame.height as usize],
                                &frame.rgb,
                            );
                            match self.previews.get_mut(&id) {
                                Some(preview) => {
                                    preview.texture.set(image, TextureOptions::LINEAR);
                                    preview.seq = frame.seq;
                                }
                                None => {
                                    let texture = ui.ctx().load_texture(
                                        format!("preview:{id}"),
                                        image,
                                        TextureOptions::LINEAR,
                                    );
                                    self.previews.insert(
                                        id.clone(),
                                        Preview {
                                            texture,
                                            seq: frame.seq,
                                        },
                                    );
                                }
                            }
                        }
                    }

                    ui.vertical(|ui| {
                        ui.label(RichText::new(&channel.config.label).strong());
                        match self.previews.get(&id) {
                            Some(preview) => {
                                let aspect = preview.texture.aspect_ratio();
                                ui.add(
                                    egui::Image::new(&preview.texture)
                                        .fit_to_exact_size(egui::vec2(width, width / aspect)),
                                );
                            }
                            None => {
                                ui.label(RichText::new("waiting for the first frame").weak());
                            }
                        }
                    });

                    if (index + 1) % columns == 0 {
                        ui.end_row();
                    }
                }
            });

        // Previews only update when egui repaints, and nothing else in this
        // panel is animating.
        ui.ctx().request_repaint();
    }

    fn detect(&mut self) {
        self.detected.clear();
        self.detect_error = None;

        #[cfg(windows)]
        {
            match crate::capture::source::webcam::list_devices() {
                Ok(devices) => {
                    tracing::info!("detected {} capture device(s)", devices.len());
                    self.detected = devices
                        .into_iter()
                        .map(|info| DetectedDevice {
                            name: info.human_name(),
                            path: info.misc(),
                        })
                        .collect();
                    if self.detected.is_empty() {
                        self.detect_error = Some("No capture devices found.".to_owned());
                    }
                }
                Err(err) => {
                    tracing::warn!("device detection failed: {err:#}");
                    self.detect_error = Some(format!("{err:#}"));
                }
            }
        }

        #[cfg(not(windows))]
        {
            self.detect_error =
                Some("Device detection is only implemented for Windows.".to_owned());
        }
    }
}

fn source_summary(source: &SourceConfig) -> String {
    match source {
        SourceConfig::Webcam {
            device_name,
            device_path,
        } => format!("{device_name} \u{2014} {device_path}"),
        SourceConfig::Synthetic { seat } => {
            format!("synthetic scene, ceiling corner {}", seat + 1)
        }
        SourceConfig::Still { path } => format!("still image, {path}"),
    }
}

fn state_text(state: CameraState) -> RichText {
    let color = match state {
        CameraState::Running => Color32::from_rgb(120, 210, 140),
        CameraState::Opening => Color32::from_rgb(230, 180, 80),
        CameraState::Failed => Color32::from_rgb(240, 100, 100),
        CameraState::Stopped => Color32::GRAY,
    };
    RichText::new(state.label()).color(color)
}
