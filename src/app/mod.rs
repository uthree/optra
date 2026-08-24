//! The eframe application shell: navigation, status bar and config lifetime.

pub mod panels;
pub mod viewer3d;

use std::time::{Duration, Instant};

use egui::RichText;

use crate::calib::{Recorder, RoomCalibration};
use crate::capture::CaptureManager;
use crate::config::Config;
use crate::logging::LogBuffer;
use crate::pipeline::Pipeline;
use crate::vr::VrLink;
use crate::worker::{Supervisor, WorkerEvent};
use panels::{Panel, PanelContext};

/// How long to wait after the last change before writing the config.
const SAVE_DEBOUNCE: Duration = Duration::from_secs(2);

pub struct OptraApp {
    config: Config,
    log: LogBuffer,
    supervisor: Supervisor,
    capture: CaptureManager,
    pipeline: Pipeline,
    vr: VrLink,
    recorder: Recorder,
    room: Option<RoomCalibration>,

    cameras: panels::cameras::CamerasPanel,
    models: panels::models::ModelsPanel,
    calibration: panels::calibration::CalibrationPanel,
    tracking: panels::tracking::TrackingPanel,
    output: panels::output::OutputPanel,
    log_panel: panels::log::LogPanel,

    /// Set when the config changed; cleared once it has been written.
    dirty_since: Option<Instant>,
    /// Workers that died unexpectedly, shown as a banner until dismissed.
    failures: Vec<String>,
}

impl OptraApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, config: Config, log: LogBuffer) -> Self {
        let mut supervisor = Supervisor::new();
        let mut capture = CaptureManager::default();
        let mut pipeline = Pipeline::default();
        let mut vr = VrLink::default();

        // A room the user calibrated earlier. Its absence is not an error: the
        // profile may have been deleted, and the application still runs.
        let room = config
            .room
            .as_deref()
            .and_then(|name| match RoomCalibration::load(name) {
                Ok(room) => Some(room),
                Err(error) => {
                    tracing::warn!(profile = %name, "could not load the room profile: {error:#}");
                    None
                }
            });

        vr.start(&config.vr, &mut supervisor);

        if config.capture.auto_start && config.cameras.iter().any(|camera| camera.enabled) {
            capture.start(&config.cameras, &mut supervisor);
            if config.inference.enabled {
                pipeline.start(
                    config.inference.clone(),
                    &config.cameras,
                    capture.channels(),
                    &mut supervisor,
                );
            }
        }

        Self {
            config,
            log,
            supervisor,
            capture,
            pipeline,
            vr,
            recorder: Recorder::default(),
            room,
            cameras: Default::default(),
            models: Default::default(),
            calibration: Default::default(),
            tracking: Default::default(),
            output: Default::default(),
            log_panel: Default::default(),
            dirty_since: None,
            failures: Vec::new(),
        }
    }

    fn mark_dirty(&mut self) {
        if self.dirty_since.is_none() {
            self.dirty_since = Some(Instant::now());
        }
    }

    fn save_if_due(&mut self, force: bool) {
        let due = match self.dirty_since {
            Some(since) => force || since.elapsed() >= SAVE_DEBOUNCE,
            None => false,
        };
        if !due {
            return;
        }
        self.dirty_since = None;
        if let Err(err) = self.config.save() {
            tracing::error!("failed to save the config: {err:#}");
        }
    }

    /// Records window geometry so the next launch reopens where the user left it.
    fn track_window_geometry(&mut self, ctx: &egui::Context) {
        let (rect, maximized) = ctx.input(|i| {
            let info = i.viewport();
            (info.inner_rect, info.maximized.unwrap_or(false))
        });

        let Some(rect) = rect else { return };
        if maximized != self.config.window.maximized {
            self.config.window.maximized = maximized;
            self.mark_dirty();
        }
        // Only the restored geometry is worth remembering; a maximized window
        // reports the screen size, which is not where the user put it.
        if maximized {
            return;
        }

        let size = [rect.width(), rect.height()];
        let pos = [rect.min.x, rect.min.y];
        if !approx_eq(size, self.config.window.size)
            || self.config.window.pos.is_none_or(|p| !approx_eq(pos, p))
        {
            self.config.window.size = size;
            self.config.window.pos = Some(pos);
            self.mark_dirty();
        }
    }

    fn drain_worker_events(&mut self) {
        self.supervisor.reap();
        while let Ok(event) = self.supervisor.events().try_recv() {
            match event {
                WorkerEvent::Started(name) => tracing::debug!(worker = %name, "worker started"),
                WorkerEvent::Finished(name) => tracing::debug!(worker = %name, "worker finished"),
                WorkerEvent::Panicked { name, message } => {
                    tracing::error!(worker = %name, "worker panicked: {message}");
                    self.failures.push(format!("{name}: {message}"));
                }
            }
        }
    }

    fn nav(&mut self, ui: &mut egui::Ui) {
        egui::Panel::left("nav")
            .resizable(false)
            .exact_size(160.0)
            .show(ui, |ui| {
                ui.add_space(8.0);
                ui.heading("Optra");
                ui.add_space(12.0);

                for panel in Panel::ALL {
                    let selected = self.config.ui.panel == panel;
                    if ui
                        .selectable_label(selected, RichText::new(panel.title()).size(15.0))
                        .clicked()
                        && !selected
                    {
                        self.config.ui.panel = panel;
                        self.mark_dirty();
                    }
                }
            });
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("v{}", env!("CARGO_PKG_VERSION"))).weak());
                ui.separator();
                ui.label(format!("{} worker(s)", self.supervisor.running()));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.dirty_since.is_some() {
                        ui.label(RichText::new("unsaved settings").weak());
                    }
                });
            });
        });
    }

    fn failure_banner(&mut self, ui: &mut egui::Ui) {
        if self.failures.is_empty() {
            return;
        }
        egui::Panel::top("failures").show(ui, |ui| {
            let mut dismiss = false;
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new("A background worker stopped unexpectedly:")
                        .color(egui::Color32::from_rgb(240, 100, 100)),
                );
                for failure in &self.failures {
                    ui.label(failure);
                }
                if ui.button("Dismiss").clicked() {
                    dismiss = true;
                }
            });
            if dismiss {
                self.failures.clear();
            }
        });
    }
}

impl eframe::App for OptraApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_worker_events();
        self.track_window_geometry(ui.ctx());

        self.nav(ui);
        self.status_bar(ui);
        self.failure_banner(ui);

        let panel = self.config.ui.panel;
        let mut dirty = false;
        egui::CentralPanel::default().show(ui, |ui| {
            ui.heading(panel.title());
            ui.label(RichText::new(panel.description()).weak());
            ui.separator();

            let mut panel_ctx = PanelContext {
                config: &mut self.config,
                log: &self.log,
                supervisor: &mut self.supervisor,
                capture: &mut self.capture,
                pipeline: &mut self.pipeline,
                vr: &mut self.vr,
                recorder: &mut self.recorder,
                room: &mut self.room,
                dirty: false,
            };

            match panel {
                Panel::Cameras => self.cameras.ui(ui, &mut panel_ctx),
                Panel::Models => self.models.ui(ui, &mut panel_ctx),
                Panel::Calibration => self.calibration.ui(ui, &mut panel_ctx),
                Panel::Tracking => self.tracking.ui(ui, &mut panel_ctx),
                Panel::Output => self.output.ui(ui, &mut panel_ctx),
                Panel::Log => self.log_panel.ui(ui, &mut panel_ctx),
            }

            dirty = panel_ctx.dirty;
        });

        if dirty {
            self.mark_dirty();
        }
        self.save_if_due(false);
    }

    fn on_exit(&mut self) {
        // Cameras first: their threads are the ones the supervisor waits on.
        self.recorder.stop();
        self.pipeline.stop();
        self.capture.stop();
        self.vr.stop();
        self.supervisor.shutdown();
        self.dirty_since.get_or_insert_with(Instant::now);
        self.save_if_due(true);
    }
}

fn approx_eq(a: [f32; 2], b: [f32; 2]) -> bool {
    (a[0] - b[0]).abs() < 1.0 && (a[1] - b[1]).abs() < 1.0
}
