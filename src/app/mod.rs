//! The eframe application shell: navigation, status bar and config lifetime.

pub mod panels;
pub mod viewer3d;

use std::time::{Duration, Instant};

use egui::RichText;

use crate::calib::{Recorder, RoomCalibration};
use crate::capture::CaptureManager;
use crate::config::Config;
use crate::fusion::bones::Skeleton;
use crate::fusion::stage::Fusion;
use crate::logging::LogBuffer;
use crate::output::stage::Output;
use crate::pipeline::Pipeline;
use crate::startup;
use crate::vr::VrLink;
use crate::worker::{Supervisor, WorkerEvent};
use panels::{Panel, PanelContext};

/// How long to wait after the last change before writing the config.
const SAVE_DEBOUNCE: Duration = Duration::from_secs(2);

/// How long the output settings must hold still before the stage is restarted
/// with them. Long enough to cover a slider drag, short enough that a user who
/// changed the port does not wonder whether it took.
const OUTPUT_SETTLE: Duration = Duration::from_millis(500);

pub struct OptraApp {
    config: Config,
    log: LogBuffer,
    supervisor: Supervisor,
    capture: CaptureManager,
    pipeline: Pipeline,
    vr: VrLink,
    recorder: Recorder,
    room: Option<RoomCalibration>,
    fusion: Fusion,
    /// The measured body, kept across restarts so tracking does not begin by
    /// re-learning a skeleton it already knows.
    body: Skeleton,
    /// Why fusion is not running, when it should be.
    fusion_problem: Option<String>,
    /// Sends the reconstructed body to VRChat or SteamVR.
    sender: Output,
    /// Why the output stage is not running, when it should be.
    output_problem: Option<String>,
    /// The output settings the running stage was started with. A change to any
    /// of them means the socket, the tracker numbering or the send clock is
    /// wrong, and none of those can be adjusted from outside the thread.
    sending_with: Option<crate::config::OutputConfig>,
    /// When the output settings last changed, so a slider being dragged does
    /// not restart the stage on every frame of the drag.
    output_changed_at: Option<Instant>,

    cameras: panels::cameras::CamerasPanel,
    models: panels::models::ModelsPanel,
    calibration: panels::calibration::CalibrationPanel,
    tracking: panels::tracking::TrackingPanel,
    output_panel: panels::output::OutputPanel,
    log_panel: panels::log::LogPanel,

    /// Set when the config changed; cleared once it has been written.
    dirty_since: Option<Instant>,
    /// Workers that died unexpectedly, shown as a banner until dismissed.
    failures: Vec<String>,
    /// What the startup check found, until the user dismisses it. Cleared
    /// rather than kept, because a check that stays on screen after it has been
    /// read is a check people learn to look past.
    startup: Option<startup::Report>,
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

        // Before anything is started, so that the report describes what the
        // user set up rather than what the last few seconds did to it. It goes
        // to the log either way: a user who dismisses the banner and asks about
        // it a week later still has the answer.
        let startup = startup::Report::gather(&config, room.as_ref());
        startup.log();

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
            fusion: Fusion::default(),
            body: Skeleton::load_or_default(),
            fusion_problem: None,
            sender: Output::default(),
            output_problem: None,
            sending_with: None,
            output_changed_at: None,
            cameras: Default::default(),
            models: Default::default(),
            calibration: Default::default(),
            tracking: Default::default(),
            output_panel: Default::default(),
            log_panel: Default::default(),
            dirty_since: None,
            failures: Vec::new(),
            startup: (!startup.is_clear()).then_some(startup),
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

    /// Starts or stops the fusion stage as its prerequisites come and go.
    ///
    /// Fusion needs a calibrated room and at least two of its cameras running
    /// with a model attached. All three are things a user changes while the
    /// application is open, so this is checked every frame rather than set up
    /// once — and when it cannot run, the reason is kept for the panel to show
    /// instead of the stage simply being absent.
    fn sync_fusion(&mut self) {
        let wanted = self.config.fusion.enabled && self.pipeline.is_running();

        let Some(room) = self.room.as_ref().filter(|_| wanted) else {
            if self.fusion.is_running() {
                self.fusion.stop();
            }
            self.fusion_problem = match (wanted, self.room.is_some()) {
                (false, _) => None,
                (true, false) => Some("no room profile is loaded".to_owned()),
                (true, true) => None,
            };
            return;
        };

        if self.fusion.is_running() {
            return;
        }

        let channels = self.pipeline.channels();
        self.fusion_problem = self
            .fusion
            .start(
                &self.config.fusion,
                channels,
                room,
                self.body.clone(),
                self.vr.channel().cloned(),
                &mut self.supervisor,
            )
            .err();
    }

    /// Starts, stops and restarts the output stage.
    ///
    /// Everything the stage needs is fixed when it is built — the socket, the
    /// tracker numbering, the clock — so a settings change means a restart
    /// rather than an adjustment. That is cheap, but not cheap enough to do on
    /// every frame of a slider being dragged, so a change has to settle first.
    fn sync_output(&mut self) {
        let wanted = self.config.output.enabled && self.fusion.is_running();

        if !wanted {
            if self.sender.is_running() {
                self.sender.stop();
            }
            self.sending_with = None;
            self.output_problem = None;
            return;
        }

        // A stage running with settings that no longer match is stopped as soon
        // as the change is noticed, rather than left sending against the old
        // ones until the drag ends. Stopping is instant; starting is what waits.
        let stale = self
            .sending_with
            .as_ref()
            .is_some_and(|running| running != &self.config.output);
        if stale {
            self.sender.stop();
            self.sending_with = None;
            self.output_changed_at = Some(Instant::now());
        }

        if self.sender.is_running() {
            return;
        }

        if self
            .output_changed_at
            .is_some_and(|at| at.elapsed() < OUTPUT_SETTLE)
        {
            return;
        }
        self.output_changed_at = None;

        let Some(fusion) = self.fusion.channel().cloned() else {
            return;
        };

        self.output_problem = self
            .sender
            .start(
                &self.config.output,
                fusion,
                self.vr.channel().cloned(),
                &mut self.supervisor,
            )
            .err();

        // Only remember the settings that actually started something. Storing
        // them regardless would make a failed start look current, and it would
        // never be retried.
        if self.output_problem.is_none() {
            self.sending_with = Some(self.config.output.clone());
        }
    }

    /// Keeps the body measurement the fusion stage refined, so the next launch
    /// starts from a skeleton rather than from nothing.
    fn save_body(&mut self) {
        let Some(measured) = self.fusion.body() else {
            return;
        };
        if measured.bones.is_empty() {
            return;
        }
        self.body = measured;
        if let Err(err) = self.body.save() {
            tracing::warn!("failed to save the body measurement: {err:#}");
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

    /// Draws the selected panel, and reports whether it changed the config.
    fn panel_body(&mut self, ui: &mut egui::Ui, panel: Panel) -> bool {
        let mut panel_ctx = PanelContext {
            config: &mut self.config,
            log: &self.log,
            supervisor: &mut self.supervisor,
            capture: &mut self.capture,
            pipeline: &mut self.pipeline,
            vr: &mut self.vr,
            recorder: &mut self.recorder,
            room: &mut self.room,
            fusion: &mut self.fusion,
            fusion_problem: self.fusion_problem.as_deref(),
            sender: &mut self.sender,
            output_problem: self.output_problem.as_deref(),
            dirty: false,
        };

        match panel {
            Panel::Cameras => self.cameras.ui(ui, &mut panel_ctx),
            Panel::Models => self.models.ui(ui, &mut panel_ctx),
            Panel::Calibration => self.calibration.ui(ui, &mut panel_ctx),
            Panel::Tracking => self.tracking.ui(ui, &mut panel_ctx),
            Panel::Output => self.output_panel.ui(ui, &mut panel_ctx),
            Panel::Log => self.log_panel.ui(ui, &mut panel_ctx),
        }

        panel_ctx.dirty
    }

    /// What the startup check found, above whichever panel the user last had
    /// open.
    ///
    /// It has to be here rather than on a panel of its own, because the whole
    /// point is to be seen by somebody who does not yet know which panel to
    /// look at.
    fn startup_banner(&mut self, ui: &mut egui::Ui) {
        let Some(report) = &self.startup else { return };

        match startup_banner_ui(ui, report) {
            BannerAction::None => {}
            BannerAction::Dismiss => self.startup = None,
            BannerAction::Recheck => {
                let report = startup::Report::gather(&self.config, self.room.as_ref());
                report.log();
                self.startup = (!report.is_clear()).then_some(report);
            }
        }
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

/// What the user did with the startup banner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BannerAction {
    None,
    Dismiss,
    Recheck,
}

/// Draws the startup report as a banner and reports what was clicked.
///
/// A free function rather than a method so that it can be laid out in a test
/// without an application around it. It is drawn only when something is wrong,
/// which is exactly the kind of UI that goes years without anybody seeing it
/// and then fails in front of the one user who needed it.
pub fn startup_banner_ui(ui: &mut egui::Ui, report: &startup::Report) -> BannerAction {
    let mut action = BannerAction::None;

    egui::Panel::top("startup").show(ui, |ui| {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let (colour, headline) = match report.verdict() {
                startup::Verdict::Blocked => (
                    egui::Color32::from_rgb(240, 100, 100),
                    "Tracking cannot start yet:",
                ),
                _ => (
                    egui::Color32::from_rgb(230, 180, 80),
                    "Tracking will run, with this worth a look:",
                ),
            };
            ui.label(RichText::new(headline).color(colour).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Dismiss").clicked() {
                    action = BannerAction::Dismiss;
                }
                // Everything here is something the user can fix without
                // restarting — plug the camera back in, install the model — so
                // the check has to be repeatable without one.
                if ui.button("Check again").clicked() {
                    action = BannerAction::Recheck;
                }
            });
        });

        for check in report.problems() {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                ui.label(RichText::new(check.title).strong());
                ui.label(&check.detail);
                if let Some(fix) = check.fix {
                    ui.label(RichText::new(fix).weak());
                }
            });
        }
        ui.add_space(4.0);
    });

    action
}

impl eframe::App for OptraApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.drain_worker_events();
        self.sync_fusion();
        self.sync_output();
        self.track_window_geometry(ui.ctx());

        self.nav(ui);
        self.status_bar(ui);
        self.failure_banner(ui);
        self.startup_banner(ui);

        let panel = self.config.ui.panel;
        let mut dirty = false;
        egui::CentralPanel::default().show(ui, |ui| {
            // The heading stays outside the scroll area: it is what tells the
            // user which panel they are looking at, and scrolling it away to
            // reach the bottom of a long one is no help to anybody.
            ui.heading(panel.title());
            ui.label(RichText::new(panel.description()).weak());
            ui.separator();

            dirty = if panel.scrolls() {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.panel_body(ui, panel))
                    .inner
            } else {
                self.panel_body(ui, panel)
            };
        });

        if dirty {
            self.mark_dirty();
        }
        self.save_if_due(false);
    }

    fn on_exit(&mut self) {
        // Cameras first: their threads are the ones the supervisor waits on.
        self.save_body();
        // The output stage before fusion: it has a goodbye to send, and it
        // cannot send it once the body it describes has stopped arriving.
        self.sender.stop();
        self.fusion.stop();
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
