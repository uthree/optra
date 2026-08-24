//! Every panel, laid out headlessly.
//!
//! egui is immediate mode: a panel that indexes past the end of a list, or
//! hands a widget a value it will not accept, fails when it is *drawn*, and the
//! panel a user is not looking at is never drawn. Running the layout without a
//! window catches that in a test run instead of in front of someone mid-walk.
//!
//! This is a smoke test, not a check of what anything looks like. What it
//! asserts is that laying the panel out does not panic.

use optra::app::panels::{Panel, PanelContext};
use optra::app::panels::{calibration, cameras, log, models, output, tracking};
use optra::calib::{Recorder, RoomCalibration};
use optra::capture::CaptureManager;
use optra::config::{CameraConfig, Config, SourceConfig};
use optra::fusion::stage::Fusion;
use optra::logging::LogBuffer;
use optra::pipeline::Pipeline;
use optra::vr::VrLink;
use optra::worker::Supervisor;

/// Lays out every panel against the given configuration.
fn draw_every_panel(config: Config) {
    let mut config = config;
    let log = LogBuffer::default();
    let mut supervisor = Supervisor::new();
    let mut capture = CaptureManager::default();
    let mut pipeline = Pipeline::default();
    let mut vr = VrLink::default();
    let mut recorder = Recorder::default();
    let mut room: Option<RoomCalibration> = None;
    let mut fusion = Fusion::default();

    let mut cameras_panel = cameras::CamerasPanel::default();
    let mut models_panel = models::ModelsPanel::default();
    let mut calibration_panel = calibration::CalibrationPanel::default();
    let mut tracking_panel = tracking::TrackingPanel;
    let mut output_panel = output::OutputPanel;
    let mut log_panel = log::LogPanel::default();

    let ctx = egui::Context::default();

    // Twice: an immediate-mode panel can behave differently once its widgets
    // have ids and stored state from a previous pass.
    for _ in 0..2 {
        let mut output = ctx.run_ui(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                for panel in Panel::ALL {
                    let mut panel_ctx = PanelContext {
                        config: &mut config,
                        log: &log,
                        supervisor: &mut supervisor,
                        capture: &mut capture,
                        pipeline: &mut pipeline,
                        vr: &mut vr,
                        recorder: &mut recorder,
                        room: &mut room,
                        fusion: &mut fusion,
                        fusion_problem: None,
                        dirty: false,
                    };

                    match panel {
                        Panel::Cameras => cameras_panel.ui(ui, &mut panel_ctx),
                        Panel::Models => models_panel.ui(ui, &mut panel_ctx),
                        Panel::Calibration => calibration_panel.ui(ui, &mut panel_ctx),
                        Panel::Tracking => tracking_panel.ui(ui, &mut panel_ctx),
                        Panel::Output => output_panel.ui(ui, &mut panel_ctx),
                        Panel::Log => log_panel.ui(ui, &mut panel_ctx),
                    }
                }
            });
        });

        // The font atlas comes out as a texture delta a real backend would
        // upload. Nothing here has a GPU, and epaint refuses to be dropped
        // holding one.
        output.textures_delta.clear();
    }

    supervisor.shutdown();
}

#[test]
fn a_fresh_install_lays_out() {
    draw_every_panel(Config::default());
}

/// The state a user is actually in most of the time: cameras configured, none
/// of the background stages running because nothing has been started yet.
#[test]
fn a_configured_room_lays_out() {
    let config = Config {
        cameras: (0..4)
            .map(|index| CameraConfig {
                id: format!("cam{index}"),
                label: format!("Camera {index}"),
                enabled: index != 3,
                source: SourceConfig::Synthetic { seat: index },
                ..CameraConfig::default()
            })
            .collect(),
        room: Some("a-profile-that-does-not-exist".to_owned()),
        ..Config::default()
    };

    draw_every_panel(config);
}

/// The result table, which is only reachable after a solve or with a profile
/// loaded — so it is the part of the wizard a normal run never draws.
#[test]
fn a_solved_room_lays_out() {
    use nalgebra::{Point3, Vector3};

    use optra::calib::recorder::Rig;
    use optra::calib::solve::CameraCalibration;
    use optra::geometry::camera::{Camera, Intrinsics};
    use optra::geometry::lens::Lens;
    use optra::models::keypoints::Joint;
    use optra::vr::Role;

    let camera = |index: usize, rms: f64, spread: f64| CameraCalibration {
        id: format!("cam{index}"),
        camera: Camera::look_at(
            Intrinsics::from_fov(1280, 720, 70f64.to_radians()),
            Lens::default(),
            Point3::new(index as f64, 2.4, -1.8),
            Point3::new(0.0, 1.0, 0.0),
            Vector3::y(),
        ),
        rms,
        sightings: 400,
        coverage: 0.6,
        spread,
        latency: None,
    };

    let room = RoomCalibration {
        // One good camera and one that came out badly, so both sides of every
        // colour decision in the table are drawn.
        cameras: vec![camera(0, 0.0009, 0.4), camera(1, 0.02, 0.01)],
        rigs: vec![
            (
                Rig {
                    role: Role::Head,
                    joint: Joint::Head,
                },
                Vector3::new(0.01, 0.06, 0.13),
            ),
            (
                Rig {
                    role: Role::LeftHand,
                    joint: Joint::LeftWrist,
                },
                Vector3::new(-0.02, 0.03, 0.09),
            ),
        ],
        rms: 0.014,
        rejected: 37,
        used: 763,
        solved_at: "2026-08-25T12:00:00+09:00".to_owned(),
    };

    let mut config = Config::default();
    let log = LogBuffer::default();
    let mut supervisor = Supervisor::new();
    let mut capture = CaptureManager::default();
    let mut pipeline = Pipeline::default();
    let mut vr = VrLink::default();
    let mut recorder = Recorder::default();
    let mut loaded = Some(room);
    let mut fusion = Fusion::default();
    let mut panel = calibration::CalibrationPanel::default();

    let ctx = egui::Context::default();
    let mut output = ctx.run_ui(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut panel_ctx = PanelContext {
                config: &mut config,
                log: &log,
                supervisor: &mut supervisor,
                capture: &mut capture,
                pipeline: &mut pipeline,
                vr: &mut vr,
                recorder: &mut recorder,
                room: &mut loaded,
                fusion: &mut fusion,
                fusion_problem: None,
                dirty: false,
            };
            panel.ui(ui, &mut panel_ctx);
        });
    });
    output.textures_delta.clear();

    supervisor.shutdown();
}
