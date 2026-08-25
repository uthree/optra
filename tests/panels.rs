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
use optra::output::stage::Output;
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
    let mut sender = Output::default();

    let mut cameras_panel = cameras::CamerasPanel::default();
    let mut models_panel = models::ModelsPanel::default();
    let mut calibration_panel = calibration::CalibrationPanel::default();
    let mut tracking_panel = tracking::TrackingPanel::default();
    let mut output_panel = output::OutputPanel::default();
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
                        sender: &mut sender,
                        output_problem: None,
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
        range: 2.4,
        // One camera that can see the legs and one that cannot, so both sides
        // of the warning are drawn.
        feet: if index == 0 { 0.9 } else { 0.05 },
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
        precision: Some(0.031),
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
    let mut sender = Output::default();
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
                sender: &mut sender,
                output_problem: None,
                dirty: false,
            };
            panel.ui(ui, &mut panel_ctx);
        });
    });
    output.textures_delta.clear();

    supervisor.shutdown();
}

/// A tracking stage with a body in it, holding every state that carries
/// formatting: coloured uncertainties, an inferred joint, a camera taken out of
/// service, a half-measured skeleton. None of it is drawn on a machine with no
/// room and no cameras, which is every machine a test runs on.
fn tracked_fusion() -> Fusion {
    use std::time::{Duration, Instant};

    use nalgebra::{Point3, Vector3};

    use optra::fusion::bones::{BONES, BoneLength, Skeleton};
    use optra::fusion::filter::{Filtered, FilteredJoint};
    use optra::fusion::fit::{Fitted, FittedJoint};
    use optra::fusion::fuse::{FusedJoint, Pose3d};
    use optra::fusion::stage::{CameraContribution, FusionFrame, FusionStats};
    use optra::models::keypoints::Joint;

    let at = Instant::now();
    let mut raw = Pose3d::empty(at);
    let mut fitted = Fitted::empty(at);
    let mut filtered = Filtered::empty(at, Duration::from_millis(60));

    let body = [
        (Joint::Hip, Point3::new(0.0, 0.95, 0.0), 0.004),
        (Joint::LeftHip, Point3::new(-0.12, 0.95, 0.0), 0.006),
        (Joint::RightHip, Point3::new(0.12, 0.95, 0.0), 0.03),
        (Joint::LeftKnee, Point3::new(-0.12, 0.51, 0.0), 0.05),
        (Joint::LeftAnkle, Point3::new(-0.12, 0.09, 0.0), 0.008),
    ];

    for (index, (joint, point, sigma)) in body.into_iter().enumerate() {
        // One joint is left out of the reconstruction entirely, so the
        // "inferred" path through every table is drawn.
        if joint != Joint::LeftKnee {
            raw.set(
                joint,
                FusedJoint {
                    point,
                    sigma,
                    residual: 0.0004 * (index as f64 + 1.0),
                    weights: vec![(0, 0.5), (1, 0.3), (2, 0.2)],
                    rejected: if index == 2 { vec![2] } else { Vec::new() },
                },
            );
        }

        fitted.set(
            joint,
            FittedJoint {
                point,
                sigma,
                inferred: joint == Joint::LeftKnee,
                correction: 0.003 * index as f64,
            },
        );
        filtered.set(
            joint,
            FilteredJoint {
                point,
                velocity: Vector3::new(0.0, 0.0, -0.8),
                predicted: point + Vector3::new(0.0, 0.0, -0.05),
                lead: 0.06,
                sigma,
                inferred: joint == Joint::LeftKnee,
            },
        );
    }

    // Half the skeleton settled, half of it still scattered.
    let mut measured = Skeleton::default();
    for (index, bone) in BONES.iter().enumerate() {
        measured.bones.push(BoneLength {
            bone: *bone,
            length: 0.3,
            samples: if index % 2 == 0 { 900 } else { 20 },
            scatter: if index % 2 == 0 { 0.002 } else { 0.09 },
        });
    }

    let stats = FusionStats {
        running: true,
        rate: 59.4,
        lag_ms: 130.0,
        joints: 5,
        inferred: 1,
        lower_body: 4,
        worst_correction: 0.031,
        disagreement: 2.4,
        head: Some(nalgebra::Vector3::new(0.02, -0.41, 0.05)),
        scale: Some(0.63),
        tally: optra::fusion::fuse::Tally {
            measured: 5,
            miss: 0.28,
            unseen: 12,
            unsure: 6,
            one_ray: 2,
            disagreed: 1,
            uncertain: 0,
        },
        shake: optra::fusion::shake::Shake {
            raw: 0.011,
            fitted: 0.009,
            filtered: 0.002,
            predicted: 0.014,
        },
        cameras: vec![
            CameraContribution {
                id: "cam0".to_owned(),
                aligned: 1.0,
                weight: 0.5,
                rejected: 0.02,
                latency_ms: 0.0,
                problem: None,
            },
            CameraContribution {
                id: "cam1".to_owned(),
                aligned: 0.6,
                weight: 0.3,
                rejected: 0.45,
                latency_ms: 42.0,
                problem: None,
            },
            CameraContribution {
                id: "cam2".to_owned(),
                aligned: 0.0,
                weight: 0.0,
                rejected: 0.0,
                latency_ms: 0.0,
                problem: Some("running at 640x480, calibrated at 1280x720".to_owned()),
            },
        ],
        body: measured,
        floor: Some(-0.68),
        measuring: true,
        warning: Some("cam1 disagrees with the others on 45% of the joints it sees".to_owned()),
    };

    Fusion::detached(
        stats,
        Some(FusionFrame {
            raw,
            fitted,
            filtered,
        }),
    )
}

/// An output stage that is sending, with one tracker in each state the panel
/// formats differently: healthy, patchy, inferred, and lost.
fn sending_output() -> Output {
    use optra::config::OutputConfig;
    use optra::output::stage::{OutputStats, TrackerReport};

    let config = OutputConfig::default();
    let trackers = optra::output::assign(&config.enabled_roles())
        .into_iter()
        .enumerate()
        .map(|(position, (index, role))| TrackerReport {
            role,
            index,
            live: [1.0, 0.6, 0.0][position % 3],
            sigma: [0.008, 0.035, 0.09][position % 3],
            inferred: position == 1,
            lost: position % 3 == 2,
        })
        .collect();

    Output::detached(OutputStats {
        running: true,
        sink: "VRChat OSC".to_owned(),
        target: "127.0.0.1:9000".to_owned(),
        rate: 89.6,
        sent: 12_403,
        lead_ms: 118.0,
        head: true,
        trackers,
        problem: None,
        warning: Some("right foot not reaching the trackers".to_owned()),
    })
}

#[test]
fn a_tracked_body_lays_out() {
    let mut config = Config::default();
    let log = LogBuffer::default();
    let mut supervisor = Supervisor::new();
    let mut capture = CaptureManager::default();
    let mut pipeline = Pipeline::default();
    let mut vr = VrLink::default();
    let mut recorder = Recorder::default();
    let mut room = None;
    let mut fusion = tracked_fusion();
    let mut sender = sending_output();
    let mut panel = tracking::TrackingPanel::default();

    let ctx = egui::Context::default();
    // Twice, because the collapsing sections only draw their bodies once they
    // have stored state saying they are open.
    for _ in 0..2 {
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
                    room: &mut room,
                    fusion: &mut fusion,
                    fusion_problem: None,
                    sender: &mut sender,
                    output_problem: None,
                    dirty: false,
                };
                panel.ui(ui, &mut panel_ctx);
            });
        });
        output.textures_delta.clear();
    }

    supervisor.shutdown();
}

/// A window shorter than the panel in it.
///
/// This is the failure the scroll area exists for, and it is invisible in a
/// test that lays panels out against an unbounded screen: the content is drawn
/// either way, and only a real window is short enough to cut it off. Every
/// panel must either fit or scroll — what none of them may do is quietly run
/// off the bottom, which is what they all did before `Panel::scrolls`.
#[test]
fn no_panel_spills_out_of_a_short_window() {
    // Short enough that the wizard, the camera list and the 3D view all
    // overrun it, and about as short as a user would ever drag the window.
    const HEIGHT: f32 = 260.0;

    let mut config = Config::default();
    let log = LogBuffer::default();
    let mut supervisor = Supervisor::new();
    let mut capture = CaptureManager::default();
    let mut pipeline = Pipeline::default();
    let mut vr = VrLink::default();
    let mut recorder = Recorder::default();
    let mut room: Option<RoomCalibration> = None;
    let mut fusion = tracked_fusion();
    let mut sender = sending_output();

    let mut cameras_panel = cameras::CamerasPanel::default();
    let mut models_panel = models::ModelsPanel::default();
    let mut calibration_panel = calibration::CalibrationPanel::default();
    let mut tracking_panel = tracking::TrackingPanel::default();
    let mut output_panel = output::OutputPanel::default();
    let mut log_panel = log::LogPanel::default();

    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1000.0, HEIGHT),
        )),
        ..egui::RawInput::default()
    };

    for panel in Panel::ALL {
        let mut used = 0.0;

        // Twice, so the collapsing sections are open on the pass that counts.
        for _ in 0..2 {
            let mut output = ctx.run_ui(input.clone(), |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
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
                        sender: &mut sender,
                        output_problem: None,
                        dirty: false,
                    };

                    // The same decision the shell makes, and the reason this
                    // test is worth having: a panel added without thinking
                    // about it gets scrolled, and one that opts out has to
                    // have brought its own.
                    let mut draw = |ui: &mut egui::Ui| match panel {
                        Panel::Cameras => cameras_panel.ui(ui, &mut panel_ctx),
                        Panel::Models => models_panel.ui(ui, &mut panel_ctx),
                        Panel::Calibration => calibration_panel.ui(ui, &mut panel_ctx),
                        Panel::Tracking => tracking_panel.ui(ui, &mut panel_ctx),
                        Panel::Output => output_panel.ui(ui, &mut panel_ctx),
                        Panel::Log => log_panel.ui(ui, &mut panel_ctx),
                    };

                    if panel.scrolls() {
                        egui::ScrollArea::vertical()
                            .auto_shrink([false, false])
                            .show(ui, |ui| draw(ui));
                    } else {
                        draw(ui);
                    }

                    used = ui.min_rect().height();
                });
            });
            output.textures_delta.clear();
        }

        assert!(
            used <= HEIGHT,
            "{} wanted {used} px of a {HEIGHT} px window, so the bottom of it is unreachable",
            panel.title()
        );
    }

    supervisor.shutdown();
}

/// The output panel with trackers actually going out. Everything that makes it
/// worth looking at — the live percentages, the uncertainties, the lost row —
/// is only drawn once a stage is running, which no test machine has.
#[test]
fn a_sending_output_lays_out() {
    let mut config = Config::default();
    let log = LogBuffer::default();
    let mut supervisor = Supervisor::new();
    let mut capture = CaptureManager::default();
    let mut pipeline = Pipeline::default();
    let mut vr = VrLink::default();
    let mut recorder = Recorder::default();
    let mut room = None;
    let mut fusion = tracked_fusion();
    let mut sender = sending_output();
    let mut panel = output::OutputPanel::default();

    let ctx = egui::Context::default();
    // Twice, so the collapsing sections draw their bodies.
    for _ in 0..2 {
        let mut drawn = ctx.run_ui(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
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
                    sender: &mut sender,
                    output_problem: None,
                    dirty: false,
                };
                panel.ui(ui, &mut panel_ctx);
            });
        });
        drawn.textures_delta.clear();
    }

    // The panel fills in any role a config from an older build is missing, and
    // getting that wrong would silently hide a tracker rather than crash.
    assert_eq!(
        config.output.trackers.len(),
        optra::output::TrackerRole::ALL.len()
    );

    supervisor.shutdown();
}

/// A config written before a tracker existed must not leave that tracker
/// unreachable: there would be no way to turn it on from the panel it is
/// missing from.
#[test]
fn a_config_missing_a_tracker_gains_it() {
    use optra::config::OutputConfig;
    use optra::output::TrackerRole;

    let mut config = OutputConfig {
        trackers: Vec::new(),
        ..OutputConfig::default()
    };
    config.complete();

    assert_eq!(config.trackers.len(), TrackerRole::ALL.len());
    // Added, not enabled: turning a tracker on is the user's decision.
    assert!(config.enabled_roles().is_empty());
}
