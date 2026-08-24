//! Guided calibration against the headset.
//!
//! The wizard is deliberately opinionated about what it will let a user do. A
//! calibration that comes out subtly wrong is worse than one that refuses to
//! start: the tracking still works, it just puts the feet in the wrong place,
//! and nothing about that points back to the walk that caused it. So the
//! prerequisites are checked before recording, the things that make a walk
//! unusable are shown while it is still running, and the result is reported in
//! terms of what went wrong rather than as a single number.

use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{Receiver, bounded};
use egui::{Color32, RichText};
use nalgebra::Point3;

use crate::app::viewer3d::{Scene, Viewer3d};
use crate::calib::recorder::{Coverage, RecorderConfig, RecorderStats};
use crate::calib::{RoomCalibration, SolveOptions, solve};
use crate::vr::{LinkState, Role};

use super::PanelContext;

/// Rotation spread below which a rig's offset cannot be separated from a shift
/// of every camera.
const OBSERVABLE: f64 = 0.15;

/// Reprojection error above which a calibration should not be trusted, in
/// degrees. Roughly two pixels on a 70 degree 720p camera.
const GOOD_RMS_DEGREES: f64 = 0.2;

#[derive(Default)]
pub struct CalibrationPanel {
    stage: Stage,
    /// Name the next save will use.
    profile: String,
    message: Option<String>,
    viewer: Viewer3d,
    /// The path the headset took during the last walk. Kept after the solve
    /// because it is the most direct answer to "do these cameras agree with
    /// the room": the walk drawn through them should look like the room.
    walk: Vec<Point3<f64>>,
}

#[derive(Default)]
enum Stage {
    #[default]
    Idle,
    Recording,
    Solving(Receiver<Result<RoomCalibration, String>>),
    Reviewing(RoomCalibration),
}

impl CalibrationPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        self.poll_solver();

        self.link(ui, ctx);
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(6.0);

        match &self.stage {
            Stage::Idle => self.idle(ui, ctx),
            Stage::Recording => self.recording(ui, ctx),
            Stage::Solving(_) => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Solving. This takes a few seconds and does not block tracking.");
                });
            }
            Stage::Reviewing(_) => self.reviewing(ui, ctx),
        }

        if let Some(message) = &self.message {
            ui.add_space(8.0);
            ui.colored_label(Color32::from_rgb(220, 150, 150), message);
        }
    }

    /// Picks up the result of a solve that was running in the background.
    fn poll_solver(&mut self) {
        let Stage::Solving(results) = &self.stage else {
            return;
        };

        match results.try_recv() {
            Ok(Ok(room)) => {
                // Framed once, when the answer arrives. Doing it every frame
                // would fight the user as soon as they turned the view.
                let mut points: Vec<Point3<f64>> =
                    room.cameras.iter().map(|c| c.camera.position()).collect();
                points.extend(self.walk.iter().copied());
                self.viewer.frame(&points);

                self.stage = Stage::Reviewing(room);
            }
            Ok(Err(error)) => {
                self.message = Some(error);
                self.stage = Stage::Idle;
            }
            Err(crossbeam_channel::TryRecvError::Empty) => {}
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                self.message = Some("the solver thread died".to_owned());
                self.stage = Stage::Idle;
            }
        }
    }

    // ---- SteamVR link -----------------------------------------------------

    fn link(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        ui.horizontal(|ui| {
            ui.strong("SteamVR");

            let mut enabled = ctx.config.vr.enabled;
            if ui.checkbox(&mut enabled, "Connect").changed() {
                ctx.config.vr.enabled = enabled;
                ctx.dirty = true;
                if enabled {
                    ctx.vr.start(&ctx.config.vr, ctx.supervisor);
                } else {
                    ctx.vr.stop();
                }
            }
        });

        let Some(channel) = ctx.vr.channel() else {
            ui.label(RichText::new("Not connecting. Camera setup does not need SteamVR.").weak());
            return;
        };

        let stats = channel.stats();
        let (colour, text) = match stats.state {
            LinkState::Connected => (
                Color32::from_rgb(120, 200, 120),
                format!("connected, {:.0} Hz", stats.measured_hz),
            ),
            LinkState::Searching if stats.installed => (
                Color32::from_rgb(220, 190, 110),
                "SteamVR is installed but not running".to_owned(),
            ),
            LinkState::Searching => (
                Color32::from_rgb(200, 120, 120),
                "no SteamVR runtime found on this machine".to_owned(),
            ),
            LinkState::Stopped => (Color32::GRAY, "stopped".to_owned()),
        };
        ui.colored_label(colour, text);

        if let Some(runtime) = &stats.runtime {
            ui.label(RichText::new(runtime.display().to_string()).weak().small());
        }
        if stats.state != LinkState::Connected
            && let Some(error) = &stats.last_error
        {
            ui.label(RichText::new(error).weak().small());
        }

        let Some(snapshot) = channel.latest() else {
            return;
        };

        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            for role in Role::RIGS {
                let device = snapshot.device(role);
                let (colour, suffix) = match device {
                    Some(device) if device.tracking => (Color32::from_rgb(120, 200, 120), "ok"),
                    Some(_) => (Color32::from_rgb(220, 190, 110), "lost"),
                    None => (Color32::from_rgb(160, 160, 160), "absent"),
                };
                ui.colored_label(colour, format!("{}: {suffix}", role.label()));
                ui.add_space(6.0);
            }
        });
    }

    // ---- idle -------------------------------------------------------------

    fn idle(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        self.profiles(ui, ctx);
        ui.add_space(10.0);

        ui.strong("Record a calibration walk");
        ui.label(
            RichText::new(
                "Put the headset on and walk slowly around the room. Cover as much floor as you \
                 can, crouch and stretch to vary your height, and keep turning your head — the \
                 offset between the headset and the head keypoint is only visible because the \
                 headset rotates.",
            )
            .weak(),
        );
        ui.add_space(6.0);

        let blockers = blockers(ctx);
        for blocker in &blockers {
            ui.colored_label(
                Color32::from_rgb(220, 190, 110),
                format!("\u{2022} {blocker}"),
            );
        }

        ui.add_space(4.0);
        if ui
            .add_enabled(blockers.is_empty(), egui::Button::new("Start recording"))
            .clicked()
        {
            self.start_recording(ctx);
        }
    }

    fn start_recording(&mut self, ctx: &mut PanelContext<'_>) {
        let Some(vr) = ctx.vr.channel().cloned() else {
            return;
        };

        let cameras: Vec<_> = ctx
            .config
            .cameras
            .iter()
            .filter(|camera| camera.enabled)
            .filter_map(|camera| {
                ctx.pipeline
                    .channel(&camera.id)
                    .map(|channel| (camera.id.clone(), Arc::clone(channel)))
            })
            .collect();

        self.message = None;
        ctx.recorder
            .start(RecorderConfig::default(), cameras, vr, ctx.supervisor);
        self.stage = Stage::Recording;
    }

    // ---- recording --------------------------------------------------------

    fn recording(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        let Some(channel) = ctx.recorder.channel() else {
            self.stage = Stage::Idle;
            return;
        };
        let stats = channel.stats();

        ui.horizontal(|ui| {
            ui.strong(format!("Recording  {}", duration(stats.elapsed)));
            ui.label(RichText::new(format!("{} samples", stats.samples)).weak());

            if ui.button("Stop and solve").clicked() {
                self.finish_recording(ctx);
            }
            if ui.button("Discard").clicked() {
                ctx.recorder.stop();
                self.stage = Stage::Idle;
            }
        });

        if let Some(warning) = &stats.warning {
            ui.add_space(4.0);
            ui.colored_label(Color32::from_rgb(230, 180, 90), warning);
        }

        ui.add_space(8.0);
        self.rigs(ui, &stats);
        ui.add_space(8.0);
        self.coverage(ui, ctx);
    }

    fn finish_recording(&mut self, ctx: &mut PanelContext<'_>) {
        let Some(recording) = ctx.recorder.finish() else {
            self.stage = Stage::Idle;
            return;
        };

        // Kept before the recording is handed to the solver: the walk is what
        // the 3D view draws to show whether the cameras agree with the room.
        self.walk = recording
            .rigs
            .iter()
            .position(|rig| rig.role == Role::Head)
            .and_then(|rig| recording.tracks.get(rig))
            .map(|track| track.positions())
            .unwrap_or_default();

        // The solve takes seconds, and a frozen window during it looks like a
        // crash. It runs on a worker like everything else.
        let (sender, receiver) = bounded(1);
        let cameras = ctx.config.cameras.clone();
        ctx.supervisor.spawn("calib:solve", move |_| {
            let result = solve(&recording, &cameras, &SolveOptions::default())
                .map_err(|error| format!("{error:#}"));
            let _ = sender.send(result);
        });

        self.stage = Stage::Solving(receiver);
    }

    /// How well each rig has turned so far, which is what decides whether its
    /// offset can be recovered at all.
    fn rigs(&self, ui: &mut egui::Ui, stats: &RecorderStats) {
        if stats.rigs.is_empty() {
            ui.label(RichText::new("no keypoints matched to a device yet").weak());
            return;
        }

        let mut rigs = stats.rigs.clone();
        rigs.sort_by_key(|progress| (progress.rig.role.order(), progress.rig.joint));

        egui::Grid::new("calib-rigs")
            .num_columns(4)
            .striped(true)
            .show(ui, |ui| {
                for heading in ["Device", "Samples", "Rotation variety", ""] {
                    ui.strong(heading);
                }
                ui.end_row();

                for progress in &rigs {
                    ui.label(progress.rig.label());
                    ui.label(RichText::new(progress.samples.to_string()).weak());
                    ui.add(
                        egui::ProgressBar::new((progress.spread / 0.4).clamp(0.0, 1.0) as f32)
                            .desired_width(160.0),
                    );
                    if progress.spread < OBSERVABLE {
                        ui.colored_label(Color32::from_rgb(230, 180, 90), "turn it more");
                    } else {
                        ui.colored_label(Color32::from_rgb(120, 200, 120), "enough");
                    }
                    ui.end_row();
                }
            });
    }

    /// Per-camera coverage, as a map of the frame rather than one number.
    ///
    /// A narrow camera sees a smaller slice of the room, so a walk that
    /// satisfies a wide camera can leave a narrow one under-constrained. The
    /// map is what tells the user *where* to walk, which a percentage cannot.
    fn coverage(&self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        let Some(channel) = ctx.recorder.channel() else {
            return;
        };
        let recording = channel.recording();

        ui.strong("Coverage");
        ui.label(
            RichText::new(
                "Walk into the dark areas. The edges of a frame are what pin down the \n                 lens distortion, and they are the part a walk usually misses.",
            )
            .weak()
            .small(),
        );
        ui.horizontal_wrapped(|ui| {
            for trail in &recording.cameras {
                ui.vertical(|ui| {
                    ui.label(RichText::new(&trail.camera).small());
                    coverage_map(ui, &trail.coverage);
                    ui.label(
                        RichText::new(format!(
                            "{} samples, {:.0}%",
                            trail.samples.len(),
                            trail.coverage.filled() * 100.0
                        ))
                        .weak()
                        .small(),
                    );
                });
                ui.add_space(12.0);
            }
        });
    }

    // ---- reviewing --------------------------------------------------------

    fn reviewing(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        let Self {
            stage,
            viewer,
            walk,
            ..
        } = self;
        let Stage::Reviewing(room) = stage else {
            return;
        };
        summary(ui, room, "calib-result", viewer, walk);

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("Save as");
            ui.add(egui::TextEdit::singleline(&mut self.profile).desired_width(180.0));

            let named = !self.profile.trim().is_empty();
            if ui
                .add_enabled(named, egui::Button::new("Save profile"))
                .clicked()
            {
                self.save(ctx);
            }
            if ui.button("Discard").clicked() {
                self.stage = Stage::Idle;
            }
        });
    }

    fn save(&mut self, ctx: &mut PanelContext<'_>) {
        let Stage::Reviewing(room) = &self.stage else {
            return;
        };
        let name = self.profile.trim().to_owned();

        match room.save(&name) {
            Ok(()) => {
                ctx.config.room = Some(name);
                ctx.dirty = true;
                *ctx.room = Some(room.clone());
                self.message = None;
                self.stage = Stage::Idle;
            }
            Err(error) => self.message = Some(format!("could not save the profile: {error:#}")),
        }
    }

    // ---- profiles ---------------------------------------------------------

    fn profiles(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        ui.horizontal(|ui| {
            ui.strong("Room profile");

            match ctx.config.room.as_deref() {
                Some(name) if ctx.room.is_some() => {
                    ui.colored_label(Color32::from_rgb(120, 200, 120), name)
                }
                Some(name) => ui.colored_label(
                    Color32::from_rgb(220, 130, 130),
                    format!("{name} (failed to load)"),
                ),
                None => ui.label(RichText::new("none loaded").weak()),
            };
        });

        let saved = RoomCalibration::list();
        if !saved.is_empty() {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("saved:").weak());
                for name in saved {
                    if ui.button(&name).clicked() {
                        match RoomCalibration::load(&name) {
                            Ok(room) => {
                                *ctx.room = Some(room);
                                ctx.config.room = Some(name);
                                ctx.dirty = true;
                                self.message = None;
                            }
                            Err(error) => {
                                self.message = Some(format!("could not load {name}: {error:#}"))
                            }
                        }
                    }
                }
            });
        }

        // The quality of the profile in force is worth seeing at any time, not
        // only in the minute after it was solved. A user whose feet are wrong
        // needs to be able to look this up.
        if let Some(room) = ctx.room.as_ref() {
            ui.add_space(8.0);
            summary(ui, room, "calib-loaded", &mut self.viewer, &self.walk);
        }
    }
}

/// The result table, shared by a fresh solve and a profile already in force.
fn summary(
    ui: &mut egui::Ui,
    room: &RoomCalibration,
    id: &str,
    viewer: &mut Viewer3d,
    walk: &[Point3<f64>],
) {
    let good = room.rms_degrees() <= GOOD_RMS_DEGREES;

    ui.horizontal(|ui| {
        ui.strong("Result");
        ui.colored_label(
            if good {
                Color32::from_rgb(120, 200, 120)
            } else {
                Color32::from_rgb(230, 180, 90)
            },
            format!("{:.3}\u{b0} RMS", room.rms_degrees()),
        );
        ui.label(
            RichText::new(format!(
                "{} of {} sightings used",
                room.used,
                room.used + room.rejected
            ))
            .weak(),
        );
    });

    if !good {
        ui.colored_label(
            Color32::from_rgb(230, 180, 90),
            "That is high enough to put the feet visibly wrong. Walking again, more slowly \
                 and covering more of the room, is usually the fix.",
        );
    }

    ui.add_space(6.0);
    egui::Grid::new(id)
        .num_columns(6)
        .striped(true)
        .show(ui, |ui| {
            for heading in ["Camera", "Position", "RMS", "Coverage", "Spread", "Latency"] {
                ui.strong(heading);
            }
            ui.end_row();

            for camera in &room.cameras {
                ui.label(&camera.id);

                let p = camera.camera.position();
                ui.label(format!("{:.2}, {:.2}, {:.2}", p.x, p.y, p.z));

                ui.colored_label(
                    if camera.rms_degrees() <= GOOD_RMS_DEGREES {
                        Color32::from_rgb(120, 200, 120)
                    } else {
                        Color32::from_rgb(230, 180, 90)
                    },
                    format!("{:.3}\u{b0}", camera.rms_degrees()),
                );

                ui.label(format!("{:.0}%", camera.coverage * 100.0));

                // Near zero means the walk was almost flat, and the answer
                // rests on nothing however small the residual looks.
                ui.colored_label(
                    if camera.spread > 0.05 {
                        Color32::from_rgb(120, 200, 120)
                    } else {
                        Color32::from_rgb(220, 130, 130)
                    },
                    format!("{:.2}", camera.spread),
                );

                // A camera whose latency could not be measured is not broken;
                // the walk simply was not brisk enough to leave a mark, and
                // saying so is more use than showing a number nothing backs.
                match camera.latency {
                    Some(estimate) if estimate.is_confident() => {
                        ui.label(format!("{:.0} ms", estimate.millis()))
                    }
                    Some(estimate) => ui.colored_label(
                        Color32::from_rgb(190, 190, 190),
                        format!("~{:.0} ms, unsure", estimate.millis()),
                    ),
                    None => ui.label(RichText::new("{2014}").weak()),
                };
                ui.end_row();
            }
        });

    ui.add_space(8.0);
    room_view(ui, room, viewer, walk);

    ui.add_space(6.0);
    ui.collapsing("Device offsets", |ui| {
        ui.label(
            RichText::new(
                "How far each keypoint sits from the device it was matched to, in the \
                     device's own frame. A few centimetres is expected.",
            )
            .weak()
            .small(),
        );
        for (rig, offset) in &room.rigs {
            ui.label(format!(
                "{}: {:.1}, {:.1}, {:.1} cm",
                rig.label(),
                offset.x * 100.0,
                offset.y * 100.0,
                offset.z * 100.0
            ));
        }
    });
}

/// Everything standing between the user and a recording.
///
/// Listed rather than reduced to a disabled button with no explanation: each of
/// these has a different fix, and the user is the only one who can apply it.
fn blockers(ctx: &PanelContext<'_>) -> Vec<String> {
    let mut out = Vec::new();

    match ctx.vr.channel() {
        None => out.push("SteamVR is not connected".to_owned()),
        Some(channel) => {
            if channel.stats().state != LinkState::Connected {
                out.push("SteamVR is not connected".to_owned());
            } else if !channel.is_tracking(Role::Head) {
                out.push("the headset is not being tracked".to_owned());
            }
        }
    }

    let streaming = ctx
        .config
        .cameras
        .iter()
        .filter(|camera| camera.enabled)
        .filter(|camera| ctx.pipeline.channel(&camera.id).is_some())
        .count();
    if streaming < 2 {
        out.push(format!(
            "{streaming} camera(s) are running keypoints; two is the minimum and three or more \
             is what survives being stood in front of"
        ));
    }

    out
}

/// Draws a coverage grid: one cell per region of the frame, brighter where the
/// walk went.
fn coverage_map(ui: &mut egui::Ui, coverage: &Coverage) {
    const CELL: f32 = 14.0;

    let size = egui::vec2(CELL * coverage.columns as f32, CELL * coverage.rows as f32);
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    // Scaled against the busiest cell rather than an absolute count: what the
    // user needs to see is which parts of the frame are thin *relative to the
    // rest*, which is where they have not walked yet.
    let busiest = (0..coverage.rows)
        .flat_map(|row| (0..coverage.columns).map(move |column| (column, row)))
        .map(|(column, row)| coverage.count(column, row))
        .max()
        .unwrap_or(0)
        .max(1);

    for row in 0..coverage.rows {
        for column in 0..coverage.columns {
            let count = coverage.count(column, row);
            let fill = (count as f32 / busiest as f32).sqrt();
            let cell = egui::Rect::from_min_size(
                egui::pos2(
                    rect.min.x + column as f32 * CELL,
                    rect.min.y + row as f32 * CELL,
                ),
                egui::vec2(CELL - 1.0, CELL - 1.0),
            );

            let colour = if count == 0 {
                Color32::from_rgb(48, 50, 56)
            } else {
                Color32::from_rgb(
                    (60.0 + 60.0 * fill) as u8,
                    (70.0 + 130.0 * fill) as u8,
                    (80.0 + 60.0 * fill) as u8,
                )
            };
            painter.rect_filled(cell, 1.0, colour);
        }
    }
}

fn duration(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    format!("{}:{:02}", seconds / 60, seconds % 60)
}

/// The solved room, drawn.
///
/// The residual says whether the cameras agree with each other. This says
/// whether they agree with the room, which is a different question and the one
/// a user can actually answer by looking: the walk should trace the floor they
/// walked on, and the cameras should be where they put them.
fn room_view(
    ui: &mut egui::Ui,
    room: &RoomCalibration,
    viewer: &mut Viewer3d,
    walk: &[Point3<f64>],
) {
    const PALETTE: [Color32; 4] = [
        Color32::from_rgb(120, 200, 250),
        Color32::from_rgb(250, 190, 90),
        Color32::from_rgb(150, 220, 150),
        Color32::from_rgb(230, 150, 220),
    ];

    let mut scene = Scene::default();
    scene.floor(3.0, 0.5);

    if walk.len() > 1 {
        // Thinned: a walk holds tens of thousands of samples and a line
        // between every pair of them is a solid smear rather than a path.
        let stride = (walk.len() / 600).max(1);
        let thinned: Vec<Point3<f64>> = walk.iter().step_by(stride).copied().collect();
        scene.path(&thinned, Color32::from_rgb(96, 104, 120));
    }

    for (index, camera) in room.cameras.iter().enumerate() {
        scene.camera(
            &camera.camera,
            &camera.id,
            PALETTE[index % PALETTE.len()],
            0.8,
        );
    }

    if ui.button("Frame the room").clicked() {
        let mut points: Vec<Point3<f64>> =
            room.cameras.iter().map(|c| c.camera.position()).collect();
        points.extend(walk.iter().copied());
        viewer.frame(&points);
    }

    viewer.show(ui, &scene, 320.0);
    ui.label(
        RichText::new("Drag to turn, scroll to zoom.")
            .weak()
            .small(),
    );
}
