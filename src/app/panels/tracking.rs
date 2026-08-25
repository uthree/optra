//! The live skeleton, and why it looks the way it does.
//!
//! When tracking misbehaves the question is never "is it wrong" — the user can
//! see that — but "which part is wrong". A skeleton on its own cannot answer
//! that. So this panel shows the reconstruction next to the three things that
//! explain it: how much each camera is contributing, how well each joint is
//! pinned down, and what the body was measured as.
//!
//! Both skeletons are drawn. The solid one is where the cameras say the body is
//! at the instant the fusion clock reconstructed; the faint one is where it is
//! predicted to be, which is what the trackers are actually told. Seeing the
//! gap between them is the only way to judge whether the prediction horizon is
//! set sensibly.

use egui::{Color32, RichText};
use nalgebra::Point3;

use crate::app::viewer3d::{Scene, Viewer3d};
use crate::fusion::bones::{BONES, MeasureOptions, Skeleton};
use crate::fusion::fuse::Missing;
use crate::fusion::stage::{FusionFrame, FusionStats};
use crate::models::Joint;
use crate::vr::Role;

use super::notice::{Level, Notice, Threshold};
use super::{BAD, FAIR, GOOD, PanelContext};

/// Positional uncertainty a joint is good at, in metres. Below this the joint
/// is as good as the cameras can make it.
const GOOD_SIGMA: f64 = 0.01;
/// Uncertainty past which a joint is not worth driving a tracker from.
const POOR_SIGMA: f64 = 0.04;

/// A joint the fit placed rather than the cameras seeing.
const INFERRED: Color32 = Color32::from_rgb(150, 150, 220);

#[derive(Default)]
pub struct TrackingPanel {
    viewer: Viewer3d,
    /// Set once, the first time a body appears, so the view starts pointed at
    /// the user instead of at the origin.
    framed: bool,
    /// Whatever is wrong with the reconstruction, held steady.
    notice: Notice,
    /// The fit correction, which hovers on its threshold while a user stands
    /// still and would otherwise strobe the button under it.
    correction: Notice,
    correcting: Threshold,
}

impl TrackingPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        let channel = ctx.fusion.channel().cloned();
        let Some(channel) = channel else {
            self.idle(ui, ctx);
            return;
        };

        let stats = channel.stats();
        let frame = channel.latest();

        self.summary(ui, &stats);
        ui.add_space(8.0);

        if let Some(frame) = frame.as_deref() {
            self.view(ui, frame, &stats);
        } else {
            ui.label(RichText::new("Waiting for the first reconstruction.").weak());
        }

        ui.add_space(10.0);
        egui::CollapsingHeader::new("Cameras")
            .default_open(true)
            .show(ui, |ui| cameras(ui, &stats));

        ui.add_space(4.0);
        egui::CollapsingHeader::new("Joints")
            .default_open(false)
            .show(ui, |ui| match frame.as_deref() {
                Some(frame) => joints(ui, frame),
                None => {
                    ui.label(RichText::new("Nothing reconstructed yet.").weak());
                }
            });

        ui.add_space(4.0);
        egui::CollapsingHeader::new("Body")
            .default_open(false)
            .show(ui, |ui| self.body(ui, ctx, &stats));

        ui.add_space(4.0);
        egui::CollapsingHeader::new("Settings")
            .default_open(false)
            .show(ui, |ui| settings(ui, ctx));

        // An immediate-mode panel only redraws when something asks it to, and
        // nothing here is driven by input.
        ui.ctx().request_repaint();
    }

    /// What to say when there is nothing to show.
    ///
    /// The useful answer is which prerequisite is missing, not that tracking is
    /// off. Everything this stage needs is something the user set up in another
    /// panel, so the message points at the one to go back to.
    fn idle(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        ui.label(RichText::new("Not tracking.").strong());
        ui.add_space(6.0);

        self.notice.show(
            ui,
            ctx.fusion_problem
                .map(|text| (text.to_owned(), Level::Warning)),
        );
        if self.notice.visible() {
            ui.add_space(6.0);
        }

        super::checklist(
            ui,
            &[
                (
                    "Fusion enabled",
                    ctx.config.fusion.enabled,
                    "turn it on under Settings below",
                ),
                (
                    "Cameras running with a model",
                    ctx.pipeline.is_running(),
                    "start them in the Cameras panel",
                ),
                (
                    "A calibrated room",
                    ctx.room.is_some(),
                    "run the wizard in the Calibration panel",
                ),
            ],
        );

        ui.add_space(10.0);
        let mut enabled = ctx.config.fusion.enabled;
        if ui
            .checkbox(&mut enabled, "Track when everything is ready")
            .changed()
        {
            ctx.config.fusion.enabled = enabled;
            ctx.dirty = true;
        }
    }

    fn summary(&mut self, ui: &mut egui::Ui, stats: &FusionStats) {
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{:.0} Hz", stats.rate)).strong());
            ui.separator();
            ui.label(format!("{} joints", stats.joints));
            if stats.inferred > 0 {
                ui.label(RichText::new(format!("({} inferred)", stats.inferred)).weak());
            }
            ui.separator();
            ui.label(format!("{} lower body", stats.lower_body));
            ui.separator();
            ui.label(RichText::new(format!("{:.0} ms behind", stats.lag_ms)).weak());
        });

        // Where shaking comes from, read left to right. The chain is four
        // stages long and each one can shake for its own reason, so "it
        // shakes" on its own has never been a report anybody could act on.
        // Anything moving at a constant velocity contributes nothing to these
        // numbers, so walking about does not inflate them.
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Shake").weak());
            for (name, metres) in [
                ("cameras", stats.shake.raw),
                ("fit", stats.shake.fitted),
                ("smoothed", stats.shake.filtered),
                ("sent", stats.shake.predicted),
            ] {
                ui.label(RichText::new(name).weak());
                ui.label(
                    RichText::new(format!("{:.0} mm", metres * 1000.0)).color(shaking(metres)),
                );
            }

            // Beside the shake because it is the same trade seen from the other
            // side. Shake is what an over-bold prediction costs; this is what a
            // timid one costs, and a user moving the settings below needs to
            // watch both or they will simply trade one for the other without
            // knowing it.
            ui.separator();
            ui.label(RichText::new("Prediction reaching").weak());
            match stats.reach {
                Some(reach) => {
                    ui.label(RichText::new(format!("{:.0}%", reach * 100.0)).color(reaching(reach)))
                }
                None => ui.label(RichText::new("\u{2014}").weak()),
            };
        });
        if stats.reach.is_some_and(|reach| reach < REACH_TIMID) {
            ui.label(
                RichText::new(
                    "The prediction is acting on a fraction of the speed the cameras \
                     measured, so the trackers are being sent a body nearer to where it \
                     was than to where it is going. Lowering the prediction caution or \
                     raising the agility below will let more of it through — at the cost \
                     of the shake figures above.",
                )
                .weak(),
            );
        }

        // Why the joints that are not measured are not measured. "Twenty-three
        // of twenty-six inferred" says something is badly wrong and nothing
        // about what: a camera that cannot see the legs, a model that will not
        // commit to them, a calibration that stopped the rays meeting and a
        // geometry that cannot place them are four unrelated problems with four
        // different next moves. All six counts are shown whether or not they
        // are zero, so the row never changes shape.
        //
        // Settling is the odd one out and is last for that reason: it is not a
        // fault in the room but a joint that keeps passing and failing the same
        // test, held back until it stops. A count that stays high says the
        // thresholds are being sat on, which is worth knowing because it means
        // the rest of this panel describes a body that keeps changing which
        // joints it is made of.
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Not measured").weak());
            for (name, count) in [
                ("unseen", stats.tally.unseen),
                ("unsure", stats.tally.unsure),
                ("one ray", stats.tally.one_ray),
                ("disagreed", stats.tally.disagreed),
                ("too uncertain", stats.tally.uncertain),
                ("settling", stats.tally.settling),
            ] {
                ui.label(RichText::new(name).weak());
                ui.label(RichText::new(format!("{count}")).color(if count == 0 {
                    GOOD
                } else if count <= 4 {
                    FAIR
                } else {
                    BAD
                }));
            }
        });

        // The one number that says whether the room is still calibrated. It is
        // measured continuously, from the user rather than from a checkerboard,
        // and it is what says whether a camera has been knocked since it was
        // solved. One means the rays landed exactly as accurately as the pose
        // models claimed they would.
        //
        // The joint count next to it is not decoration. This is measured only
        // over the joints that could be solved, and a joint whose rays missed
        // each other entirely is not among them — so on a body where most of
        // the joints failed outright, a fine-looking figure here is a fine
        // figure about the handful that survived. It read 1.2x on a room where
        // eleven joints could not be triangulated at all.
        if stats.disagreement > 0.0 {
            let solid = stats.tally.measured >= 8;
            ui.horizontal(|ui| {
                ui.label(RichText::new("Cameras agree to").weak());
                ui.colored_label(
                    if !solid {
                        FAIR
                    } else if stats.disagreement <= 1.5 {
                        GOOD
                    } else if stats.disagreement <= 3.0 {
                        FAIR
                    } else {
                        BAD
                    },
                    format!("{:.1}\u{00d7} their own keypoints", stats.disagreement),
                );
                ui.label(
                    RichText::new(format!(
                        "measured on the {} joint(s) that could be solved",
                        stats.tally.measured
                    ))
                    .weak(),
                );
                // The other half of the sentence. "The cameras disagreed" is
                // only half a fact until it says what they were asked for: a
                // gate tighter than the room profile that set it throws out
                // rays that were right, and the two numbers side by side are
                // what makes that visible instead of just "18 disagreed".
                if stats.gate > 0.0 {
                    ui.label(
                        RichText::new(format!(
                            "\u{2014} rays had to pass within {:.1}\u{00b0}",
                            stats.gate.to_degrees()
                        ))
                        .weak(),
                    );
                }
            });
        }

        // The sharpest check here: two independent answers to where the user's
        // head is, one of them from a device that knows to a millimetre. It
        // measures the total error of the calibration, the lens models, the
        // room transform and the clock in one number, and the direction is most
        // of the diagnosis — nearly all vertical is a room setup run at the
        // wrong height, anything else is the calibration itself.
        if let Some(head) = stats.head {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Head").weak());
                ui.colored_label(
                    if head.norm() <= 0.30 {
                        GOOD
                    } else if head.norm() <= 0.60 {
                        FAIR
                    } else {
                        BAD
                    },
                    format!("{:.0} cm from the headset", head.norm() * 100.0),
                );
                ui.label(
                    RichText::new(format!(
                        "({:+.0} across, {:+.0} up, {:+.0} forward)",
                        head.x * 100.0,
                        head.y * 100.0,
                        head.z * 100.0
                    ))
                    .weak(),
                );
            });
        }

        // The one measurement nothing else here can make. A uniformly scaled
        // set of cameras agrees with itself perfectly — every ray still meets
        // every other ray and every reprojection residual is still zero — so
        // the wizard's RMS is happy and the agreement factor reads 1.0 while
        // the body comes out two-thirds life size. It takes an external metre
        // rule, and the headset is one. Measured by moving, not by standing.
        ui.horizontal(|ui| {
            ui.label(RichText::new("Room scale").weak());
            match stats.scale {
                Some(scale) => ui.colored_label(
                    if (scale - 1.0).abs() <= 0.05 {
                        GOOD
                    } else if (scale - 1.0).abs() <= 0.10 {
                        FAIR
                    } else {
                        BAD
                    },
                    format!("{scale:.2}\u{00d7} life size"),
                ),
                None => ui.label(RichText::new("walk about to measure it").weak()),
            };
        });

        // Reported next to the rate rather than buried in a section, because it
        // invalidates everything else on the panel. The cameras watching the
        // feet are an independent measurement of a quantity the rest of the
        // application inherits from SteamVR and never checks.
        if let Some(floor) = stats.floor {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Floor").weak());
                ui.colored_label(
                    if floor.abs() <= 0.06 { GOOD } else { FAIR },
                    format!("{:+.0} cm against SteamVR's", floor * 100.0),
                );
            });
        }

        // Latched, because the whole panel hangs below it. Every warning here
        // is a live statistic against a fixed limit, and a camera or a floor
        // estimate sitting on its limit would otherwise move the skeleton, the
        // tables and every button under them at the repaint rate.
        if self.notice.visible() {
            ui.add_space(4.0);
        }
        self.notice.show(
            ui,
            stats
                .warning
                .as_ref()
                .map(|text| (text.clone(), Level::Warning)),
        );
    }

    fn view(&mut self, ui: &mut egui::Ui, frame: &FusionFrame, stats: &FusionStats) {
        let mut scene = Scene::default();
        scene.floor(3.0, 0.5);

        // The prediction first, so the measured skeleton draws over it.
        for bone in BONES {
            let (Some(from), Some(to)) = (
                frame.filtered.predicted(bone.from),
                frame.filtered.predicted(bone.to),
            ) else {
                continue;
            };
            scene.line(from, to, Color32::from_rgb(90, 90, 110), 1.0);
        }

        for bone in BONES {
            let (Some(from), Some(to)) =
                (frame.filtered.get(bone.from), frame.filtered.get(bone.to))
            else {
                continue;
            };
            // A bone is only as trustworthy as its worse end.
            let colour = quality(from.sigma.max(to.sigma), from.inferred || to.inferred);
            scene.line(from.point, to.point, colour, 2.0);
        }

        // A short cross at each joint, so a joint with no bone drawn to it is
        // still visible rather than silently missing.
        for (_, joint) in frame.filtered.iter() {
            let colour = quality(joint.sigma, joint.inferred);
            for axis in [
                nalgebra::Vector3::x(),
                nalgebra::Vector3::y(),
                nalgebra::Vector3::z(),
            ] {
                scene.line(
                    joint.point - axis * 0.02,
                    joint.point + axis * 0.02,
                    colour,
                    1.5,
                );
            }
        }

        if !self.framed && frame.filtered.count() > 0 {
            let points: Vec<Point3<f64>> = frame
                .filtered
                .iter()
                .map(|(_, joint)| joint.point)
                .collect();
            self.viewer.frame(&points);
            self.framed = true;
        }

        self.viewer.show(ui, &scene, 320.0);

        ui.horizontal(|ui| {
            ui.label(RichText::new("Solid: measured.").weak());
            ui.label(RichText::new("Faint: predicted, which is what is sent.").weak());
            if ui.button("Frame the body").clicked() {
                let points: Vec<Point3<f64>> = frame
                    .filtered
                    .iter()
                    .map(|(_, joint)| joint.point)
                    .collect();
                self.viewer.frame(&points);
            }
        });

        // Two centimetres is a limit the correction wanders across constantly
        // on a body standing still, and this line sits under the button that
        // frames the view.
        let correcting = self
            .correcting
            .over(stats.worst_correction, 0.02, 0.25)
            .then(|| {
                (
                    format!(
                        "The fit is moving a joint {:.0} cm to keep the body together.",
                        stats.worst_correction * 100.0
                    ),
                    Level::Warning,
                )
            });
        self.correction.show(ui, correcting);
    }

    fn body(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>, stats: &FusionStats) {
        let measure = MeasureOptions::default();
        let coverage = stats.body.coverage(&measure);

        ui.horizontal(|ui| {
            ui.add(
                egui::ProgressBar::new(coverage)
                    .desired_width(180.0)
                    .text(format!("{:.0}% measured", coverage * 100.0)),
            );

            let mut measuring = ctx.config.fusion.measure_body;
            if ui.checkbox(&mut measuring, "Keep measuring").changed() {
                ctx.config.fusion.measure_body = measuring;
                ctx.dirty = true;
                ctx.fusion.stop();
            }
        });

        height_check(ui, &stats.body, ctx);

        ui.add_space(6.0);
        egui::Grid::new("bones").striped(true).show(ui, |ui| {
            ui.label(RichText::new("Bone").strong());
            ui.label(RichText::new("Length").strong());
            ui.label(RichText::new("Spread").strong());
            ui.label(RichText::new("Samples").strong());
            ui.end_row();

            for bone in BONES {
                let Some(measured) = stats.body.get(*bone) else {
                    continue;
                };
                let settled = measured.is_settled(&measure);

                ui.label(bone.label());
                ui.label(format!("{:.1} cm", measured.length * 100.0));
                ui.label(
                    RichText::new(format!("\u{b1}{:.1} cm", measured.scatter * 100.0))
                        .color(if settled { GOOD } else { FAIR }),
                );
                ui.label(format!("{}", measured.samples));
                ui.end_row();
            }
        });
    }
}

/// Compares the measured leg against the headset's own height.
///
/// Two independent measurements of the same person: one from the cameras
/// through the calibration, one from the runtime's floor. They should agree,
/// and when they do not it is the calibration that is wrong, not the body — so
/// this reports rather than corrects.
fn height_check(ui: &mut egui::Ui, body: &Skeleton, ctx: &PanelContext<'_>) {
    let Some(leg) = body.leg_length() else { return };
    let Some(head) = ctx
        .vr
        .channel()
        .and_then(|channel| channel.latest())
        .and_then(|snapshot| snapshot.device(Role::Head).map(|device| device.pose))
    else {
        return;
    };

    // Hip to floor is a little more than the leg: the ankle sits above the sole
    // and the foot has a heel.
    let expected = head.translation.y * 0.53;
    let difference = leg - expected;

    let text = format!(
        "Leg measures {:.0} cm; the headset at {:.0} cm implies about {:.0} cm.",
        leg * 100.0,
        head.translation.y * 100.0,
        expected * 100.0
    );

    if difference.abs() > 0.08 {
        ui.label(
            RichText::new(format!(
                "{text} That is a large disagreement — check the room profile."
            ))
            .color(FAIR),
        );
    } else {
        ui.label(RichText::new(text).weak());
    }
}

fn cameras(ui: &mut egui::Ui, stats: &FusionStats) {
    egui::Grid::new("contributions")
        .striped(true)
        .show(ui, |ui| {
            for heading in ["Camera", "Keeping up", "Share", "Outvoted", "Delay"] {
                ui.label(RichText::new(heading).strong());
            }
            ui.end_row();

            for camera in &stats.cameras {
                ui.label(&camera.id);

                if let Some(problem) = &camera.problem {
                    ui.label(RichText::new(problem).color(BAD));
                    ui.end_row();
                    continue;
                }

                ui.label(
                    RichText::new(format!("{:.0}%", camera.aligned * 100.0))
                        .color(if camera.aligned > 0.9 { GOOD } else { FAIR }),
                );
                ui.label(format!("{:.0}%", camera.weight * 100.0));
                ui.label(
                    RichText::new(format!("{:.0}%", camera.rejected * 100.0)).color(
                        if camera.rejected < 0.2 {
                            Color32::from_rgb(160, 160, 160)
                        } else {
                            BAD
                        },
                    ),
                );
                ui.label(match camera.latency_ms {
                    Some(ms) => format!("{ms:.0} ms"),
                    None => "not measured".to_owned(),
                });
                ui.end_row();
            }
        });

    ui.add_space(4.0);
    ui.label(
        RichText::new(
            "Share is how much of the answer a camera carried, over the joints it voted \
             on. Outvoted is how often it disagreed with the rest — steadily high means \
             it is mis-calibrated or badly placed.",
        )
        .weak(),
    );
}

fn joints(ui: &mut egui::Ui, frame: &FusionFrame) {
    egui::Grid::new("joints").striped(true).show(ui, |ui| {
        for heading in [
            "Joint",
            "Rays",
            "Uncertainty",
            "Residual",
            "Dropped",
            "Moved by fit",
        ] {
            ui.label(RichText::new(heading).strong());
        }
        ui.end_row();

        for joint in Joint::ALL {
            let Some(fitted) = frame.fitted.get(joint) else {
                continue;
            };

            ui.label(joint.name());

            match frame.raw.get(joint) {
                Some(fused) => {
                    ui.label(format!("{}", fused.rays()));
                    ui.label(
                        RichText::new(format!("{:.0} mm", fused.sigma * 1000.0))
                            .color(quality(fused.sigma, false)),
                    );
                    ui.label(format!("{:.2}\u{b0}", fused.residual_degrees()));
                    ui.label(if fused.rejected.is_empty() {
                        String::new()
                    } else {
                        format!("{:?}", fused.rejected)
                    });
                }
                // The fit put it where the joints around it say it has to be,
                // and the interesting part is why the cameras did not. One dash
                // here used to cover five unrelated faults.
                None => {
                    let why = frame.raw.missing(joint);
                    ui.label(match why {
                        Some(Missing::Unsure { offered, .. }) => format!("{offered} unsure"),
                        Some(Missing::OneRay) => "1".to_owned(),
                        Some(Missing::Disagreed { rays, .. }) => format!("{rays} split"),
                        _ => "\u{2014}".to_owned(),
                    });
                    ui.label(match why {
                        Some(Missing::Uncertain { sigma }) => {
                            RichText::new(format!("{:.0} mm", sigma * 1000.0)).color(BAD)
                        }
                        // How far apart the rays were is the whole diagnosis for
                        // a joint that could not be solved.
                        Some(Missing::Disagreed { miss, .. }) => {
                            RichText::new(format!("{:.0} cm apart", miss * 100.0)).color(BAD)
                        }
                        _ => RichText::new("inferred").color(INFERRED),
                    });
                    ui.label(match why {
                        Some(Missing::Unsure { best, .. }) => {
                            RichText::new(format!("best {best:.2}")).weak()
                        }
                        _ => RichText::new("\u{2014}").weak(),
                    });
                    ui.label(match why {
                        Some(why) => RichText::new(why.remedy()).weak(),
                        None => RichText::new(String::new()),
                    });
                }
            }

            // How far the anatomy had to disagree with the cameras. A
            // millimetre is the fit doing its job; centimetres mean the
            // measured skeleton and the room do not describe the same person.
            ui.label(if fitted.inferred {
                RichText::new("\u{2014}").weak()
            } else {
                RichText::new(format!("{:.0} mm", fitted.correction * 1000.0))
            });
            ui.end_row();
        }
    });
}

fn settings(ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
    let mut fusion = ctx.config.fusion.clone();
    let mut changed = false;

    changed |= ui
        .checkbox(&mut fusion.enabled, "Track when everything is ready")
        .changed();

    changed |= ui
        .add(egui::Slider::new(&mut fusion.prediction_ms, 0..=150).text("Prediction horizon (ms)"))
        .changed();
    ui.label(
        RichText::new(
            "Only the delay Optra cannot see: the hop to the consumer and whatever it does \
             before drawing. The time from a camera exposing a frame to a body existing is \
             measured and added on top, so this does not have to track the camera setup.",
        )
        .weak(),
    );

    ui.add_space(4.0);
    changed |= ui
        .add(egui::Slider::new(&mut fusion.smoothing_hz, 0.2..=6.0).text("Smoothing (Hz)"))
        .changed();
    ui.label(RichText::new("Lower is stiller at rest and slower to react.").weak());

    // The two below are the same trade from two directions, and there is no
    // setting of them that is right everywhere: it depends on how noisy this
    // room's cameras and pose model are, and on whether the person is standing
    // at a desk or dancing. Measured on the simulated walk, lowering the
    // agility roughly halves the error on a walking body and makes it half as
    // good again on one whose legs move at a couple of metres per second.
    //
    // So they are presented together, with what to watch named in both
    // directions, rather than as two numbers in a list.
    ui.add_space(8.0);
    ui.strong("Following movement");
    ui.label(
        RichText::new(
            "These two decide how much of a movement reaches the trackers and how much \
             is held back as noise. There is no setting that is right for every room: a \
             sharp pose model on well-lit cameras leaves a signal worth following where a \
             480p webcam leaves noise. Move them while walking about and watch the two \
             figures at the top — Shake for going too far, Prediction reaching for not \
             going far enough.",
        )
        .weak(),
    );

    ui.add_space(4.0);
    changed |= ui
        .add(
            egui::Slider::new(&mut fusion.agility_mps2, 0.5..=12.0)
                .text("Body agility (m/s\u{b2})"),
        )
        .changed();
    ui.label(
        RichText::new(
            "How hard the body is expected to change speed between frames. Too low and \
             the filter will not believe a stride; too high and it follows whatever the \
             pose model did that frame.",
        )
        .weak(),
    );

    ui.add_space(4.0);
    changed |= ui
        .add(
            egui::Slider::new(&mut fusion.prediction_caution, 0.0..=1.0).text("Prediction caution"),
        )
        .changed();
    ui.label(
        RichText::new(
            "How much the prediction holds back on a speed it is unsure of. Higher is \
             stiller when you stand still, at the cost of arriving late with a movement. \
             Zero acts on the measured speed whatever the filter thinks of it.",
        )
        .weak(),
    );

    ui.add_space(4.0);
    changed |= ui
        .add(
            egui::Slider::new(&mut fusion.min_confidence, 0.05..=0.9)
                .text("Keypoint confidence gate"),
        )
        .changed();

    ui.add_space(4.0);
    changed |= ui
        .add(
            egui::Slider::new(&mut fusion.max_joint_sigma, 0.02..=0.40)
                .text("Withhold a joint past (m)"),
        )
        .changed();
    ui.label(
        RichText::new(
            "A joint the cameras cannot place this well is left out of the body, and the \
             fit places it from the skeleton instead. The figure it is compared against \
             includes how far the cameras turned out to disagree, so a room that has drifted \
             withholds more.",
        )
        .weak(),
    );

    ui.add_space(4.0);
    changed |= ui
        .add(egui::Slider::new(&mut fusion.rate_hz, 30..=120).text("Fusion rate (Hz)"))
        .changed();

    ui.add_space(4.0);
    changed |= ui
        .add(egui::Slider::new(&mut fusion.align_slack_ms, 10..=150).text("Alignment slack (ms)"))
        .changed();
    ui.label(
        RichText::new(
            "Headroom the clock keeps on top of what the cameras are measured to be \
             delivering. How far back it has to sit is worked out from what they \
             actually manage; this absorbs the variation between one tick and the next.",
        )
        .weak(),
    );

    ui.add_space(4.0);
    changed |= ui
        .add(egui::Slider::new(&mut fusion.max_lag_ms, 60..=500).text("Wait at most (ms)"))
        .changed();
    ui.label(
        RichText::new(
            "The clock follows whichever camera delivers latest, because a camera it does \
             not wait for drops in and out of ticks, and a joint reconstructed from a \
             different set of cameras every few ticks jumps by the disagreement between \
             them. This is where waiting stops being worth it: a camera later than this is \
             left out of the reconstruction instead, and said so above. Raise it if a \
             camera you need keeps being dropped; lower it if the body feels late.",
        )
        .weak(),
    );

    if changed {
        ctx.config.fusion = fusion;
        ctx.dirty = true;
        // The worker took a copy of the settings when it started, so it has to
        // be replaced rather than told.
        ctx.fusion.stop();
    }
}

/// Colour for a positional uncertainty, in metres.
fn quality(sigma: f64, inferred: bool) -> Color32 {
    if inferred {
        return INFERRED;
    }
    if sigma <= GOOD_SIGMA {
        GOOD
    } else if sigma <= POOR_SIGMA {
        FAIR
    } else {
        BAD
    }
}

/// Colours a stage's wobble.
///
/// A millimetre or two is the quantisation of a well-seen joint and is what a
/// good chain looks like. A centimetre is visible in the game.
fn shaking(metres: f64) -> Color32 {
    if metres <= 0.003 {
        GOOD
    } else if metres <= 0.010 {
        FAIR
    } else {
        BAD
    }
}

/// Below this share of the measured speed, the prediction is said to be timid
/// and the panel says what to do about it.
///
/// Half, because the prediction is what pays back the whole delay from a shutter
/// opening to a tracker moving. Acting on less than half the speed the cameras
/// measured leaves more than half of that delay unpaid, whatever the horizon is
/// set to — which is the failure the accuracy harness found and which nothing in
/// the application could previously see.
const REACH_TIMID: f64 = 0.5;

/// Colours how much of the measured speed the prediction is acting on.
///
/// Unlike the shake figures, more is not simply better: a hundred per cent on a
/// noisy room means the prediction is acting on speeds that were measurement
/// error, and the shake beside it is where that shows up. So only the timid end
/// is coloured as a fault, and the rest is left plain.
fn reaching(share: f64) -> Color32 {
    if share >= REACH_TIMID {
        GOOD
    } else if share >= 0.25 {
        FAIR
    } else {
        BAD
    }
}
