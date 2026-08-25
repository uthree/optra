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

        for (needed, met, where_to_go) in [
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
        ] {
            let (mark, colour) = if met {
                ("\u{2713}", GOOD)
            } else {
                ("\u{2717}", BAD)
            };
            ui.horizontal(|ui| {
                ui.label(RichText::new(mark).color(colour));
                ui.label(needed);
                if !met {
                    ui.label(RichText::new(format!("\u{2014} {where_to_go}")).weak());
                }
            });
        }

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
                ui.label(if camera.latency_ms > 0.0 {
                    format!("{:.0} ms", camera.latency_ms)
                } else {
                    "not measured".to_owned()
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
                // Nothing saw it; the fit put it where the joints around it
                // say it has to be.
                None => {
                    ui.label(RichText::new("\u{2014}").weak());
                    ui.label(RichText::new("inferred").color(Color32::from_rgb(150, 150, 220)));
                    ui.label(RichText::new("\u{2014}").weak());
                    ui.label(String::new());
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

    ui.add_space(4.0);
    changed |= ui
        .add(
            egui::Slider::new(&mut fusion.min_confidence, 0.05..=0.9)
                .text("Keypoint confidence gate"),
        )
        .changed();

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
            "How far behind the slowest camera's measured delay the clock runs. It needs \
             about one frame interval of the slowest camera; less and cameras start \
             dropping out of ticks.",
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
        return Color32::from_rgb(150, 150, 220);
    }
    if sigma <= GOOD_SIGMA {
        GOOD
    } else if sigma <= POOR_SIGMA {
        FAIR
    } else {
        BAD
    }
}
