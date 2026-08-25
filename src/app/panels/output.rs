//! Where the trackers go, and whether they are arriving.
//!
//! This is the last panel in the chain and the one with the least of its own to
//! say: if the body is right, sending it is a solved problem. What it has to be
//! good at is the case where nothing appears in the game — which is almost never
//! a fault here. It is a camera that cannot see the feet, a room profile that
//! was never loaded, or a consumer listening on a different port. So the panel
//! leads with which of those it is rather than with a row of zeroes.

use egui::RichText;

use crate::config::{OutputConfig, SinkKind};
use crate::output::TrackerRole;
use crate::output::stage::{OutputStats, TrackerReport};

use super::notice::{Level, Notice};
use super::{BAD, FAIR, GOOD, PanelContext};

/// Uncertainty a tracker is good at, in metres.
const GOOD_SIGMA: f64 = 0.02;
/// Uncertainty past which a tracker is not worth driving a limb from.
const POOR_SIGMA: f64 = 0.05;

#[derive(Default)]
pub struct OutputPanel {
    /// Whatever is wrong, held steady. The trackers section under it is a
    /// grid of checkboxes, and a line appearing above them moves every one.
    notice: Notice,
}

impl OutputPanel {
    pub fn ui(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        let stats = ctx.sender.channel().map(|channel| channel.stats());

        match &stats {
            Some(stats) if stats.running => self.summary(ui, stats),
            _ => self.idle(ui, ctx),
        }

        ui.add_space(10.0);
        egui::CollapsingHeader::new("Trackers")
            .default_open(true)
            .show(ui, |ui| self.trackers(ui, ctx, stats.as_ref()));

        ui.add_space(4.0);
        egui::CollapsingHeader::new("Destination")
            .default_open(true)
            .show(ui, |ui| self.destination(ui, ctx));

        ui.add_space(4.0);
        egui::CollapsingHeader::new("Settings")
            .default_open(false)
            .show(ui, |ui| self.settings(ui, ctx));

        ui.ctx().request_repaint();
    }

    /// What is going out, in one line.
    fn summary(&mut self, ui: &mut egui::Ui, stats: &OutputStats) {
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("{:.0} Hz", stats.rate)).strong());
            ui.separator();
            ui.label(format!("{} \u{2192} {}", stats.sink, stats.target));
            ui.separator();

            let live = stats.trackers.iter().filter(|t| !t.lost).count();
            ui.label(format!("{live} of {} trackers", stats.trackers.len()));
            ui.separator();

            // The number that explains everything else about how it feels.
            ui.label(
                RichText::new(format!("{:.0} ms ahead", stats.lead_ms))
                    .color(if stats.lead_ms > 250.0 { FAIR } else { GOOD }),
            );
        });

        // Latched: this line sits directly above a grid of checkboxes, and
        // the statistics behind it cross their thresholds every repaint.
        let wanted = stats
            .problem
            .as_ref()
            .map(|text| (text.clone(), Level::Problem))
            .or_else(|| {
                stats
                    .warning
                    .as_ref()
                    .map(|text| (text.clone(), Level::Warning))
            });
        self.notice.show(ui, wanted);
    }

    /// What to say when nothing is going out.
    ///
    /// Which prerequisite is missing, and which panel to go and fix it in. The
    /// output stage sits on top of every other one, so "not sending" on its own
    /// tells a user nothing they did not already know.
    fn idle(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        ui.label(RichText::new("Not sending.").strong());
        ui.add_space(6.0);

        self.notice.show(
            ui,
            ctx.output_problem
                .map(|text| (text.to_owned(), Level::Problem)),
        );
        if self.notice.visible() {
            ui.add_space(6.0);
        }

        super::checklist(
            ui,
            &[
                (
                    "Output enabled",
                    ctx.config.output.enabled,
                    "turn it on under Settings below",
                ),
                (
                    "At least one tracker chosen",
                    !ctx.config.output.enabled_roles().is_empty(),
                    "choose them under Trackers below",
                ),
                (
                    "A body being reconstructed",
                    ctx.fusion.is_running(),
                    "start it in the Tracking panel",
                ),
            ],
        );
    }

    /// The tracker list: what is on, what index it goes out as, how it is doing.
    fn trackers(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &mut PanelContext<'_>,
        stats: Option<&OutputStats>,
    ) {
        ctx.config.output.complete();

        let mut changed = false;
        egui::Grid::new("trackers").striped(true).show(ui, |ui| {
            for heading in [
                "",
                "Tracker",
                "Sent as",
                "Arriving",
                "Uncertainty",
                "Offset (m)",
            ] {
                ui.label(RichText::new(heading).strong());
            }
            ui.end_row();

            // The index a tracker will go out as depends on which others are
            // on, so it is worked out from the same list the checkboxes edit
            // rather than from whatever the running stage happens to have.
            let mut index = 0u8;
            for slot in 0..ctx.config.output.trackers.len() {
                let role = ctx.config.output.trackers[slot].role;
                let enabled = ctx.config.output.trackers[slot].enabled;
                if enabled {
                    index += 1;
                }

                changed |= ui
                    .checkbox(&mut ctx.config.output.trackers[slot].enabled, "")
                    .changed();

                let name = RichText::new(role.label());
                ui.label(if role.is_essential() {
                    name.strong()
                } else {
                    name
                });

                if enabled {
                    ui.label(format!("{index}"));
                } else {
                    ui.label(RichText::new("\u{2014}").weak());
                }

                let report = stats
                    .and_then(|stats| stats.trackers.iter().find(|tracker| tracker.role == role));
                arriving(ui, enabled, report);
                uncertainty(ui, report);

                ui.horizontal(|ui| {
                    for axis in 0..3 {
                        changed |= ui
                            .add(
                                egui::DragValue::new(
                                    &mut ctx.config.output.trackers[slot].offset[axis],
                                )
                                .speed(0.005)
                                .range(-0.5..=0.5)
                                .fixed_decimals(3),
                            )
                            .changed();
                    }
                });
                ui.end_row();
            }
        });

        ui.add_space(4.0);
        ui.label(
            RichText::new(
                "Hips and both feet are what full-body tracking needs; the rest are extra. \
                 Offsets move a tracker along its own axes, which is where a real puck would \
                 have sat. Turning one on or off renumbers the ones after it, so recalibrate \
                 in the game afterwards.",
            )
            .weak(),
        );

        if changed {
            ctx.dirty = true;
        }
    }

    /// Which backend, and where it is pointed.
    fn destination(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        let mut changed = false;

        for kind in SinkKind::ALL {
            let selected = ctx.config.output.sink == kind;
            if ui.radio(selected, kind.label()).clicked() && !selected {
                ctx.config.output.sink = kind;
                changed = true;
            }
            ui.label(RichText::new(kind.description()).weak());
            ui.add_space(4.0);
        }

        ui.separator();
        ui.horizontal(|ui| {
            ui.label("Address");
            changed |= ui
                .add(
                    egui::TextEdit::singleline(ctx.config.output.target_mut()).desired_width(180.0),
                )
                .changed();
            // Checked here rather than left to the send loop: a typo in a port
            // number is the likeliest reason for nothing arriving, and finding
            // out when the stage starts means finding out in the log.
            if !resolves(ctx.config.output.target()) {
                ui.label(RichText::new("not an address").color(BAD));
            }
        });

        if ctx.config.output.sink == SinkKind::Vmt {
            changed |= ui
                .checkbox(
                    &mut ctx.config.output.vmt_send_room_matrix,
                    "Send SteamVR's room setup to VMT",
                )
                .changed();
            ui.label(
                RichText::new(
                    "VMT places its devices in SteamVR's raw space, which sits a floor height \
                     away from the one everything here works in. Optra can read the real \
                     transform and hand it over for this run only. Turn it off if you have \
                     already set VMT's room matrix yourself.",
                )
                .weak(),
            );
        }

        if changed {
            ctx.dirty = true;
        }
    }

    fn settings(&mut self, ui: &mut egui::Ui, ctx: &mut PanelContext<'_>) {
        let mut changed = false;

        changed |= ui
            .checkbox(
                &mut ctx.config.output.enabled,
                "Send trackers when a body is being reconstructed",
            )
            .changed();

        ui.add_space(6.0);
        changed |= ui
            .add(egui::Slider::new(&mut ctx.config.output.rate_hz, 30..=144).text("Send rate (Hz)"))
            .changed();
        ui.label(
            RichText::new(
                "Faster than fusion runs on purpose: each send predicts to a later instant \
                 from the same reconstruction, so the poses really do move between them.",
            )
            .weak(),
        );

        ui.add_space(6.0);
        changed |= ui
            .add(
                egui::Slider::new(&mut ctx.config.output.max_lead_ms, 0..=300)
                    .text("Predict at most (ms)"),
            )
            .changed();
        ui.label(
            RichText::new(
                "The time from a camera exposing a frame to a body existing is measured and \
                 predicted through, so this caps the total rather than setting it. Turn it \
                 down to zero if the trackers are shaking: nothing is guessed at all then, so \
                 the body comes out late, and if it also comes out steady the trouble is here \
                 rather than in the cameras.",
            )
            .weak(),
        );

        ui.add_space(6.0);
        changed |= ui
            .add(
                egui::Slider::new(&mut ctx.config.output.max_sigma, 0.02..=0.30)
                    .text("Withhold past (m)"),
            )
            .changed();
        ui.label(
            RichText::new(
                "A tracker the cameras cannot place this well is not sent at all. Holding one \
                 back reads as a limb that stopped; sending it reads as a limb somewhere it \
                 never was.",
            )
            .weak(),
        );

        if changed {
            ctx.dirty = true;
        }
    }
}

/// How reliably a tracker is reaching the consumer.
fn arriving(ui: &mut egui::Ui, enabled: bool, report: Option<&TrackerReport>) {
    if !enabled {
        ui.label(RichText::new("off").weak());
        return;
    }

    let Some(report) = report else {
        ui.label(RichText::new("\u{2014}").weak());
        return;
    };

    if report.lost {
        ui.label(RichText::new("lost").color(BAD));
        return;
    }

    let live = report.live * 100.0;
    ui.label(RichText::new(format!("{live:.0}%")).color(if live > 90.0 {
        GOOD
    } else if live > 50.0 {
        FAIR
    } else {
        BAD
    }));
}

fn uncertainty(ui: &mut egui::Ui, report: Option<&TrackerReport>) {
    let Some(report) = report.filter(|report| !report.lost) else {
        ui.label(RichText::new("\u{2014}").weak());
        return;
    };

    let text = format!("{:.0} mm", report.sigma * 1000.0);
    let text = if report.inferred {
        format!("{text}  (inferred)")
    } else {
        text
    };

    ui.label(RichText::new(text).color(if report.sigma <= GOOD_SIGMA {
        GOOD
    } else if report.sigma <= POOR_SIGMA {
        FAIR
    } else {
        BAD
    }));
}

/// Whether a string is something a socket could be pointed at.
fn resolves(target: &str) -> bool {
    use std::net::ToSocketAddrs;
    target
        .to_socket_addrs()
        .ok()
        .and_then(|mut found| found.next())
        .is_some()
}

/// Roles the essential set is missing.
pub fn missing_essentials(config: &OutputConfig) -> Vec<TrackerRole> {
    let enabled = config.enabled_roles();
    TrackerRole::ALL
        .iter()
        .filter(|role| role.is_essential() && !enabled.contains(role))
        .copied()
        .collect()
}
