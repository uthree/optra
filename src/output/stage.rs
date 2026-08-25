//! The output thread.
//!
//! One loop on its own clock, faster than fusion runs: take the newest
//! reconstruction, carry it forward to the instant the consumer will act on it,
//! build the trackers that can be built, send.
//!
//! Sending faster than fusion reconstructs is not padding. Every send predicts
//! from the same filter state but to a later instant, so the poses genuinely
//! advance between them — and the consumer, which is running its own render
//! loop at whatever rate it likes, gets a pose closer to when it asked. What it
//! cannot do is invent detail: a body that fusion has not seen move is a body
//! extrapolated along a straight line, and the further this loop runs ahead of
//! the last reconstruction the more that is all it is doing.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::config::{OutputConfig, SinkKind};
use crate::fusion::stage::FusionChannel;
use crate::vr::{Role, VrChannel};
use crate::worker::timing::{Rate, Ticker, ema};
use crate::worker::{Shutdown, Supervisor};

use super::pose::{Posture, TrackerPose, TrackerRole};
use super::sink::{TrackerFrame, TrackerSink, assign};
use super::vmt::Vmt;
use super::vrchat::VrchatOsc;

/// How long a tracker may be missing before it is called lost rather than
/// merely occluded.
///
/// A limb passes behind the other one constantly, and a tracker that switched
/// itself off every time would be unusable. Anything longer than this is not an
/// occlusion, and holding a pose past it puts a foot on the floor that the user
/// is no longer standing on.
const PATIENCE: Duration = Duration::from_millis(500);

/// How old a reconstruction may be before nothing is sent from it at all.
///
/// Past this the extrapolation is most of the answer, and a straight line
/// through half a second of a walking body is not a body.
const STALE: Duration = Duration::from_millis(400);

/// One tracker as the panel sees it.
#[derive(Debug, Clone)]
pub struct TrackerReport {
    pub role: TrackerRole,
    /// Index it is being sent as, which is what a consumer's own calibration
    /// refers to.
    pub index: u8,
    /// Fraction of recent sends this tracker was actually in.
    pub live: f32,
    /// Worst uncertainty among the joints it is built from, in metres.
    pub sigma: f64,
    /// True when the fit placed one of those joints rather than the cameras
    /// seeing it.
    pub inferred: bool,
    /// True when it has been missing long enough to have been switched off.
    pub lost: bool,
}

#[derive(Debug, Clone, Default)]
pub struct OutputStats {
    pub running: bool,
    /// Which backend, and where it is pointed.
    pub sink: String,
    pub target: String,
    /// Measured sends per second.
    pub rate: f32,
    pub sent: u64,
    /// How far ahead of the reconstruction the last send predicted, in
    /// milliseconds. Not a setting: it is the fusion lag, which is measured,
    /// plus the configured horizon.
    pub lead_ms: f32,
    /// Whether a headset pose is going out with the trackers.
    pub head: bool,
    pub trackers: Vec<TrackerReport>,
    /// Why nothing is being sent, when nothing is.
    pub problem: Option<String>,
    /// Something worth saying about output that is otherwise working.
    pub warning: Option<String>,
}

/// The shared face of the output thread.
#[derive(Default)]
pub struct OutputChannel {
    stop: Shutdown,
    stats: Mutex<OutputStats>,
}

impl OutputChannel {
    pub fn stats(&self) -> OutputStats {
        self.stats.lock().clone()
    }
}

/// Owns the output thread, if one is running.
#[derive(Default)]
pub struct Output {
    channel: Option<Arc<OutputChannel>>,
}

impl Output {
    pub fn channel(&self) -> Option<&Arc<OutputChannel>> {
        self.channel.as_ref()
    }

    pub fn is_running(&self) -> bool {
        self.channel.is_some()
    }

    /// A stage with no thread behind it, holding one fixed set of statistics,
    /// so the panel's populated states can be laid out in a test.
    pub fn detached(stats: OutputStats) -> Self {
        Self {
            channel: Some(Arc::new(OutputChannel {
                stop: Shutdown::default(),
                stats: Mutex::new(stats),
            })),
        }
    }

    /// Starts sending, or says why it cannot.
    pub fn start(
        &mut self,
        config: &OutputConfig,
        fusion: Arc<FusionChannel>,
        vr: Option<Arc<VrChannel>>,
        supervisor: &mut Supervisor,
    ) -> Result<(), String> {
        self.stop();

        let enabled = config.enabled_roles();
        if enabled.is_empty() {
            return Err("no trackers are turned on".to_owned());
        }

        let indices = assign(&enabled);
        let sink = open(config, &indices, vr.as_deref()).map_err(|error| format!("{error:#}"))?;

        let channel = Arc::new(OutputChannel {
            stop: Shutdown::default(),
            stats: Mutex::new(OutputStats {
                running: true,
                sink: sink.name().to_owned(),
                target: sink.target(),
                trackers: indices
                    .iter()
                    .map(|(index, role)| TrackerReport {
                        role: *role,
                        index: *index,
                        live: 0.0,
                        sigma: 0.0,
                        inferred: false,
                        lost: true,
                    })
                    .collect(),
                ..OutputStats::default()
            }),
        });
        self.channel = Some(channel.clone());

        let config = config.clone();
        supervisor.spawn("output", move |global| {
            run(channel, config, indices, sink, fusion, vr, global)
        });

        Ok(())
    }

    pub fn stop(&mut self) {
        if let Some(channel) = self.channel.take() {
            channel.stop.cancel();
            channel.stats.lock().running = false;
        }
    }
}

/// Builds the sink the configuration asks for.
fn open(
    config: &OutputConfig,
    indices: &[(u8, TrackerRole)],
    vr: Option<&VrChannel>,
) -> anyhow::Result<Box<dyn TrackerSink>> {
    Ok(match config.sink {
        SinkKind::VrchatOsc => Box::new(VrchatOsc::open(&config.vrchat_target, indices.to_vec())?),
        SinkKind::Vmt => {
            // Only if the user asked for it and SteamVR actually told us: a
            // room matrix guessed at is worse than the one VMT already has.
            let room = config
                .vmt_send_room_matrix
                .then(|| vr.and_then(|vr| vr.standing_to_raw()))
                .flatten();
            Box::new(Vmt::open(&config.vmt_target, indices.to_vec(), room)?)
        }
    })
}

/// One tracker's history, for deciding when it has been gone long enough to
/// call lost.
struct Watch {
    role: TrackerRole,
    index: u8,
    seen: Option<Instant>,
    live: f32,
    sigma: f64,
    inferred: bool,
}

#[allow(clippy::too_many_arguments)]
fn run(
    channel: Arc<OutputChannel>,
    config: OutputConfig,
    indices: Vec<(u8, TrackerRole)>,
    mut sink: Box<dyn TrackerSink>,
    fusion: Arc<FusionChannel>,
    vr: Option<Arc<VrChannel>>,
    global: Shutdown,
) {
    let mut ticker = Ticker::at_hz(config.rate_hz.max(1) as f32);
    let mut rate = Rate::default();
    let mut sent = 0u64;

    let mut watches: Vec<Watch> = indices
        .iter()
        .map(|(index, role)| Watch {
            role: *role,
            index: *index,
            seen: None,
            live: 0.0,
            sigma: 0.0,
            inferred: false,
        })
        .collect();

    let max_sigma = config.max_sigma as f64;
    let ceiling = config.max_lead_ms as f64 / 1000.0;
    let offsets = config.offsets();

    loop {
        if channel.stop.is_cancelled() || global.is_cancelled() {
            break;
        }

        let now = Instant::now();
        let frame = fusion.latest();

        let (trackers, lead, problem) = match frame.as_deref() {
            Some(frame) if now.saturating_duration_since(frame.filtered.at) < STALE => {
                let posture = Posture::predicted(&frame.filtered, now, ceiling);
                let lead = (now
                    .saturating_duration_since(frame.filtered.at)
                    .as_secs_f64()
                    + frame.filtered.horizon.as_secs_f64())
                .min(ceiling);

                let trackers = indices
                    .iter()
                    .filter_map(|(_, role)| posture.derive(*role))
                    // A tracker nobody could place well is not sent at all.
                    // Driving a limb from a joint the cameras have lost track
                    // of is how a leg ends up somewhere behind the user.
                    .filter(|tracker| tracker.sigma <= max_sigma)
                    .map(|tracker| offset(tracker, &offsets))
                    .collect::<Vec<_>>();

                (trackers, lead, None)
            }
            Some(_) => (
                Vec::new(),
                0.0,
                Some("the reconstruction has stopped arriving".to_owned()),
            ),
            None => (
                Vec::new(),
                0.0,
                Some("fusion has not produced a body yet".to_owned()),
            ),
        };

        // Which trackers have been gone long enough to switch off.
        let mut lost = Vec::new();
        for watch in &mut watches {
            match trackers.iter().find(|tracker| tracker.role == watch.role) {
                Some(tracker) => {
                    watch.seen = Some(now);
                    watch.sigma = tracker.sigma;
                    watch.inferred = tracker.inferred;
                    watch.live = ema(watch.live, 1.0);
                }
                None => {
                    watch.live = ema(watch.live, 0.0);
                    if watch
                        .seen
                        .is_none_or(|seen| now.saturating_duration_since(seen) > PATIENCE)
                    {
                        lost.push(watch.role);
                    }
                }
            }
        }

        let head = vr.as_ref().and_then(|vr| vr.latest()).and_then(|snapshot| {
            snapshot
                .device(Role::Head)
                .filter(|device| device.tracking)
                .map(|device| device.pose)
        });

        let outgoing = TrackerFrame {
            at: now + Duration::from_secs_f64(lead.max(0.0)),
            lead,
            trackers,
            lost,
            head,
        };

        let mut failure = None;
        if !outgoing.is_empty() || !outgoing.lost.is_empty() {
            match sink.send(&outgoing) {
                Ok(()) => sent += 1,
                Err(error) => failure = Some(format!("{error:#}")),
            }
        }

        publish(
            &channel,
            &config,
            &watches,
            rate.tick(now),
            sent,
            lead,
            head.is_some(),
            failure.or(problem),
        );

        if !ticker.wait(&channel.stop) || global.is_cancelled() {
            break;
        }
    }

    if let Err(error) = sink.close() {
        tracing::warn!(
            "could not tell {} the trackers are going: {error:#}",
            sink.name()
        );
    }

    channel.stats.lock().running = false;
}

/// Moves a tracker along its own axes.
///
/// In the tracker's frame rather than the world's, because what an offset is
/// for is the difference between where a joint is and where a real puck would
/// have been strapped — two centimetres in front of the shin, not two
/// centimetres north.
fn offset(
    mut tracker: TrackerPose,
    offsets: &[(TrackerRole, nalgebra::Vector3<f64>)],
) -> TrackerPose {
    if let Some((_, offset)) = offsets.iter().find(|(role, _)| *role == tracker.role) {
        tracker.pose.translation.vector += tracker.pose.rotation * offset;
    }
    tracker
}

#[allow(clippy::too_many_arguments)]
fn publish(
    channel: &OutputChannel,
    config: &OutputConfig,
    watches: &[Watch],
    rate: f32,
    sent: u64,
    lead: f64,
    head: bool,
    problem: Option<String>,
) {
    let trackers: Vec<TrackerReport> = watches
        .iter()
        .map(|watch| TrackerReport {
            role: watch.role,
            index: watch.index,
            live: watch.live,
            sigma: watch.sigma,
            inferred: watch.inferred,
            lost: watch.live < 0.05,
        })
        .collect();

    let mut stats = channel.stats.lock();
    stats.rate = rate;
    stats.sent = sent;
    stats.lead_ms = (lead * 1000.0) as f32;
    stats.head = head;
    stats.warning = warning(config, &trackers, head, lead);
    stats.trackers = trackers;
    stats.problem = problem;
}

/// The one thing most worth saying about an output that is otherwise running.
fn warning(
    config: &OutputConfig,
    trackers: &[TrackerReport],
    head: bool,
    lead: f64,
) -> Option<String> {
    // Ordered by how badly it breaks things. A missing hip is not a degraded
    // experience, it is no full-body tracking at all.
    let missing: Vec<&str> = trackers
        .iter()
        .filter(|tracker| tracker.lost && tracker.role.is_essential())
        .map(|tracker| tracker.role.label())
        .collect();
    if !missing.is_empty() {
        return Some(format!(
            "{} not reaching the trackers; the cameras cannot see enough of the body",
            missing.join(" and ")
        ));
    }

    if config.sink == SinkKind::VrchatOsc && !head {
        return Some(
            "no headset pose is going out, so VRChat has nothing to place the trackers against"
                .to_owned(),
        );
    }

    // Whatever the horizon is set to, the fusion lag is the larger half of it,
    // and a lag this size means the cameras themselves are slow.
    if lead > 0.25 {
        return Some(format!(
            "predicting {:.0} ms ahead, which is further than a straight line stays true",
            lead * 1000.0
        ));
    }

    None
}

#[cfg(test)]
mod tests {
    use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

    use super::*;

    #[test]
    fn an_offset_moves_along_the_tracker_and_not_along_the_room() {
        // Facing its own -Z, turned a quarter turn so that its forward is the
        // world's -X.
        let turned =
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), std::f64::consts::FRAC_PI_2);
        let tracker = TrackerPose {
            role: TrackerRole::LeftFoot,
            pose: Isometry3::from_parts(Translation3::new(0.0, 0.1, 0.0), turned),
            sigma: 0.01,
            inferred: false,
        };

        // Ten centimetres forward, in the tracker's own frame.
        let offsets = vec![(TrackerRole::LeftFoot, Vector3::new(0.0, 0.0, -0.1))];
        let moved = offset(tracker, &offsets);

        let position = moved.pose.translation.vector;
        assert!(
            (position.x + 0.1).abs() < 1e-9,
            "the offset went into the room's frame: {position:?}"
        );
        assert!(position.z.abs() < 1e-9, "{position:?}");
    }

    #[test]
    fn a_tracker_with_no_offset_is_left_where_it_was() {
        let tracker = TrackerPose {
            role: TrackerRole::Hip,
            pose: Isometry3::from_parts(
                Translation3::new(0.2, 0.95, -1.0),
                UnitQuaternion::identity(),
            ),
            sigma: 0.01,
            inferred: false,
        };
        let moved = offset(
            tracker.clone(),
            &[(TrackerRole::Chest, Vector3::new(0.0, 0.5, 0.0))],
        );
        assert_eq!(moved.pose, tracker.pose);
    }

    fn reports(lost: &[TrackerRole]) -> Vec<TrackerReport> {
        TrackerRole::ALL
            .iter()
            .enumerate()
            .map(|(index, role)| TrackerReport {
                role: *role,
                index: index as u8 + 1,
                live: if lost.contains(role) { 0.0 } else { 1.0 },
                sigma: 0.01,
                inferred: false,
                lost: lost.contains(role),
            })
            .collect()
    }

    #[test]
    fn a_missing_foot_outranks_a_missing_head_reference() {
        let config = OutputConfig::default();
        let said = warning(&config, &reports(&[TrackerRole::LeftFoot]), false, 0.1)
            .expect("a missing foot is worth saying");
        assert!(said.contains("left foot"), "{said}");
    }

    #[test]
    fn a_working_output_says_nothing() {
        let config = OutputConfig::default();
        assert_eq!(warning(&config, &reports(&[]), true, 0.1), None);
    }

    /// An elbow nobody enabled is not a fault. Only the three that full-body
    /// tracking cannot do without are worth interrupting for.
    #[test]
    fn a_missing_elbow_is_not_a_warning() {
        let config = OutputConfig::default();
        assert_eq!(
            warning(&config, &reports(&[TrackerRole::LeftElbow]), true, 0.1),
            None
        );
    }
}
