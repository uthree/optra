//! Turning a recorded walk into solved cameras.
//!
//! Two steps, in the order the design document sets out. Each camera is first
//! resected on its own from the device positions, which ignores the offset
//! between a device and its keypoint and so lands a few centimetres out. Then
//! every camera is refined together with those offsets free, which is what
//! removes the bias.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Result, bail};
use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};

use crate::config::{CameraConfig, LensKind};
use crate::geometry::camera::{Camera, Intrinsics};
use crate::geometry::lens::Lens;
use crate::geometry::refine::{RefineOptions, Sighting, refine};
use crate::geometry::resection::{Correspondence, ResectionOptions, resect};
use crate::geometry::triangulate::{Observation, triangulate};
use crate::paths;
use crate::vr::Role;

use super::latency::{self, Estimate, LatencyOptions};
use super::recorder::{CameraTrail, Recording, Rig};

#[derive(Debug, Clone)]
pub struct SolveOptions {
    pub resection: ResectionOptions,
    pub refine: RefineOptions,
    /// Sightings a camera needs before it is worth solving at all.
    pub min_samples: usize,
    pub latency: LatencyOptions,
    /// Measure each camera latency and fit again against the corrected
    /// timestamps.
    pub estimate_latency: bool,
}

impl Default for SolveOptions {
    fn default() -> Self {
        Self {
            resection: ResectionOptions {
                // Loose on purpose. At resection time the offset between each
                // device and its keypoint is still unknown, so every
                // correspondence is wrong by the same handful of centimetres —
                // which at three metres is a couple of degrees. A threshold
                // tight enough to reject that would reject the whole walk.
                inlier_threshold: 0.08,
                ..ResectionOptions::default()
            },
            refine: RefineOptions::default(),
            min_samples: 40,
            latency: LatencyOptions::default(),
            estimate_latency: true,
        }
    }
}

/// What a camera came out as.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraCalibration {
    pub id: String,
    pub camera: Camera,
    /// RMS angular reprojection error after refinement, in radians.
    pub rms: f64,
    /// Sightings that survived to the final fit.
    pub sightings: usize,
    /// Fraction of the frame the walk covered.
    pub coverage: f32,
    /// How far the correspondences were from lying in a plane when this camera
    /// was resected. Near zero and the answer rests on nothing.
    pub spread: f64,
    /// How far behind this camera is, when the walk was brisk enough to tell.
    pub latency: Option<Estimate>,
    /// Typical distance from this camera to the person during the walk, in
    /// metres.
    #[serde(default)]
    pub range: f64,
    /// Fraction of the frames this camera saw a person in that also had a foot
    /// in them.
    ///
    /// Nothing to do with the calibration, which never looks at a foot, and
    /// everything to do with whether this camera can contribute to tracking
    /// one. A camera can come out of the solve perfect and be aimed somewhere
    /// that will never see a leg.
    #[serde(default)]
    pub feet: f32,
}

impl CameraCalibration {
    /// The reprojection error in degrees, which is the number to put in front
    /// of a user.
    pub fn rms_degrees(&self) -> f64 {
        self.rms.to_degrees()
    }

    /// What the reprojection error is worth in metres, at the distance the
    /// person actually walked.
    ///
    /// The angle is the right quantity to *solve* on, because it compares
    /// across cameras of different resolutions and fields of view. It is the
    /// wrong quantity to *judge* on, because the same angle is a different
    /// error at a different distance: half a degree is four millimetres for a
    /// ceiling camera watching from four metres and one millimetre for a camera
    /// on a desk a metre away. A user asking whether the calibration is good
    /// enough is asking how far out their feet will be, and that is this.
    pub fn error_metres(&self) -> Option<f64> {
        (self.range > 0.0).then_some(self.rms * self.range)
    }
}

/// A solved room.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomCalibration {
    pub cameras: Vec<CameraCalibration>,
    /// The device-to-keypoint offsets that were solved for, one per rig.
    pub rigs: Vec<(Rig, Vector3<f64>)>,
    /// RMS angular reprojection error over every camera, in radians.
    pub rms: f64,
    /// Sightings discarded as outliers before the final fit.
    pub rejected: usize,
    /// Sightings the final fit used.
    pub used: usize,
    /// How well these camera positions can locate a joint at all, in metres.
    ///
    /// A property of where the cameras are rather than of how well they were
    /// solved, and the number that answers "is this a good placement".
    #[serde(default)]
    pub precision: Option<f64>,
    pub solved_at: String,
}

impl RoomCalibration {
    pub fn rms_degrees(&self) -> f64 {
        self.rms.to_degrees()
    }

    pub fn camera(&self, id: &str) -> Option<&CameraCalibration> {
        self.cameras.iter().find(|camera| camera.id == id)
    }

    /// Writes the profile under the rooms directory.
    pub fn save(&self, name: &str) -> Result<()> {
        let path = paths::rooms_dir()?.join(format!("{name}.toml"));
        let text = toml::to_string_pretty(self)?;

        // Written beside the target and renamed, so an interrupted write
        // cannot leave a half-finished profile where a good one was.
        let temporary = path.with_extension("toml.tmp");
        std::fs::write(&temporary, text)?;
        std::fs::rename(&temporary, &path)?;

        tracing::info!(path = %path.display(), "saved a room profile");
        Ok(())
    }

    pub fn load(name: &str) -> Result<Self> {
        let path = paths::rooms_dir()?.join(format!("{name}.toml"));
        Ok(toml::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// Profiles saved on this machine, for the UI to offer.
    pub fn list() -> Vec<String> {
        let Ok(directory) = paths::rooms_dir() else {
            return Vec::new();
        };
        let Ok(entries) = std::fs::read_dir(directory) else {
            return Vec::new();
        };

        let mut names: Vec<String> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "toml")
            })
            .filter_map(|path| Some(path.file_stem()?.to_string_lossy().into_owned()))
            .collect();
        names.sort();
        names
    }
}

/// How many times the latency is measured and the room solved again against it.
///
/// Three is one more than it takes to stop moving on any delay a webcam can
/// have, which makes it a bound rather than a schedule: the loop leaves as soon
/// as a round changes nothing.
const LATENCY_ROUNDS: usize = 3;

/// Solves every camera in the recording.
pub fn solve(
    recording: &Recording,
    cameras: &[CameraConfig],
    options: &SolveOptions,
) -> Result<RoomCalibration> {
    if recording.rigs.is_empty() {
        bail!("the recording has no tracked devices in it");
    }

    let configs: HashMap<&str, &CameraConfig> =
        cameras.iter().map(|c| (c.id.as_str(), c)).collect();

    let mut ids = Vec::new();
    let mut coverages = Vec::new();
    let mut feet = Vec::new();

    for trail in &recording.cameras {
        if trail.samples.len() < options.min_samples {
            tracing::warn!(
                camera = %trail.camera,
                samples = trail.samples.len(),
                "not enough of the walk was visible; skipping this camera"
            );
            continue;
        }

        if !configs.contains_key(trail.camera.as_str()) {
            bail!(
                "the recording names a camera that is not configured: {}",
                trail.camera
            );
        }

        ids.push(trail.camera.clone());
        coverages.push(trail.coverage.filled());
        feet.push(if trail.frames > 0 {
            trail.feet_seen as f32 / trail.frames as f32
        } else {
            0.0
        });
    }

    if ids.len() < 2 {
        bail!(
            "only {} camera(s) saw enough of the walk; tracking needs at least two",
            ids.len()
        );
    }

    let index: HashMap<&str, usize> = ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.as_str(), index))
        .collect();

    let offsets = vec![Vector3::zeros(); recording.rigs.len()];
    let mut lags = vec![Duration::ZERO; ids.len()];

    // A camera too late to resect against prompt timestamps is solved at
    // whatever delay does work, and that delay is kept: pairing the sightings
    // as though it were prompt would undo the only thing that got it solved.
    // It is a twenty-millisecond grid rather than an answer, and the estimator
    // below refines it.
    let (mut seeds, mut spreads, seeded_at) = seed(recording, &configs, &ids, &lags, options)?;
    lags = seeded_at;
    let mut sightings = pair(recording, &index, &lags);
    let mut refined = refine(&seeds, &offsets, &sightings, &options.refine);

    // Each camera hands over its frames a little late, and by a different
    // amount. Measuring that needs cameras to reproject through, which is why
    // it happens here rather than before the first fit — and once it is known,
    // fitting again against the corrected timestamps is what turns it from a
    // number into accuracy.
    //
    // It has to go round more than once. The first fit had no delays to work
    // with, so it did the only thing it could and moved each camera to wherever
    // best explained a walk it believed happened forty milliseconds after it
    // did. Part of every delay is therefore already hidden in the extrinsics
    // by the time the search runs, and the search finds only the part that is
    // left: a camera really ninety milliseconds late measured as fifty-two on
    // the first pass. Refitting puts the pose back where it belongs, which
    // exposes the rest, and a second search finds it.
    let mut estimates = vec![None; ids.len()];
    if options.estimate_latency {
        for round in 0..LATENCY_ROUNDS {
            let before = lags.clone();

            for (slot, id) in ids.iter().enumerate() {
                let Some(trail) = recording.trail(id) else {
                    continue;
                };

                let estimate = latency::estimate(
                    &refined.cameras[slot],
                    trail,
                    recording,
                    &refined.offsets,
                    &options.latency,
                );

                match estimate {
                    // A walk too slow to leave a mark produces a
                    // confident-looking minimum in noise. Applying that is
                    // worse than applying nothing, so it is reported and not
                    // used.
                    Some(estimate) if estimate.is_confident() && estimate.is_plausible() => {
                        tracing::info!(
                            camera = %id,
                            round,
                            latency_ms = estimate.millis(),
                            "measured the camera latency"
                        );
                        lags[slot] = estimate.latency;
                    }
                    Some(estimate) if !estimate.is_plausible() => tracing::warn!(
                        camera = %id,
                        latency_ms = estimate.millis(),
                        "that is too long for a webcam to be behind; not applying it"
                    ),
                    Some(estimate) => tracing::warn!(
                        camera = %id,
                        latency_ms = estimate.millis(),
                        sharpness = estimate.sharpness,
                        "the walk was too slow to measure this camera's latency"
                    ),
                    None => {}
                }
                estimates[slot] = estimate;
            }

            // Nothing moved, so another round would measure the same thing
            // against the same cameras. This covers the case where no delay was
            // found at all, since `before` starts out zero.
            if lags == before {
                break;
            }

            // Falling out of the loop still moving means the delays and the
            // extrinsics were still trading places when the rounds ran out, and
            // the room being returned is one iterate short of wherever they
            // were heading. Worth saying: the symptom downstream is a camera a
            // centimetre or two off with nothing obviously wrong with it.
            if round + 1 == LATENCY_ROUNDS {
                tracing::warn!(
                    ?lags,
                    "the camera delays had not settled after {LATENCY_ROUNDS} rounds"
                );
            }

            // Re-seeded as well as re-paired. Correcting only the sightings
            // leaves the refinement starting from a resection that was itself
            // done against a walk the camera had not caught up with, and a seed
            // that far out is not a seed the refinement pulls back: its outlier
            // rejection throws away the sightings that disagree with it
            // instead. A camera forty milliseconds late kept 51 of its 190
            // sightings and came out 39 cm from where it was, with every other
            // camera in the room fine.
            (seeds, spreads, _) = seed(recording, &configs, &ids, &lags, options)?;
            sightings = pair(recording, &index, &lags);
            refined = refine(&seeds, &offsets, &sightings, &options.refine);
        }
    }

    let cameras = ids
        .into_iter()
        .enumerate()
        .map(|(slot, id)| CameraCalibration {
            id,
            camera: refined.cameras[slot].clone(),
            rms: refined.per_camera[slot].rms,
            sightings: refined.per_camera[slot].sightings,
            coverage: coverages[slot],
            spread: spreads[slot],
            latency: estimates[slot],
            range: median_range(&refined.cameras[slot], slot, &sightings, &refined.offsets),
            feet: feet[slot],
        })
        .collect();

    // A rig nothing was ever seen through has an offset of exactly the value it
    // started at, which is zero, and reporting that alongside the solved ones
    // presents a seed as an answer.
    let rigs = recording
        .rigs
        .iter()
        .copied()
        .zip(refined.offsets.iter().copied())
        .enumerate()
        .filter(|(index, _)| {
            sightings.iter().filter(|s| s.rig == *index).count() >= MIN_RIG_SIGHTINGS
        })
        .map(|(_, pair)| pair)
        .collect();

    Ok(RoomCalibration {
        precision: precision(&refined.cameras, &sightings, &refined.offsets),
        cameras,
        rigs,
        rms: refined.rms,
        rejected: refined.rejected,
        used: sightings.len() - refined.rejected,
        solved_at: chrono::Local::now().to_rfc3339(),
    })
}

/// Pairs each recorded pixel with where its device was when the frame was
/// actually exposed, which is `lag` before the timestamp it carries.
/// Resects every camera on its own, pairing each pixel with where the device
/// was when the shutter opened rather than when the frame was stamped.
///
/// Every rig contributes, not just the head. The head alone traces a narrow
/// band of heights, and a set of points that close to a plane is degenerate for
/// the linear solve; a hand raised and lowered is what gives the resection
/// something to work with.
/// The resection is also a delay detector, and this is how wide it looks when
/// the delay it was handed does not solve.
///
/// Past the hundred and twenty milliseconds [`latency::Estimate::is_plausible`]
/// will accept, so that a camera slower than that produces a solved room with a
/// number the user can be shown and act on, rather than an error about
/// correspondences that names neither the camera's problem nor its cause.
const SEED_LAG_MAX: Duration = Duration::from_millis(160);

/// Resolution of that search. The resection tolerates being about twenty
/// milliseconds out and nothing like forty, so this is the coarsest step that
/// cannot fall through the gap; the latency estimator refines from there.
const SEED_LAG_STEP: Duration = Duration::from_millis(20);

/// What one camera's resection came out as, and the delay it was solved at.
struct Seed {
    camera: Camera,
    spread: f64,
    lag: Duration,
    inliers: usize,
}

fn seed(
    recording: &Recording,
    configs: &HashMap<&str, &CameraConfig>,
    ids: &[String],
    lags: &[Duration],
    options: &SolveOptions,
) -> Result<(Vec<Camera>, Vec<f64>, Vec<Duration>)> {
    let mut cameras = Vec::with_capacity(ids.len());
    let mut spreads = Vec::with_capacity(ids.len());
    let mut solved_at = Vec::with_capacity(ids.len());

    for (slot, id) in ids.iter().enumerate() {
        let (Some(trail), Some(config)) = (recording.trail(id), configs.get(id.as_str())) else {
            bail!("camera {id} left the recording between one pass and the next");
        };
        let lag = lags.get(slot).copied().unwrap_or(Duration::ZERO);

        // The delay this camera was handed, and then — only if that does not
        // solve — every delay it might have instead.
        //
        // A camera eighty milliseconds late does not resect at all against
        // pixels paired with poses from eighty milliseconds after the shutter:
        // the correspondences do not agree with any one camera, and the search
        // ends with no consensus rather than with a bad answer. Before this,
        // that ended the whole calibration on a message about correspondences,
        // and it ended it *before* the latency estimator ran — which needs a
        // camera to reproject through and so could never have rescued it. Four
        // cameras with sixty milliseconds between them and nothing solved.
        //
        // The resection is a sharp detector of its own delay, which is what
        // makes the sweep cheap: it fails outright at zero, twenty, forty and
        // sixty and comes back clean at eighty. Taking the most inliers rather
        // than the first success is what keeps it from stopping at the edge of
        // that window.
        let solved = resect_at(recording, trail, config, lag, options).or_else(|| {
            let mut steps = 0;
            let mut best: Option<Seed> = None;
            while SEED_LAG_STEP * steps <= SEED_LAG_MAX {
                let candidate = resect_at(recording, trail, config, SEED_LAG_STEP * steps, options);
                if let Some(candidate) = candidate
                    && best
                        .as_ref()
                        .is_none_or(|best| candidate.inliers > best.inliers)
                {
                    best = Some(candidate);
                }
                steps += 1;
            }
            if let Some(found) = &best {
                tracing::warn!(
                    camera = %id,
                    lag_ms = found.lag.as_secs_f64() * 1000.0,
                    "this camera only resects if its frames are treated as late"
                );
            }
            best
        });

        let Some(solved) = solved else {
            bail!(
                "camera {id} could not be solved from {} sightings at any delay \
                 up to {} ms; either it moved during the walk or it is seeing \
                 something other than the headset",
                trail.samples.len(),
                SEED_LAG_MAX.as_millis()
            );
        };

        cameras.push(solved.camera);
        spreads.push(solved.spread);
        solved_at.push(solved.lag);
    }

    Ok((cameras, spreads, solved_at))
}

/// Resects one camera with its pixels paired against poses from `lag` before
/// the timestamps they carry.
fn resect_at(
    recording: &Recording,
    trail: &CameraTrail,
    config: &CameraConfig,
    lag: Duration,
    options: &SolveOptions,
) -> Option<Seed> {
    let correspondences: Vec<Correspondence> = trail
        .samples
        .iter()
        .filter_map(|sample| {
            let pose = recording.tracks[sample.rig].at(sample.at.checked_sub(lag)?)?;
            Some(Correspondence {
                world: Point3::from(pose.translation.vector),
                pixel: sample.pixel,
            })
        })
        .collect();

    let lens = Lens::for_kind(config.lens);
    let guess = Intrinsics::from_fov(
        trail.width,
        trail.height,
        seed_fov(config.lens).to_radians(),
    );

    let resection = resect(&guess, lens, &correspondences, &options.resection)?;
    Some(Seed {
        camera: resection.camera,
        spread: resection.spread,
        lag,
        inliers: resection.inliers.len(),
    })
}

fn pair(recording: &Recording, index: &HashMap<&str, usize>, lags: &[Duration]) -> Vec<Sighting> {
    let mut out = Vec::new();

    for trail in &recording.cameras {
        let Some(camera) = index.get(trail.camera.as_str()).copied() else {
            continue;
        };
        let lag = lags.get(camera).copied().unwrap_or(Duration::ZERO);

        for sample in &trail.samples {
            let Some(when) = sample.at.checked_sub(lag) else {
                continue;
            };
            let Some(anchor) = recording.tracks[sample.rig].at(when) else {
                continue;
            };
            out.push(Sighting {
                camera,
                rig: sample.rig,
                anchor,
                pixel: sample.pixel,
                weight: weight_of(recording.rigs[sample.rig], sample.confidence),
            });
        }
    }

    out
}

/// Sightings a rig needs before its offset is an answer rather than the value
/// it was seeded with.
const MIN_RIG_SIGHTINGS: usize = 30;

/// How well these cameras, in these positions, can locate a point at all.
///
/// This is a property of the *geometry*, and it is a different question from
/// the reprojection error. A calibration can be flawless and still describe a
/// set of cameras clustered in one corner, all looking the same way: they will
/// agree with each other beautifully about a point none of them can place along
/// their shared line of sight. The residual cannot see that, and it is the
/// first thing a user moving cameras around needs to know.
///
/// Answered by asking the machinery that will actually do the work: put a
/// keypoint of ordinary quality at each place the person stood, triangulate it
/// through the solved cameras, and report how well it comes out.
fn precision(cameras: &[Camera], sightings: &[Sighting], offsets: &[Vector3<f64>]) -> Option<f64> {
    /// A keypoint neither especially good nor especially bad.
    const NOMINAL: f64 = 0.8;
    /// One in this many sightings is sampled. The answer varies smoothly with
    /// position and thousands of them would say the same thing.
    const STRIDE: usize = 37;

    let mut spreads = Vec::new();

    for sighting in sightings.iter().step_by(STRIDE) {
        let Some(offset) = offsets.get(sighting.rig) else {
            continue;
        };
        let world = sighting.anchor * Point3::from(*offset);

        let observations: Vec<Observation> = cameras
            .iter()
            .enumerate()
            .filter_map(|(index, camera)| {
                let pixel = camera.project(world)?;
                let inside = pixel.x >= 0.0
                    && pixel.y >= 0.0
                    && pixel.x < camera.intrinsics.width as f64
                    && pixel.y < camera.intrinsics.height as f64;
                inside.then(|| Observation::new(index, camera, pixel, NOMINAL, 1.0))
            })
            .collect();

        if observations.len() < 2 {
            continue;
        }
        // Generous, because these are synthetic sightings with no noise in
        // them: nothing here should ever be rejected as an outlier.
        if let Some(solved) = triangulate(cameras, &observations, 1.0) {
            spreads.push(solved.sigma());
        }
    }

    if spreads.is_empty() {
        return None;
    }
    spreads.sort_by(f64::total_cmp);
    Some(spreads[spreads.len() / 2])
}

/// How far this camera typically was from the person, in metres.
///
/// A median, so that one sighting at the far wall does not decide it.
fn median_range(
    camera: &Camera,
    slot: usize,
    sightings: &[Sighting],
    offsets: &[Vector3<f64>],
) -> f64 {
    let origin = camera.position();
    let mut ranges: Vec<f64> = sightings
        .iter()
        .filter(|sighting| sighting.camera == slot)
        .filter_map(|sighting| {
            let offset = offsets.get(sighting.rig)?;
            Some((sighting.anchor * Point3::from(*offset) - origin).norm())
        })
        .collect();

    if ranges.is_empty() {
        return 0.0;
    }
    ranges.sort_by(f64::total_cmp);
    ranges[ranges.len() / 2]
}

/// A starting field of view for a lens kind, in degrees.
///
/// Only the order of magnitude matters: the solve recovers the real focal
/// length, and needs no better than a factor of two to get there.
fn seed_fov(kind: LensKind) -> f64 {
    match kind {
        LensKind::Standard => 70.0,
        LensKind::Wide => 100.0,
        LensKind::Fisheye => 150.0,
    }
}

/// How much a controller sighting counts against a headset one.
///
/// A controller is held rather than worn: the wrist keypoint sits against it
/// less rigidly than the head sits against a headset, and a change of grip
/// moves the offset by a centimetre or two. They earn their place by reaching
/// heights the head never does, but they should not outvote it.
///
/// It is a constant here rather than a setting on the recorder, where it used
/// to be declared and never read. A number a user can change and that nothing
/// consults is worse than one they cannot: the second is a decision and the
/// first is a lie.
const CONTROLLER_RIGIDITY: f64 = 0.5;

/// How much a sighting is trusted, before the solver's own outlier handling.
fn weight_of(rig: Rig, confidence: f64) -> f64 {
    let rigidity = if rig.role == Role::Head {
        1.0
    } else {
        CONTROLLER_RIGIDITY
    };
    confidence.clamp(0.0, 1.0) * rigidity
}

#[cfg(test)]
mod tests {
    use nalgebra::{Isometry3, Point2, Translation3, UnitQuaternion};

    use super::*;

    fn camera_at(x: f64, y: f64, z: f64) -> Camera {
        Camera::look_at(
            Intrinsics::from_fov(1280, 720, 70f64.to_radians()),
            Lens::default(),
            Point3::new(x, y, z),
            Point3::new(0.0, 1.0, 0.0),
            Vector3::y(),
        )
    }

    /// A short walk in the middle of the room, as sightings the precision
    /// estimate can be asked about.
    fn walk() -> Vec<Sighting> {
        (0..200)
            .map(|step| {
                let t = step as f64 * 0.07;
                Sighting {
                    camera: 0,
                    rig: 0,
                    anchor: Isometry3::from_parts(
                        Translation3::new(0.5 * t.sin(), 1.4, 0.5 * (0.7 * t).cos()),
                        UnitQuaternion::identity(),
                    ),
                    pixel: Point2::new(640.0, 360.0),
                    weight: 1.0,
                }
            })
            .collect()
    }

    /// The check a user moving cameras around needs: cameras bunched into one
    /// corner see a joint from nearly the same direction, and agree perfectly
    /// about a point none of them can place.
    #[test]
    fn clustered_cameras_are_reported_as_a_poor_placement() {
        let offsets = [Vector3::zeros()];
        let walk = walk();

        let spread = [
            camera_at(-2.0, 2.2, -2.0),
            camera_at(2.0, 2.2, -2.0),
            camera_at(0.0, 2.2, 2.2),
        ];
        let clustered = [
            camera_at(0.7, 0.05, -1.6),
            camera_at(0.5, 0.05, -1.7),
            camera_at(0.2, 0.6, -1.9),
        ];

        let good = precision(&spread, &walk, &offsets).expect("the walk is visible");
        let bad = precision(&clustered, &walk, &offsets).expect("the walk is visible");

        assert!(
            good < 0.02,
            "cameras around the room should be good to a centimetre or two, got {good:.3} m"
        );
        assert!(
            bad > 3.0 * good,
            "a cluster should be visibly worse: {bad:.3} m against {good:.3} m"
        );
    }

    /// Nothing can be said about a placement no sighting is visible from.
    #[test]
    fn a_placement_nothing_can_see_has_no_precision() {
        let facing_away = [
            Camera::look_at(
                Intrinsics::from_fov(1280, 720, 70f64.to_radians()),
                Lens::default(),
                Point3::new(0.0, 1.4, 0.0),
                Point3::new(0.0, 1.4, -5.0),
                Vector3::y(),
            ),
            camera_at(0.0, 1.4, 0.1),
        ];

        assert_eq!(precision(&facing_away, &walk(), &[Vector3::zeros()]), None);
    }
}
