//! Joint refinement of every camera at once.
//!
//! Resection solves each camera on its own from a linear system that ignores
//! distortion. This is the step that fixes both: one Levenberg-Marquardt solve
//! over every camera's intrinsics, distortion and pose together, with the
//! constant offset between a tracked device and the keypoint that stands in for
//! it shared across all of them.
//!
//! The residual is angular rather than measured in pixels. A 1080p camera has
//! smaller pixels than a 480p one, and a pixel-based objective would let it
//! dominate the solve purely for that reason, dragging the wide camera's
//! calibration off to buy a fraction of a pixel on the narrow one.

use nalgebra::{DMatrix, DVector, Matrix3, Point3, UnitQuaternion, Vector3};

use super::camera::Camera;

/// One sighting from the calibration walk.
///
/// The world position is not stored directly: it is wherever the tracked device
/// was, plus a constant offset in the device's own frame, and that offset is
/// one of the things being solved for. A headset does not sit on the head
/// keypoint the pose model reports, and the gap between them is a fixed few
/// centimetres that would otherwise bias every camera.
#[derive(Debug, Clone, Copy)]
pub struct Sighting {
    /// Index into the camera list.
    pub camera: usize,
    /// Which rigid body this is: the headset, the left controller, the right
    /// controller. Each carries its own offset.
    pub rig: usize,
    /// Where the device was in the world when the frame was taken.
    pub anchor: nalgebra::Isometry3<f64>,
    /// Where the matching keypoint appeared in the image.
    pub pixel: nalgebra::Point2<f64>,
    /// Relative trust in this sighting, from the keypoint confidence.
    pub weight: f64,
}

#[derive(Debug, Clone)]
pub struct RefineOptions {
    pub iterations: usize,
    pub free_intrinsics: bool,
    pub free_distortion: bool,
    pub free_extrinsics: bool,
    pub free_offsets: bool,
    /// Angular residual past which a sighting is progressively discounted, in
    /// radians. A missed keypoint is a plausible-looking observation in the
    /// wrong place, and without this a handful of them bend the whole solve.
    pub huber: f64,
    /// Multiple of the median residual past which a sighting is discarded
    /// before the final fit.
    pub reject: f64,
    /// Stop once an iteration improves the cost by less than this fraction.
    pub tolerance: f64,
    /// Furthest a keypoint may sit from the device it was matched to, in
    /// metres.
    ///
    /// Not a guess at the real distance -- that is what the solve is for -- but
    /// a bound on what a body permits, so that a direction the walk left
    /// unobservable cannot slide the whole room away.
    pub max_offset: f64,
}

impl Default for RefineOptions {
    fn default() -> Self {
        Self {
            iterations: 60,
            free_intrinsics: true,
            free_distortion: true,
            free_extrinsics: true,
            free_offsets: true,
            huber: 0.02,
            reject: 5.0,
            tolerance: 1e-6,
            // Generous. A head keypoint sits ten or fifteen centimetres from a
            // headset and a wrist about the same from a controller, so this
            // never binds on a real body — it only stops the solver walking off
            // into an answer no body could take.
            max_offset: 0.25,
        }
    }
}

/// How well one camera came out.
#[derive(Debug, Clone)]
pub struct CameraResidual {
    pub sightings: usize,
    /// RMS angular reprojection error, in radians.
    pub rms: f64,
    /// Largest single angular residual, in radians.
    pub worst: f64,
}

#[derive(Debug, Clone)]
pub struct Refinement {
    pub cameras: Vec<Camera>,
    /// Device-to-keypoint offset per rig, in the device's own frame.
    pub offsets: Vec<Vector3<f64>>,
    /// RMS angular reprojection error over every sighting, in radians.
    pub rms: f64,
    pub per_camera: Vec<CameraResidual>,
    /// Sightings discarded as outliers before the final fit.
    pub rejected: usize,
    pub iterations: usize,
    /// Whether the solve stopped because it had converged rather than because
    /// it ran out of iterations or stalled.
    pub converged: bool,
}

/// Units the solver works in, one per kind of parameter.
///
/// The parameter vector is dimensionless: a step of one means one of these.
/// Without it a single finite-difference epsilon would have to serve both a
/// focal length near a thousand pixels and a distortion coefficient near a
/// hundredth, and it cannot.
const FOCAL_STEP: f64 = 1.0;
const PRINCIPAL_STEP: f64 = 1.0;
const DISTORTION_STEP: f64 = 1e-3;
const ROTATION_STEP: f64 = 1e-3;
const TRANSLATION_STEP: f64 = 1e-3;
const OFFSET_STEP: f64 = 1e-3;

/// Residual charged for a point that falls behind the camera, in radians. Large
/// enough to push the solver away, finite enough not to wreck the linear
/// system.
const BEHIND_PENALTY: f64 = 1.0;

/// Finite-difference step, in parameter units.
const EPSILON: f64 = 1.0;

/// Refines every camera and every rig offset against the recorded sightings.
///
/// The solve runs twice. The first pass fits everything, with the Huber weight
/// limiting how far a bad keypoint can pull; the second drops the sightings that
/// pass left behind and fits the rest cleanly. Huber bounds an outlier's
/// influence but does not remove it, and a walk of several hundred frames
/// carries enough missed keypoints for what remains to matter.
pub fn refine(
    cameras: &[Camera],
    offsets: &[Vector3<f64>],
    sightings: &[Sighting],
    options: &RefineOptions,
) -> Refinement {
    // The residual divides by focal length to become an angle. That divisor is
    // frozen at the starting value: were it the live parameter, the solver
    // could shrink the objective by growing the focal length instead of by
    // fitting anything.
    let reference: Vec<f64> = cameras.iter().map(|c| c.intrinsics.fx.max(1.0)).collect();
    let layout = Layout::new(options, cameras.len(), offsets.len());

    let start = State {
        cameras: cameras.to_vec(),
        offsets: offsets.to_vec(),
    };

    let (mut state, mut iterations, mut converged) =
        descend(start.clone(), sightings, &reference, &layout, options);

    let mut kept: Vec<Sighting> = sightings.to_vec();
    let survivors = accepted(&state, sightings, &reference, options);
    let rejected = sightings.len() - survivors.len();

    // Restart from the original guess rather than from the first pass: its
    // answer was shaped by the very sightings just discarded.
    if rejected > 0 && survivors.len() * 2 >= layout.total() {
        let (second, more, stopped) = descend(start, &survivors, &reference, &layout, options);
        state = second;
        iterations += more;
        converged = stopped;
        kept = survivors;
    }

    let residual = raw_residuals(&state, &kept, &reference);
    let (rms, per_camera) = report(&residual, &kept, state.cameras.len());

    Refinement {
        cameras: state.cameras,
        offsets: state.offsets,
        rms,
        per_camera,
        rejected,
        iterations,
        converged,
    }
}

/// One Levenberg-Marquardt descent.
fn descend(
    mut state: State,
    sightings: &[Sighting],
    reference: &[f64],
    layout: &Layout,
    options: &RefineOptions,
) -> (State, usize, bool) {
    if layout.total() == 0 || sightings.len() * 2 < layout.total() {
        return (state, 0, false);
    }

    let mut lambda = 1e-3;
    let mut residual = raw_residuals(&state, sightings, reference);
    let mut scale = huber_scale(&residual, sightings, options.huber);
    let mut cost = weighted_cost(&residual, &scale);

    let mut delta = DVector::zeros(layout.total());
    let mut iterations = 0;
    let mut converged = false;

    for _ in 0..options.iterations {
        iterations += 1;

        let jacobian = jacobian(&state, sightings, reference, &scale, layout, &mut delta);
        let weighted = component_product(&residual, &scale);
        let jtj = jacobian.transpose() * &jacobian;
        let jtr = jacobian.transpose() * &weighted;

        let mut stepped = false;
        for _ in 0..8 {
            let mut system = jtj.clone();
            for index in 0..system.nrows() {
                // Marquardt's scaling: damp each parameter in proportion to its
                // own curvature, plus a floor so a parameter no sighting
                // constrains cannot make the system singular.
                system[(index, index)] += lambda * jtj[(index, index)] + 1e-9;
            }

            let Some(cholesky) = system.cholesky() else {
                lambda *= 4.0;
                continue;
            };
            let step = cholesky.solve(&(-&jtr));

            let mut candidate = apply(&state, &step, layout);
            clamp_offsets(&mut candidate, options.max_offset);
            let candidate_residual = raw_residuals(&candidate, sightings, reference);
            let candidate_cost = weighted_cost(&candidate_residual, &scale);

            if candidate_cost < cost {
                let improvement = (cost - candidate_cost) / cost.max(1e-30);
                state = candidate;
                residual = candidate_residual;
                scale = huber_scale(&residual, sightings, options.huber);
                cost = weighted_cost(&residual, &scale);
                lambda = (lambda / 3.0).max(1e-12);
                stepped = true;
                converged = improvement < options.tolerance;
                break;
            }

            lambda *= 4.0;
        }

        if !stepped || converged {
            break;
        }
    }

    (state, iterations, converged)
}

/// The sightings a fitted state explains well enough to keep.
///
/// The cut is relative to how well the walk fits overall, so a camera in a dim
/// corner with noisier keypoints is not thrown away wholesale, with an absolute
/// floor so that a near-perfect fit does not reject its own rounding error.
fn accepted(
    state: &State,
    sightings: &[Sighting],
    reference: &[f64],
    options: &RefineOptions,
) -> Vec<Sighting> {
    let residual = raw_residuals(state, sightings, reference);
    let mut magnitudes: Vec<f64> = (0..sightings.len())
        .map(|index| magnitude(&residual, index))
        .collect();
    if magnitudes.is_empty() {
        return Vec::new();
    }

    magnitudes.sort_by(f64::total_cmp);
    let median = magnitudes[magnitudes.len() / 2];
    let threshold = (options.reject * median).max(options.huber);

    sightings
        .iter()
        .enumerate()
        .filter(|(index, _)| magnitude(&residual, *index) <= threshold)
        .map(|(_, sighting)| *sighting)
        .collect()
}

fn magnitude(residual: &DVector<f64>, sighting: usize) -> f64 {
    (residual[sighting * 2].powi(2) + residual[sighting * 2 + 1].powi(2)).sqrt()
}

/// How well the recorded motion pins down a rig's offset, from zero to one.
///
/// The offset only becomes visible because the device *rotates*: a constant
/// offset in the device's frame traces a different world path than the device
/// origin, and that difference is the whole signal. A user who walks the room
/// without ever turning their head leaves the offset indistinguishable from a
/// shift of every camera, and the number here goes to zero.
pub fn offset_observability(sightings: &[Sighting], rig: usize) -> f64 {
    let mut sum = Matrix3::zeros();
    let mut count = 0usize;

    for sighting in sightings.iter().filter(|s| s.rig == rig) {
        sum += sighting.anchor.rotation.to_rotation_matrix().into_inner();
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }

    rotation_observability(sum / count as f64)
}

/// How well a set of rotations, given by their mean matrix, constrains a
/// constant offset in the rotating frame.
///
/// Moving every camera by some `d` leaves the reprojections untouched if the
/// rig offset can absorb it, which needs `Rᵢᵀ d` to be the same for every
/// sample — that is, `d` has to be a common fixed axis of the rotations. So the
/// question is not whether the device turned but whether it turned about more
/// than one axis, and the mean rotation matrix answers it: averaging unit
/// vectors `Rᵢᵀ d` gives back something of unit length only when they were all
/// equal, so a **largest** singular value of one is exactly the degenerate
/// case.
///
/// The largest, not the smallest. This is the correction to what this function
/// used to do, and the difference matters more than it sounds: a user walking
/// a room turns their head from side to side and hardly at all up and down, so
/// the yaw axis averages away nicely and the vertical does not. The smallest
/// singular value sees the yaw and reports an excellent walk; the largest sees
/// that the vertical is untouched and reports the truth, which is that the
/// whole room is free to slide up and down. The synthetic walks in the tests
/// never caught it because the simulated user obligingly nods.
///
/// The value returned is the fraction of a unit shift that the offset *cannot*
/// absorb, `sqrt(1 - σ_max²)`, which is roughly the radians of rotation the
/// walk varied by in its worst direction.
pub fn rotation_observability(mean: Matrix3<f64>) -> f64 {
    let largest = mean
        .svd(false, false)
        .singular_values
        .iter()
        .copied()
        .fold(0.0, f64::max);

    (1.0 - largest * largest).max(0.0).sqrt().clamp(0.0, 1.0)
}

#[derive(Debug, Clone)]
struct State {
    cameras: Vec<Camera>,
    offsets: Vec<Vector3<f64>>,
}

/// Which parameters are free, and where each one sits in the vector.
struct Layout {
    intrinsics: bool,
    distortion: bool,
    extrinsics: bool,
    offsets: bool,
    per_camera: usize,
    cameras: usize,
    rigs: usize,
}

impl Layout {
    fn new(options: &RefineOptions, cameras: usize, rigs: usize) -> Self {
        let per_camera = usize::from(options.free_intrinsics) * 4
            + usize::from(options.free_distortion) * 4
            + usize::from(options.free_extrinsics) * 6;

        Self {
            intrinsics: options.free_intrinsics,
            distortion: options.free_distortion,
            extrinsics: options.free_extrinsics,
            offsets: options.free_offsets,
            per_camera,
            cameras,
            rigs,
        }
    }

    fn total(&self) -> usize {
        self.cameras * self.per_camera + if self.offsets { 3 * self.rigs } else { 0 }
    }

    fn camera_base(&self, camera: usize) -> usize {
        camera * self.per_camera
    }

    fn offset_base(&self, rig: usize) -> usize {
        self.cameras * self.per_camera + 3 * rig
    }
}

/// Builds the state a parameter step describes, leaving the base untouched.
fn apply(base: &State, delta: &DVector<f64>, layout: &Layout) -> State {
    let mut state = base.clone();

    for (index, camera) in state.cameras.iter_mut().enumerate() {
        let mut at = layout.camera_base(index);

        if layout.intrinsics {
            camera.intrinsics.fx += delta[at] * FOCAL_STEP;
            camera.intrinsics.fy += delta[at + 1] * FOCAL_STEP;
            camera.intrinsics.cx += delta[at + 2] * PRINCIPAL_STEP;
            camera.intrinsics.cy += delta[at + 3] * PRINCIPAL_STEP;
            at += 4;
        }

        if layout.distortion {
            let mut parameters = camera.lens.parameters();
            for (slot, value) in parameters.iter_mut().enumerate() {
                *value += delta[at + slot] * DISTORTION_STEP;
            }
            camera.lens = camera.lens.with_parameters(parameters);
            at += 4;
        }

        if layout.extrinsics {
            let turn = Vector3::new(delta[at], delta[at + 1], delta[at + 2]) * ROTATION_STEP;
            let shift =
                Vector3::new(delta[at + 3], delta[at + 4], delta[at + 5]) * TRANSLATION_STEP;

            // The rotation step is a world-frame increment on top of the
            // current orientation, which keeps the parameterization free of the
            // singularities any three-angle form has.
            camera.pose.rotation = UnitQuaternion::from_scaled_axis(turn) * camera.pose.rotation;
            camera.pose.translation.vector += shift;
            at += 6;
        }

        debug_assert_eq!(at, layout.camera_base(index) + layout.per_camera);
    }

    if layout.offsets {
        for (rig, offset) in state.offsets.iter_mut().enumerate() {
            let at = layout.offset_base(rig);
            *offset += Vector3::new(delta[at], delta[at + 1], delta[at + 2]) * OFFSET_STEP;
        }
    }

    state
}

/// Pulls any rig offset back to something a body could actually have.
///
/// A headset does not sit half a metre from the head it is strapped to, and a
/// hand does not hold a controller at arm's length. When the walk leaves a
/// direction unobservable — which is most walks, in the vertical — the solver
/// has no reason to prefer any answer along it and will take one that is
/// anatomically impossible, carrying every camera with it, since a shift of the
/// offset and a shift of the whole room are the same thing to it.
///
/// This is a bound, not a prior: it does nothing at all while the answer is
/// plausible, and it is the difference between a bounded error and an unbounded
/// one when it is not. It is applied to the accepted iterate rather than inside
/// the step, so the Jacobian is still differentiating the thing it thinks it is.
fn clamp_offsets(state: &mut State, max_offset: f64) {
    for offset in &mut state.offsets {
        let reach = offset.norm();
        if reach > max_offset {
            *offset *= max_offset / reach;
        }
    }
}

/// Reprojection error of every sighting, two components each, in radians.
fn raw_residuals(state: &State, sightings: &[Sighting], reference: &[f64]) -> DVector<f64> {
    let mut out = DVector::zeros(sightings.len() * 2);

    for (index, sighting) in sightings.iter().enumerate() {
        let camera = &state.cameras[sighting.camera];
        let offset = state.offsets[sighting.rig];
        let world = sighting.anchor * Point3::from(offset);

        let local = camera.pose.inverse_transform_point(&world);
        let (x, y) = if local.z <= 1e-4 {
            (BEHIND_PENALTY, BEHIND_PENALTY)
        } else {
            let (dx, dy) = camera.lens.distort(local.x / local.z, local.y / local.z);
            let px = camera.intrinsics.fx * dx + camera.intrinsics.cx;
            let py = camera.intrinsics.fy * dy + camera.intrinsics.cy;
            let focal = reference[sighting.camera];
            (
                (px - sighting.pixel.x) / focal,
                (py - sighting.pixel.y) / focal,
            )
        };

        out[index * 2] = x;
        out[index * 2 + 1] = y;
    }

    out
}

/// Per-component multipliers combining each sighting's trust with its Huber
/// discount.
///
/// These are held fixed across an iteration. Letting them move with the
/// parameters would mean the solver differentiates through its own outlier
/// rejection, and it would happily reduce the cost by declaring everything an
/// outlier.
fn huber_scale(residual: &DVector<f64>, sightings: &[Sighting], huber: f64) -> DVector<f64> {
    let mut out = DVector::zeros(residual.len());

    for (index, sighting) in sightings.iter().enumerate() {
        let magnitude = (residual[index * 2].powi(2) + residual[index * 2 + 1].powi(2)).sqrt();
        let discount = if huber > 0.0 && magnitude > huber {
            (huber / magnitude).sqrt()
        } else {
            1.0
        };

        let scale = sighting.weight.max(0.0).sqrt() * discount;
        out[index * 2] = scale;
        out[index * 2 + 1] = scale;
    }

    out
}

fn component_product(residual: &DVector<f64>, scale: &DVector<f64>) -> DVector<f64> {
    residual.component_mul(scale)
}

fn weighted_cost(residual: &DVector<f64>, scale: &DVector<f64>) -> f64 {
    component_product(residual, scale).norm_squared()
}

/// Numeric Jacobian of the weighted residual, by central differences.
///
/// The analytic form would be faster, but every term of it — the lens models,
/// the pose parameterization, the shared offset — is a place to make a mistake
/// that shows up as slow convergence rather than as a failure, and the solve
/// runs once per calibration.
fn jacobian(
    state: &State,
    sightings: &[Sighting],
    reference: &[f64],
    scale: &DVector<f64>,
    layout: &Layout,
    delta: &mut DVector<f64>,
) -> DMatrix<f64> {
    let mut out = DMatrix::zeros(sightings.len() * 2, layout.total());
    delta.fill(0.0);

    for parameter in 0..layout.total() {
        delta[parameter] = EPSILON;
        let forward = raw_residuals(&apply(state, delta, layout), sightings, reference);

        delta[parameter] = -EPSILON;
        let backward = raw_residuals(&apply(state, delta, layout), sightings, reference);

        delta[parameter] = 0.0;

        for row in 0..out.nrows() {
            out[(row, parameter)] = scale[row] * (forward[row] - backward[row]) / (2.0 * EPSILON);
        }
    }

    out
}

fn report(
    residual: &DVector<f64>,
    sightings: &[Sighting],
    cameras: usize,
) -> (f64, Vec<CameraResidual>) {
    let mut totals = vec![(0usize, 0.0f64, 0.0f64); cameras];
    let mut sum = 0.0;

    for (index, sighting) in sightings.iter().enumerate() {
        let magnitude = (residual[index * 2].powi(2) + residual[index * 2 + 1].powi(2)).sqrt();
        sum += magnitude * magnitude;

        if let Some(entry) = totals.get_mut(sighting.camera) {
            entry.0 += 1;
            entry.1 += magnitude * magnitude;
            entry.2 = entry.2.max(magnitude);
        }
    }

    let rms = if sightings.is_empty() {
        0.0
    } else {
        (sum / sightings.len() as f64).sqrt()
    };

    let per_camera = totals
        .into_iter()
        .map(|(count, squared, worst)| CameraResidual {
            sightings: count,
            rms: if count == 0 {
                0.0
            } else {
                (squared / count as f64).sqrt()
            },
            worst,
        })
        .collect();

    (rms, per_camera)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::camera::Intrinsics;
    use crate::geometry::lens::Lens;
    use nalgebra::{Isometry3, Point2, Translation3};

    fn corner(x: f64, z: f64, lens: Lens) -> Camera {
        Camera::look_at(
            Intrinsics::from_fov(1280, 720, 72f64.to_radians()),
            lens,
            Point3::new(x, 2.4, z),
            Point3::new(0.0, 1.0, 0.0),
            Vector3::y(),
        )
    }

    fn room(lens: Lens) -> Vec<Camera> {
        vec![
            corner(-1.9, -1.9, lens),
            corner(1.9, -1.9, lens),
            corner(1.9, 1.9, lens),
        ]
    }

    /// The head offset the solver is meant to find: the head keypoint sits a
    /// little above and behind the headset origin.
    fn head_offset() -> Vector3<f64> {
        Vector3::new(0.01, 0.06, 0.12)
    }

    /// A calibration walk, as headset poses. The orientation varies, which is
    /// the only reason the offset is recoverable at all.
    fn walk() -> Vec<Isometry3<f64>> {
        (0..140)
            .map(|step| {
                let t = step as f64 * 0.11;
                let position = Translation3::new(
                    1.2 * t.sin(),
                    1.35 + 0.3 * (1.7 * t).sin(),
                    1.0 * (0.7 * t).cos(),
                );
                let rotation = UnitQuaternion::from_euler_angles(
                    0.25 * (1.3 * t).sin(),
                    0.9 * t,
                    0.12 * (0.6 * t).cos(),
                );
                Isometry3::from_parts(position, rotation)
            })
            .collect()
    }

    fn record(cameras: &[Camera], offset: Vector3<f64>) -> Vec<Sighting> {
        let mut out = Vec::new();

        for anchor in walk() {
            let world = anchor * Point3::from(offset);
            for (index, camera) in cameras.iter().enumerate() {
                let Some(pixel) = camera.project(world) else {
                    continue;
                };
                if pixel.x < 0.0
                    || pixel.y < 0.0
                    || pixel.x >= camera.intrinsics.width as f64
                    || pixel.y >= camera.intrinsics.height as f64
                {
                    continue;
                }
                out.push(Sighting {
                    camera: index,
                    rig: 0,
                    anchor,
                    pixel,
                    weight: 1.0,
                });
            }
        }

        out
    }

    /// Cameras as a resection would hand them over: roughly right, not right.
    fn perturbed(cameras: &[Camera]) -> Vec<Camera> {
        cameras
            .iter()
            .enumerate()
            .map(|(index, camera)| {
                let sign = if index % 2 == 0 { 1.0 } else { -1.0 };
                let mut shifted = camera.clone();
                shifted.intrinsics.fx *= 1.0 + 0.03 * sign;
                shifted.intrinsics.fy *= 1.0 + 0.03 * sign;
                shifted.intrinsics.cx += 6.0 * sign;
                shifted.intrinsics.cy -= 4.0 * sign;
                shifted.pose.rotation = UnitQuaternion::from_scaled_axis(Vector3::new(
                    0.012 * sign,
                    -0.009,
                    0.006 * sign,
                )) * shifted.pose.rotation;
                shifted.pose.translation.vector += Vector3::new(0.04 * sign, -0.03, 0.05);
                shifted
            })
            .collect()
    }

    fn largest_position_error(solved: &[Camera], truth: &[Camera]) -> f64 {
        solved
            .iter()
            .zip(truth)
            .map(|(a, b)| (a.position() - b.position()).norm())
            .fold(0.0, f64::max)
    }

    fn largest_angle_error(solved: &[Camera], truth: &[Camera]) -> f64 {
        solved
            .iter()
            .zip(truth)
            .map(|(a, b)| a.pose.rotation.angle_to(&b.pose.rotation))
            .fold(0.0, f64::max)
    }

    #[test]
    fn a_perturbed_room_is_pulled_back_onto_the_data() {
        let truth = room(Lens::default());
        let sightings = record(&truth, head_offset());
        let start = perturbed(&truth);

        let result = refine(
            &start,
            &[Vector3::zeros()],
            &sightings,
            &RefineOptions::default(),
        );

        assert!(result.rms < 1e-6, "rms was {} rad", result.rms);
        assert!(
            largest_position_error(&result.cameras, &truth) < 2e-3,
            "worst camera moved {} m from the truth",
            largest_position_error(&result.cameras, &truth)
        );
        assert!(
            largest_angle_error(&result.cameras, &truth) < 1e-3,
            "worst camera is off by {} rad",
            largest_angle_error(&result.cameras, &truth)
        );
    }

    /// The headset does not sit on the head keypoint. If that gap is not solved
    /// for, it is absorbed into the camera positions instead, and every one of
    /// them ends up wrong by a few centimetres.
    #[test]
    fn the_head_offset_is_recovered() {
        let truth = room(Lens::default());
        let sightings = record(&truth, head_offset());

        let result = refine(
            &perturbed(&truth),
            &[Vector3::zeros()],
            &sightings,
            &RefineOptions::default(),
        );

        let error = (result.offsets[0] - head_offset()).norm();
        assert!(
            error < 2e-3,
            "recovered {:?}, expected {:?}",
            result.offsets[0],
            head_offset()
        );
    }

    #[test]
    fn distortion_is_recovered_from_a_zeroed_start() {
        let lens = Lens::RadialTangential {
            k1: -0.22,
            k2: 0.06,
            p1: 0.0,
            p2: 0.0,
        };
        let truth = room(lens);
        let sightings = record(&truth, head_offset());

        // The starting point is the truth with the lens forgotten, which is
        // what a fresh camera profile looks like.
        let start: Vec<Camera> = truth
            .iter()
            .map(|camera| Camera::new(camera.intrinsics, Lens::default(), camera.pose))
            .collect();

        let result = refine(
            &start,
            &[head_offset()],
            &sightings,
            &RefineOptions::default(),
        );

        assert!(result.rms < 1e-5, "rms was {} rad", result.rms);
        let recovered = result.cameras[0].lens.parameters();
        assert!(
            (recovered[0] - (-0.22)).abs() < 0.02,
            "recovered k1 {}, expected -0.22",
            recovered[0]
        );
    }

    /// A missed keypoint is a confident observation in the wrong place, and a
    /// least-squares fit will happily bend every camera to accommodate one.
    #[test]
    fn stray_keypoints_do_not_drag_the_solution() {
        let truth = room(Lens::default());
        let mut sightings = record(&truth, head_offset());
        for (index, sighting) in sightings.iter_mut().enumerate() {
            if index % 11 == 0 {
                sighting.pixel.x += 150.0;
                sighting.pixel.y -= 110.0;
            }
        }

        let result = refine(
            &perturbed(&truth),
            &[Vector3::zeros()],
            &sightings,
            &RefineOptions::default(),
        );

        assert!(
            largest_position_error(&result.cameras, &truth) < 1e-2,
            "the outliers moved a camera {} m",
            largest_position_error(&result.cameras, &truth)
        );
    }

    #[test]
    fn a_camera_that_saw_nothing_is_left_alone() {
        let truth = room(Lens::default());
        let mut sightings = record(&truth, head_offset());
        sightings.retain(|s| s.camera != 2);

        let result = refine(
            &truth,
            &[head_offset()],
            &sightings,
            &RefineOptions::default(),
        );

        assert_eq!(result.per_camera[2].sightings, 0);
        assert!(
            (result.cameras[2].position() - truth[2].position()).norm() < 1e-9,
            "an unobserved camera must not drift"
        );
    }

    #[test]
    fn per_camera_residuals_are_reported() {
        let truth = room(Lens::default());
        let sightings = record(&truth, head_offset());

        let result = refine(
            &truth,
            &[head_offset()],
            &sightings,
            &RefineOptions::default(),
        );

        assert_eq!(result.per_camera.len(), truth.len());
        let counted: usize = result.per_camera.iter().map(|c| c.sightings).sum();
        assert_eq!(counted, sightings.len());
        for camera in &result.per_camera {
            assert!(camera.worst < 1e-9, "worst residual {}", camera.worst);
        }
    }

    /// Walking the room without ever turning the head leaves the offset
    /// indistinguishable from a shift of every camera. The wizard has to be
    /// able to say so rather than returning a confident wrong answer.
    #[test]
    fn a_walk_without_rotation_cannot_pin_the_offset() {
        let still: Vec<Sighting> = walk()
            .into_iter()
            .map(|anchor| Sighting {
                camera: 0,
                rig: 0,
                anchor: Isometry3::from_parts(anchor.translation, UnitQuaternion::identity()),
                pixel: Point2::new(640.0, 360.0),
                weight: 1.0,
            })
            .collect();

        let turning: Vec<Sighting> = walk()
            .into_iter()
            .map(|anchor| Sighting {
                camera: 0,
                rig: 0,
                anchor,
                pixel: Point2::new(640.0, 360.0),
                weight: 1.0,
            })
            .collect();

        assert!(offset_observability(&still, 0) < 1e-9);
        assert!(
            offset_observability(&turning, 0) > 0.15,
            "a varied walk should be observable, got {}",
            offset_observability(&turning, 0)
        );
    }

    /// The failure the synthetic walks never showed, because the simulated user
    /// obligingly nods. A real one walks a room looking left and right and
    /// almost never up, and that leaves the vertical completely free: rotating
    /// about an axis tells you nothing about a shift along it.
    #[test]
    fn turning_on_the_spot_leaves_the_vertical_unobservable() {
        let yaw_only: Vec<Sighting> = (0..400)
            .map(|step| {
                let t = step as f64 * 0.05;
                Sighting {
                    camera: 0,
                    rig: 0,
                    anchor: Isometry3::from_parts(
                        Translation3::new(1.2 * t.sin(), 1.5, 1.0 * (0.7 * t).cos()),
                        UnitQuaternion::from_euler_angles(0.0, 0.9 * t, 0.0),
                    ),
                    pixel: Point2::new(640.0, 360.0),
                    weight: 1.0,
                }
            })
            .collect();

        assert!(
            offset_observability(&yaw_only, 0) < 0.02,
            "a walk that only turns should report as unobservable, got {}",
            offset_observability(&yaw_only, 0)
        );
    }

    /// With a direction the walk cannot see, nothing stops the offset and every
    /// camera sliding along it together. The bound is what makes the error
    /// finite.
    #[test]
    fn an_unobservable_direction_cannot_slide_the_room_away() {
        let truth = room(Lens::default());
        let offset = Vector3::new(0.02, 0.09, 0.12);

        // A walk with no pitch or roll at all, so the vertical is free.
        let anchors: Vec<Isometry3<f64>> = (0..400)
            .map(|step| {
                let t = step as f64 * 0.05;
                Isometry3::from_parts(
                    Translation3::new(1.2 * t.sin(), 1.4, 1.0 * (0.7 * t).cos()),
                    UnitQuaternion::from_euler_angles(0.0, 0.9 * t, 0.0),
                )
            })
            .collect();

        let mut sightings = Vec::new();
        for anchor in &anchors {
            let world = anchor * Point3::from(offset);
            for (index, camera) in truth.iter().enumerate() {
                if let Some(pixel) = camera.project(world) {
                    sightings.push(Sighting {
                        camera: index,
                        rig: 0,
                        anchor: *anchor,
                        pixel,
                        weight: 1.0,
                    });
                }
            }
        }

        // Start every camera half a metre below where it belongs. Without the
        // bound the offset absorbs it and the solve is perfectly happy there.
        let sunk: Vec<Camera> = truth
            .iter()
            .map(|camera| {
                let mut moved = camera.clone();
                moved.pose.translation.vector.y -= 0.5;
                moved
            })
            .collect();

        let result = refine(
            &sunk,
            &[Vector3::zeros()],
            &sightings,
            &RefineOptions::default(),
        );

        assert!(
            result.offsets[0].norm() <= RefineOptions::default().max_offset + 1e-9,
            "the offset reached {} m",
            result.offsets[0].norm()
        );

        let worst = truth
            .iter()
            .zip(&result.cameras)
            .map(|(a, b)| (a.position().y - b.position().y).abs())
            .fold(0.0, f64::max);
        assert!(
            worst < 0.5,
            "the room stayed {worst:.2} m below where it belongs"
        );
    }
}
