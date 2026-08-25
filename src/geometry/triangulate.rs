//! Turning the same keypoint seen from several cameras into one 3D point.
//!
//! Weights are angular, not pixel-based. Confidence scores are not comparable
//! across models and a pixel is not comparable across cameras, so both are
//! converted into the same physical quantity first: the uncertainty of the
//! ray's *direction*. That is what lets a 1080p narrow camera correctly outvote
//! a 480p fisheye looking at the same joint.

use nalgebra::{Matrix3, Matrix4, Point2, Point3, Vector3};

use super::camera::Camera;

/// One camera's sighting of a point.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    /// Index into the camera list this observation belongs to.
    pub camera: usize,
    pub pixel: Point2<f64>,
    /// Uncertainty of the ray direction, in radians.
    pub sigma: f64,
}

impl Observation {
    /// Builds an observation, converting a keypoint confidence into an angular
    /// uncertainty for the camera that saw it.
    ///
    /// `interpolation_penalty` inflates the uncertainty when the keypoint had
    /// to be interpolated in time, which is how a slow camera loses influence
    /// during fast motion without being switched off.
    pub fn new(
        camera_index: usize,
        camera: &Camera,
        pixel: Point2<f64>,
        confidence: f64,
        interpolation_penalty: f64,
    ) -> Self {
        Self {
            camera: camera_index,
            pixel,
            sigma: angular_sigma(camera, confidence) * interpolation_penalty.max(1.0),
        }
    }
}

/// Localization noise implied by a keypoint confidence, in pixels.
///
/// A confident keypoint is worth about a pixel; a weak one is worth several.
/// The exact curve matters less than the fact that everything else that
/// competes with keypoint noise is expressed against this same scale.
pub fn pixel_sigma(confidence: f64) -> f64 {
    const BEST_PIXELS: f64 = 1.0;
    const WORST_PIXELS: f64 = 8.0;

    let confidence = confidence.clamp(0.0, 1.0);
    WORST_PIXELS + (BEST_PIXELS - WORST_PIXELS) * confidence
}

/// The same noise as an angle, which is the form that compares across cameras.
///
/// A pixel is not the same quantity on a 480p fisheye as on a 1080p narrow
/// lens; the angle it subtends is.
pub fn angular_sigma(camera: &Camera, confidence: f64) -> f64 {
    pixel_sigma(confidence) * camera.intrinsics.radians_per_pixel()
}

/// A triangulated point and how well the rays agreed about it.
#[derive(Debug, Clone)]
pub struct Triangulation {
    pub point: Point3<f64>,
    /// Angular reprojection residual per contributing observation, in radians.
    pub residuals: Vec<(usize, f64)>,
    /// Observations that were used, after outlier rejection.
    pub inliers: Vec<usize>,
    /// Weight each contributing camera carried, normalized to sum to one.
    pub weights: Vec<(usize, f64)>,
    /// Position covariance implied by the rays, in square metres.
    ///
    /// Implied is the word: it is built from the uncertainty each ray *claimed*
    /// and knows nothing about whether they turned out to agree.
    pub covariance: Matrix3<f64>,
    /// Sum of each inlier.s squared residual measured in units of the noise it
    /// claimed, and the degrees of freedom that sum was taken over.
    ///
    /// The other half of the answer, and the half the covariance cannot give.
    /// A well conditioned bundle of rays that disagree by three times what they
    /// claimed has a small covariance and a large chi-square, and it is the
    /// chi-square that is telling the truth about where the point is. Pooled
    /// across the body it becomes the factor the covariance was missing.
    pub chi_square: f64,
    /// Two constraints per ray, three unknowns in the point.
    pub dof: f64,
}

impl Triangulation {
    /// Largest angular residual among the inliers, in radians.
    pub fn worst_residual(&self) -> f64 {
        self.residuals
            .iter()
            .map(|(_, residual)| *residual)
            .fold(0.0, f64::max)
    }

    pub fn rms_residual(&self) -> f64 {
        if self.residuals.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.residuals.iter().map(|(_, r)| r * r).sum();
        (sum / self.residuals.len() as f64).sqrt()
    }

    /// Standard deviation along the least constrained direction, in metres.
    ///
    /// The single number worth putting in front of a user, because it is the
    /// one that says how much this joint can be believed. Two cameras twenty
    /// degrees apart agree perfectly and still leave the point free to slide
    /// along the bisector; the residual cannot see that and this can.
    pub fn sigma(&self) -> f64 {
        self.covariance
            .symmetric_eigenvalues()
            .iter()
            .fold(0.0, |worst: f64, value| worst.max(*value))
            .max(0.0)
            .sqrt()
    }
}

/// Triangulates one point from two or more observations.
///
/// With three or more, subsets are tried and the observations that disagree are
/// dropped: a joint hidden from one camera produces a confident keypoint in the
/// wrong place, and one such ray is enough to pull a point metres away.
pub fn triangulate(
    cameras: &[Camera],
    observations: &[Observation],
    inlier_threshold: f64,
) -> Option<Triangulation> {
    if observations.len() < 2 {
        return None;
    }

    let all: Vec<usize> = (0..observations.len()).collect();
    let mut best: Option<(Vec<usize>, Point3<f64>)> = None;

    if observations.len() == 2 {
        let point = solve(cameras, observations, &all)?;
        best = Some((all.clone(), point));
    } else {
        // Every pair is a hypothesis. With at most four cameras this is six
        // solves, which is cheaper than any sampling scheme and never unlucky.
        let mut best_score = (0usize, f64::INFINITY);
        for i in 0..observations.len() {
            for j in (i + 1)..observations.len() {
                let Some(candidate) = solve(cameras, observations, &[i, j]) else {
                    continue;
                };

                let inliers: Vec<usize> = all
                    .iter()
                    .copied()
                    .filter(|index| {
                        residual(cameras, &observations[*index], candidate)
                            .map(|r| r <= inlier_threshold)
                            .unwrap_or(false)
                    })
                    .collect();

                if inliers.len() < 2 {
                    continue;
                }
                let error: f64 = inliers
                    .iter()
                    .filter_map(|index| residual(cameras, &observations[*index], candidate))
                    .sum();

                let score = (inliers.len(), error);
                if score.0 > best_score.0 || (score.0 == best_score.0 && score.1 < best_score.1) {
                    best_score = score;
                    best = Some((inliers, candidate));
                }
            }
        }
    }

    let (inliers, seed) = best?;

    // Re-solve using every inlier, which is what actually uses the extra
    // cameras rather than just the pair that happened to seed the hypothesis,
    // then polish that against the objective the answer is judged by.
    let linear = solve(cameras, observations, &inliers).unwrap_or(seed);
    let polished = polish(cameras, observations, &inliers, linear);
    let (point, covariance, metric) = match polished {
        Some(polished) => polished,
        None => (linear, Matrix3::zeros(), Vec::new()),
    };

    let residuals: Vec<(usize, f64)> = inliers
        .iter()
        .filter_map(|index| {
            residual(cameras, &observations[*index], point)
                .map(|r| (observations[*index].camera, r))
        })
        .collect();

    // How much each camera actually pinned the point down. The angular
    // uncertainty alone would say a distant camera is as good as a near one
    // looking at the same joint through the same lens, and it is not: the same
    // angle covers more room the further away it is measured from.
    let mut weights: Vec<(usize, f64)> = if metric.len() == inliers.len() {
        inliers
            .iter()
            .zip(&metric)
            .map(|(index, weight)| (observations[*index].camera, *weight))
            .collect()
    } else {
        inliers
            .iter()
            .map(|index| {
                (
                    observations[*index].camera,
                    1.0 / observations[*index].sigma.max(1e-9).powi(2),
                )
            })
            .collect()
    };

    let total: f64 = weights.iter().map(|(_, weight)| *weight).sum();
    for (_, weight) in &mut weights {
        *weight /= total.max(1e-12);
    }

    // How much worse the rays agreed than they said they would, in the units
    // the claim was made in. Nothing here decides what to do with it — that is
    // a question about the whole body rather than about one joint, because one
    // joint has one or three degrees of freedom and no estimate worth acting
    // on comes out of three.
    let chi_square: f64 = inliers
        .iter()
        .filter_map(|index| {
            let observation = observations[*index];
            residual(cameras, &observation, point)
                .map(|error| (error / observation.sigma.max(1e-9)).powi(2))
        })
        .sum();
    let dof = (2 * inliers.len()) as f64 - 3.0;

    Some(Triangulation {
        point,
        covariance,
        residuals,
        inliers: inliers
            .iter()
            .map(|index| observations[*index].camera)
            .collect(),
        weights,
        chi_square,
        dof,
    })
}

/// Refines a point against the objective it is actually judged by, and reports
/// how well the rays pinned it down.
///
/// The linear solve minimizes an algebraic quantity that happens to vanish at
/// the right answer; away from it, it is biased towards the cameras nearest the
/// point. This minimizes the weighted perpendicular distance to each ray in
/// metres, which is the maximum-likelihood estimate under the angular noise
/// model, and it is what converts an angular uncertainty into the positional
/// one a user can read.
///
/// Returns the point, its covariance in square metres, and the weight each
/// inlier carried, in the order they were given.
fn polish(
    cameras: &[Camera],
    observations: &[Observation],
    use_indices: &[usize],
    seed: Point3<f64>,
) -> Option<(Point3<f64>, Matrix3<f64>, Vec<f64>)> {
    /// A joint closer to a camera than this is inside it, and the range is
    /// only there to scale an angle into a distance.
    const MIN_RANGE: f64 = 0.05;
    /// The weights depend on the ranges, which depend on the point. Three
    /// passes is past where it stops moving for any real room.
    const PASSES: usize = 3;

    let mut point = seed;
    let mut covariance = Matrix3::zeros();
    let mut weights = Vec::with_capacity(use_indices.len());

    for _ in 0..PASSES {
        // Sum of w * P, where P projects out the ray direction: what is left is
        // the displacement the ray can actually see.
        let mut normal = Matrix3::zeros();
        let mut moment = Vector3::zeros();
        weights.clear();

        for index in use_indices {
            let observation = observations[*index];
            let camera = cameras.get(observation.camera)?;
            let origin = camera.position();
            let direction = camera.ray(observation.pixel).into_inner();

            // An angle is not a distance until it is multiplied by one. The
            // same half-degree of ray uncertainty is a millimetre up close and
            // a centimetre across the room.
            let range = (point - origin).dot(&direction).max(MIN_RANGE);
            let weight = 1.0 / (observation.sigma.max(1e-9) * range).powi(2);

            let projector = Matrix3::identity() - direction * direction.transpose();
            normal += weight * projector;
            moment += weight * projector * origin.coords;
            weights.push(weight);
        }

        // Rays that are all but parallel leave the point free to slide along
        // them, and the inverse is what says so rather than something to guard
        // against.
        covariance = normal.try_inverse()?;
        point = Point3::from(covariance * moment);
    }

    Some((point, covariance, weights))
}

/// Weighted linear triangulation over the given observations.
fn solve(
    cameras: &[Camera],
    observations: &[Observation],
    use_indices: &[usize],
) -> Option<Point3<f64>> {
    // Each ray contributes two constraints: the point lies on the line through
    // the camera centre in the ray's direction. Written as the two components
    // of the cross product being zero, the system is linear in the point.
    let mut normal = Matrix4::zeros();

    for index in use_indices {
        let observation = observations[*index];
        let camera = cameras.get(observation.camera)?;
        let origin = camera.position();
        let direction = camera.ray(observation.pixel).into_inner();

        // Two directions perpendicular to the ray; the point must have no
        // component along either, measured from the camera centre.
        let (u, v) = perpendicular_basis(direction);
        let weight = 1.0 / observation.sigma.max(1e-9).powi(2);

        for axis in [u, v] {
            let row = nalgebra::RowVector4::new(axis.x, axis.y, axis.z, -axis.dot(&origin.coords));
            normal += weight * row.transpose() * row;
        }
    }

    // The solution is the null space of the accumulated system, which is the
    // eigenvector of the smallest eigenvalue.
    let eigen = normal.symmetric_eigen();
    let (min_index, _) = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))?;

    let solution = eigen.eigenvectors.column(min_index);
    if solution[3].abs() < 1e-12 {
        return None;
    }

    Some(Point3::new(
        solution[0] / solution[3],
        solution[1] / solution[3],
        solution[2] / solution[3],
    ))
}

/// How near the nearest pair of rays came to meeting, in metres.
///
/// Only asked when triangulation has already failed, and only to say by how
/// much. The size of it is the diagnosis: a couple of centimetres is keypoint
/// noise and a threshold set too tight, twenty is the two cameras looking at
/// different legs of the same person, and a metre is them looking at different
/// people.
pub fn closest_approach(cameras: &[Camera], observations: &[Observation]) -> Option<f64> {
    let mut closest = f64::INFINITY;

    for i in 0..observations.len() {
        for j in (i + 1)..observations.len() {
            let first = cameras.get(observations[i].camera)?;
            let second = cameras.get(observations[j].camera)?;
            let a = first.ray(observations[i].pixel).into_inner();
            let b = second.ray(observations[j].pixel).into_inner();

            let normal = a.cross(&b);
            let length = normal.norm();
            // Parallel rays never meet and never miss either; there is no
            // distance between them worth reporting.
            if length < 1e-9 {
                continue;
            }

            let between = second.position() - first.position();
            closest = closest.min((between.dot(&normal) / length).abs());
        }
    }

    closest.is_finite().then_some(closest)
}

fn residual(cameras: &[Camera], observation: &Observation, point: Point3<f64>) -> Option<f64> {
    cameras
        .get(observation.camera)?
        .angular_error(point, observation.pixel)
}

/// Two unit vectors perpendicular to `direction` and to each other.
fn perpendicular_basis(direction: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let direction = direction.normalize();
    // Pick the world axis least aligned with the ray, so the cross product is
    // never near zero.
    let helper = if direction.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let u = direction.cross(&helper).normalize();
    let v = direction.cross(&u).normalize();
    (u, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::camera::Intrinsics;
    use crate::geometry::lens::Lens;

    fn corner_camera(x: f64, z: f64, width: u32, fov_degrees: f64) -> Camera {
        Camera::look_at(
            Intrinsics::from_fov(width, width * 9 / 16, fov_degrees.to_radians()),
            Lens::default(),
            Point3::new(x, 2.4, z),
            Point3::new(0.0, 1.0, 0.0),
            Vector3::y(),
        )
    }

    fn room() -> Vec<Camera> {
        vec![
            corner_camera(-1.8, -1.8, 1280, 70.0),
            corner_camera(1.8, -1.8, 1280, 70.0),
            corner_camera(1.8, 1.8, 1280, 70.0),
        ]
    }

    fn sightings(cameras: &[Camera], point: Point3<f64>, confidence: f64) -> Vec<Observation> {
        cameras
            .iter()
            .enumerate()
            .filter_map(|(index, camera)| {
                camera
                    .project(point)
                    .map(|pixel| Observation::new(index, camera, pixel, confidence, 1.0))
            })
            .collect()
    }

    #[test]
    fn two_cameras_recover_a_point_exactly() {
        let cameras = room();
        let truth = Point3::new(0.3, 0.9, -0.4);
        let observations = sightings(&cameras[..2], truth, 0.9);

        let result = triangulate(&cameras, &observations, 0.01).expect("two rays should solve");
        assert!(
            (result.point - truth).norm() < 1e-6,
            "recovered {:?}, expected {truth:?}",
            result.point
        );
    }

    #[test]
    fn a_single_ray_is_not_enough() {
        let cameras = room();
        let observations = sightings(&cameras[..1], Point3::new(0.0, 1.0, 0.0), 0.9);
        assert!(triangulate(&cameras, &observations, 0.01).is_none());
    }

    #[test]
    fn three_cameras_agree_and_all_count_as_inliers() {
        let cameras = room();
        let truth = Point3::new(-0.5, 0.4, 0.6);
        let observations = sightings(&cameras, truth, 0.9);

        let result = triangulate(&cameras, &observations, 0.005).expect("three rays should solve");
        assert_eq!(result.inliers.len(), 3);
        assert!((result.point - truth).norm() < 1e-6);
        assert!(result.worst_residual() < 1e-6);
    }

    /// A joint hidden from one camera still produces a confident keypoint, just
    /// in the wrong place. One such ray must not be able to drag the point.
    #[test]
    fn a_confidently_wrong_camera_is_rejected() {
        let cameras = room();
        let truth = Point3::new(0.2, 0.7, 0.1);
        let mut observations = sightings(&cameras, truth, 0.9);

        let bad = observations.last_mut().unwrap();
        bad.pixel.x += 90.0;
        bad.pixel.y -= 60.0;

        let result = triangulate(&cameras, &observations, 0.01).expect("the good pair should win");
        assert_eq!(result.inliers.len(), 2, "the bad ray should be dropped");
        assert!(
            (result.point - truth).norm() < 1e-5,
            "the outlier moved the point to {:?}",
            result.point
        );
    }

    /// The point of angular weighting: a sharp camera should dominate a coarse
    /// one, and the weights should reflect the optics rather than the pixels.
    #[test]
    fn a_sharper_camera_carries_more_weight() {
        let cameras = vec![
            corner_camera(-1.8, -1.8, 640, 100.0),
            corner_camera(1.8, -1.8, 1920, 60.0),
        ];
        let truth = Point3::new(0.1, 0.8, 0.0);
        let observations = sightings(&cameras, truth, 0.9);

        let result = triangulate(&cameras, &observations, 0.02).expect("both rays see it");
        let wide = result.weights.iter().find(|(c, _)| *c == 0).unwrap().1;
        let narrow = result.weights.iter().find(|(c, _)| *c == 1).unwrap().1;

        assert!(
            narrow > 5.0 * wide,
            "the sharp camera should dominate, got {narrow} against {wide}"
        );
    }

    /// A noisy keypoint should pull the result less than a confident one.
    #[test]
    fn a_weak_keypoint_is_trusted_less() {
        let cameras = room();
        let truth = Point3::new(0.0, 0.9, 0.0);

        let mut confident = sightings(&cameras[..2], truth, 0.95);
        confident[1].pixel.x += 6.0;

        let mut weak = confident.clone();
        weak[1].sigma = angular_sigma(&cameras[1], 0.1);

        let a = triangulate(&cameras, &confident, 1.0).unwrap();
        let b = triangulate(&cameras, &weak, 1.0).unwrap();

        assert!(
            (b.point - truth).norm() < (a.point - truth).norm(),
            "down-weighting the offset ray should move the answer closer to the truth"
        );
    }

    /// Two cameras looking from nearly the same place agree perfectly and still
    /// have no idea how far away the point is. The residual cannot see that;
    /// the covariance is the thing that can.
    #[test]
    fn rays_that_barely_cross_report_an_uncertain_point() {
        let truth = Point3::new(0.0, 0.9, 0.0);

        let crossed = vec![
            corner_camera(-1.8, -1.8, 1280, 70.0),
            corner_camera(1.8, -1.8, 1280, 70.0),
        ];
        let grazing = vec![
            corner_camera(-1.8, -1.8, 1280, 70.0),
            corner_camera(-1.7, -1.8, 1280, 70.0),
        ];

        let wide = triangulate(&crossed, &sightings(&crossed, truth, 0.9), 0.02).unwrap();
        let narrow = triangulate(&grazing, &sightings(&grazing, truth, 0.9), 0.02).unwrap();

        assert!(
            narrow.sigma() > 10.0 * wide.sigma(),
            "a grazing pair should be far less certain, got {} against {}",
            narrow.sigma(),
            wide.sigma()
        );
        assert!(
            narrow.rms_residual() < 1e-6,
            "and it should say so despite fitting its own rays perfectly"
        );
    }

    #[test]
    fn a_third_camera_tightens_the_answer() {
        let cameras = room();
        let truth = Point3::new(0.2, 0.8, -0.1);

        let pair = triangulate(&cameras, &sightings(&cameras[..2], truth, 0.9), 0.02).unwrap();
        let trio = triangulate(&cameras, &sightings(&cameras, truth, 0.9), 0.02).unwrap();

        assert!(trio.sigma() < pair.sigma());
    }

    /// The uncertainty is only worth reporting if it is the size of the error
    /// that actually turns up. Noise of the size the model assumes is injected,
    /// and the spread of the answers is compared against what was predicted.
    #[test]
    fn the_predicted_uncertainty_matches_the_error_that_occurs() {
        let cameras = room();
        let truth = Point3::new(0.1, 0.85, -0.2);
        let clean = sightings(&cameras, truth, 0.9);

        let predicted = triangulate(&cameras, &clean, 0.05).unwrap().sigma();

        let mut noise = Gaussian::new(0x5EED_1234);
        let spread = pixel_sigma(0.9);
        let mut sum = 0.0;
        const TRIALS: usize = 400;

        for _ in 0..TRIALS {
            let jittered: Vec<Observation> = clean
                .iter()
                .map(|observation| Observation {
                    pixel: Point2::new(
                        observation.pixel.x + noise.next() * spread,
                        observation.pixel.y + noise.next() * spread,
                    ),
                    ..*observation
                })
                .collect();

            // A threshold loose enough that the noise never costs a ray: this
            // measures the uncertainty, not the outlier rejection.
            let result = triangulate(&cameras, &jittered, 1.0).unwrap();
            sum += (result.point - truth).norm_squared();
        }

        let actual = (sum / TRIALS as f64).sqrt();
        // The prediction is the worst of the three axes, so the error over all
        // three lands between one and the square root of three times it.
        assert!(
            actual > 0.7 * predicted && actual < 2.2 * predicted,
            "predicted {predicted:.5} m, saw {actual:.5} m"
        );
    }

    /// Deterministic standard normals, so the test above means the same thing
    /// on every machine and every run.
    struct Gaussian {
        state: u64,
        spare: Option<f64>,
    }

    impl Gaussian {
        fn new(seed: u64) -> Self {
            Self {
                state: seed,
                spare: None,
            }
        }

        fn uniform(&mut self) -> f64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn next(&mut self) -> f64 {
            if let Some(spare) = self.spare.take() {
                return spare;
            }
            let (u, v) = (self.uniform(), self.uniform());
            let radius = (-2.0 * u.ln()).sqrt();
            let angle = std::f64::consts::TAU * v;
            self.spare = Some(radius * angle.sin());
            radius * angle.cos()
        }
    }

    #[test]
    fn interpolated_observations_lose_influence() {
        let camera = corner_camera(0.0, -2.0, 1280, 70.0);
        let pixel = Point2::new(640.0, 360.0);

        let fresh = Observation::new(0, &camera, pixel, 0.9, 1.0);
        let stale = Observation::new(0, &camera, pixel, 0.9, 3.0);

        assert!((stale.sigma - 3.0 * fresh.sigma).abs() < 1e-12);
    }
}
