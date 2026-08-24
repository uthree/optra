//! Resection: solving where a camera is from points whose world position is
//! already known.
//!
//! The calibration walk produces exactly that. The headset reports where the
//! head is in the world while each camera reports where it appears in the
//! image, and six well-spread correspondences determine the whole projection
//! matrix, which decomposes into the intrinsics, the rotation and the position.
//!
//! What comes out here is a seed, not the answer. The linear solve ignores
//! distortion and weights every correspondence equally, so RANSAC guards it
//! against bad detections and the bundle refinement in [`super::refine`]
//! finishes the job.

use nalgebra::{
    Isometry3, Matrix3, Matrix3x4, Matrix4, Point2, Point3, Rotation3, SMatrix, Translation3,
    UnitQuaternion, Vector3,
};

use super::camera::{Camera, Intrinsics};
use super::lens::Lens;

/// Correspondences needed for one linear solve.
const SAMPLE: usize = 6;

/// A known world point and the pixel a camera saw it at.
#[derive(Debug, Clone, Copy)]
pub struct Correspondence {
    pub world: Point3<f64>,
    pub pixel: Point2<f64>,
}

#[derive(Debug, Clone)]
pub struct ResectionOptions {
    /// Random hypotheses to try before settling on the best one.
    pub hypotheses: usize,
    /// Angular error below which a correspondence counts as an inlier, in
    /// radians. Angular rather than pixel-based so that the same setting means
    /// the same thing on every camera in a mixed room.
    pub inlier_threshold: f64,
    /// Inliers a solve must reach to be accepted at all.
    pub min_inliers: usize,
    /// Fraction of the correspondences, nearest the image centre, that
    /// hypotheses are drawn from when the camera has a distorted lens.
    /// Distortion is smallest near the centre, so a seed taken from there is
    /// least damaged by the linear solve ignoring it.
    pub central_fraction: f64,
    /// Largest departure from square pixels a solve may show before it is
    /// discarded as nonsense.
    pub max_aspect_error: f64,
    /// How far from the frame centre the principal point may land, as a
    /// fraction of the image size.
    pub max_principal_offset: f64,
    /// Seed for the hypothesis sampler, so a calibration run is reproducible.
    pub seed: u64,
}

impl Default for ResectionOptions {
    fn default() -> Self {
        Self {
            hypotheses: 512,
            inlier_threshold: 0.01,
            min_inliers: 12,
            central_fraction: 0.6,
            max_aspect_error: 0.25,
            max_principal_offset: 0.3,
            seed: 0x00FF_1CE5_u64,
        }
    }
}

/// What a resection recovered.
#[derive(Debug, Clone)]
pub struct Resection {
    pub camera: Camera,
    /// Indices of the correspondences that agreed with the solution.
    pub inliers: Vec<usize>,
    /// RMS angular reprojection error over the inliers, in radians, measured
    /// against the original pixels through the camera's own lens model.
    pub rms: f64,
    /// How far the inlier world points are from lying in a plane, as the ratio
    /// of their smallest to their largest principal extent. Near zero the
    /// solve is close to degenerate and the result should not be trusted.
    pub spread: f64,
}

impl Resection {
    /// Whether the correspondences filled enough of the room for the linear
    /// solve to be well posed.
    ///
    /// A calibration walk that stays on one line, or that only ever puts the
    /// head at one height, lands here.
    pub fn is_well_conditioned(&self) -> bool {
        self.spread > 0.05
    }
}

/// Solves for a camera from correspondences, rejecting outliers.
///
/// `guess` supplies the image size and a starting focal length; only its
/// resolution is kept, the rest is solved for. `lens` is the camera's declared
/// lens model, used to straighten the pixels before the linear solve with
/// whatever coefficients are known so far.
pub fn resect(
    guess: &Intrinsics,
    lens: Lens,
    points: &[Correspondence],
    options: &ResectionOptions,
) -> Option<Resection> {
    if points.len() < SAMPLE.max(options.min_inliers) {
        return None;
    }

    let world: Vec<Point3<f64>> = points.iter().map(|c| c.world).collect();
    let straight = straighten(guess, &lens, points);
    let pool = hypothesis_pool(guess, &lens, points, options);
    if pool.len() < SAMPLE {
        return None;
    }

    let mut rng = Rng::new(options.seed);
    let mut sample = Vec::with_capacity(SAMPLE);
    let mut best: Option<(Vec<usize>, f64)> = None;

    for _ in 0..options.hypotheses {
        if !draw(&mut rng, &pool, &mut sample) {
            continue;
        }
        let Some(camera) = solve(guess, &world, &straight, &sample) else {
            continue;
        };
        if !plausible(&camera.intrinsics, guess, options) {
            continue;
        }

        let (inliers, error) = score(&camera, &world, &straight, options.inlier_threshold);
        if inliers.len() < options.min_inliers {
            continue;
        }

        let better = match &best {
            None => true,
            Some((current, current_error)) => {
                inliers.len() > current.len()
                    || (inliers.len() == current.len() && error < *current_error)
            }
        };
        if better {
            let exhausted = inliers.len() == points.len();
            best = Some((inliers, error));
            // Every correspondence already agrees; more hypotheses cannot
            // improve on that.
            if exhausted {
                break;
            }
        }
    }

    let (mut inliers, _) = best?;

    // Re-solve from every inlier, which is what actually uses the whole walk
    // rather than the six points that happened to seed the hypothesis.
    let mut camera = solve(guess, &world, &straight, &inliers)?;

    // Straightening the pixels needed intrinsics, and the only ones available
    // for the first pass were the initial guess, which is merely the right
    // order of magnitude. Feeding the solved intrinsics back in and repeating
    // converges on the pair that is consistent with itself; two or three passes
    // is normally all it takes.
    if !lens.is_identity() {
        for _ in 0..8 {
            let restraightened = straighten(&camera.intrinsics, &lens, points);
            let Some(next) = solve(guess, &world, &restraightened, &inliers) else {
                break;
            };
            let settled = (next.intrinsics.fx - camera.intrinsics.fx).abs() < 1e-9;
            camera = next;
            if settled {
                break;
            }
        }
    }

    // The answer carries the real lens, so its error is measured the way the
    // rest of the application will measure it.
    let camera = Camera::new(camera.intrinsics, lens, camera.pose);
    let pixels: Vec<Point2<f64>> = points.iter().map(|c| c.pixel).collect();
    let (final_inliers, _) = score(&camera, &world, &pixels, options.inlier_threshold);
    if final_inliers.len() >= options.min_inliers {
        inliers = final_inliers;
    }

    let residuals: Vec<f64> = inliers
        .iter()
        .filter_map(|index| camera.angular_error(points[*index].world, points[*index].pixel))
        .collect();
    let rms = if residuals.is_empty() {
        f64::INFINITY
    } else {
        (residuals.iter().map(|r| r * r).sum::<f64>() / residuals.len() as f64).sqrt()
    };

    let spread = planarity(&inliers.iter().map(|i| points[*i].world).collect::<Vec<_>>());

    Some(Resection {
        camera,
        inliers,
        rms,
        spread,
    })
}

/// Removes lens distortion from the observed pixels, leaving the pixels a
/// pinhole camera would have produced.
fn straighten(intrinsics: &Intrinsics, lens: &Lens, points: &[Correspondence]) -> Vec<Point2<f64>> {
    if lens.is_identity() {
        return points.iter().map(|c| c.pixel).collect();
    }

    points
        .iter()
        .map(|c| {
            let x = (c.pixel.x - intrinsics.cx) / intrinsics.fx;
            let y = (c.pixel.y - intrinsics.cy) / intrinsics.fy;
            let (x, y) = lens.undistort(x, y);
            Point2::new(
                intrinsics.fx * x + intrinsics.cx,
                intrinsics.fy * y + intrinsics.cy,
            )
        })
        .collect()
}

/// Which correspondences hypotheses may be drawn from.
fn hypothesis_pool(
    intrinsics: &Intrinsics,
    lens: &Lens,
    points: &[Correspondence],
    options: &ResectionOptions,
) -> Vec<usize> {
    let all: Vec<usize> = (0..points.len()).collect();
    if lens.is_identity() {
        return all;
    }

    let radius = |index: usize| {
        let dx = points[index].pixel.x - intrinsics.cx;
        let dy = points[index].pixel.y - intrinsics.cy;
        (dx * dx + dy * dy).sqrt()
    };

    let mut sorted = all;
    sorted.sort_by(|a, b| radius(*a).total_cmp(&radius(*b)));

    let keep = ((points.len() as f64 * options.central_fraction).round() as usize)
        .max(SAMPLE.max(options.min_inliers))
        .min(points.len());
    sorted.truncate(keep);
    sorted
}

/// Counts the correspondences a candidate camera explains, and their total
/// error.
fn score(
    camera: &Camera,
    world: &[Point3<f64>],
    pixel: &[Point2<f64>],
    threshold: f64,
) -> (Vec<usize>, f64) {
    let mut inliers = Vec::new();
    let mut error = 0.0;

    for index in 0..world.len() {
        if let Some(residual) = camera.angular_error(world[index], pixel[index])
            && residual <= threshold
        {
            inliers.push(index);
            error += residual;
        }
    }

    (inliers, error)
}

/// One linear solve over the given correspondence indices.
fn solve(
    guess: &Intrinsics,
    world: &[Point3<f64>],
    pixel: &[Point2<f64>],
    use_indices: &[usize],
) -> Option<Camera> {
    let world: Vec<Point3<f64>> = use_indices.iter().map(|i| world[*i]).collect();
    let pixel: Vec<Point2<f64>> = use_indices.iter().map(|i| pixel[*i]).collect();

    let projection = dlt(&world, &pixel)?;
    let (k, rotation, translation) = decompose(&projection)?;

    // Every point must be in front of the camera. A projection matrix is only
    // defined up to sign, and a solve that puts the room behind the lens is a
    // sign error rather than an answer.
    if world
        .iter()
        .any(|point| (rotation * point.coords + translation).z <= 0.0)
    {
        return None;
    }

    let intrinsics = Intrinsics {
        // Skew is dropped: no webcam has any, and carrying it would leave the
        // rest of the application with a projection it cannot invert.
        fx: k[(0, 0)],
        fy: k[(1, 1)],
        cx: k[(0, 2)],
        cy: k[(1, 2)],
        width: guess.width,
        height: guess.height,
    };

    let camera_to_world = rotation.transpose();
    let centre = -camera_to_world * translation;
    let pose = Isometry3::from_parts(
        Translation3::from(centre),
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(camera_to_world)),
    );

    Some(Camera::new(intrinsics, Lens::default(), pose))
}

/// Rejects solutions no webcam could produce, before they are scored.
fn plausible(solved: &Intrinsics, guess: &Intrinsics, options: &ResectionOptions) -> bool {
    if !solved.fx.is_finite()
        || !solved.fy.is_finite()
        || !solved.cx.is_finite()
        || !solved.cy.is_finite()
        || solved.fx <= 0.0
        || solved.fy <= 0.0
    {
        return false;
    }

    let aspect = solved.fx / solved.fy;
    if (aspect - 1.0).abs() > options.max_aspect_error {
        return false;
    }

    let width = guess.width as f64;
    let height = guess.height as f64;
    (solved.cx - width * 0.5).abs() <= width * options.max_principal_offset
        && (solved.cy - height * 0.5).abs() <= height * options.max_principal_offset
}

/// The direct linear transform: the 3x4 projection matrix taking these world
/// points to these pixels.
fn dlt(world: &[Point3<f64>], pixel: &[Point2<f64>]) -> Option<Matrix3x4<f64>> {
    if world.len() < SAMPLE {
        return None;
    }

    // Both sets are centred and scaled first. Without it the system mixes
    // metres with pixels and the solve loses most of its precision.
    let t = normalizer_2d(pixel);
    let u = normalizer_3d(world);

    let mut normal = SMatrix::<f64, 12, 12>::zeros();
    for (point, observed) in world.iter().zip(pixel) {
        let w = u * point.to_homogeneous();
        let p = t * observed.to_homogeneous();
        let (x, y) = (p.x, p.y);

        let rows = [
            [
                -w.x,
                -w.y,
                -w.z,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                x * w.x,
                x * w.y,
                x * w.z,
                x,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                -w.x,
                -w.y,
                -w.z,
                -1.0,
                y * w.x,
                y * w.y,
                y * w.z,
                y,
            ],
        ];
        for row in rows {
            let row = SMatrix::<f64, 1, 12>::from_row_slice(&row);
            normal += row.transpose() * row;
        }
    }

    // The solution is the null space of the accumulated system: the eigenvector
    // belonging to the smallest eigenvalue.
    let eigen = normal.symmetric_eigen();
    let (index, _) = eigen
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))?;

    let column = eigen.eigenvectors.column(index);
    let normalized = Matrix3x4::from_row_slice(column.as_slice());
    let projection = t.try_inverse()? * normalized * u;

    projection
        .iter()
        .all(|value| value.is_finite())
        .then_some(projection)
}

/// Splits a projection matrix into intrinsics, rotation and translation.
///
/// `P = K [R | t]`, where `K` is upper triangular with a positive diagonal and
/// `R` is a rotation. That is an RQ decomposition, done here with the three
/// Givens rotations that zero the lower triangle in turn.
fn decompose(projection: &Matrix3x4<f64>) -> Option<(Matrix3<f64>, Matrix3<f64>, Vector3<f64>)> {
    let mut projection = *projection;

    // Fix the overall sign first. A negative determinant means the solve
    // produced a mirrored camera, and no amount of later sign juggling turns
    // that back into a rotation.
    if projection.fixed_view::<3, 3>(0, 0).determinant() < 0.0 {
        projection = -projection;
    }

    let m = projection.fixed_view::<3, 3>(0, 0).into_owned();
    let (mut k, r) = rq(&m)?;

    // K is homogeneous; normalizing its corner puts the focal lengths in
    // pixels.
    let scale = k[(2, 2)];
    if scale.abs() < 1e-12 {
        return None;
    }
    k /= scale;
    projection /= scale;

    let translation = k.try_inverse()? * projection.fixed_view::<3, 1>(0, 3);
    if !translation.iter().all(|value| value.is_finite()) {
        return None;
    }

    Some((k, r, translation.into_owned()))
}

/// RQ decomposition of a 3x3 matrix into an upper triangular factor with a
/// positive diagonal and an orthogonal one.
///
/// The positive diagonal is what makes the result unique, and it is also what
/// the intrinsics need: a negative focal length describes the same camera, but
/// nothing downstream expects one.
fn rq(m: &Matrix3<f64>) -> Option<(Matrix3<f64>, Matrix3<f64>)> {
    let mut a = *m;
    let hypot = |x: f64, y: f64| (x * x + y * y).sqrt();

    // Rotate about x until the (2, 1) entry vanishes.
    let d = hypot(a[(2, 2)], a[(2, 1)]);
    if d < 1e-12 {
        return None;
    }
    let (c, s) = (-a[(2, 2)] / d, a[(2, 1)] / d);
    let qx = Matrix3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c);
    a *= qx;

    // Then about y for the (2, 0) entry.
    let d = hypot(a[(2, 2)], a[(2, 0)]);
    if d < 1e-12 {
        return None;
    }
    let (c, s) = (a[(2, 2)] / d, a[(2, 0)] / d);
    let qy = Matrix3::new(c, 0.0, s, 0.0, 1.0, 0.0, -s, 0.0, c);
    a *= qy;

    // And about z for the (1, 0) entry, which leaves A upper triangular.
    let d = hypot(a[(1, 1)], a[(1, 0)]);
    if d < 1e-12 {
        return None;
    }
    let (c, s) = (-a[(1, 1)] / d, a[(1, 0)] / d);
    let qz = Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0);
    a *= qz;

    let mut k = a;
    let mut r = (qx * qy * qz).transpose();

    // A negative diagonal entry is absorbed by flipping the matching axis in
    // both factors, which leaves their product unchanged.
    for axis in 0..3 {
        if k[(axis, axis)] < 0.0 {
            for row in 0..3 {
                k[(row, axis)] = -k[(row, axis)];
            }
            for column in 0..3 {
                r[(axis, column)] = -r[(axis, column)];
            }
        }
    }

    Some((k, r))
}

/// How far a set of points is from lying in a plane, as the ratio of the
/// smallest to the largest principal extent.
fn planarity(points: &[Point3<f64>]) -> f64 {
    if points.len() < 4 {
        return 0.0;
    }

    let count = points.len() as f64;
    let centroid = points
        .iter()
        .fold(Vector3::zeros(), |sum, p| sum + p.coords)
        / count;

    let mut covariance = Matrix3::zeros();
    for point in points {
        let d = point.coords - centroid;
        covariance += d * d.transpose();
    }
    covariance /= count;

    let eigen = covariance.symmetric_eigen();
    let smallest = eigen
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let largest = eigen.eigenvalues.iter().copied().fold(0.0, f64::max);
    if largest <= 1e-15 {
        return 0.0;
    }
    (smallest.max(0.0) / largest).sqrt()
}

fn normalizer_2d(points: &[Point2<f64>]) -> Matrix3<f64> {
    let count = points.len() as f64;
    let cx = points.iter().map(|p| p.x).sum::<f64>() / count;
    let cy = points.iter().map(|p| p.y).sum::<f64>() / count;
    let mean = points
        .iter()
        .map(|p| ((p.x - cx).powi(2) + (p.y - cy).powi(2)).sqrt())
        .sum::<f64>()
        / count;

    let s = if mean > 1e-12 {
        2f64.sqrt() / mean
    } else {
        1.0
    };
    Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0)
}

fn normalizer_3d(points: &[Point3<f64>]) -> Matrix4<f64> {
    let count = points.len() as f64;
    let centroid = points
        .iter()
        .fold(Vector3::zeros(), |sum, p| sum + p.coords)
        / count;
    let mean = points
        .iter()
        .map(|p| (p.coords - centroid).norm())
        .sum::<f64>()
        / count;

    let s = if mean > 1e-12 {
        3f64.sqrt() / mean
    } else {
        1.0
    };

    let mut m = Matrix4::identity();
    m[(0, 0)] = s;
    m[(1, 1)] = s;
    m[(2, 2)] = s;
    m[(0, 3)] = -s * centroid.x;
    m[(1, 3)] = -s * centroid.y;
    m[(2, 3)] = -s * centroid.z;
    m
}

/// A small deterministic generator, so a calibration run gives the same answer
/// twice. SplitMix64.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

/// Draws distinct indices for one hypothesis.
fn draw(rng: &mut Rng, pool: &[usize], out: &mut Vec<usize>) -> bool {
    out.clear();
    for _ in 0..SAMPLE * 8 {
        if out.len() == SAMPLE {
            break;
        }
        let candidate = pool[rng.below(pool.len())];
        if !out.contains(&candidate) {
            out.push(candidate);
        }
    }
    out.len() == SAMPLE
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intrinsics() -> Intrinsics {
        Intrinsics::from_fov(1280, 720, 72f64.to_radians())
    }

    /// A camera in a ceiling corner, looking down at the middle of the room.
    fn truth(lens: Lens) -> Camera {
        Camera::look_at(
            intrinsics(),
            lens,
            Point3::new(-1.8, 2.4, -1.8),
            Point3::new(0.0, 1.0, 0.0),
            Vector3::y(),
        )
    }

    /// A calibration walk: a path through the room that changes height, so the
    /// correspondences fill a volume rather than a plane.
    fn walk() -> Vec<Point3<f64>> {
        (0..160)
            .map(|step| {
                let t = step as f64 * 0.13;
                Point3::new(
                    1.3 * (t).sin(),
                    1.15 + 0.4 * (1.9 * t).sin(),
                    1.1 * (0.7 * t).cos(),
                )
            })
            .collect()
    }

    fn seen_by(camera: &Camera, points: &[Point3<f64>]) -> Vec<Correspondence> {
        points
            .iter()
            .filter_map(|world| {
                let pixel = camera.project(*world)?;
                let inside = pixel.x >= 0.0
                    && pixel.y >= 0.0
                    && pixel.x < camera.intrinsics.width as f64
                    && pixel.y < camera.intrinsics.height as f64;
                inside.then_some(Correspondence {
                    world: *world,
                    pixel,
                })
            })
            .collect()
    }

    /// The focal guess is deliberately far off: the solver is supposed to
    /// recover it, not be told it.
    fn guess() -> Intrinsics {
        Intrinsics::from_fov(1280, 720, 100f64.to_radians())
    }

    #[test]
    fn a_clean_walk_recovers_the_camera_exactly() {
        let truth = truth(Lens::default());
        let points = seen_by(&truth, &walk());
        assert!(points.len() > 40, "the test walk should be visible");

        let result = resect(
            &guess(),
            Lens::default(),
            &points,
            &ResectionOptions::default(),
        )
        .expect("a clean walk should solve");

        assert!(
            (result.camera.position() - truth.position()).norm() < 1e-6,
            "recovered {:?}, expected {:?}",
            result.camera.position(),
            truth.position()
        );
        assert!(
            result.camera.forward().angle(&truth.forward()) < 1e-6,
            "the camera should point where it was pointed"
        );
        assert!(
            (result.camera.intrinsics.fx - truth.intrinsics.fx).abs() < 1e-3,
            "recovered fx {} against {}",
            result.camera.intrinsics.fx,
            truth.intrinsics.fx
        );
        assert_eq!(result.inliers.len(), points.len());
        assert!(result.rms < 1e-7, "rms was {}", result.rms);
        assert!(result.is_well_conditioned());
    }

    /// A mis-detected keypoint is a confident observation in the wrong place.
    /// Without RANSAC one of them is enough to move the camera metres.
    #[test]
    fn bad_detections_are_rejected() {
        let truth = truth(Lens::default());
        let mut points = seen_by(&truth, &walk());
        let clean = points.len();

        for (index, point) in points.iter_mut().enumerate() {
            if index % 5 == 0 {
                point.pixel.x += 130.0 - (index % 3) as f64 * 90.0;
                point.pixel.y -= 70.0 + (index % 4) as f64 * 40.0;
            }
        }

        let result = resect(
            &guess(),
            Lens::default(),
            &points,
            &ResectionOptions::default(),
        )
        .expect("four fifths of the walk is still good");

        assert!(
            (result.camera.position() - truth.position()).norm() < 1e-3,
            "the outliers moved the camera to {:?}",
            result.camera.position()
        );
        let corrupted = clean.div_ceil(5);
        assert!(
            result.inliers.len() >= clean - corrupted,
            "kept {} of {} good correspondences",
            result.inliers.len(),
            clean - corrupted
        );
    }

    #[test]
    fn a_distorted_lens_still_resolves() {
        let lens = Lens::RadialTangential {
            k1: -0.25,
            k2: 0.08,
            p1: 0.0004,
            p2: -0.0006,
        };
        let truth = truth(lens);
        let points = seen_by(&truth, &walk());

        let result = resect(&guess(), lens, &points, &ResectionOptions::default())
            .expect("a known lens should solve");

        assert!(
            (result.camera.position() - truth.position()).norm() < 5e-3,
            "recovered {:?}, expected {:?}",
            result.camera.position(),
            truth.position()
        );
        assert!(result.rms < 1e-3, "rms was {}", result.rms);
    }

    /// A walk that never changes height leaves the points on a plane, where the
    /// linear solve has no unique answer. That has to be reported, not returned
    /// as a confident wrong camera.
    #[test]
    fn a_flat_walk_is_reported_as_degenerate() {
        let truth = truth(Lens::default());
        let flat: Vec<Point3<f64>> = (0..160)
            .map(|step| {
                let t = step as f64 * 0.13;
                Point3::new(1.3 * t.sin(), 1.2, 1.1 * (0.7 * t).cos())
            })
            .collect();
        let points = seen_by(&truth, &flat);

        let result = resect(
            &guess(),
            Lens::default(),
            &points,
            &ResectionOptions::default(),
        );
        assert!(
            result.is_none_or(|r| !r.is_well_conditioned()),
            "a planar walk must not pass as a good calibration"
        );
    }

    #[test]
    fn too_few_correspondences_do_not_solve() {
        let truth = truth(Lens::default());
        let points = seen_by(&truth, &walk());
        assert!(
            resect(
                &guess(),
                Lens::default(),
                &points[..5],
                &ResectionOptions::default()
            )
            .is_none()
        );
    }

    #[test]
    fn the_solve_is_reproducible() {
        let truth = truth(Lens::default());
        let points = seen_by(&truth, &walk());
        let options = ResectionOptions::default();

        let first = resect(&guess(), Lens::default(), &points, &options).unwrap();
        let second = resect(&guess(), Lens::default(), &points, &options).unwrap();
        assert_eq!(first.camera.pose, second.camera.pose);
    }

    #[test]
    fn rq_splits_a_product_back_into_its_factors() {
        let k = Matrix3::new(900.0, 0.0, 640.0, 0.0, 880.0, 360.0, 0.0, 0.0, 1.0);
        let r = Rotation3::from_euler_angles(0.3, -0.7, 1.1)
            .matrix()
            .into_owned();

        let (recovered_k, recovered_r) = rq(&(k * r)).expect("a product of the two splits");

        let scaled = recovered_k / recovered_k[(2, 2)];
        assert!((scaled - k).norm() < 1e-9, "recovered {scaled}");
        assert!((recovered_r - r).norm() < 1e-9);
    }
}
