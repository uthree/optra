//! Smoothing what is left, and predicting past the delay.
//!
//! Two different problems that are easy to confuse. The fit upstream removed
//! the part of the error that was anatomically impossible; what remains is
//! ordinary noise, in directions the joint really could have moved, and only
//! time can tell it apart from motion. That is smoothing, and it costs lag.
//!
//! The other problem is that everything here is late. A frame is exposed,
//! travels over USB, is decoded, run through two networks, fused and sent — and
//! the user's foot has been somewhere else for fifty milliseconds by the time
//! anything renders. No amount of smoothing helps, because the answer is not
//! wrong, only old. It has to be extrapolated forward.
//!
//! So the joint carries a velocity as well as a position. A constant-velocity
//! Kalman filter estimates both, weighting each measurement by the uncertainty
//! the triangulation reported, and that velocity is what the prediction runs
//! on. A One Euro filter then takes the residual jitter out of the position.
//!
//! The order matters and is the reverse of the obvious one. Smoothing first and
//! differentiating afterwards would measure the velocity of the smoothed signal,
//! which lags the real one and would leave every prediction short. Estimating
//! the velocity first keeps it honest.
//!
//! The smoothing then costs no latency at all, which is worth being precise
//! about. A first-order low pass running on something moving steadily sits
//! exactly its own time constant behind — a known quantity, not an unknown
//! one — so that time constant is added to the prediction horizon and paid
//! straight back. What the smoothing still costs is how quickly the output can
//! follow a genuine change of direction, and that is the trade actually being
//! made.

use std::time::{Duration, Instant};

use nalgebra::{Matrix2, Point3, Vector2, Vector3};

use crate::models::Joint;

use super::fit::Fitted;

#[derive(Debug, Clone)]
pub struct FilterOptions {
    /// Lowest cutoff the position filter will use, in hertz. This is what a
    /// motionless joint is smoothed at, and lowering it trades responsiveness
    /// for stillness.
    pub min_cutoff: f64,
    /// How fast the cutoff opens up with speed, in hertz per metre per second.
    ///
    /// The whole idea of the One Euro filter: a still joint is smoothed hard
    /// because lag is invisible when nothing is moving, and a moving one is
    /// barely smoothed because lag is all that is visible then.
    pub beta: f64,
    /// Acceleration the constant-velocity model expects to be surprised by, in
    /// metres per second squared.
    ///
    /// Too low and the filter refuses to believe a step; too high and it
    /// follows noise. A foot changes direction over roughly a tenth of a second
    /// at a couple of metres per second.
    pub agility: f64,
    /// How far ahead to predict, which should be the end-to-end delay from
    /// exposure to the consumer seeing it.
    pub horizon: Duration,
    /// Largest distance a prediction may extrapolate, in metres.
    ///
    /// A velocity estimate that has gone wrong should show as a foot that
    /// stopped tracking, not as one that left the room.
    pub max_prediction: f64,
    /// How long a joint may be missing before its filter is discarded rather
    /// than resumed.
    pub patience: Duration,
}

impl Default for FilterOptions {
    fn default() -> Self {
        Self {
            min_cutoff: 1.2,
            beta: 0.5,
            agility: 8.0,
            horizon: Duration::from_millis(60),
            max_prediction: 0.35,
            patience: Duration::from_millis(400),
        }
    }
}

/// One joint after filtering.
#[derive(Debug, Clone, Copy)]
pub struct FilteredJoint {
    /// Smoothed position at the pose's own instant.
    pub point: Point3<f64>,
    /// Estimated velocity, in metres per second.
    pub velocity: Vector3<f64>,
    /// Where the joint is expected to be one horizon from now. This is what
    /// goes out to the trackers.
    pub predicted: Point3<f64>,
    /// Uncertainty the measurement came in with, in metres.
    pub sigma: f64,
    /// True when the fit placed this joint rather than the cameras seeing it.
    pub inferred: bool,
}

impl FilteredJoint {
    pub fn speed(&self) -> f64 {
        self.velocity.norm()
    }
}

/// A whole body, filtered and predicted.
#[derive(Debug, Clone)]
pub struct Filtered {
    /// The instant the measurements describe.
    pub at: Instant,
    /// How far ahead `predicted` looks.
    pub horizon: Duration,
    joints: Vec<Option<FilteredJoint>>,
}

impl Filtered {
    pub fn empty(at: Instant, horizon: Duration) -> Self {
        Self {
            at,
            horizon,
            joints: (0..Joint::ALL.len()).map(|_| None).collect(),
        }
    }

    pub fn get(&self, joint: Joint) -> Option<FilteredJoint> {
        self.joints[joint.index()]
    }

    pub fn position(&self, joint: Joint) -> Option<Point3<f64>> {
        self.get(joint).map(|filtered| filtered.point)
    }

    pub fn predicted(&self, joint: Joint) -> Option<Point3<f64>> {
        self.get(joint).map(|filtered| filtered.predicted)
    }

    pub fn iter(&self) -> impl Iterator<Item = (Joint, FilteredJoint)> + '_ {
        Joint::ALL
            .iter()
            .filter_map(|joint| self.get(*joint).map(|filtered| (*joint, filtered)))
    }

    pub fn count(&self) -> usize {
        self.joints.iter().filter(|joint| joint.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }
}

/// Filters a stream of fitted poses.
#[derive(Debug, Clone, Default)]
pub struct PoseFilter {
    options: FilterOptions,
    tracks: Vec<Option<Track>>,
}

impl PoseFilter {
    pub fn new(options: FilterOptions) -> Self {
        Self {
            options,
            tracks: Vec::new(),
        }
    }

    pub fn options(&self) -> &FilterOptions {
        &self.options
    }

    pub fn set_options(&mut self, options: FilterOptions) {
        self.options = options;
    }

    /// Drops every joint's state, so nothing carries across a break.
    pub fn reset(&mut self) {
        self.tracks.clear();
    }

    pub fn push(&mut self, fitted: &Fitted) -> Filtered {
        if self.tracks.len() != Joint::ALL.len() {
            self.tracks = (0..Joint::ALL.len()).map(|_| None).collect();
        }

        let mut out = Filtered::empty(fitted.at, self.options.horizon);

        for joint in Joint::ALL {
            let slot = &mut self.tracks[joint.index()];

            let Some(measured) = fitted.get(joint) else {
                // A joint that has been gone a while is not the same joint when
                // it returns: the user has moved, and resuming a stale velocity
                // would throw the prediction across the room.
                if slot.as_ref().is_some_and(|track| {
                    fitted.at.saturating_duration_since(track.at) > self.options.patience
                }) {
                    *slot = None;
                }
                continue;
            };

            let track = match slot {
                Some(track) => track,
                none => none.insert(Track::new(measured.point, fitted.at)),
            };

            out.joints[joint.index()] = Some(track.step(
                measured.point,
                measured.sigma,
                fitted.at,
                measured.inferred,
                &self.options,
            ));
        }

        out
    }
}

/// One joint's filter state.
#[derive(Debug, Clone)]
struct Track {
    axes: [Kalman; 3],
    /// Smoothed position, which trails the Kalman's own estimate.
    smoothed: Point3<f64>,
    at: Instant,
}

impl Track {
    fn new(point: Point3<f64>, at: Instant) -> Self {
        Self {
            axes: [
                Kalman::new(point.x),
                Kalman::new(point.y),
                Kalman::new(point.z),
            ],
            smoothed: point,
            at,
        }
    }

    fn step(
        &mut self,
        measured: Point3<f64>,
        sigma: f64,
        at: Instant,
        inferred: bool,
        options: &FilterOptions,
    ) -> FilteredJoint {
        let dt = at.saturating_duration_since(self.at).as_secs_f64();
        // Two poses stamped at the same instant would divide by zero, and a
        // clock that ran backwards is not a negative time step.
        let dt = if dt > 1e-6 { dt } else { 1.0 / 60.0 };
        self.at = at;

        let variance = (sigma.max(1e-4)).powi(2);
        for (axis, measurement) in self
            .axes
            .iter_mut()
            .zip([measured.x, measured.y, measured.z])
        {
            axis.predict(dt, options.agility);
            axis.update(measurement, variance);
        }

        let estimate = Point3::new(self.axes[0].x, self.axes[1].x, self.axes[2].x);
        let velocity = Vector3::new(self.axes[0].v, self.axes[1].v, self.axes[2].v);

        // The speed the cutoff opens up with comes from the Kalman rather than
        // from a low-passed difference, which is what the One Euro filter would
        // normally use. It is the same quantity, already estimated better.
        let cutoff = options.min_cutoff + options.beta * velocity.norm();
        let tau = time_constant(cutoff);
        self.smoothed += (estimate - self.smoothed) * smoothing_factor(tau, dt);

        // The smoothing costs lag, and a first-order low pass costs exactly its
        // own time constant of it. That is not something to accept: it is a
        // known quantity, so it is simply added to the horizon. The prediction
        // then lands where the joint will be, and the smoothing costs nothing in
        // latency — only in how quickly the output can follow a real change of
        // direction.
        let step = velocity * (options.horizon.as_secs_f64() + tau);
        let distance = step.norm();
        let step = if distance > options.max_prediction {
            step * (options.max_prediction / distance)
        } else {
            step
        };

        FilteredJoint {
            point: self.smoothed,
            velocity,
            predicted: self.smoothed + step,
            sigma,
            inferred,
        }
    }
}

/// Time constant of a first-order low pass at `cutoff` hertz, in seconds.
///
/// Also exactly how far behind that filter runs on a signal moving at a
/// constant rate, which is what makes the lag correctable rather than a cost.
fn time_constant(cutoff: f64) -> f64 {
    1.0 / (std::f64::consts::TAU * cutoff.max(1e-3))
}

/// How much of the new value to take, sampling `dt` seconds apart.
fn smoothing_factor(tau: f64, dt: f64) -> f64 {
    (dt / (tau + dt)).clamp(0.0, 1.0)
}

/// A constant-velocity Kalman filter along one axis.
///
/// Position and velocity, with the measurement variance supplied per frame from
/// the triangulation. That is the part worth having: a joint two cameras
/// suddenly disagree about is trusted less that frame, automatically, rather
/// than at a rate chosen in advance.
#[derive(Debug, Clone, Copy)]
struct Kalman {
    x: f64,
    v: f64,
    covariance: Matrix2<f64>,
}

impl Kalman {
    fn new(x: f64) -> Self {
        Self {
            x,
            v: 0.0,
            // The position is known about as well as the first measurement; the
            // velocity is not known at all, and saying so is what lets the
            // second measurement set it rather than being averaged into zero.
            covariance: Matrix2::new(0.01, 0.0, 0.0, 4.0),
        }
    }

    fn predict(&mut self, dt: f64, agility: f64) {
        self.x += self.v * dt;

        let transition = Matrix2::new(1.0, dt, 0.0, 1.0);
        // White acceleration noise of the given spectral density, integrated
        // over the step. This is the standard form and the reason `agility` is
        // expressed as an acceleration rather than as a tuning number.
        let q = agility * agility;
        let process = Matrix2::new(
            q * dt * dt * dt / 3.0,
            q * dt * dt / 2.0,
            q * dt * dt / 2.0,
            q * dt,
        );
        self.covariance = transition * self.covariance * transition.transpose() + process;
    }

    fn update(&mut self, measurement: f64, variance: f64) {
        let innovation = self.covariance[(0, 0)] + variance;
        if innovation < 1e-12 {
            return;
        }

        let gain = Vector2::new(
            self.covariance[(0, 0)] / innovation,
            self.covariance[(1, 0)] / innovation,
        );
        let residual = measurement - self.x;
        self.x += gain.x * residual;
        self.v += gain.y * residual;

        let update = Matrix2::new(1.0 - gain.x, 0.0, -gain.y, 1.0);
        self.covariance = update * self.covariance;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fusion::fit::FittedJoint;

    /// A repeatable jitter, so the tests mean the same thing every run.
    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        }
    }

    fn pose(at: Instant, point: Point3<f64>, sigma: f64) -> Fitted {
        let mut fitted = Fitted::empty(at);
        fitted.set(
            Joint::LeftAnkle,
            FittedJoint {
                point,
                sigma,
                inferred: false,
                correction: 0.0,
            },
        );
        fitted
    }

    const STEP: Duration = Duration::from_millis(1000 / 60);

    #[test]
    fn a_still_joint_comes_out_much_stiller() {
        let mut filter = PoseFilter::default();
        let mut noise = Noise(0xB0A7);
        let start = Instant::now();
        let truth = Point3::new(0.1, 0.09, -0.2);

        let mut raw = 0.0;
        let mut smoothed = 0.0;
        let mut counted = 0;

        for step in 0..300 {
            let jitter = Vector3::new(noise.next(), noise.next(), noise.next()) * 0.02;
            let measured = truth + jitter;
            let out = filter.push(&pose(start + STEP * step, measured, 0.01));

            // Give the filter a moment to settle before judging it.
            if step > 60 {
                raw += (measured - truth).norm_squared();
                smoothed += (out.position(Joint::LeftAnkle).unwrap() - truth).norm_squared();
                counted += 1;
            }
        }

        let raw = (raw / counted as f64).sqrt();
        let smoothed = (smoothed / counted as f64).sqrt();
        assert!(
            smoothed < raw / 3.0,
            "the filter left {smoothed:.4} m of {raw:.4} m"
        );
    }

    /// The point of estimating velocity: a joint moving steadily should be
    /// predicted where it is going, not where it was.
    #[test]
    fn a_steady_joint_is_predicted_where_it_is_going() {
        let options = FilterOptions::default();
        let mut filter = PoseFilter::new(options.clone());
        let start = Instant::now();
        let speed = Vector3::new(0.0, 0.0, -1.4);

        let mut last = None;
        for step in 0..180 {
            let at = start + STEP * step;
            let truth = Point3::new(0.1, 0.09, 0.0) + speed * (STEP * step).as_secs_f64();
            last = Some((filter.push(&pose(at, truth, 0.008)), truth));
        }

        let (filtered, truth) = last.unwrap();
        let joint = filtered.get(Joint::LeftAnkle).unwrap();

        assert!(
            (joint.velocity - speed).norm() < 0.05,
            "velocity came out as {:?}",
            joint.velocity
        );

        // The smoothing really does lag — that is what smoothing is — and the
        // prediction has to be paying it back rather than starting from it.
        assert!(
            (joint.point - truth).norm() > 0.05,
            "the smoothing was expected to lag, and did not"
        );

        let expected = truth + speed * options.horizon.as_secs_f64();
        assert!(
            (joint.predicted - expected).norm() < 0.02,
            "predicted {:?}, should be near {expected:?}",
            joint.predicted
        );
    }

    /// A prediction is only as good as the velocity behind it. When that goes
    /// wrong, the foot should stop rather than leave the room.
    #[test]
    fn a_runaway_prediction_is_clamped() {
        let options = FilterOptions {
            max_prediction: 0.1,
            ..FilterOptions::default()
        };
        let mut filter = PoseFilter::new(options.clone());
        let start = Instant::now();

        let mut last = None;
        for step in 0..60 {
            // Absurdly fast, as a mis-tracked limb would be.
            let truth = Point3::new(0.0, 0.5, -20.0 * (STEP * step).as_secs_f64());
            last = Some(filter.push(&pose(start + STEP * step, truth, 0.01)));
        }

        let joint = last.unwrap().get(Joint::LeftAnkle).unwrap();
        assert!(
            (joint.predicted - joint.point).norm() <= options.max_prediction + 1e-9,
            "the prediction reached {} m",
            (joint.predicted - joint.point).norm()
        );
    }

    /// The measurement uncertainty is supplied per frame, and a frame the
    /// cameras disagreed about should move the answer less.
    #[test]
    fn an_uncertain_measurement_moves_the_answer_less() {
        let start = Instant::now();
        let settled = Point3::new(0.0, 0.09, 0.0);

        let mut moved = Vec::new();
        for sigma in [0.002, 0.2] {
            let mut filter = PoseFilter::default();
            for step in 0..120 {
                filter.push(&pose(start + STEP * step, settled, 0.002));
            }
            // One frame that says the joint jumped ten centimetres.
            let jumped = filter.push(&pose(
                start + STEP * 120,
                settled + Vector3::new(0.1, 0.0, 0.0),
                sigma,
            ));
            moved.push((jumped.position(Joint::LeftAnkle).unwrap() - settled).norm());
        }

        assert!(
            moved[0] > 5.0 * moved[1],
            "confident frame moved {:.4} m, uncertain one {:.4} m",
            moved[0],
            moved[1]
        );
    }

    /// A joint that vanishes for a moment should carry on where it left off; one
    /// that vanishes for a long time is somewhere else entirely by the time it
    /// returns, and resuming its velocity would fling the prediction.
    #[test]
    fn a_long_absence_forgets_the_joint() {
        let options = FilterOptions::default();
        let mut filter = PoseFilter::new(options.clone());
        let start = Instant::now();

        for step in 0..60 {
            let truth = Point3::new(0.0, 0.09, -2.0 * (STEP * step).as_secs_f64());
            filter.push(&pose(start + STEP * step, truth, 0.01));
        }

        // Gone for well past the patience, then back somewhere else.
        let gone = start + STEP * 60 + options.patience + Duration::from_millis(200);
        filter.push(&Fitted::empty(gone));

        let back = filter.push(&pose(gone + STEP, Point3::new(1.0, 0.09, 1.0), 0.01));
        let joint = back.get(Joint::LeftAnkle).unwrap();

        assert!(
            (joint.point - Point3::new(1.0, 0.09, 1.0)).norm() < 1e-9,
            "it resumed at {:?} instead of where it reappeared",
            joint.point
        );
        assert_eq!(joint.velocity, Vector3::zeros());
    }

    #[test]
    fn a_brief_absence_is_carried_through() {
        let mut filter = PoseFilter::default();
        let start = Instant::now();

        for step in 0..60 {
            filter.push(&pose(
                start + STEP * step,
                Point3::new(0.0, 0.09, -2.0 * (STEP * step).as_secs_f64()),
                0.01,
            ));
        }
        filter.push(&Fitted::empty(start + STEP * 61));

        let resumed = filter.push(&pose(
            start + STEP * 62,
            Point3::new(0.0, 0.09, -2.0 * (STEP * 62).as_secs_f64()),
            0.01,
        ));
        let joint = resumed.get(Joint::LeftAnkle).unwrap();

        assert!(
            (joint.velocity.z + 2.0).abs() < 0.2,
            "the velocity was lost across the gap: {:?}",
            joint.velocity
        );
    }

    /// A moving joint must not be smoothed as hard as a still one, or every
    /// step lags. This is the whole reason the cutoff is adaptive.
    #[test]
    fn a_moving_joint_is_smoothed_less_than_a_still_one() {
        let dt = 1.0 / 60.0;
        let options = FilterOptions::default();

        let still = smoothing_factor(time_constant(options.min_cutoff), dt);
        let walking = smoothing_factor(time_constant(options.min_cutoff + options.beta * 2.0), dt);

        assert!(
            walking > 1.5 * still,
            "still {still:.3} against walking {walking:.3}"
        );
    }
}
