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

use crate::models::{Joint, JointMap};

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
    /// Cutoff of the low pass on the speed that opens the position filter, in
    /// hertz.
    ///
    /// The speed estimate is itself noisy, and a joint standing still shows an
    /// apparent speed that never quite reaches zero. Feeding that straight into
    /// the cutoff would open the filter exactly when there is nothing to track
    /// and everything to smooth, so the *velocity* is low-passed first —
    /// vector, not magnitude, since zero-mean noise only averages away before
    /// the absolute value is taken.
    ///
    /// Higher than the one hertz the One Euro filter is usually given, because
    /// a walking leg swings at over one and a half. At a hertz the speed of a
    /// swinging foot is attenuated to the point that the filter never notices
    /// it is moving, and every stride comes out lagged.
    pub derivative_cutoff: f64,
    /// Acceleration the constant-velocity model expects to be surprised by, in
    /// metres per second squared.
    ///
    /// Too low and the filter refuses to believe a step; too high and it
    /// follows noise. A foot changes direction over roughly a tenth of a second
    /// at a couple of metres per second.
    ///
    /// It also sets the floor under how well the velocity can ever be known,
    /// which is what [`caution`](Self::caution) is measured against: five
    /// metres per second squared at sixty hertz leaves rather less uncertainty
    /// than the same figure at twenty, so the two are worth reading together.
    pub agility: f64,
    /// How much the prediction holds back on a speed it is unsure of, from one
    /// for the full amount down to zero for none.
    ///
    /// The velocity estimate never reaches zero on a joint standing still — it
    /// wanders around its own noise floor — and a prediction that acts on every
    /// wander is how a body smoothed to a little over a hertz reached VRChat
    /// vibrating. So the velocity is weighed against how well it is known
    /// before it is allowed to move anything, and this is how hard.
    ///
    /// Where it should sit depends on the room rather than on this code. The
    /// noise floor it is defending against comes from the cameras and the pose
    /// model — a sharp model on a well-lit 1080p camera leaves a velocity worth
    /// acting on where a 480p webcam leaves noise — and it is traded against
    /// latency, since holding a speed back means arriving late with it. What
    /// the two look like is on the Tracking panel: `shake` for the cost of too
    /// little, and how much of the measured speed the prediction is reaching
    /// for the cost of too much.
    pub caution: f64,
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
    /// Gap in a joint's measurements past which its velocity is thrown away
    /// rather than inferred across the gap.
    ///
    /// A constant-velocity filter given a measurement after a long silence
    /// divides the whole distance the joint travelled by the whole time it was
    /// missing, and calls the result a velocity. It is not one: nothing was
    /// observed in between, and the body may have changed direction twice. That
    /// bogus velocity is then multiplied by the prediction horizon, which is
    /// how a foot that was merely occluded ends up thrown across the room.
    ///
    /// A few frames, so that an ordinary dropped frame costs nothing and an
    /// actual occlusion is not guessed through. Zero is the honest claim after
    /// being blind: it says the joint is where it was seen, which is the only
    /// thing that is known.
    pub reacquire: Duration,
}

impl Default for FilterOptions {
    fn default() -> Self {
        Self {
            min_cutoff: 1.2,
            beta: 4.0,
            derivative_cutoff: 3.0,
            agility: 5.0,
            caution: 1.0,
            horizon: Duration::from_millis(60),
            max_prediction: 0.35,
            patience: Duration::from_millis(400),
            reacquire: Duration::from_millis(120),
        }
    }
}

/// One joint after filtering.
#[derive(Debug, Clone, Copy)]
pub struct FilteredJoint {
    /// Smoothed position at the pose's own instant.
    pub point: Point3<f64>,
    /// Velocity the prediction is built from, in metres per second.
    ///
    /// Low-passed, not the Kalman's raw estimate. It is the velocity the system
    /// acts on, so it is the one worth reporting.
    pub velocity: Vector3<f64>,
    /// Where the joint is expected to be one horizon from now. This is what
    /// goes out to the trackers.
    pub predicted: Point3<f64>,
    /// How far past the pose's own instant `predicted` looks, in seconds.
    ///
    /// Not the same as the configured horizon: the smoothing lag being paid
    /// back is added to it, and that varies with how fast this particular joint
    /// is moving. The output stage sends faster than fusion runs and has to
    /// extrapolate the rest of the way itself, which it cannot do without
    /// knowing which instant it is extrapolating *from*.
    pub lead: f64,
    /// Uncertainty the measurement came in with, in metres.
    pub sigma: f64,
    /// True when the fit placed this joint rather than the cameras seeing it.
    pub inferred: bool,
}

impl FilteredJoint {
    pub fn speed(&self) -> f64 {
        self.velocity.norm()
    }

    /// Where this joint is expected to be `ahead` seconds past the pose's own
    /// instant, extrapolating no further than `limit` metres.
    ///
    /// The limit is measured from the smoothed position rather than added to
    /// each step, so extrapolating twice as far never travels twice as far
    /// wrong. A velocity estimate that has gone bad should show as a joint
    /// that stopped tracking, not as one that left the room.
    pub fn extrapolate(&self, ahead: f64, limit: f64) -> Point3<f64> {
        let step = self.velocity * ahead;
        let distance = step.norm();
        let step = if distance > limit {
            step * (limit / distance)
        } else {
            step
        };
        self.point + step
    }
}

/// A whole body, filtered and predicted.
#[derive(Debug, Clone)]
pub struct Filtered {
    /// The instant the measurements describe.
    pub at: Instant,
    /// How far ahead the prediction was asked to look. Each joint's own `lead`
    /// is this plus the smoothing lag it is owed.
    pub horizon: Duration,
    /// How far any joint may be extrapolated from where it was measured, in
    /// metres. Carried here so that a stage predicting further can apply the
    /// same bound rather than inventing its own.
    pub limit: f64,
    /// The share of the lower body's measured speed the prediction acted on,
    /// weighted by that speed. `None` when nothing was moving.
    ///
    /// This is the cost of [`FilterOptions::caution`] made visible. The filter
    /// holds back on a velocity it cannot distinguish from standing still, and
    /// how much that costs depends entirely on how noisy this room's cameras
    /// and pose model are — which is not something this code can know and is
    /// something a user watching the number can. Near one and the prediction is
    /// paying the latency back in full; near zero and the trackers are being
    /// sent a body that is where it was rather than where it is going.
    pub reach: Option<f64>,
    joints: JointMap<FilteredJoint>,
}

impl Filtered {
    pub fn empty(at: Instant, horizon: Duration) -> Self {
        Self {
            at,
            horizon,
            limit: FilterOptions::default().max_prediction,
            reach: None,
            joints: JointMap::default(),
        }
    }

    pub fn get(&self, joint: Joint) -> Option<FilteredJoint> {
        self.joints.copied(joint)
    }

    pub fn position(&self, joint: Joint) -> Option<Point3<f64>> {
        self.get(joint).map(|filtered| filtered.point)
    }

    pub fn predicted(&self, joint: Joint) -> Option<Point3<f64>> {
        self.get(joint).map(|filtered| filtered.predicted)
    }

    pub fn set(&mut self, joint: Joint, filtered: FilteredJoint) {
        self.joints.set(joint, filtered);
    }

    pub fn iter(&self) -> impl Iterator<Item = (Joint, FilteredJoint)> + '_ {
        self.joints
            .iter()
            .map(|(joint, filtered)| (joint, *filtered))
    }

    pub fn count(&self) -> usize {
        self.joints.count()
    }

    pub fn is_empty(&self) -> bool {
        self.joints.is_empty()
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
        out.limit = self.options.max_prediction;
        let (mut measured_speed, mut acted_on_speed) = (0.0, 0.0);

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

            let (filtered, reach) = track.step(
                measured.point,
                measured.sigma,
                fitted.at,
                measured.inferred,
                &self.options,
            );
            out.set(joint, filtered);

            // Over the lower body only. It is the half the trackers are built
            // from, and an arm the cameras are unsure about would otherwise
            // report the feet as timid.
            if joint.is_lower_body() {
                measured_speed += reach.measured;
                acted_on_speed += reach.acted_on;
            }
        }

        out.reach = (measured_speed > 1e-9).then(|| acted_on_speed / measured_speed);
        out
    }
}

/// One joint's filter state.
#[derive(Debug, Clone)]
struct Track {
    axes: [Kalman; 3],
    /// Smoothed position, which trails the Kalman's own estimate.
    smoothed: Point3<f64>,
    /// Velocity low-passed at a fixed three hertz, used only to decide how hard
    /// to smooth. Fast, so that motion starting is noticed quickly.
    steady: Vector3<f64>,
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
            steady: Vector3::zeros(),
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
    ) -> (FilteredJoint, Reach) {
        let gap = at.saturating_duration_since(self.at);
        // A joint that has been out of sight is not resumed across the gap.
        // The Kalman would divide the distance it travelled while invisible by
        // the time it was invisible for and call that a velocity, and the
        // prediction would then act on it — which is a foot flung across the
        // room for what was only an occlusion.
        if gap > options.reacquire {
            *self = Track::new(measured, at);
            return (
                FilteredJoint {
                    point: measured,
                    velocity: Vector3::zeros(),
                    predicted: measured,
                    lead: options.horizon.as_secs_f64(),
                    sigma,
                    inferred,
                },
                Reach::default(),
            );
        }

        let dt = gap.as_secs_f64();
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
        let raw = Vector3::new(self.axes[0].v, self.axes[1].v, self.axes[2].v);

        // The speed the cutoff opens up with comes from the Kalman rather than
        // from a finite difference, which is what the One Euro filter would
        // normally use. It is the same quantity, already estimated better — but
        // it still has to be low-passed before it decides anything, or the
        // noise in it opens the filter on a joint that is standing still. The
        // vector is smoothed and then measured, not the other way round: taking
        // the length first would turn zero-mean noise into a speed that never
        // averages away.
        self.steady +=
            (raw - self.steady) * smoothing_factor(time_constant(options.derivative_cutoff), dt);
        let cutoff = options.min_cutoff + options.beta * self.steady.norm();
        let tau = time_constant(cutoff);
        self.smoothed += (estimate - self.smoothed) * smoothing_factor(tau, dt);

        // The smoothing costs lag, and a first-order low pass costs exactly its
        // own time constant of it. That is not something to accept: it is a
        // known quantity, so it is simply added to the horizon. The prediction
        // then lands where the joint will be, and the smoothing costs nothing in
        // latency — only in how quickly the output can follow a real change of
        // direction.
        // The velocity the prediction runs on is smoothed at the same cutoff
        // the position is, and that is the whole point rather than a detail.
        //
        // What goes out is `smoothed + velocity * (horizon + tau)`. The first
        // term has been low-passed hard; adding an unsmoothed second term to it
        // puts all the noise straight back, scaled by a fifth of a second. And
        // it is worst exactly where it can least be afforded: tau is largest
        // when the joint is *still*, so the term that is meant to pay back a
        // smoothing lag there is no motion to have lagged behind is multiplied
        // by the largest number in the filter. That is how a body smoothed to a
        // little over a hertz reached VRChat vibrating.
        //
        // Filtering the velocity at the position's own cutoff makes the pair
        // coherent: it is then the rate of change of the signal actually being
        // extrapolated. The fixed three-hertz `steady` stays where it is,
        // because the two are doing different jobs — that one has to notice
        // motion *starting*, quickly, and a velocity smoothed at rest cutoffs
        // never would.
        // Low-passing the velocity was tried here and is a trap. It does quiet
        // a still joint, but a low pass lags by its time constant and the error
        // that costs is the acceleration times that lag — so it is worst during
        // a stride, which is the one moment the prediction is earning its keep.
        // Measured on the simulated walk: a velocity smoothed at the position's
        // own cutoff put the prediction 9 cm out, against 5 cm unsmoothed, and
        // that is most of the benefit of predicting at all.
        //
        // What a still joint needs is not a slower velocity but an honest one.
        // Its velocity estimate never reaches zero — it wanders around its own
        // noise floor — and the prediction faithfully acts on every wander. So
        // the velocity is weighed against how well it is known before it is
        // allowed to move anything.
        //
        // The Kalman has carried a velocity variance all along and nothing has
        // ever read it. Subtracting that noise power from the signal power
        // leaves the part of the velocity that is genuinely distinguishable
        // from standing still, which is precisely the question being asked.
        //
        // Nothing is thresholded and nothing snaps: a joint moving well clear
        // of its noise floor passes through untouched, one buried in it scales
        // to nothing, and in between it fades. Crucially it costs no lag, which
        // is what makes it usable during a stride. What it does cost is that a
        // genuinely slow drift goes uncompensated — but a drift slower than the
        // filter can measure is one there was never a prediction to make.
        //
        // Axis by axis rather than on the vector as a whole: a foot travelling
        // along one axis is standing still along the other two, and pooling the
        // three would let the noise it is still in eat the motion it is making.
        //
        // The square root is not decoration. What is observed has the true
        // power plus the noise power in it, so subtracting the noise leaves the
        // true *power*, and the scale that recovers the velocity from it is the
        // root of the ratio. Using the ratio itself takes a second bite out of
        // a velocity that was already correct.
        //
        // Judged on this sample's velocity, not on a running average of it.
        // Averaging was tried, on the reasoning that a swinging leg passes
        // through zero twice a stride and should not be called noise there. It
        // is worse — 7.3 cm against 6.5 on the simulated walk — because the
        // average keeps the gain open at exactly the moments the velocity is
        // small and mostly noise, and lets that noise through. Sample by sample
        // the gain closes wherever the velocity is not worth acting on, which
        // is the whole point.
        let mut velocity = Vector3::zeros();
        for (axis, kalman) in self.axes.iter().enumerate() {
            let power = raw[axis] * raw[axis];
            let noise = options.caution.clamp(0.0, 1.0) * kalman.velocity_variance();
            let credible = (1.0 - noise / power.max(1e-12)).clamp(0.0, 1.0);
            velocity[axis] = raw[axis] * credible.sqrt();
        }

        let lead = options.horizon.as_secs_f64() + tau;
        let mut filtered = FilteredJoint {
            point: self.smoothed,
            velocity,
            predicted: self.smoothed,
            lead,
            sigma,
            inferred,
        };
        filtered.predicted = filtered.extrapolate(lead, options.max_prediction);
        (
            filtered,
            Reach {
                measured: raw.norm(),
                acted_on: velocity.norm(),
            },
        )
    }
}

/// How much of the speed a joint was measured to have the prediction acted on.
///
/// The two numbers are kept apart rather than divided here, because they are
/// summed over the body before the ratio means anything. A joint standing still
/// has a measured speed of almost nothing and acts on almost none of it, which
/// is the filter working; averaging that in as "nought per cent" alongside a
/// swinging foot would report the body as timid whenever most of it was at rest.
/// Weighting by the speed actually measured asks the useful question instead:
/// of the movement there was, how much reached the trackers.
#[derive(Debug, Clone, Copy, Default)]
pub struct Reach {
    /// Speed the Kalman estimated, in metres per second.
    pub measured: f64,
    /// Speed the prediction was allowed to use, in metres per second.
    pub acted_on: f64,
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

    /// How wrong the velocity estimate is expected to be, as a variance.
    ///
    /// The filter has always computed this and nothing has ever read it, which
    /// is a shame, because it is the only thing that can say whether a joint is
    /// moving slowly or standing still and being measured badly.
    fn velocity_variance(&self) -> f64 {
        self.covariance[(1, 1)]
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

    /// The output stage sends the *prediction*, not the smoothed position, and
    /// it reaches further than one horizon because the reconstruction it starts
    /// from is already some way in the past. So the prediction is what has to
    /// be still, and a test that only judges the smoothed position is testing
    /// the wrong end of the filter — which is how a carefully smoothed body
    /// reached VRChat vibrating.
    #[test]
    fn a_still_joint_is_predicted_still() {
        let mut filter = PoseFilter::default();
        let mut noise = Noise(0x5EED);
        let start = Instant::now();
        let truth = Point3::new(0.1, 0.09, -0.2);

        // As far ahead as the output stage really asks for: the fusion lag on
        // top of the configured horizon.
        const AGE: f64 = 0.08;

        let mut raw = 0.0;
        let mut predicted = 0.0;
        let mut counted = 0;

        for step in 0..300 {
            let jitter = Vector3::new(noise.next(), noise.next(), noise.next()) * 0.02;
            let measured = truth + jitter;
            let out = filter.push(&pose(start + STEP * step, measured, 0.02));

            if step > 60 {
                let joint = out.get(Joint::LeftAnkle).unwrap();
                let sent = joint.extrapolate(AGE + joint.lead, out.limit);
                raw += (measured - truth).norm_squared();
                predicted += (sent - truth).norm_squared();
                counted += 1;
            }
        }

        let raw = (raw / counted as f64).sqrt();
        let predicted = (predicted / counted as f64).sqrt();
        assert!(
            predicted < raw / 2.0,
            "a joint that never moved was sent {:.1} mm from where it was, \
             against {:.1} mm of measurement noise",
            predicted * 1000.0,
            raw * 1000.0
        );
    }

    /// A joint the cameras lose for a moment comes back somewhere else. The
    /// distance it covered while invisible is not a velocity — nothing was
    /// watching, and the body may have turned round twice — and multiplying it
    /// by the prediction horizon throws the foot past where it actually is.
    #[test]
    fn a_joint_that_reappears_is_not_flung() {
        let mut filter = PoseFilter::default();
        let start = Instant::now();
        let here = Point3::new(0.1, 0.09, -0.2);
        let there = Point3::new(0.1, 0.09, -0.45);

        for step in 0..60 {
            filter.push(&pose(start + STEP * step, here, 0.01));
        }

        // Gone for a fifth of a second, back a quarter of a metre away.
        let back = filter.push(&pose(start + Duration::from_millis(1200), there, 0.01));
        let joint = back.get(Joint::LeftAnkle).unwrap();

        assert!(
            (joint.point - there).norm() < 1e-9,
            "the joint should be where it was seen, and is at {:?}",
            joint.point
        );
        assert!(
            joint.velocity.norm() < 1e-9,
            "a velocity of {:.2} m/s was inferred across a gap nothing was \
             observed in",
            joint.velocity.norm()
        );
        assert!(
            (joint.predicted - there).norm() < 0.01,
            "the prediction landed {:.0} cm past the only place the joint was \
             actually seen",
            (joint.predicted - there).norm() * 100.0
        );
    }

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

        // Not tighter than this, and deliberately. The measurements here are
        // noiseless, so the filter's velocity is exactly right — but the filter
        // does not know that, and it weighs every velocity against the variance
        // it believes it has before acting on it. Demanding the full 1.4 m/s
        // back would be demanding that it act on a confidence it cannot have on
        // real data, which is the behaviour that reached VRChat vibrating.
        assert!(
            (joint.velocity - speed).norm() < 0.15,
            "velocity came out as {:?}",
            joint.velocity
        );

        // The smoothing really does lag — that is what smoothing is — and the
        // prediction has to be paying it back rather than starting from it.
        assert!(
            (joint.point - truth).norm() > 0.02,
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

    /// Walks a noisy joint at a steady speed and reports what the filter did
    /// with it: how far the prediction reached past the smoothed position, and
    /// how much of the measured speed it acted on.
    fn walked(caution: f64, speed: f64) -> (f64, f64) {
        let mut filter = PoseFilter::new(FilterOptions {
            caution,
            ..FilterOptions::default()
        });
        let mut noise = Noise(0xA11E);
        let start = Instant::now();

        let (mut lead, mut reach, mut counted) = (0.0, 0.0, 0);

        for step in 0..240 {
            let t = STEP.as_secs_f64() * step as f64;
            let jitter = Vector3::new(noise.next(), noise.next(), noise.next()) * 0.01;
            let measured = Point3::new(speed * t, 1.0, 0.0) + jitter;
            let out = filter.push(&pose(start + STEP * step, measured, 0.01));

            // After the Kalman has settled, so this measures the filter rather
            // than its first few frames.
            if step > 120 {
                let joint = out.get(Joint::LeftAnkle).unwrap();
                lead += (joint.predicted - joint.point).norm();
                reach += out.reach.unwrap_or(0.0);
                counted += 1;
            }
        }

        let counted = counted as f64;
        (lead / counted, reach / counted)
    }

    /// The prediction caution does what the panel says it does.
    ///
    /// A setting a user can move has to move the thing its label names, and
    /// this one names two: how far the prediction reaches, and how much of the
    /// measured speed it is allowed to use. The default holds a walking joint
    /// back to a fraction of its own speed, which is the finding that made the
    /// setting worth exposing rather than a number chosen here.
    #[test]
    fn lowering_the_caution_lets_the_prediction_reach_further() {
        // Two speeds, because how much the caution costs is not a constant. It
        // weighs a velocity against a noise floor, so a brisk joint passes
        // through nearly untouched and a slow one is held back hard — which is
        // the behaviour, and also why the panel shows the figure live instead of
        // this file claiming a number for it.
        for speed in [1.0, 0.3] {
            let (cautious_lead, cautious_reach) = walked(1.0, speed);
            let (bold_lead, bold_reach) = walked(0.0, speed);

            println!(
                "at {speed:.1} m/s — caution 1.0: {:.0} mm ahead on {:.0}% of the speed; \
                 caution 0.0: {:.0} mm ahead on {:.0}% of the speed",
                cautious_lead * 1000.0,
                cautious_reach * 100.0,
                bold_lead * 1000.0,
                bold_reach * 100.0
            );

            assert!(
                bold_reach > 0.95,
                "with no caution the prediction should act on the speed it measured, \
                 and at {speed:.1} m/s it acted on {:.0}%",
                bold_reach * 100.0
            );
            assert!(
                cautious_reach < 0.9,
                "the caution held nothing back at {speed:.1} m/s: {:.0}% of the speed \
                 went through",
                cautious_reach * 100.0
            );
            assert!(
                bold_lead > 1.2 * cautious_lead && bold_lead - cautious_lead > 0.01,
                "at {speed:.1} m/s the prediction reached {:.0} mm ahead cautiously and \
                 {:.0} mm boldly, which is not a difference a user could act on",
                cautious_lead * 1000.0,
                bold_lead * 1000.0
            );
        }
    }

    /// And what that costs, which is the other half of the same setting.
    ///
    /// A user turning the caution down to follow their stride is buying it with
    /// stillness, and the panel says so. This is that sentence as a number, so
    /// that the two directions cannot drift apart.
    #[test]
    fn lowering_the_caution_costs_stillness() {
        fn wobble(caution: f64) -> f64 {
            let mut filter = PoseFilter::new(FilterOptions {
                caution,
                ..FilterOptions::default()
            });
            let mut noise = Noise(0x5EED);
            let start = Instant::now();
            let truth = Point3::new(0.1, 0.09, -0.2);

            let (mut sum, mut counted) = (0.0, 0);
            for step in 0..300 {
                let jitter = Vector3::new(noise.next(), noise.next(), noise.next()) * 0.02;
                let out = filter.push(&pose(start + STEP * step, truth + jitter, 0.02));
                if step > 60 {
                    let joint = out.get(Joint::LeftAnkle).unwrap();
                    let sent = joint.extrapolate(0.08 + joint.lead, out.limit);
                    sum += (sent - truth).norm_squared();
                    counted += 1;
                }
            }
            (sum / counted as f64).sqrt()
        }

        let cautious = wobble(1.0);
        let bold = wobble(0.0);
        println!(
            "a joint that never moved was sent {:.1} mm out cautiously and {:.1} mm boldly",
            cautious * 1000.0,
            bold * 1000.0
        );
        assert!(
            bold > 2.0 * cautious,
            "the caution is not buying any stillness: {:.1} mm against {:.1} mm",
            cautious * 1000.0,
            bold * 1000.0
        );
    }

    /// Nothing moving is not the same as a timid prediction, and the panel
    /// would read as the second if this reported zero for the first.
    #[test]
    fn a_body_that_is_not_moving_reports_no_reach_rather_than_none_reached() {
        let mut filter = PoseFilter::default();
        let start = Instant::now();
        let truth = Point3::new(0.0, 0.5, 0.0);

        let mut last = None;
        for step in 0..90 {
            last = Some(filter.push(&pose(start + STEP * step, truth, 0.005)));
        }

        assert_eq!(
            last.expect("a pose").reach,
            None,
            "a joint that never moved reported a share of a speed it did not have"
        );
    }
}
