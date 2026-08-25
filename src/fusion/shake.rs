//! How much the body is moving in ways a body cannot move.
//!
//! "The trackers are shaking" is a complaint about the whole chain, and the
//! chain is four stages long. The reconstruction can shake; the fit can shake
//! while the reconstruction is calm; the smoothing can fail to remove either;
//! and the prediction can put shake back into a position that was smooth. Each
//! of those has a different cause and a different repair, and nothing in the
//! application could tell them apart — a user could only report that it shook,
//! and so could the skeleton on screen.
//!
//! The measurement is the second difference: where the joint is now, minus
//! twice where it was, plus where it was before that. Anything moving at a
//! constant velocity contributes nothing to it, so ordinary motion — a leg
//! swinging, a body walking — barely registers, while noise registers at
//! roughly two and a half times its own amplitude. At sixty hertz a real
//! acceleration of five metres per second squared moves a joint under a
//! millimetre between ticks, and millimetres of jitter are worth several, which
//! is the separation that makes the number worth printing.
//!
//! The median across joints rather than the mean, so that one badly seen ankle
//! does not become the whole body's verdict.

use nalgebra::Point3;

use crate::models::Joint;

/// How settled one stage of the chain is.
#[derive(Debug, Clone, Default)]
pub struct ShakeMeter {
    /// The last two positions of each joint, and how long it has been present.
    /// A joint that blinked has no usable history.
    history: Vec<Track>,
    value: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Track {
    previous: Option<Point3<f64>>,
    before: Option<Point3<f64>>,
    seen: u32,
}

impl ShakeMeter {
    /// How much smoothing to apply to the reported figure.
    ///
    /// A second difference is a noisy statistic even when what it is measuring
    /// is steady, and this is a number a user reads off a panel rather than one
    /// anything acts on.
    const SMOOTHING: f64 = 0.05;

    /// Takes one tick's worth of positions and updates the estimate.
    ///
    /// The tick is assumed regular, which the fusion clock makes true. A second
    /// difference over uneven samples measures the unevenness as well.
    pub fn observe(&mut self, points: impl Iterator<Item = (Joint, Point3<f64>)>) {
        if self.history.len() != Joint::ALL.len() {
            self.history = vec![Track::default(); Joint::ALL.len()];
        }

        let mut present = vec![false; Joint::ALL.len()];
        let mut shakes = Vec::new();

        for (joint, point) in points {
            let track = &mut self.history[joint.index()];
            present[joint.index()] = true;

            // Only once all three samples are consecutive. A joint that came
            // back after a gap would otherwise report the gap as shake.
            if let (Some(previous), Some(before), true) =
                (track.previous, track.before, track.seen >= 2)
            {
                shakes.push((point - previous * 2.0 + before.coords).norm());
            }

            track.before = track.previous;
            track.previous = Some(point);
            track.seen = track.seen.saturating_add(1);
        }

        for (index, seen) in present.iter().enumerate() {
            if !seen {
                self.history[index] = Track::default();
            }
        }

        if shakes.is_empty() {
            return;
        }
        shakes.sort_by(f64::total_cmp);
        let median = shakes[shakes.len() / 2];
        self.value += (median - self.value) * Self::SMOOTHING;
    }

    /// The typical joint's tick-to-tick wobble, in metres.
    pub fn metres(&self) -> f64 {
        self.value
    }
}

/// The same measurement at each stage of the chain.
///
/// Read left to right it says where shaking comes from. Raw high and filtered
/// low is the smoothing doing its job on ordinary noise. All four high is the
/// cameras, and no amount of tuning downstream will help. Filtered low and sent
/// high is the prediction, which is the one thing here that can add movement
/// rather than remove it.
#[derive(Debug, Clone, Copy, Default)]
pub struct Shake {
    /// Straight out of the triangulation.
    pub raw: f64,
    /// After the skeleton fit has held the body together.
    pub fitted: f64,
    /// After smoothing, which is what the solid skeleton draws.
    pub filtered: f64,
    /// After prediction, which is what the trackers are told.
    pub predicted: f64,
}

impl Shake {
    /// The stage that added the most, and how much, in metres.
    ///
    /// None when nothing stands out, which is the common and uninteresting
    /// case of a chain that is simply quiet.
    pub fn worst_stage(&self) -> Option<(&'static str, f64)> {
        let stages = [
            ("the cameras", self.raw),
            ("the skeleton fit", self.fitted - self.raw),
            ("the smoothing", self.filtered - self.fitted),
            ("the prediction", self.predicted - self.filtered),
        ];
        stages
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .filter(|(_, added)| *added > 0.002)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settle(meter: &mut ShakeMeter, mut at: impl FnMut(u32) -> Point3<f64>) {
        for tick in 0..400 {
            meter.observe(std::iter::once((Joint::Hip, at(tick))));
        }
    }

    #[test]
    fn a_joint_moving_steadily_does_not_count_as_shaking() {
        let mut meter = ShakeMeter::default();
        settle(&mut meter, |tick| Point3::new(tick as f64 * 0.02, 1.0, 0.0));
        assert!(
            meter.metres() < 1e-9,
            "a metre a second registered as {:.4} m of shake",
            meter.metres()
        );
    }

    #[test]
    fn a_joint_wobbling_in_place_counts_as_shaking() {
        let mut meter = ShakeMeter::default();
        // A centimetre, alternating: a, b, a differences to twice the gap, so
        // a centimetre of wobble should read as two.
        settle(&mut meter, |tick| {
            Point3::new(if tick % 2 == 0 { 0.0 } else { 0.01 }, 1.0, 0.0)
        });
        assert!(
            meter.metres() > 0.019,
            "a centimetre of wobble registered as only {:.4} m",
            meter.metres()
        );
    }

    #[test]
    fn a_joint_that_blinks_reports_nothing_rather_than_the_gap() {
        let mut meter = ShakeMeter::default();
        for tick in 0..400 {
            // Present every other tick, and a long way from where it was.
            if tick % 2 == 0 {
                let x = if tick % 4 == 0 { 0.0 } else { 0.5 };
                meter.observe(std::iter::once((Joint::Hip, Point3::new(x, 1.0, 0.0))));
            } else {
                meter.observe(std::iter::empty());
            }
        }
        assert_eq!(
            meter.metres(),
            0.0,
            "a joint never present three ticks running has no measurable shake"
        );
    }

    #[test]
    fn the_stage_that_added_the_most_is_the_one_named() {
        let quiet = Shake {
            raw: 0.001,
            fitted: 0.001,
            filtered: 0.001,
            predicted: 0.001,
        };
        assert_eq!(quiet.worst_stage(), None);

        let predicting = Shake {
            raw: 0.004,
            fitted: 0.004,
            filtered: 0.001,
            predicted: 0.020,
        };
        assert_eq!(
            predicting.worst_stage().map(|(name, _)| name),
            Some("the prediction")
        );
    }
}
