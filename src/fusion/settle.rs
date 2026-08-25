//! Keeping a joint out until it means to stay in.
//!
//! Every test that decides whether a joint is reconstructed — enough rays, the
//! rays agreeing, the result certain enough to be worth having — is a threshold
//! applied to a quantity that moves from tick to tick. A joint sitting on one
//! of those thresholds does not settle on an answer. It passes, fails, passes,
//! fails, at the tick rate.
//!
//! That would be harmless if the two outcomes were near each other, and they
//! are not. A joint that is reconstructed sits where the cameras put it; a
//! joint that is not is invented by the fit, from the skeleton, wherever the
//! bones say it has to be. The distance between those two answers is the
//! calibration error, which is centimetres in a good room and was seventeen of
//! them in the report that prompted this. So the joint does not degrade when it
//! flips, it teleports, sixty times a second, between two positions that are
//! each defensible. No filter downstream removes that: it is a square wave
//! whose amplitude is a real disagreement, not noise around a true value.
//!
//! This is the same failure the fusion clock already has an answer for one
//! level up, where a camera dropping in and out of ticks changed which cameras
//! a joint was built from. The answer is the same shape and the reasoning is
//! the same: decide, and then hold the decision.
//!
//! Asymmetric, because the two directions are not alike. A joint that fails its
//! test goes out at once — there is nothing else available, the reconstruction
//! genuinely has no position for it — but coming back takes a run of ticks that
//! all agree. The cost is that a joint the cameras can only solve half the time
//! stays inferred, which is the right trade: an inferred joint is smooth and
//! anatomically possible and may be in the wrong place, while an alternating
//! one is in the wrong place half the time *and* unusable the rest.
//!
//! A joint starts admitted, so acquiring the body at startup costs nothing.
//! The dwell exists to stop a joint coming *back* too eagerly, and there is
//! nothing to come back from until something has gone out.

use crate::models::Joint;

/// Per-joint admission, so a joint on the edge of a threshold stops flipping.
#[derive(Debug, Default, Clone)]
pub struct Settling {
    state: Vec<State>,
}

#[derive(Debug, Clone, Copy)]
struct State {
    /// Consecutive ticks the reconstruction has solved this joint.
    passes: u32,
    /// Whether it is currently allowed through.
    inside: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            passes: 0,
            // Admitted until something takes it out. See the module docs.
            inside: true,
        }
    }
}

impl Settling {
    /// How many more consecutive solved ticks this joint owes before it may be
    /// used. Zero means it may be used now.
    ///
    /// Call once per joint per tick, whether or not it was solved: a tick the
    /// joint missed is what starts it owing anything.
    pub fn wait(&mut self, joint: Joint, solved: bool, dwell: u32) -> u32 {
        if self.state.len() != Joint::ALL.len() {
            self.state = vec![State::default(); Joint::ALL.len()];
        }
        let state = &mut self.state[joint.index()];

        if !solved {
            state.passes = 0;
            state.inside = false;
            return dwell;
        }

        state.passes = state.passes.saturating_add(1);
        if state.inside || state.passes >= dwell {
            state.inside = true;
            return 0;
        }
        dwell - state.passes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DWELL: u32 = 6;

    /// Nothing has gone wrong yet, so nothing is owed. A user who starts the
    /// application should not watch the body assemble itself.
    #[test]
    fn a_joint_seen_from_the_start_is_used_from_the_start() {
        let mut settling = Settling::default();
        for _ in 0..10 {
            assert_eq!(settling.wait(Joint::LeftAnkle, true, DWELL), 0);
        }
    }

    /// The case the whole module is for.
    #[test]
    fn a_joint_that_alternates_is_held_out_rather_than_flipping() {
        let mut settling = Settling::default();
        let mut used = 0;

        for tick in 0..600 {
            if settling.wait(Joint::LeftAnkle, tick % 2 == 0, DWELL) == 0 {
                used += 1;
            }
        }

        // The first tick is solved and the joint has not been out yet, so it
        // is used once. After that it can never gather six in a row.
        assert_eq!(
            used, 1,
            "a joint alternating at the tick rate was let through {used} times"
        );
    }

    /// A joint that genuinely comes back should come back, and not much later.
    #[test]
    fn a_joint_that_returns_and_stays_is_readmitted() {
        let mut settling = Settling::default();

        assert_eq!(settling.wait(Joint::Hip, false, DWELL), DWELL);
        for expected in (1..DWELL).rev() {
            assert_eq!(settling.wait(Joint::Hip, true, DWELL), expected);
        }
        assert_eq!(
            settling.wait(Joint::Hip, true, DWELL),
            0,
            "a joint solved on {DWELL} ticks running is still being held out"
        );

        // And stays in, rather than owing the dwell again every tick.
        assert_eq!(settling.wait(Joint::Hip, true, DWELL), 0);
    }

    /// One bad tick is enough to take a joint out. There is no position to
    /// keep it in with — the reconstruction did not produce one.
    #[test]
    fn a_single_miss_takes_a_settled_joint_out() {
        let mut settling = Settling::default();
        for _ in 0..100 {
            settling.wait(Joint::RightKnee, true, DWELL);
        }
        settling.wait(Joint::RightKnee, false, DWELL);
        assert!(settling.wait(Joint::RightKnee, true, DWELL) > 0);
    }

    /// Joints do not share a verdict.
    #[test]
    fn one_joint_settling_does_not_hold_another_out() {
        let mut settling = Settling::default();
        settling.wait(Joint::LeftAnkle, false, DWELL);
        assert!(settling.wait(Joint::LeftAnkle, true, DWELL) > 0);
        assert_eq!(settling.wait(Joint::RightAnkle, true, DWELL), 0);
    }

    /// Zero turns it off, which is how a test that is not about settling gets
    /// to ignore it.
    #[test]
    fn a_dwell_of_zero_admits_everything() {
        let mut settling = Settling::default();
        settling.wait(Joint::Hip, false, 0);
        assert_eq!(settling.wait(Joint::Hip, true, 0), 0);
    }
}
