//! What a tracker consumer looks like from here.

use std::time::Instant;

use anyhow::Result;
use nalgebra::Isometry3;

use super::pose::{TrackerPose, TrackerRole};

/// One send: every tracker Optra can place, at one instant.
#[derive(Debug, Clone)]
pub struct TrackerFrame {
    /// The instant the poses describe, which is in the future — the whole
    /// frame has been predicted forward to when the consumer will act on it.
    pub at: Instant,
    /// How far ahead of the reconstruction that is, in seconds. Sinks that can
    /// tell their consumer the age of a pose need it; the rest report it.
    pub lead: f64,
    pub trackers: Vec<TrackerPose>,
    /// Roles that were enabled but have been missing long enough to call lost.
    ///
    /// Separate from simply being absent, because the two want different
    /// handling: a tracker missing for one frame should hold still, and one
    /// missing for a second should visibly stop, not stay behind pretending.
    pub lost: Vec<TrackerRole>,
    /// The headset, in the same world frame, when SteamVR is reporting it.
    pub head: Option<Isometry3<f64>>,
}

impl TrackerFrame {
    pub fn is_empty(&self) -> bool {
        self.trackers.is_empty()
    }
}

/// Somewhere to send trackers.
///
/// Each sink owns its own coordinate conversion. The frame handed to it is
/// always in Optra's world frame — the OpenVR standing universe, right-handed,
/// +Y up, metres — and what a consumer wants instead is that consumer's
/// business, not something to be negotiated upstream.
pub trait TrackerSink: Send {
    /// Short name, for logs and for the panel.
    fn name(&self) -> &str;

    /// Where it is sending, in a form a user can check against the other
    /// application's settings.
    fn target(&self) -> String;

    fn send(&mut self, frame: &TrackerFrame) -> Result<()>;

    /// Called once when the stage stops, for a sink that has to tell its
    /// consumer the trackers are going away. A sink with nothing to say may
    /// leave this alone.
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Assigns tracker indices to a chosen set of roles.
///
/// One-based and contiguous, in [`TrackerRole::ALL`] order, skipping the roles
/// that are off. Contiguous rather than one fixed index per role because a
/// consumer handed trackers 1, 5 and 6 with nothing between them is a consumer
/// being asked to cope with something no other tracking system produces, and
/// the point of this milestone is that it works.
///
/// What the assignment does *not* depend on is what happens to be visible: a
/// knee that drops out for a moment must not renumber the feet behind it, or a
/// consumer calibrated against index four finds a different limb there. Turning
/// a tracker on or off does renumber, and that is a moment the user is already
/// going to recalibrate.
pub fn assign(enabled: &[TrackerRole]) -> Vec<(u8, TrackerRole)> {
    TrackerRole::ALL
        .iter()
        .filter(|role| enabled.contains(role))
        .enumerate()
        .map(|(index, role)| (index as u8 + 1, *role))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_follow_the_fixed_order_not_the_chosen_one() {
        let chosen = [
            TrackerRole::RightFoot,
            TrackerRole::Hip,
            TrackerRole::LeftFoot,
        ];
        assert_eq!(
            assign(&chosen),
            vec![
                (1, TrackerRole::Hip),
                (2, TrackerRole::LeftFoot),
                (3, TrackerRole::RightFoot),
            ]
        );
    }

    /// Indices are contiguous from one whatever the chosen set is. Nothing
    /// downstream has to cope with a gap.
    #[test]
    fn indices_have_no_gaps_in_them() {
        for chosen in [
            vec![TrackerRole::Hip],
            vec![TrackerRole::LeftFoot, TrackerRole::RightElbow],
            TrackerRole::ALL.to_vec(),
        ] {
            let assigned = assign(&chosen);
            assert_eq!(assigned.len(), chosen.len());
            for (position, (index, _)) in assigned.iter().enumerate() {
                assert_eq!(*index as usize, position + 1);
            }
        }
    }

    #[test]
    fn nothing_is_assigned_when_nothing_is_enabled() {
        assert!(assign(&[]).is_empty());
    }
}
