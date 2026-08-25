//! A message that is allowed to change, but not quickly.
//!
//! Every panel here reports what is wrong from live statistics, and live
//! statistics cross their thresholds constantly. A camera that is right on the
//! edge of keeping up, a floor estimate wandering either side of six
//! centimetres, a fit correction hovering at two — each of those turns a
//! warning on and off at the repaint rate, and because a warning is a line of
//! text, everything under it moves by a line of text sixty times a second. The
//! buttons underneath become unclickable, which is a worse fault than whatever
//! the warning was about.
//!
//! So a notice is latched. It has to be asked for continuously before it
//! appears, and it stays for a good while after it stops being asked for. The
//! asymmetry is the point: appearing costs the user a line of layout, so it is
//! worth being sure, and disappearing costs them the same line back, so it is
//! worth being slow. A warning that is genuinely blinking then reads as a
//! warning that is on, which is also the truer description of it.

use std::time::{Duration, Instant};

use egui::RichText;

use super::{BAD, FAIR};

/// How urgent a notice is, which is all that decides how it is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Something is wrong and the stage is still running.
    Warning,
    /// Something is wrong and it is why nothing is happening.
    Problem,
}

impl Level {
    fn colour(self) -> egui::Color32 {
        match self {
            Level::Warning => FAIR,
            Level::Problem => BAD,
        }
    }
}

/// A latched message.
///
/// Held by the panel across frames, which is what gives it somewhere to
/// remember the last thing it was asked for.
#[derive(Debug, Default)]
pub struct Notice {
    shown: Option<(String, Level)>,
    /// When the panel started asking for something other than what is shown.
    /// Cleared as soon as the two agree again.
    since: Option<Instant>,
}

impl Notice {
    /// How long a message has to be wanted before it appears.
    ///
    /// Long enough to outlast a statistic crossing its threshold and coming
    /// straight back, short enough that a real fault is reported while the user
    /// is still looking at what caused it.
    const APPEAR: Duration = Duration::from_millis(400);
    /// How long a message stays after nothing is asking for it any more.
    ///
    /// Much longer, because this is the direction that takes a line of layout
    /// away from under the pointer.
    const LINGER: Duration = Duration::from_millis(2500);

    /// Draws the message that has settled, given the one wanted right now.
    pub fn show(&mut self, ui: &mut egui::Ui, wanted: Option<(String, Level)>) {
        self.settle(wanted, Instant::now());
        if let Some((text, level)) = &self.shown {
            ui.label(RichText::new(text).color(level.colour()));
        }
    }

    /// Whether anything is currently being shown, for a caller that has to
    /// space around it.
    pub fn visible(&self) -> bool {
        self.shown.is_some()
    }

    fn settle(&mut self, wanted: Option<(String, Level)>, now: Instant) {
        if wanted == self.shown {
            self.since = None;
            return;
        }

        let since = *self.since.get_or_insert(now);
        // Taking a line away is the change that hurts, so it is the slow one.
        let dwell = if wanted.is_none() {
            Self::LINGER
        } else {
            Self::APPEAR
        };

        if now.duration_since(since) >= dwell {
            self.shown = wanted;
            self.since = None;
        }
    }
}

/// A threshold that has to be crossed properly to count.
///
/// The other half of the same problem: a notice only settles quickly when the
/// thing it is reporting settles, and a bare `value > limit` never does. This
/// makes the way in and the way out different numbers, so a statistic sitting
/// on the limit stays wherever it already was rather than reporting both
/// answers at once.
#[derive(Debug, Default, Clone, Copy)]
pub struct Threshold {
    over: bool,
}

impl Threshold {
    /// True while `value` is over `limit`, and only false again once it has
    /// come back under `limit * (1 - margin)`.
    pub fn over(&mut self, value: f64, limit: f64, margin: f64) -> bool {
        self.over = if self.over {
            value > limit * (1.0 - margin)
        } else {
            value > limit
        };
        self.over
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warning(text: &str) -> Option<(String, Level)> {
        Some((text.to_owned(), Level::Warning))
    }

    #[test]
    fn a_message_asked_for_once_never_appears() {
        let mut notice = Notice::default();
        let start = Instant::now();

        // Sixty repaints a second, every other one wanting the warning: what a
        // statistic sitting exactly on its threshold looks like.
        for tick in 0..600u32 {
            let at = start + Duration::from_millis(tick as u64 * 16);
            let wanted = if tick % 2 == 0 {
                warning("on the edge")
            } else {
                None
            };
            notice.settle(wanted, at);
        }

        assert!(
            !notice.visible(),
            "a warning that blinks at thirty hertz should not be drawn at all"
        );
    }

    #[test]
    fn a_message_that_holds_appears_and_then_stays() {
        let mut notice = Notice::default();
        let start = Instant::now();

        for tick in 0..60u32 {
            notice.settle(
                warning("cam1 is not being used"),
                start + Duration::from_millis(tick as u64 * 16),
            );
        }
        assert!(notice.visible(), "half a second of asking should be enough");

        // It goes away, and the layout under it must not move for a while.
        let gone = start + Duration::from_secs(1);
        notice.settle(None, gone);
        notice.settle(None, gone + Duration::from_millis(1000));
        assert!(notice.visible(), "a second later it should still be there");

        notice.settle(None, gone + Notice::LINGER);
        assert!(!notice.visible(), "and by the linger it should be gone");
    }

    #[test]
    fn a_threshold_does_not_report_both_answers_at_once() {
        let mut threshold = Threshold::default();

        assert!(!threshold.over(0.019, 0.02, 0.25));
        assert!(threshold.over(0.021, 0.02, 0.25));
        // Back under the limit, but not far enough under to count as settled.
        assert!(threshold.over(0.019, 0.02, 0.25));
        assert!(!threshold.over(0.014, 0.02, 0.25));
    }
}
