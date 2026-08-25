//! Asking Windows for a timer it can actually keep.
//!
//! By default the scheduler wakes sleeping threads on a roughly 15.6 ms tick,
//! so a loop that asks to sleep 8 ms gets 15.6 ms and runs at half the rate it
//! was configured for, without anything reporting a problem. Optra has three
//! loops that care: pose sampling, the fusion clock, and the output thread —
//! and on the last of those, timing jitter becomes tracker jitter that the user
//! sees.
//!
//! `timeBeginPeriod` is the documented way to ask for better, and on Windows 11
//! it applies to the calling process rather than the whole machine. It costs
//! some power, which is the right trade for an application that only runs while
//! the user is in VR.

use std::time::{Duration, Instant};

use super::Shutdown;

/// Holds the raised timer resolution for as long as it lives.
pub struct TimerResolution {
    /// The period that was granted, in milliseconds. `None` if the request was
    /// refused, in which case dropping this does nothing.
    granted: Option<u32>,
}

/// One millisecond is the finest `timeBeginPeriod` accepts, and is well beyond
/// what any loop here needs.
const REQUESTED_MS: u32 = 1;

impl TimerResolution {
    /// Raises the timer resolution, or leaves it alone if the system refuses.
    ///
    /// A refusal is not an error worth failing over: the application still
    /// works, it just holds its rates less exactly.
    #[cfg(windows)]
    pub fn request() -> Self {
        // SAFETY: `timeBeginPeriod` takes a period in milliseconds and returns
        // a status. It has no preconditions beyond a matching `timeEndPeriod`,
        // which `Drop` provides.
        let result = unsafe { windows::Win32::Media::timeBeginPeriod(REQUESTED_MS) };

        if result == windows::Win32::Media::TIMERR_NOERROR {
            tracing::debug!("timer resolution raised to {REQUESTED_MS} ms");
            Self {
                granted: Some(REQUESTED_MS),
            }
        } else {
            tracing::warn!("the system refused a {REQUESTED_MS} ms timer resolution");
            Self { granted: None }
        }
    }

    #[cfg(not(windows))]
    pub fn request() -> Self {
        Self { granted: None }
    }

    /// Whether the request was granted.
    pub fn is_raised(&self) -> bool {
        self.granted.is_some()
    }
}

impl Drop for TimerResolution {
    fn drop(&mut self) {
        #[cfg(windows)]
        if let Some(period) = self.granted {
            // SAFETY: paired with the `timeBeginPeriod` that granted it.
            unsafe {
                let _ = windows::Win32::Media::timeEndPeriod(period);
            }
        }
    }
}

/// A fixed-rate clock for a worker loop.
///
/// Sleeping for the period at the end of each pass drifts: the sleep overshoots
/// a little, the work takes a little, and both accumulate. A ticker keeps the
/// schedule instead of the interval, so an overshoot on one tick is absorbed by
/// a shorter sleep on the next and the average rate is the one that was asked
/// for.
pub struct Ticker {
    period: Duration,
    /// When the next tick is due.
    next: Instant,
}

impl Ticker {
    pub fn new(period: Duration) -> Self {
        Self {
            period: period.max(Duration::from_micros(100)),
            // The first tick is one period from now, not immediately: the
            // caller runs a pass and then waits.
            next: Instant::now() + period.max(Duration::from_micros(100)),
        }
    }

    /// From a rate in hertz, which is how every caller thinks about it.
    pub fn at_hz(hz: f32) -> Self {
        Self::new(Duration::from_secs_f32(1.0 / hz.max(1.0)))
    }

    /// Waits until the next tick is due.
    ///
    /// Returns `false` if the shutdown signal fired instead, in which case the
    /// caller should stop rather than run another pass.
    pub fn wait(&mut self, shutdown: &Shutdown) -> bool {
        match self.advance(Instant::now()) {
            Some(remaining) => shutdown.sleep(remaining),
            // Already late; run immediately rather than sleeping zero.
            None => !shutdown.is_cancelled(),
        }
    }

    /// Schedules the next tick and reports how long to wait for this one.
    ///
    /// Split out from [`Ticker::wait`] so the schedule can be tested without
    /// waiting for real time to pass.
    fn advance(&mut self, now: Instant) -> Option<Duration> {
        if self.next > now {
            let remaining = self.next - now;
            self.next += self.period;
            return Some(remaining);
        }

        // Behind schedule. Skip the ticks that have already gone by instead of
        // running them back to back: a loop that stalled for a second should
        // resume, not spend the next second catching up on stale work.
        let late = now - self.next;
        let missed = (late.as_nanos() / self.period.as_nanos()) as u32;
        self.next += self.period * (missed + 1);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: fn(u64) -> Duration = Duration::from_millis;

    /// A ticker rooted at `start`, as if it had just been constructed then.
    fn at(start: Instant, period: Duration) -> Ticker {
        let mut ticker = Ticker::new(period);
        ticker.next = start + period;
        ticker
    }

    /// The ticks stay a period apart no matter how long each pass takes, which
    /// a plain sleep-for-the-period loop does not manage.
    #[test]
    fn a_ticker_keeps_the_schedule_rather_than_the_interval() {
        let start = Instant::now();
        let mut ticker = at(start, MS(10));

        // Three ticks, with the work taking 3 ms each time. Every sleep is
        // shortened by what the pass before it spent.
        assert_eq!(ticker.advance(start + MS(3)), Some(MS(7)));
        assert_eq!(ticker.advance(start + MS(13)), Some(MS(7)));
        assert_eq!(ticker.advance(start + MS(23)), Some(MS(7)));
    }

    /// The case a plain sleep gets wrong: a sleep that overshoots must not
    /// push the whole schedule along with it.
    #[test]
    fn an_overshooting_sleep_does_not_accumulate() {
        let start = Instant::now();
        let mut ticker = at(start, MS(10));

        // The first tick was meant for 10 ms but the sleep returned at 12, and
        // the pass took 3 ms on top.
        assert_eq!(ticker.advance(start + MS(3)), Some(MS(7)));
        assert_eq!(
            ticker.advance(start + MS(15)),
            Some(MS(5)),
            "the next tick is still due at 20 ms, not at 25"
        );
    }

    #[test]
    fn a_stalled_loop_skips_the_ticks_it_missed() {
        let start = Instant::now();
        let mut ticker = at(start, MS(10));

        // Half a second of stall is a great many missed ticks.
        let late = start + MS(500);
        assert_eq!(ticker.advance(late), None);

        // The next tick is one period away, not fifty ticks in the past.
        assert_eq!(ticker.advance(late + MS(2)), Some(MS(8)));
    }

    #[test]
    fn a_rate_becomes_the_matching_period() {
        let start = Instant::now();
        let mut ticker = Ticker::at_hz(50.0);
        ticker.next = start + MS(20);

        assert_eq!(ticker.advance(start + MS(5)), Some(MS(15)));
    }

    /// Nothing should be able to ask for a zero period and spin the machine.
    #[test]
    fn an_absurd_rate_is_clamped() {
        assert!(Ticker::at_hz(0.0).period >= Duration::from_micros(100));
        assert!(Ticker::new(Duration::ZERO).period >= Duration::from_micros(100));
    }
}

/// A smoothed count of events per second.
///
/// Reported rather than counted over a window, because what a user wants to
/// know from it is whether a loop is keeping up *now* — and a plain average
/// over the last second takes a second to admit that it has stopped.
#[derive(Debug, Default, Clone)]
pub struct Rate {
    last: Option<Instant>,
    rate: f32,
}

impl Rate {
    pub fn tick(&mut self, now: Instant) -> f32 {
        if let Some(previous) = self.last.replace(now) {
            let dt = now.duration_since(previous).as_secs_f32();
            if dt > 0.0 {
                self.rate = ema(self.rate, 1.0 / dt);
            }
        }
        self.rate
    }

    pub fn get(&self) -> f32 {
        self.rate
    }
}

/// One step of an exponential moving average, seeded by its first sample.
///
/// Seeding matters: starting from zero would make every rate and every fraction
/// in the UI climb slowly out of nothing for the first second, which reads as a
/// stage struggling to start rather than as a filter warming up.
pub fn ema(current: f32, sample: f32) -> f32 {
    const ALPHA: f32 = 0.05;
    if current == 0.0 {
        sample
    } else {
        current * (1.0 - ALPHA) + sample * ALPHA
    }
}
