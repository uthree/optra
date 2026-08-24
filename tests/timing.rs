//! What rate a fixed-rate worker loop can actually hold.
//!
//! Windows runs its timers at roughly 64 Hz unless a process asks for better,
//! so a loop asking to sleep 8 ms gets 15.6 ms and quietly runs at half the
//! rate it was configured for. Optra has three such loops — pose sampling, the
//! fusion clock and the output thread — and the last of those turns timing
//! jitter into visible tracker jitter.
//!
//! Ignored by default: it measures wall-clock behaviour, which is not something
//! to assert on in a normal test run.

use std::time::{Duration, Instant};

use optra::worker::{Shutdown, timing};

/// Ticks a shutdown-aware sleep as fast as the requested period allows, and
/// reports what it managed.
fn achieved(period: Duration, over: Duration) -> f32 {
    let shutdown = Shutdown::default();
    let start = Instant::now();
    let mut ticks = 0u32;

    while start.elapsed() < over {
        shutdown.sleep(period);
        ticks += 1;
    }

    ticks as f32 / start.elapsed().as_secs_f32()
}

/// The same, driven by a ticker rather than a plain sleep.
fn achieved_with_ticker(period: Duration, over: Duration) -> f32 {
    let shutdown = Shutdown::default();
    let mut ticker = timing::Ticker::new(period);
    let start = Instant::now();
    let mut ticks = 0u32;

    while start.elapsed() < over {
        ticker.wait(&shutdown);
        ticks += 1;
    }

    ticks as f32 / start.elapsed().as_secs_f32()
}

#[test]
#[ignore = "measures wall-clock timing"]
fn a_fixed_rate_loop_holds_its_rate() {
    let period = Duration::from_millis(8);
    let over = Duration::from_secs(2);
    let target = 1.0 / period.as_secs_f32();

    let coarse = achieved(period, over);
    println!("plain sleep, default timer: {coarse:6.1} Hz asking for {target:.0}");

    let guard = timing::TimerResolution::request();
    assert!(guard.is_raised(), "the system refused a 1 ms timer");

    let fine = achieved(period, over);
    println!("plain sleep, raised timer:  {fine:6.1} Hz asking for {target:.0}");

    let ticked = achieved_with_ticker(period, over);
    println!("ticker,      raised timer:  {ticked:6.1} Hz asking for {target:.0}");
    drop(guard);

    assert!(
        fine > coarse * 1.3,
        "raising the timer resolution should have helped: {coarse:.1} -> {fine:.1} Hz"
    );
    assert!(
        ticked > target * 0.95,
        "a ticker should hold the rate it was asked for: {ticked:.1} against {target:.0} Hz"
    );
}
