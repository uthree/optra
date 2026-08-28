//! Timing a model on the machine it will actually run on.
//!
//! The catalogue can say what a model is good for; it cannot say what it costs
//! *here*, because that depends on the GPU, the driver, and whether DirectML
//! accepted the graph or the session quietly fell back to the CPU. Those are
//! exactly the questions a user choosing between two models is asking, and
//! until now the only way to answer them was to assign the model and watch the
//! Cameras panel struggle.

use std::time::Instant;

use anyhow::Result;

use crate::models::ModelSpec;
use crate::models::manifest::ModelKind;

use super::arch;
use super::session::{Backend, ProviderChoice};
use super::traits::{Detection, ImageView};

/// Runs that are timed. At the tens of milliseconds a slow model takes this is
/// still well under two seconds, and fewer runs would let one scheduler hiccup
/// pass itself off as the model's worst case.
const RUNS: usize = 20;

/// Runs thrown away first. The first DirectML run compiles kernels and can be
/// an order of magnitude slower than every run after it; charging that to the
/// model would make every benchmark mostly measure the driver.
const WARMUP: usize = 3;

/// What one benchmark measured.
#[derive(Debug, Clone, Copy)]
pub struct Benchmark {
    /// The backend the session actually got, which is half the point of
    /// running one: a model DirectML rejected runs on the CPU and nothing else
    /// in the UI says so.
    pub backend: Backend,
    /// Building and warming the session, in milliseconds. Paid once per model
    /// swap, not per frame.
    pub build_ms: f32,
    /// Median time for one frame's work, in milliseconds.
    pub median_ms: f32,
    /// The slowest timed run. Frame pacing answers to this, not to the median.
    pub worst_ms: f32,
}

impl Benchmark {
    /// The frame rate the median run time could sustain, for the label.
    pub fn fps(&self) -> f32 {
        if self.median_ms > 0.0 {
            1000.0 / self.median_ms
        } else {
            0.0
        }
    }
}

/// Builds the model with `provider` and times it against a synthetic frame.
///
/// The input is a fixed gradient rather than anything meaningful: inference
/// cost does not depend on what is in the picture, and a deterministic input
/// keeps two runs of the same benchmark comparable.
pub fn run(spec: &ModelSpec, provider: ProviderChoice) -> Result<Benchmark> {
    let (width, height) = (640u32, 480u32);
    let rgb = gradient(width, height);
    let view = ImageView::new(width, height, &rgb);

    // A person-sized box in the middle of the frame, for the pose crop.
    let person = Detection {
        x1: 240.0,
        y1: 60.0,
        x2: 400.0,
        y2: 420.0,
        score: 0.9,
    };

    let started = Instant::now();
    let (backend, mut work): (Backend, Box<dyn FnMut() -> Result<()>>) = match spec.kind {
        ModelKind::Detector => {
            let mut detector = arch::build_detector(spec, provider)?;
            (
                detector.backend(),
                Box::new(move || detector.detect(&[view]).map(|_| ())),
            )
        }
        ModelKind::Pose2d => {
            let mut pose = arch::build_pose2d(spec, provider)?;
            (
                pose.backend(),
                Box::new(move || pose.estimate(&[(view, person)]).map(|_| ())),
            )
        }
    };

    for _ in 0..WARMUP {
        work()?;
    }
    let build_ms = started.elapsed().as_secs_f32() * 1000.0;

    let mut times = Vec::with_capacity(RUNS);
    for _ in 0..RUNS {
        let run = Instant::now();
        work()?;
        times.push(run.elapsed().as_secs_f32() * 1000.0);
    }

    let (median_ms, worst_ms) = summarize(&mut times);
    Ok(Benchmark {
        backend,
        build_ms,
        median_ms,
        worst_ms,
    })
}

/// Median and worst of a set of run times.
fn summarize(times: &mut [f32]) -> (f32, f32) {
    times.sort_by(|a, b| a.total_cmp(b));
    let median = times[times.len() / 2];
    let worst = *times.last().unwrap_or(&0.0);
    (median, worst)
}

/// A deterministic RGB test card.
fn gradient(width: u32, height: u32) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            rgb.push((x * 255 / width.max(1)) as u8);
            rgb.push((y * 255 / height.max(1)) as u8);
            rgb.push(((x + y) % 256) as u8);
        }
    }
    rgb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_median_is_not_the_worst() {
        let mut times = [5.0, 4.0, 60.0, 5.5, 4.5];
        let (median, worst) = summarize(&mut times);
        // One stalled run must show up as the worst case without dragging the
        // median, which is the whole reason both are reported.
        assert_eq!(median, 5.0);
        assert_eq!(worst, 60.0);
    }

    #[test]
    fn the_test_card_is_deterministic() {
        assert_eq!(gradient(64, 48), gradient(64, 48));
    }
}
