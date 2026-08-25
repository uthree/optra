# Optra Implementation Roadmap

Milestones are ordered so that each one is independently testable and leaves the
application in a runnable state. See [design.md](design.md) for the architecture
these steps build toward.

| Milestone | Status |
|---|---|
| M0 - Project skeleton | done |
| M1 - Camera capture | done |
| M2 - Inference | done |
| M3 - VR link and calibration | done, pending a room |
| M4 - Fusion | done, pending a room |
| M5 - Tracker output | done, pending a consumer |
| M6 - Tuning and release | in progress |

## M0 - Project skeleton

- Workspace-less single crate, `Cargo.toml` with the dependency set from the
  design document.
- `tracing` logging, `anyhow` error handling, config load/save under
  `%APPDATA%/optra`.
- `eframe` window with the six empty panels and a shared `AppState`.
- Thread supervision helper: named threads, shutdown signal, panic reporting
  back to the UI.

**Done when:** the app starts, shows the panel layout, and persists window and
config state across restarts.

## M1 - Camera capture

- Device enumeration and format negotiation via `nokhwa`.
- One capture thread per camera with monotonic timestamps and a single-slot
  mailbox.
- Synthetic frame source: a virtual room rendered from a ceiling corner, so the
  multi-camera paths can be developed and tested without the hardware.
- Device property control (exposure, gain, focus, white balance) through
  DirectShow, applied at open and adjustable live.
- Camera panel: device list, per-camera resolution/FPS/format selection, lens
  type, rotation, property sliders, live preview grid, measured FPS and
  missed-frame counters.
- Camera identity persisted by device path.

**Done when:** cameras of different models, resolutions and frame rates stream
simultaneously with stable measured FPS, and the UI makes bandwidth problems
visible.

Two findings from this milestone changed the design, both recorded in
[design.md](design.md):

- Media Foundation misreports the frame rate of a format after it is set, and
  cannot be trusted for what the camera panel shows. The value returned by the
  setter is used instead.
- Automatic exposure halves the frame rate in normal room lighting. Manual
  exposure control is not optional, and nokhwa cannot provide it.

## M2 - Inference

- `ort` session setup with the DirectML execution provider and a CPU fallback.
- Model registry: `manifest.toml` and model spec parsing, download with
  progress, SHA-256 verification, license display and gate, graph validation
  against the declared tensors.
- `Detector` and `Pose2d` traits and the architecture adapter registry, with the
  `mmdet_end2end` and `simcc` adapters.
- Per-camera model assignment and a detector stride, with the box carried by the
  previous keypoints between detector runs.
- Canonical keypoint mapping driven by `keypoints.toml`, covering COCO-17,
  Halpe-26 and COCO-WholeBody-133.
- Runtime model swap: background session build and warm-up, and a camera keeps
  its current model until the replacement is ready.
- Model panel: catalogue with license badges, install progress, execution
  provider and default model selection.
- Keypoint overlay on the camera previews.
- A still-image source, so the stages downstream can be tested against a known
  scene.

**Done when:** all cameras show live 2D skeletons, each camera can run a
different model, and swapping a model at runtime neither blocks the UI nor
interrupts tracking. Adding a new checkpoint of a supported architecture
requires editing only the manifest.

What this milestone found:

- PINTO's model zoo publishes every conversion of a model in one archive, which
  runs to gigabytes for a single ONNX file. The catalogue therefore fetches the
  same models from their upstream publishers and records the zoo entry
  alongside. See [design.md](design.md).
- Halpe 26 replaced the 133-point whole-body model as the default: it carries
  the same heel and toe points without the face and hand keypoints.
- Only the `mmdet_end2end` and `simcc` adapters were needed for the shipped
  catalogue; the others in the design remain unwritten until a model needs them.

Left for later: a per-model benchmark in the model panel, and a UI for
registering a local ONNX file. The manifest already supports a local source, so
that one is a hand-edited entry in the user manifest for now.

## M3 - VR link and calibration

- OpenVR background client reading HMD and controller poses. Done: the runtime
  is loaded at run time, poses are sampled into a short history, and the
  calibration panel shows the live device list.
- Correspondence recorder for the calibration walk. Done.
- Lens models: radial-tangential and equidistant fisheye, selected per camera.
  Done.
- DLT resection with RANSAC, RQ decomposition into `K`, `R`, `t`. Done.
- Levenberg-Marquardt bundle refinement on angular residuals, including
  distortion and the HMD-to-head-keypoint offset. Done.
- Per-camera latency estimation. Done, as a search over the reprojection error
  rather than a correlation; see [design.md](design.md).
- Calibration wizard UI: guidance, live coverage map, residual reporting,
  degenerate-configuration detection, profile save/load. Done.
- 3D viewport showing camera frusta and the floor grid. Done. Play-space
  bounds need the chaperone interface, which is a separate function table and
  is left until something else needs it.

**Done when:** a calibration run converges to a low RMS reprojection error and
the reconstructed camera positions match the physical room layout.

Progress: the solver is written and verified against synthetic rooms whose
answer is known — lens models, projection, triangulation, resection and the
joint refinement. `tests/calibration.rs` runs the whole procedure on four
unlike cameras and recovers their positions to within 3 mm and the head offset
to within 3 mm, including a run with a camera blocked for half the walk and one
keypoint in twelve thrown somewhere else.

The SteamVR link is written and verified against a Quest 3: the headset and
both controllers are read with correct classes, roles and poses. Testing it
turned up two things, both recorded in [design.md](design.md):

- An OpenVR connection is a process singleton rather than a handle. A second
  one is now refused instead of crashing the first.
- A loop asking to sleep 8 ms was running at 64 Hz, because that is the rate
  Windows wakes sleeping threads at by default. This affects the fusion clock
  and the output thread as much as pose sampling, so the fix lives in
  `worker::timing`: a raised timer resolution for the process, and a ticker
  that keeps a schedule rather than an interval.

The recorder and the solver are written: a walk is turned into a room profile
by `calib::recorder` and `calib::solve`, with `tests/calibration.rs` running a
synthetic recording through the whole path and recovering four camera positions
to within 5 mm and every rig offset to within 5 mm.

The wizard is written and the whole path is reachable from the UI: check the
prerequisites, record a walk with a live coverage map and a warning when the
walk cannot be solved, solve on a worker thread, review, and save a room
profile. `tests/panels.rs` lays every panel out headlessly, which is how an
immediate-mode UI gets tested at all — a panel nobody is looking at is never
drawn, and a layout bug in it waits until someone is mid-walk.

One more thing came out later, when a synthetic recording was finally given the
camera delays that every real recording has. Both halves of the latency
correction were wrong, and neither could show up against a prompt recording:

- The measured delay was an underestimate, because the first fit had already
  hidden part of it in the extrinsics. A camera ninety milliseconds late was
  reported at fifty-two. The measurement and the fit now alternate.
- Correcting the pairings was not enough on its own. The refinement was still
  starting from a resection done against a walk the camera had not caught up
  with, and its outlier rejection threw away the sightings that disagreed with
  that seed rather than pulling the camera back — 51 of 190 kept, and the camera
  39 cm from where it was. The resection is redone as well.

A room whose cameras are 0, 20, 40 and 90 ms late now comes out to 5 mm.
Ignoring the delays puts a camera nine metres away, which is the calibration
every real installation would have had. See [design.md](design.md) section 8.3.

That test picked delays that happened to work, and sweeping past them found two
more failures — which is the lesson as much as the fixes are.

- **A camera much more than sixty milliseconds late did not solve at all**, and
  ended the whole calibration on a message about correspondences, *before* the
  latency estimator ran. Three of the four test cameras failed at eighty
  milliseconds and all four at a hundred and ten, which is inside the range this
  same code calls plausible for a webcam. The resection turns out to be a sharp
  detector of its own delay, so the seed searches for one that solves. Forty to
  a hundred and twenty milliseconds now comes out within half a millimetre, and
  four cameras all seventy milliseconds late — a room furnished with one model
  of webcam — within nine.
- **A camera solved to nonsense did not look like one.** The outlier rejection
  throws away everything that disagreed with wherever the camera ended up, so
  what comes back is a few sightings with a small error over them and an `Ok`.
  One kept 5 of 175 sightings at 36° and stood two and a half metres from where
  it hangs. The solve now refuses that, and refuses a room fitted at a delay
  other than the one measured on it — which is the only way to see a timing
  error, since it moves every sighting the same way and leaves the reprojection
  perfectly happy about a camera in the wrong place.

What is left is a walk through a real room with the cameras where they will
live. Everything the milestone calls for is written and green against synthetic
rooms and against a Quest 3.

## M4 - Fusion

- Fixed-rate fusion clock with per-camera interpolation to a common timestamp.
  Done. The clock runs behind real time by the slowest camera's measured delay
  plus a slack, because interpolating onto an instant needs a frame after it.
- Angular-weighted triangulation with RANSAC and non-linear refinement, with
  per-camera contribution weights exposed to the UI. Done.
- Bone-length measurement step and constrained skeleton fit. Done, without the
  T-pose step the design called for; see [design.md](design.md).
- One Euro filtering and the constant-velocity Kalman predictor. Done, in that
  order rather than the design's; see [design.md](design.md).
- 3D viewport shows the live skeleton with per-joint confidence and residuals.
  Done, with the predicted skeleton drawn behind the measured one.

**Done when:** the 3D skeleton tracks the user smoothly, holds up under partial
occlusion, and per-joint residuals stay within a few pixels.

What this milestone found, all recorded in [design.md](design.md):

- The linear triangulation is biased towards the nearest cameras, and inverting
  the refinement's normal matrix gives the position covariance for free. The
  square root of its largest eigenvalue — how far a joint could be wrong along
  the direction the cameras constrain least — turned out to be far more useful
  than the reprojection residual. Two cameras a hand's width apart agree
  perfectly about a point neither can place: the residual says zero and the
  uncertainty says a third of a metre.
- The Kalman filter has to come before the One Euro filter, not after. Reversed,
  the velocity is measured from the smoothed signal and every prediction comes
  out short by exactly the amount the smoothing lagged.
- The smoothing costs no latency, because a first-order low pass lags by exactly
  its own time constant and that can simply be added to the prediction horizon.
- The speed that opens the adaptive cutoff has to be low-passed, or a joint
  standing still shows enough apparent speed to open the filter. At the one
  hertz that filter is usually given, a swinging leg is attenuated enough that
  every stride lags; three hertz is the trade.
- Bone lengths belong in their own file, not the room profile. A body belongs to
  a person and a room profile to a set of cameras.

`tests/fusion.rs` runs a simulated walk past three unlike cameras — 30/60/25 fps,
three resolutions, three fields of view, delays of 0, 40 and 90 ms — through the
whole chain and recovers the body to 4 mm. Treating those delays as zero gives
6.7 cm, which is the argument for temporal alignment in one number. Hiding a
knee from two of the three cameras leaves the fit to place it, and it does, to
4 cm.

What is left is the same thing M3 is waiting on: a real room.

## M5 - Tracker output

- Limb-frame orientation derivation for each tracker role.
- `TrackerSink` trait, VRChat OSC backend, VMT backend.
- Output thread with configurable send rate and prediction horizon.
- Output panel: sink selection, tracker enable list (3-8), per-tracker offsets.

**Done when:** VRChat full-body tracking works end to end with hips and both
feet, and the same session works through VMT under SteamVR.

Three things came out of building it, all recorded in
[design.md](design.md):

- **A knee and an elbow need opposite signs.** They are the same three points
  and the same cross product, and the bend plane normal points forwards for one
  and backwards for the other, because the kneecap is on the front of the body
  and the point of the elbow is on the back. Caught by a test, not by reading.
- **The prediction horizon should never have been a camera setting.** The
  distance from a frame being exposed to a reconstruction existing is measured
  by the fusion stage. Only the hop out and the consumer's own delay are
  unknowable from here, so that is all the setting covers now, and its default
  dropped from 60 ms to 20.
- **The fallbacks are the common case, not the exception.** A straight knee has
  no bend plane and a standing person's knees are straight; a foot without heel
  and toe keypoints has no yaw of its own. Both fall back to the hip axis.

First run in VRChat: the trackers were recognised over OSC and the coordinate
conversions held, but the body vibrated on the spot. The cause was in the
filter rather than in anything this milestone added -- the prediction was
riding on an unsmoothed velocity, and the term that pays back the smoothing lag
is scaled by the largest number in the filter exactly when the joint is still.
See section 9.5 of [design.md](design.md); the fix cost 1.9 cm on a simulated
walk and made a standing body steadier than its own measurement noise.

The layout tests could not have caught it, and neither could the filter tests as
they stood: they judged the *smoothed position*, and what the output stage sends
is the *prediction*, which reaches further. Testing the wrong end of a filter is
an easy thing to do for a long time.

Untested against a real consumer: everything downstream is another process, so
the tests send to a loopback socket and decode what arrived. The conversions
they check are the ones this project believes are right, which is not the same
as the ones VRChat implements.

Second run: it still shook, and the skeleton on screen shook with it — which
ruled out the prediction and everything else past the reconstruction. Two rounds
of guessing at a cause was one too many, so the first thing added was a way to
tell the four stages apart: the second difference of each joint's position,
measured at the reconstruction, the fit, the smoothing and the prediction, and
printed as a row on the Tracking panel. Constant-velocity motion contributes
nothing to it, so it separates shaking from walking. See section 9.6 of
[design.md](design.md).

Two faults upstream came out of looking properly, and both had been visible on
the panel all along without being recognised:

- **The fusion clock did not wait for the cameras.** Its lag was the largest
  calibrated camera latency plus a fixed margin, worked out once before any
  camera had delivered anything, and it left out the term that dominates: the
  time from a camera grabbing a frame to that frame's keypoints existing, which
  here was nearly two hundred milliseconds against a forty-millisecond margin.
  The consequence is not a graceful loss of accuracy. A camera the clock does
  not wait for has a bracketing pair on some ticks and not others, so it drops
  in and out of the reconstruction, and a joint reconstructed from a different
  set of cameras every few ticks moves by the disagreement between them each
  time the set changes — the calibration error, centimetres, arriving as a
  square wave rather than as noise. The panel had been reporting alignment
  fractions of 44-79% since the milestone was written. The clock now follows
  what the cameras actually deliver, and a camera past a ceiling is dropped
  outright with a reason rather than flickering.
- **The reported uncertainty was a prediction of the error, never a measurement
  of one.** A triangulation's covariance is built entirely from the noise each
  ray *claimed*, and three well spread rays pin a point down beautifully whether
  or not the cameras agree about where anything is — so a room calibrated to
  three centimetres reported joints good to five millimetres, and the filter,
  the panels and both withholding limits all believed it. The residuals were the
  missing measurement, computed and printed in degrees and never once allowed to
  say anything about the answer. Scaling the covariance by the ratio is the
  standard a posteriori variance factor; the Tracking panel now shows the factor
  on its own, since it is the only thing in the application that can notice a
  camera has been knocked since it was solved.

Both were reported by the existing diagnostics and neither was read as a fault.
The lesson is not that more numbers were needed but that a number nobody has
been told the healthy value of is not a diagnostic.

## M6 - Tuning and release

- A way to measure accuracy without a room, so that a change to a model or to
  the chain can be judged before anyone puts cameras on a ceiling. Done; see
  section 14 of [design.md](design.md).
- Room profile management, multiple named profiles.
- Startup self-check: cameras present, calibration loaded, models available.
- Performance pass: allocation reuse in the hot path, preview downscaling,
  optional per-camera inference rate limits.
- User documentation: camera placement guide, calibration guide,
  troubleshooting.
- Release build configuration and packaging.

**Done when:** a fresh install can be taken from download to working full-body
tracking using only the in-app documentation.

The harness came first because every other test in the project begins after
inference. `sim` renders a walking figure in a tiled room from four unlike
ceiling cameras and `tests/accuracy.rs` runs the real detector and the real pose
model over those frames, then the real triangulation, fit and filter. The figure
is forward kinematics from stated bone lengths, so the answer is known exactly.

Numbers over a fourteen second walk: a person found in all 1120 camera frames,
the lower body 2.8 cm from the truth with 1.9 cm of spread once the model's own
labelling offset is taken out, against 0.5 cm from perfectly projected
keypoints. The gap between those is what inference contributes, and it had
never been possible to look at.

Six things came out of building it and then reviewing it adversarially:

- **A pose model's joints are where its training set was annotated, not where
  the bone is.** Halpe's hip-to-neck measures six centimetres longer than the
  body it was drawn from, every frame, with almost no scatter. That is a
  convention rather than an error, and the per-tracker offsets already in the
  output stage are what absorb it — but a report that added it into one error
  figure would have made a well-behaved model look four times worse than it is.
  The harness reports the constant part and the scatter separately.
- **The hips come back mirrored on 9% of ticks.** A foot tracker on the wrong
  foot is a specific failure with a name, and a mean would have buried it in a
  tail. The hips are the worst of it and matter more than they look: the pelvis
  yaw is taken from the vector between them, so a pair that trades places turns
  the tracker round. Ankles, heels and toes swap on under one per cent.
- **The harness was measuring a value the product never sends.** It scored the
  smoother, and what reaches a tracker is the filter's extrapolation one horizon
  further on. Scoring the right thing, against a sweep of instants rather than
  one, says the geometry is accurate to half a centimetre and the filter loses
  most of it: what goes out is 4.6 cm from the truth and 80 ms behind the moment
  it claims to describe. The cause is the velocity credibility gate, which holds
  a joint walking at 0.3 m/s to none of its own speed at all.
- **Two synthetic bodies wanting opposite settings is the answer, not a
  blocker.** Lowering the process noise halves the error on the walking body and
  makes the fast one in `tests/fusion.rs` half as good again. The gate weighs a
  velocity against a noise floor set by the cameras and the pose model, so where
  it belongs is a property of the user's room. Both ends are now settings at
  their existing defaults — `Body agility` and `Prediction caution` — and the
  Tracking panel shows how much of the measured speed the prediction is reaching
  for, beside the shake figures, so that neither can be traded for the other
  blind. Sections 9.5 and 14.5 of [design.md](design.md).
- **Rendering has to be deterministic to be worth asserting on.** The renderer
  is a software rasteriser rather than the GPU already in the process, because a
  threshold tight enough to be useful would otherwise fail on somebody else's
  driver.
- **The simulation was wrong first.** The bone meter reported forty per cent
  scatter on ankle-to-heel and refused to name a length, correctly: the foot was
  hung off the floor rather than off the ankle, so it stretched every time the
  ankle lifted, and the two feet were drawn to different lengths. Every foot
  number would have been measured against that.
- **A test that passes is not the same as a feature that works.** The
  calibration test picked camera delays that happened to sit inside a narrow
  working range. Sweeping past it found that three of the four cameras failed to
  solve at all at eighty milliseconds — inside the range the code itself calls
  plausible for a webcam — and that two more delays produced a room metres out
  and returned it as `Ok`. Both are fixed; M3 has the detail.

What it cannot do is predict a real room. The figure is a rendered
approximation and real skin, fabric, motion blur and lighting are all absent.
These numbers are a floor and a regression detector.

## Deferred

These are deliberately out of scope until the milestones above are complete:

- CUDA / TensorRT execution providers.
- Linux support.
- Multi-person tracking.
- Hand and finger tracking.
