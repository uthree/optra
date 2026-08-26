# Troubleshooting

Optra reports itself through three places: the banner that appears at startup
when something is missing, the tables on the Tracking panel while it runs, and
the log. Start with whichever one is already telling you something.

**The log is a file.** `%APPDATA%\optra\logs\optra.log`, opened from the button
on the Log panel. The panel itself only holds the last few thousand records,
which during tracking is a matter of seconds — anything that happened more than
a moment ago is in the file and nowhere else. Send that file with any report.

## Nothing starts

**A banner says a camera is not attached.** It names the camera as Windows
enumerated it. Replug it — the same port, if you can — and press *Check again*.
Optra identifies cameras by device path so that re-plugging cannot silently swap
two of them and quietly invalidate a calibration.

**A banner says a model has not been downloaded.** Install it from the Models
panel. Nothing is bundled with the application.

**A banner says no room profile is selected.** See [calibration.md](calibration.md).

**The Tracking panel says "Not tracking".** It lists its own prerequisites:
fusion enabled, and at least two cameras running with a model. The Cameras and
Models panels are where those are turned on.

**No capture devices found at all.** Another application may hold the camera
open — Windows lets one process at a time have a webcam. Close anything else
that might be using it, including a browser tab in a video call.

## The picture is fine and the frame rate is not

**Half the frame rate you asked for.** Almost always automatic exposure: the
camera lengthens its shutter in dim light until it cannot deliver the rate it
advertises. Turn Auto off for exposure in the Cameras panel and use *Fit frame
rate*. See [cameras.md](cameras.md).

**Dropped frames climbing on some cameras.** USB bandwidth. Move cameras onto
different controllers, prefer MJPEG, drop resolution before frame rate.

**A camera shows a low "Keeping up" percentage on the Tracking panel.** The
fusion clock could not find frames either side of a tick for that camera, so it
drops in and out of the reconstruction — and a joint reconstructed from a
different set of cameras every few ticks moves by the disagreement between them
each time the set changes. Check that camera's frame rate and dropped frames
first; *Alignment slack* and *Wait at most* in the Tracking panel's settings are
the knobs for a camera that is simply slow.

## The body shakes

The Tracking panel prints how much each stage of the chain moves in a way a body
could not: **cameras**, **fit**, **smoothed**, **sent**. Read them left to
right; the first one that is large is where the shaking starts. Constant-velocity
motion contributes nothing to these, so walking around does not inflate them.

- **Large at "cameras"** — the reconstruction itself is noisy. Look at the
  camera table: a camera with a steadily high *Outvoted* figure is
  mis-calibrated or badly placed. Check *Cameras agree to*, which reads as a
  multiple of what the pose models claimed their own keypoints were worth: well
  above 1.5x means the profile no longer describes where the cameras are.
- **Large first at "fit"** — the skeleton fit is fighting the measurement,
  usually because a bone length was measured from a bad stretch of tracking.
  Turn *Keep measuring* back on in the Body section and move around.
- **Large first at "smoothed" or "sent"** — a filter setting. Lower *Body
  agility* to trust the measured velocity less, or raise *Prediction caution*.
  Watch **Prediction reaching** while you do: it is the same trade from the
  other side, and a prediction that reaches for almost none of your real speed
  is a body that lags behind you instead of shaking.

## The feet are in the wrong place

In this order, on the Tracking panel:

1. **Room scale.** Anything other than about 1.0 means the room was solved at
   the wrong size, and every other number is expressed in that wrong frame. A
   uniformly scaled set of cameras is perfectly self-consistent — the residual
   is happy, the cameras all agree — so nothing else can see this. Fix it by
   calibrating again over more floor and more height.
2. **Head.** How far the reconstructed head sits from the headset. A nearly
   vertical offset is SteamVR's room setup at the wrong height; anything else is
   the calibration.
3. **Floor.** Where the feet land against where SteamVR thinks the floor is.
   Reported, never corrected: a wrong floor means the profile is wrong, and a
   correction built on it would be wrong too.

If the room was calibrated with cameras that have since been moved, none of
these can be fixed from the panel. Calibrate again.

## Nothing arrives in VRChat

- OSC has to be enabled in VRChat itself; it is off by default.
- The default destination is `127.0.0.1:9000`, which is what VRChat listens on
  when both run on the same machine. The Output panel's *Destination* is where
  to change it.
- Trackers have to be enabled in the Output panel. Hips and both feet are the
  default; the rest are individual.
- VRChat places the trackers relative to the head, so Optra sends the headset
  pose alongside them. That is on by construction and not a setting.
- After the trackers appear, calibrate full-body tracking in VRChat as you would
  with any other trackers.

## Nothing arrives in SteamVR through VMT

- VMT's default destination is `127.0.0.1:39570`.
- VMT places devices in the runtime's *raw* space, which differs from the
  standing universe by whatever SteamVR's room setup did. The Output panel can
  send SteamVR's own room matrix to VMT as a temporary setting, which saves
  configuring the same thing twice. Without it, everything is offset by your
  floor height and play-space centre.

## A tracker is on the wrong foot

The pose models occasionally swap left and right, most often at the hips. A
swapped pair turns the pelvis around, because its yaw comes from the vector
between them. Measured against the synthetic harness this happens on about 9% of
ticks for the hips and under 1% for ankles, heels and toes; it is a known
limitation of the current chain rather than a setting.

## The numbers all look fine and it still feels wrong

Compare what is drawn on the Tracking panel: solid is the measurement, faint is
the prediction, and the prediction is what is actually sent. A body that looks
steady in solid and shakes in faint is a filter problem, not a camera one.

If you are reporting a problem, the log file is worth more than a description of
the symptom. The solve writes its whole result there, including where each
camera came out and what delay it was measured at.
