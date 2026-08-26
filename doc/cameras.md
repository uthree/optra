# Placing the cameras

Where the cameras go decides how well Optra can ever work. The calibration can
only tell you where they are; it cannot make a pair of cameras that both look at
your legs from the same direction place a foot any better. This is the part
worth spending an afternoon on.

## How many

Two is the minimum, because one camera gives a direction to a joint and no
distance along it. Three or four is what a room should have: a leg that one
camera loses behind the other leg is still seen by the rest, and every extra
view narrows the answer.

Optra supports two to four, and they do not have to match. Different
resolutions, fields of view, frame rates and even different pose models are all
fine — everything downstream of capture works in angles rather than pixels
precisely so that a wide 480p camera and a narrow 1080p one can be combined
without one drowning the other out.

## Where

- **High, and spread around the room.** Ceiling corners are the intended
  arrangement. What matters is the angle *between* cameras as seen from where
  you stand: two cameras side by side agree perfectly about a point neither of
  them can place, and the calibration report will say so — that is what the
  precision figures mean.
- **Aim at knee height, not at your chest.** A ceiling camera is already looking
  down, and aiming it at the middle of a body pushes the feet of anyone standing
  nearby out of the bottom of the frame. Everything above the waist is seen by
  every camera anyway; the feet are the part that is easily lost, and they are
  the reason this application exists.
- **Every camera should see the floor you actually stand on.** After a
  calibration, each camera reports what fraction of its sightings of you
  included a foot. A camera that solved perfectly and never sees a leg
  contributes nothing to what you are tracking.
- **Fixed, and left alone.** The extrinsics are solved once and reused. A camera
  that gets knocked or re-aimed invalidates the profile it was solved in, and
  the symptom is not an error message — it is feet in the wrong place. Mount
  them so that a cable tug cannot move them.

## Lenses

A wide-angle camera in a ceiling corner sees far more of a small room than a
narrow one, and that is usually the right trade. Past roughly 120 degrees, pick
the fisheye (equidistant) lens model for that camera in the Cameras panel: the
radial model does not merely lose accuracy that far out, it stops being
invertible, and the solve will not converge.

## USB and frame rate

Several cameras streaming at once is a bandwidth problem before it is anything
else. If cameras drop frames or refuse to start together:

- Spread them across different USB controllers rather than one hub.
- Prefer MJPEG over raw formats.
- Drop resolution before dropping frame rate. Triangulation cares more about
  seeing the leg move than about seeing it in detail.

The Cameras panel shows the measured frame rate and a dropped-frame count per
camera. Those are the numbers to watch while you try combinations; the frame
rate a device *claims* after a format is set cannot be trusted on Windows.

## Exposure is not optional

A webcam left on automatic exposure lengthens its shutter in dim light until it
can no longer deliver the frame rate it advertises, and smears anything moving
across the frame. Both are fatal here. Measured during development: a Logitech
C920 at 1280x720 delivered 15.0 fps on automatic exposure and 30.1 fps with the
shutter pinned, in the same room and the same light.

So, in the Cameras panel, under **Device properties**:

1. Turn **Auto** off for exposure.
2. Use **Fit frame rate** to pin the shutter short enough for the frame rate you
   asked for, then raise gain or add light until the picture is bright enough.
3. Do the same for focus, if the camera hunts.

A slightly noisy bright image beats a clean blurred one: the pose model is
robust to noise and cannot recover a limb that was smeared across four
centimetres of sensor.

## Then

Once the cameras stream steadily with keypoints drawn on the previews, the room
is ready to be [calibrated](calibration.md).
