# Calibrating a room

Calibration is what tells Optra where each camera hangs, which way it points and
how far behind it runs. It is done once per camera layout and reused; it has to
be redone when a camera moves.

There is no printed board to hold. The headset you are already wearing is the
reference: Optra records where SteamVR says your head is and where each camera
saw your head, over a few hundred frames of you walking around, and solves for
the cameras from those pairs. That is also why the result lands directly in your
play space, with nothing to line up afterwards.

## Before you start

The Calibration panel lists these, and will not start a walk without them:

- SteamVR running, with the headset and both controllers tracked.
- Every camera you intend to use streaming, with a pose model attached, and
  keypoints visible on its preview.
- The cameras in their final positions. A camera moved after the walk is a
  camera the profile is wrong about.

## The walk

Press record, put the headset on, and walk. Aim for two to three minutes.

- **Cover the floor.** Every part of the room you will actually stand in, and
  the parts near each camera as well as the middle.
- **Change height.** Stand, crouch, kneel, reach up. A walk that never changes
  height is close to a flat set of points, which is degenerate for the solve: it
  can produce a small residual and a camera metres from where it really is. The
  panel warns while this is happening rather than after.
- **Turn your head — including up and down.** The solver has to work out how far
  the headset sits from the head keypoint the cameras see, and it can only do
  that from your head *rotating*. Looking only left and right is not enough: yaw
  alone leaves the whole room free to slide vertically against that offset, and
  nothing in the numbers objects. Look at the ceiling and at the floor a few
  times.
- **Raise and lower your hands.** The controllers contribute their own pairs
  against your wrists, and they reach places your head does not.
- **Walk, do not stroll.** The camera delays are measured from how the walk
  moves. A walk too slow to leave a mark makes the delays unmeasurable, and the
  panel says so rather than guessing.

The coverage map fills in per camera, not for the room as a whole: a narrow
camera sees a smaller slice, so a walk that satisfies a wide one can leave a
narrow one thin. Keep going until every camera's map is reasonably filled.

## Reading the result

The solve reports several things because several different things can be wrong,
and no single number catches them all.

| What | What it means |
|---|---|
| **RMS error** | How well the cameras agree with each other, in degrees, and what that is worth in millimetres at the distance you actually walked. The millimetres are the number to judge on. |
| **Coverage** | How much of that camera's frame the walk visited. Low coverage means the camera is solved from one corner of its own view. |
| **Spread** | How far the correspondences were from lying in a plane. Near zero means the answer rests on nothing, however small the error looks. |
| **Delay** | How far behind that camera runs, in milliseconds. Anything up to about 120 ms is ordinary for a webcam. |
| **Placement precision** | How well these camera positions can locate a joint at all — a property of where they hang, not of how well they were solved. Reported separately at ankle height, which is the one that matters here and the one a walk done entirely above the waist would otherwise flatter. |
| **Feet** | The fraction of this camera's sightings of you that included a foot. A camera that solved beautifully and never sees a leg is aimed at the wrong place. |

Look at the 3D view as well as the table. A calibration can be perfectly
self-consistent and still have a camera on the wrong side of a wall, and that is
obvious in a picture and invisible in a residual.

## When it refuses

A calibration that comes out subtly wrong is worse than one that will not
finish, so the solve refuses several things outright:

- **A camera that kept only a handful of its sightings**, or came out tens of
  degrees away from the ones it kept. That is a camera solved to nonsense, and
  without the check it would be returned as a success.
- **A room fitted at a delay other than the one measured on it.** A timing error
  moves every sighting the same way, so the reprojection stays perfectly happy
  about a camera in the wrong place; this is the only check that can see it.
- **Not enough of the walk visible to a camera.** Usually the camera is aimed
  somewhere you did not walk, or it is dropping most of its frames.

If a camera is named, check its measured frame rate and dropped-frame count in
the Cameras panel first. Walking again does not help a camera that is not
delivering frames.

## Saving and reusing

Give the profile a name and save it. Profiles belong to a camera layout, so keep
one per room, and pick the one you want at the top of the Calibration panel. A
profile for a layout that no longer exists is worth deleting from the same list:
it is not clutter but a trap, since loading it gives you cameras solved for a
room that has been dismantled.
Your measured bone lengths are *not* part of it — a body belongs to a person and
a profile belongs to a set of cameras — so moving between rooms does not make
you re-measure yourself.

## Checking it later

The Tracking panel keeps checking the room against the headset while you use it,
which is the only thing in the application that can notice a camera has been
knocked since it was solved:

- **Room scale** — how far the headset moved against how far the cameras thought
  your head moved. One is a room solved life size. Anything else is a body that
  will never line up with an avatar, however carefully the trackers are
  calibrated in game, and it means solving the room again over more floor and
  more height.
- **Head** — the offset between where the headset is and where the cameras place
  your head, as a vector. A nearly vertical offset points at SteamVR's room
  setup; anything else points at the calibration.
- **Cameras agree to** — the factor by which the cameras' actual disagreement
  exceeds what they claimed. It rises when a camera has moved.

Read those in that order. Everything else on the panel is expressed in the frame
the head check is judging.
