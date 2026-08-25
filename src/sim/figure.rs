//! Hanging a surface on a posture.
//!
//! The figure is built from lofted cross-sections and tapered limbs rather than
//! from a rigged model file, for one reason: the harness that measures accuracy
//! has to be able to say where every joint truly is, and a mesh generated from
//! the same joint positions cannot drift out of step with them. An imported
//! model would need its own rig, its own retarget, and its own separate claim
//! about where a hip is.
//!
//! It does not need to be beautiful. It needs a person-shaped silhouette with
//! the limbs in the right places, shaded, against a background it stands out
//! from — which is what the detector and the pose model are looking for.

use nalgebra::{Point3, Vector3};

use crate::models::Joint;
use crate::sim::body::{Anatomy, Posture};
use crate::sim::mesh::{Mesh, Section};

/// The flesh on the bones, and what it is wearing.
///
/// Limb radii are derived from the anatomy rather than listed, so a taller
/// figure is proportioned rather than merely stretched. `build` scales all of
/// them together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    /// Scales every limb radius. 1.0 is an average adult.
    pub build: f64,
    pub skin: [f32; 3],
    pub hair: [f32; 3],
    /// Torso and upper arms.
    pub shirt: [f32; 3],
    /// Pelvis and legs.
    pub trousers: [f32; 3],
    pub shoes: [f32; 3],
}

impl Default for Shape {
    fn default() -> Self {
        Self {
            build: 1.0,
            skin: [0.88, 0.72, 0.62],
            hair: [0.16, 0.13, 0.12],
            // Deliberately unlike the floor and unlike each other: a body that
            // matches its background is a body a detector has to guess at, and
            // a torso that matches its own legs hides the waist a pose model
            // uses to place the hips.
            shirt: [0.28, 0.42, 0.70],
            trousers: [0.26, 0.27, 0.32],
            shoes: [0.10, 0.10, 0.12],
        }
    }
}

/// Builds the figure's surface for one posture.
pub fn build(anatomy: &Anatomy, shape: &Shape, posture: &Posture) -> Mesh {
    let mut mesh = Mesh::default();
    let scale = shape.build;

    // Radii as fractions of the bone they sit on. A thigh is roughly a quarter
    // of its own length across at the top and half that at the knee.
    let thigh_top = anatomy.thigh * 0.24 * scale;
    let knee = anatomy.thigh * 0.16 * scale;
    let calf = anatomy.shin * 0.17 * scale;
    let ankle = anatomy.shin * 0.10 * scale;
    let shoulder = anatomy.upper_arm * 0.20 * scale;
    let elbow = anatomy.upper_arm * 0.14 * scale;
    let wrist = anatomy.forearm * 0.10 * scale;

    let at = |joint: Joint| posture.get(joint).expect("the walk fills every joint");

    for (hip, knee_joint, ankle_joint, heel, big_toe, small_toe) in [
        (
            Joint::LeftHip,
            Joint::LeftKnee,
            Joint::LeftAnkle,
            Joint::LeftHeel,
            Joint::LeftBigToe,
            Joint::LeftSmallToe,
        ),
        (
            Joint::RightHip,
            Joint::RightKnee,
            Joint::RightAnkle,
            Joint::RightHeel,
            Joint::RightBigToe,
            Joint::RightSmallToe,
        ),
    ] {
        mesh.add_limb(at(hip), at(knee_joint), thigh_top, knee, shape.trousers);
        // The calf is widest a little below the knee, so the shin is two
        // segments rather than one straight taper.
        let shin = at(ankle_joint) - at(knee_joint);
        let belly = at(knee_joint) + shin * 0.3;
        mesh.add_limb(at(knee_joint), belly, knee, calf, shape.trousers);
        mesh.add_limb(belly, at(ankle_joint), calf, ankle, shape.trousers);

        shoe(
            &mut mesh,
            at(heel),
            nalgebra::center(&at(big_toe), &at(small_toe)),
            posture.right,
            shape.shoes,
            scale,
        );
    }

    torso(&mut mesh, anatomy, shape, posture, scale);

    for (shoulder_joint, elbow_joint, wrist_joint) in [
        (Joint::LeftShoulder, Joint::LeftElbow, Joint::LeftWrist),
        (Joint::RightShoulder, Joint::RightElbow, Joint::RightWrist),
    ] {
        mesh.add_limb(
            at(shoulder_joint),
            at(elbow_joint),
            shoulder,
            elbow,
            shape.shirt,
        );
        mesh.add_limb(at(elbow_joint), at(wrist_joint), elbow, wrist, shape.skin);
        // A hand, as one blunt shape. The wrist keypoint is the last one any
        // tracker is built from, so nothing downstream cares about fingers.
        let forearm = (at(wrist_joint) - at(elbow_joint)).normalize();
        mesh.add_limb(
            at(wrist_joint),
            at(wrist_joint) + forearm * (0.09 * scale),
            wrist * 1.2,
            wrist * 0.9,
            shape.skin,
        );
    }

    head(&mut mesh, anatomy, shape, posture, scale);

    mesh
}

/// The torso, as two lofted stacks: the pelvis in one colour and the chest in
/// another, so the waist is visible from directly above. A ceiling camera sees
/// a person almost entirely as a torso, and a torso with no features on it is
/// the hardest thing in the room to place hips on.
fn torso(mesh: &mut Mesh, anatomy: &Anatomy, shape: &Shape, posture: &Posture, scale: f64) {
    let hip = posture.get(Joint::Hip).expect("the pelvis");
    let neck = posture.get(Joint::Neck).expect("the neck");
    let up = (neck - hip).normalize();
    let (right, forward) = (posture.right, posture.facing);

    let section = |height: f64, half_width: f64, half_depth: f64| Section {
        centre: hip + up * height,
        right,
        forward,
        half_right: half_width * scale,
        half_forward: half_depth * scale,
    };

    let spine = anatomy.spine;
    mesh.add_loft(
        &[
            section(-0.12, 0.140, 0.100),
            section(-0.04, 0.158, 0.112),
            section(0.06, 0.150, 0.106),
            section(0.14, 0.138, 0.100),
        ],
        shape.trousers,
        16,
        true,
    );
    mesh.add_loft(
        &[
            section(0.10, 0.140, 0.101),
            section(spine * 0.45, 0.136, 0.100),
            section(spine * 0.72, 0.166, 0.117),
            section(spine * 0.95, 0.172, 0.112),
            section(spine, 0.160, 0.104),
        ],
        shape.shirt,
        16,
        true,
    );

    // The neck, which is what stops the head floating.
    mesh.add_limb(
        neck,
        neck + up * (anatomy.neck * 0.45),
        0.055 * scale,
        0.050 * scale,
        shape.skin,
    );
}

/// The head, with just enough face to tell front from back.
///
/// Which way a head is pointing is most of what a pose model has to go on for
/// the direction the body is facing, and a featureless sphere is symmetric.
fn head(mesh: &mut Mesh, anatomy: &Anatomy, shape: &Shape, posture: &Posture, scale: f64) {
    let centre = posture.get(Joint::Head).expect("the head");
    let (right, facing, up) = (posture.right, posture.facing, Vector3::y());
    let r = anatomy.head_radius;

    mesh.add_ellipsoid(
        centre,
        [right * (r * 0.86), up * (r * 1.12), facing * (r * 0.98)],
        shape.skin,
        20,
        14,
    );
    // A cap sitting above the eye line, so it reads as hair rather than as a
    // mask over the face.
    mesh.add_ellipsoid(
        centre + up * (r * 0.34),
        [right * (r * 0.90), up * (r * 0.86), facing * (r * 1.02)],
        shape.hair,
        20,
        12,
    );

    for eye in [Joint::LeftEye, Joint::RightEye] {
        let at = posture.get(eye).expect("an eye");
        mesh.add_sphere(at, 0.014 * scale, [0.12, 0.11, 0.10], 10, 6);
    }
    let nose = posture.get(Joint::Nose).expect("the nose");
    mesh.add_sphere(nose, 0.022 * scale, shape.skin, 10, 6);
    mesh.add_box(
        nose - up * (r * 0.45) - facing * (r * 0.05),
        [
            right * (0.026 * scale),
            up * (0.006 * scale),
            facing * (0.010 * scale),
        ],
        [0.52, 0.30, 0.28],
    );
}

/// A shoe, as one box from the heel to the toes.
fn shoe(
    mesh: &mut Mesh,
    heel: Point3<f64>,
    toe: Point3<f64>,
    body_right: Vector3<f64>,
    color: [f32; 3],
    scale: f64,
) {
    let along = toe - heel;
    let length = along.norm();
    if length < 1e-6 {
        return;
    }
    // The foot has its own heading, which is not the body's once a leg swings
    // out; the body's right only breaks the tie about which way is up.
    let forward = along / length;
    let side = forward.cross(&Vector3::y());
    let side = if side.norm() > 1e-6 {
        side.normalize()
    } else {
        body_right
    };

    mesh.add_box(
        nalgebra::center(&heel, &toe) + Vector3::y() * (0.012 * scale),
        [
            forward * (length * 0.5 + 0.025 * scale),
            Vector3::y() * (0.038 * scale),
            side * (0.048 * scale),
        ],
        color,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::body::Walk;

    fn figure() -> (Anatomy, Posture, Mesh) {
        let anatomy = Anatomy::default();
        let posture = Walk::default().posture(&anatomy, 1.7);
        let mesh = build(&anatomy, &Shape::default(), &posture);
        (anatomy, posture, mesh)
    }

    #[test]
    fn the_figure_is_built_from_a_useful_number_of_triangles() {
        let (_, _, mesh) = figure();
        // Enough to be round, few enough that rendering a walk is not an
        // overnight job.
        assert!(mesh.triangle_count() > 2_000, "{}", mesh.triangle_count());
        assert!(mesh.triangle_count() < 40_000, "{}", mesh.triangle_count());
    }

    /// Every joint has to be inside the surface. A knee sticking out of the
    /// trouser leg would mean the pixels a pose model sees and the truth the
    /// harness compares against disagree about where the leg is.
    #[test]
    fn every_joint_lies_inside_the_surface() {
        let (_, posture, mesh) = figure();
        for (joint, point) in posture.iter() {
            let nearest = mesh
                .vertices
                .iter()
                .map(|vertex| (vertex.position - point).norm())
                .fold(f64::MAX, f64::min);
            assert!(
                nearest < 0.12,
                "{joint:?} is {nearest:.3} m from any surface, so nothing is drawn there"
            );
        }
    }

    #[test]
    fn the_figure_stands_on_the_floor_and_no_lower() {
        let (anatomy, _, mesh) = figure();
        let lowest = mesh
            .vertices
            .iter()
            .map(|vertex| vertex.position.y)
            .fold(f64::MAX, f64::min);
        let highest = mesh
            .vertices
            .iter()
            .map(|vertex| vertex.position.y)
            .fold(f64::MIN, f64::max);

        assert!(lowest > -0.02, "the figure sank into the floor: {lowest}");
        assert!(lowest < 0.05, "the figure floats: {lowest}");
        assert!(
            (highest - anatomy.standing_height()).abs() < 0.10,
            "the figure is {highest:.2} m tall, not {:.2} m",
            anatomy.standing_height()
        );
    }

    #[test]
    fn the_figure_is_the_width_of_a_person_and_not_of_a_room() {
        let (_, posture, mesh) = figure();
        let hip = posture.get(Joint::Hip).unwrap();
        let widest = mesh
            .vertices
            .iter()
            .map(|vertex| (vertex.position - hip).dot(&posture.right).abs())
            .fold(0.0f64, f64::max);
        assert!(
            (0.25..0.55).contains(&widest),
            "half a shoulder span should be about 0.3 m, got {widest:.3}"
        );
    }

    /// The colours have to differ, or the figure is one silhouette with no
    /// waist, no sleeves and no shoes in it.
    #[test]
    fn the_figure_is_not_all_one_colour() {
        let (_, _, mesh) = figure();
        let mut seen: Vec<[f32; 3]> = Vec::new();
        for vertex in &mesh.vertices {
            if !seen.contains(&vertex.color) {
                seen.push(vertex.color);
            }
        }
        assert!(seen.len() >= 5, "only {} colours on the figure", seen.len());
    }

    #[test]
    fn a_walk_moves_the_surface_with_the_body() {
        let anatomy = Anatomy::default();
        let shape = Shape::default();
        let walk = Walk::default();

        let first = build(&anatomy, &shape, &walk.posture(&anatomy, 0.0));
        let later = build(&anatomy, &shape, &walk.posture(&anatomy, 1.4));

        assert_eq!(first.vertices.len(), later.vertices.len());
        let moved = first
            .vertices
            .iter()
            .zip(&later.vertices)
            .map(|(a, b)| (a.position - b.position).norm())
            .fold(0.0f64, f64::max);
        assert!(moved > 0.5, "the figure barely moved: {moved:.3} m");
    }
}
