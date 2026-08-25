//! Triangle meshes, and the handful of shapes the simulated room is built from.
//!
//! Everything here is deliberately plain: positions, normals and a colour per
//! vertex, with no textures, no materials and no scene graph. The renderer that
//! consumes it is a software rasteriser, and the reason it exists is that a
//! test asserting a millimetre figure has to produce the same pixels on every
//! machine that runs it.

use nalgebra::{Point3, Vector3};

/// One vertex. The colour is the surface albedo, before any lighting.
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Point3<f64>,
    pub normal: Vector3<f64>,
    pub color: [f32; 3],
}

/// One cross-section of a lofted shape: an ellipse in the plane spanned by
/// `right` and `forward`, centred on `centre`.
#[derive(Debug, Clone, Copy)]
pub struct Section {
    pub centre: Point3<f64>,
    pub right: Vector3<f64>,
    pub forward: Vector3<f64>,
    pub half_right: f64,
    pub half_forward: f64,
}

/// An indexed triangle mesh.
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub triangles: Vec<[u32; 3]>,
}

impl Mesh {
    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.triangles.is_empty()
    }

    /// Appends another mesh, renumbering its triangles.
    pub fn merge(&mut self, other: &Mesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.triangles.extend(
            other
                .triangles
                .iter()
                .map(|[a, b, c]| [a + base, b + base, c + base]),
        );
    }

    fn vertex(&mut self, position: Point3<f64>, normal: Vector3<f64>, color: [f32; 3]) -> u32 {
        let index = self.vertices.len() as u32;
        self.vertices.push(Vertex {
            position,
            normal: unit_or(normal, Vector3::y()),
            color,
        });
        index
    }

    fn face(&mut self, a: u32, b: u32, c: u32) {
        self.triangles.push([a, b, c]);
    }

    /// A flat quad, wound so that its normal follows the right-hand rule around
    /// the corners in the order given.
    pub fn add_quad(&mut self, corners: [Point3<f64>; 4], color: [f32; 3]) {
        let normal = (corners[1] - corners[0]).cross(&(corners[3] - corners[0]));
        let indices: Vec<u32> = corners
            .iter()
            .map(|corner| self.vertex(*corner, normal, color))
            .collect();
        self.face(indices[0], indices[1], indices[2]);
        self.face(indices[0], indices[2], indices[3]);
    }

    /// A box, given its centre and three half-extent vectors.
    pub fn add_box(&mut self, centre: Point3<f64>, axes: [Vector3<f64>; 3], color: [f32; 3]) {
        let [x, y, z] = axes;
        // Each face is a quad wound outwards, so the normals point out of the
        // solid rather than into it.
        for (u, v, w) in [
            (x, y, z),
            (-x, z, y),
            (y, z, x),
            (-y, x, z),
            (z, x, y),
            (-z, y, x),
        ] {
            let face = centre + u;
            self.add_quad(
                [face - v - w, face + v - w, face + v + w, face - v + w],
                color,
            );
        }
    }

    /// A UV sphere.
    pub fn add_sphere(
        &mut self,
        centre: Point3<f64>,
        radius: f64,
        color: [f32; 3],
        segments: usize,
        rings: usize,
    ) {
        self.add_ellipsoid(
            centre,
            [
                Vector3::x() * radius,
                Vector3::y() * radius,
                Vector3::z() * radius,
            ],
            color,
            segments,
            rings,
        );
    }

    /// A sphere scaled and rotated by three half-extent vectors, which are
    /// assumed orthogonal. Used for the head, which is taller than it is wide.
    pub fn add_ellipsoid(
        &mut self,
        centre: Point3<f64>,
        axes: [Vector3<f64>; 3],
        color: [f32; 3],
        segments: usize,
        rings: usize,
    ) {
        let segments = segments.max(3);
        let rings = rings.max(2);
        let [ax, ay, az] = axes;
        let base = self.vertices.len() as u32;

        for ring in 0..=rings {
            // Latitude from the south pole to the north.
            let phi = std::f64::consts::PI * (ring as f64 / rings as f64 - 0.5);
            let (sin_phi, cos_phi) = phi.sin_cos();
            for segment in 0..=segments {
                let theta = std::f64::consts::TAU * segment as f64 / segments as f64;
                let (sin_theta, cos_theta) = theta.sin_cos();
                let unit = Vector3::new(cos_phi * cos_theta, sin_phi, cos_phi * sin_theta);
                let offset = ax * unit.x + ay * unit.y + az * unit.z;
                // The normal of a scaled sphere is the unit direction divided
                // by the squared half-extents, not the direction itself;
                // getting this wrong shades a flattened head like a round one.
                let normal = ax * (unit.x / ax.norm_squared().max(1e-12))
                    + ay * (unit.y / ay.norm_squared().max(1e-12))
                    + az * (unit.z / az.norm_squared().max(1e-12));
                self.vertex(centre + offset, normal, color);
            }
        }

        let stride = segments as u32 + 1;
        for ring in 0..rings as u32 {
            for segment in 0..segments as u32 {
                let a = base + ring * stride + segment;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                self.face(a, c, b);
                self.face(b, c, d);
            }
        }
    }

    /// A tapered tube between two points, without end caps.
    pub fn add_tube(
        &mut self,
        from: Point3<f64>,
        to: Point3<f64>,
        from_radius: f64,
        to_radius: f64,
        color: [f32; 3],
        segments: usize,
    ) {
        let axis = to - from;
        let length = axis.norm();
        if length < 1e-9 {
            return;
        }
        let (right, forward) = frame(axis / length);

        self.add_loft(
            &[
                Section {
                    centre: from,
                    right,
                    forward,
                    half_right: from_radius,
                    half_forward: from_radius,
                },
                Section {
                    centre: to,
                    right,
                    forward,
                    half_right: to_radius,
                    half_forward: to_radius,
                },
            ],
            color,
            segments,
            false,
        );
    }

    /// A limb: a tapered tube with a sphere at each end, which is what gives a
    /// knee and an elbow their shape without any skinning.
    pub fn add_limb(
        &mut self,
        from: Point3<f64>,
        to: Point3<f64>,
        from_radius: f64,
        to_radius: f64,
        color: [f32; 3],
    ) {
        self.add_tube(from, to, from_radius, to_radius, color, 12);
        self.add_sphere(from, from_radius, color, 12, 8);
        self.add_sphere(to, to_radius, color, 12, 8);
    }

    /// A surface swept through a series of elliptical cross-sections.
    ///
    /// `capped` closes the two ends with a fan, which is what the torso needs
    /// and a limb does not, since a limb has a sphere there instead.
    pub fn add_loft(
        &mut self,
        sections: &[Section],
        color: [f32; 3],
        segments: usize,
        capped: bool,
    ) {
        if sections.len() < 2 {
            return;
        }
        let segments = segments.max(3);
        let base = self.vertices.len() as u32;

        // The direction the loft sweeps in at each section, and how fast it is
        // narrowing there. The taper tilts the surface away from the
        // cross-section plane, so the normal leans along the sweep by the same
        // slope; without this a tapered limb shades as though it were a
        // cylinder.
        let sweep: Vec<(Vector3<f64>, f64)> = (0..sections.len())
            .map(|index| {
                let (from, to) = if index + 1 < sections.len() {
                    (&sections[index], &sections[index + 1])
                } else {
                    (&sections[index - 1], &sections[index])
                };
                let step = to.centre - from.centre;
                let length = step.norm();
                if length < 1e-9 {
                    return (sections[index].right.cross(&sections[index].forward), 0.0);
                }
                (step / length, -(to.half_right - from.half_right) / length)
            })
            .collect();

        for (index, section) in sections.iter().enumerate() {
            let (along, slope) = sweep[index];

            for segment in 0..=segments {
                let theta = std::f64::consts::TAU * segment as f64 / segments as f64;
                let (sin_theta, cos_theta) = theta.sin_cos();
                let position = section.centre
                    + section.right * (section.half_right * cos_theta)
                    + section.forward * (section.half_forward * sin_theta);
                let outward = section.right * (cos_theta / section.half_right.max(1e-9))
                    + section.forward * (sin_theta / section.half_forward.max(1e-9));
                let normal = unit_or(outward, section.right) + along * slope;
                self.vertex(position, normal, color);
            }
        }

        let stride = segments as u32 + 1;
        for ring in 0..sections.len() as u32 - 1 {
            for segment in 0..segments as u32 {
                let a = base + ring * stride + segment;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                self.face(a, c, b);
                self.face(b, c, d);
            }
        }

        if capped {
            let last = sections.len() as u32 - 1;
            for (section, ring, outward) in [
                (sections[0], 0u32, -sweep[0].0),
                (*sections.last().unwrap(), last, sweep[last as usize].0),
            ] {
                let centre = self.vertex(section.centre, outward, color);
                for segment in 0..segments as u32 {
                    let a = base + ring * stride + segment;
                    self.face(centre, a, a + 1);
                }
            }
        }
    }
}

/// Two unit vectors perpendicular to `axis` and to each other.
///
/// The seed is chosen away from the axis so that the cross product is well
/// conditioned; a vertical limb crossed with the vertical is a zero vector, and
/// the frame it produces collapses the tube to a line.
fn frame(axis: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    let seed = if axis.y.abs() > 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    };
    let right = axis.cross(&seed).normalize();
    (right, axis.cross(&right).normalize())
}

fn unit_or(v: Vector3<f64>, fallback: Vector3<f64>) -> Vector3<f64> {
    let norm = v.norm();
    if norm > 1e-9 { v / norm } else { fallback }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn centroid(mesh: &Mesh) -> Point3<f64> {
        let sum = mesh
            .vertices
            .iter()
            .fold(Vector3::zeros(), |acc, vertex| acc + vertex.position.coords);
        Point3::from(sum / mesh.vertices.len() as f64)
    }

    #[test]
    fn every_triangle_indexes_a_vertex_that_exists() {
        let mut mesh = Mesh::default();
        mesh.add_sphere(Point3::origin(), 0.5, [1.0, 0.0, 0.0], 10, 6);
        mesh.add_limb(
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            0.1,
            0.08,
            [0.0, 1.0, 0.0],
        );
        mesh.add_box(
            Point3::new(1.0, 0.0, 0.0),
            [Vector3::x() * 0.2, Vector3::y() * 0.3, Vector3::z() * 0.1],
            [0.0, 0.0, 1.0],
        );

        assert!(mesh.triangle_count() > 100);
        for triangle in &mesh.triangles {
            for index in triangle {
                assert!((*index as usize) < mesh.vertices.len());
            }
        }
    }

    #[test]
    fn merging_keeps_both_meshes_intact() {
        let mut left = Mesh::default();
        left.add_sphere(Point3::origin(), 0.5, [1.0, 1.0, 1.0], 8, 4);
        let mut right = Mesh::default();
        right.add_sphere(Point3::new(3.0, 0.0, 0.0), 0.5, [1.0, 1.0, 1.0], 8, 4);

        let (before, faces) = (left.vertices.len(), left.triangle_count());
        left.merge(&right);

        assert_eq!(left.vertices.len(), before + right.vertices.len());
        assert_eq!(left.triangle_count(), faces + right.triangle_count());
        for triangle in &left.triangles {
            for index in triangle {
                assert!((*index as usize) < left.vertices.len());
            }
        }
    }

    /// A normal pointing into the solid turns a lit surface black, and a whole
    /// body shaded that way is a silhouette no detector was trained on.
    #[test]
    fn sphere_normals_point_away_from_the_centre() {
        let mut mesh = Mesh::default();
        mesh.add_sphere(Point3::new(1.0, 2.0, 3.0), 0.4, [1.0, 1.0, 1.0], 16, 10);

        for vertex in &mesh.vertices {
            let outward = vertex.position - Point3::new(1.0, 2.0, 3.0);
            if outward.norm() < 1e-9 {
                continue;
            }
            assert!(
                vertex.normal.dot(&outward.normalize()) > 0.99,
                "a sphere normal should be the outward direction"
            );
        }
    }

    #[test]
    fn box_normals_point_out_of_the_solid() {
        let mut mesh = Mesh::default();
        let centre = Point3::new(0.5, 0.5, 0.5);
        mesh.add_box(
            centre,
            [Vector3::x() * 0.2, Vector3::y() * 0.4, Vector3::z() * 0.3],
            [1.0, 1.0, 1.0],
        );

        for vertex in &mesh.vertices {
            assert!(
                vertex.normal.dot(&(vertex.position - centre)) > 0.0,
                "a box face should face outwards"
            );
        }
    }

    /// A vertical limb is the common case — every bone in a standing body is
    /// near vertical — and it is exactly the case a naive frame degenerates on.
    #[test]
    fn a_vertical_tube_still_has_a_width() {
        let mut mesh = Mesh::default();
        mesh.add_tube(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            0.1,
            0.1,
            [1.0, 1.0, 1.0],
            12,
        );

        let widest = mesh
            .vertices
            .iter()
            .map(|vertex| vertex.position.coords.xz().norm())
            .fold(0.0f64, f64::max);
        assert!((widest - 0.1).abs() < 1e-9, "the tube collapsed to a line");
    }

    #[test]
    fn a_tube_between_two_points_sits_between_them() {
        let mut mesh = Mesh::default();
        let (from, to) = (Point3::new(-1.0, 0.5, 2.0), Point3::new(1.0, 1.5, 2.0));
        let radius = 0.05;
        mesh.add_tube(from, to, radius, radius, [1.0, 1.0, 1.0], 12);

        // Along the axis the centroid is the midpoint exactly. Across it the
        // rings are only nearly balanced, because the seam vertex is emitted
        // twice, so the surface can only be said to surround the axis.
        let axis = (to - from).normalize();
        let offset = centroid(&mesh) - nalgebra::center(&from, &to);
        assert!(offset.dot(&axis).abs() < 1e-9);
        assert!(offset.norm() < radius);

        for vertex in &mesh.vertices {
            let along = (vertex.position - from).dot(&axis);
            assert!((-1e-9..=(to - from).norm() + 1e-9).contains(&along));
        }
    }

    #[test]
    fn a_degenerate_tube_produces_nothing_rather_than_a_crash() {
        let mut mesh = Mesh::default();
        mesh.add_tube(
            Point3::origin(),
            Point3::origin(),
            0.1,
            0.1,
            [1.0, 1.0, 1.0],
            12,
        );
        assert!(mesh.is_empty());
    }
}
