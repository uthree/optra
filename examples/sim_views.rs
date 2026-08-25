//! Writes out what each simulated camera sees, for looking at.
//!
//! The accuracy harness reports numbers, and a number cannot say that the
//! figure has its arm through its chest or that the room went dark. This is how
//! the scene gets checked by eye after it is changed.
//!
//! ```text
//! cargo run --release --example sim_views -- out/ 2.1
//! ```

use anyhow::Result;

use optra::sim::Scene;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let directory = args.next().unwrap_or_else(|| ".".to_owned());
    let t: f64 = args.next().map_or(Ok(2.1), |value| value.parse())?;

    std::fs::create_dir_all(&directory)?;
    let scene = Scene::default();

    for (seat, camera) in scene.cameras(4).iter().enumerate() {
        let image = scene.view(camera, t);
        let path = format!("{directory}/camera-{seat}.png");
        image.save(&path)?;
        println!(
            "{path}: {}x{}, {:.0} degrees horizontal, from {:.2?}",
            image.width,
            image.height,
            camera.intrinsics.horizontal_fov().to_degrees(),
            camera.position()
        );
    }

    Ok(())
}
