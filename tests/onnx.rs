//! ONNX Runtime tests.
//!
//! These need model files on disk, so they are ignored by default. Point
//! `OPTRA_TEST_MODEL` at an `.onnx` file and run:
//!
//! ```text
//! cargo test --release --test onnx -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use optra::infer::session::{self, ProviderChoice};

fn model_path() -> PathBuf {
    PathBuf::from(
        std::env::var("OPTRA_TEST_MODEL")
            .expect("set OPTRA_TEST_MODEL to the path of an .onnx file"),
    )
}

#[test]
#[ignore = "requires a model file"]
fn reports_model_inputs_and_outputs() {
    let path = model_path();
    let handle = session::load(&path, ProviderChoice::Cpu).expect("failed to load the model");

    println!("{}", path.display());
    println!("backend: {}", handle.backend.label());
    print!("{}", session::describe_io(&handle.session));

    assert!(!handle.session.inputs().is_empty());
    assert!(!handle.session.outputs().is_empty());
}

/// DirectML is the whole GPU story for this project, so whether it initializes
/// at all is worth knowing explicitly rather than discovering as a silent
/// fallback to CPU.
#[test]
#[ignore = "requires a model file"]
fn directml_is_available() {
    let path = model_path();

    let started = Instant::now();
    let handle = session::load(&path, ProviderChoice::DirectMl).expect("failed to load the model");
    println!(
        "backend: {} (session built in {:.2} s)",
        handle.backend.label(),
        started.elapsed().as_secs_f32()
    );

    assert_eq!(
        handle.backend,
        session::Backend::DirectMl,
        "DirectML was requested but the session fell back to CPU"
    );
}
