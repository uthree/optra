//! Model catalogue tests.
//!
//! The install test downloads a real model into the user's model directory, so
//! it is ignored by default:
//!
//! ```text
//! cargo test --release --test models -- --ignored --nocapture
//! ```

use optra::infer::session::{self, ProviderChoice};
use optra::models::manifest::{Manifest, ModelKind};
use optra::models::store::{self, Stage};
use optra::models::{keypoints, store as model_store};

#[test]
fn the_catalogue_loads_and_every_entry_is_coherent() {
    let models = Manifest::load().expect("the catalogue should load");
    assert!(!models.is_empty());

    for spec in &models {
        // The licence gate applies to what Optra downloads; an entry the user
        // registered for a file of their own carries whatever its licence is,
        // and this test runs against the real user manifest.
        assert!(
            matches!(
                spec.source,
                optra::models::manifest::ModelSource::Local { .. }
            ) || ["Apache-2.0", "MIT"].contains(&spec.license.as_str()),
            "{} has license {}",
            spec.id,
            spec.license
        );
        assert!(
            !spec.arch.is_empty(),
            "{} names no architecture adapter",
            spec.id
        );

        if spec.kind == ModelKind::Pose2d {
            let layout = spec
                .output
                .keypoints
                .as_deref()
                .expect("a pose model declares a keypoint layout");
            assert!(
                keypoints::layout(layout).is_some(),
                "{} refers to unknown layout {layout}",
                spec.id
            );
        }

        // A model that cannot be located is worse than one that is missing: it
        // fails at the point the user tries to track.
        assert!(
            model_store::local_path(spec).is_ok(),
            "{} has no resolvable local path",
            spec.id
        );
    }
}

#[test]
fn the_default_pose_model_provides_feet() {
    let models = Manifest::load().expect("the catalogue should load");
    let default = models
        .iter()
        .find(|spec| spec.id == "rtmpose-m-halpe26-256x192")
        .expect("the default pose model is in the catalogue");

    let layout = keypoints::layout(default.output.keypoints.as_deref().unwrap()).unwrap();
    assert!(
        layout.has_feet(),
        "the default pose model must carry heel and toe points"
    );
}

/// Downloads the smallest catalogue model, verifies its checksum, unpacks it
/// and loads it, which exercises every step between the manifest and a usable
/// session.
#[test]
#[ignore = "downloads a model"]
fn installs_and_loads_a_real_model() {
    let models = Manifest::load().expect("the catalogue should load");
    let spec = models
        .iter()
        .find(|spec| spec.id == "rtmpose-t-halpe26-256x192")
        .expect("the tiny pose model is in the catalogue");

    let mut last = String::new();
    let path = store::install(spec, &mut |stage: Stage| {
        let label = stage.label();
        if label != last {
            println!("{label}");
            last = label;
        }
    })
    .expect("install should succeed");

    println!("installed at {}", path.display());
    assert!(path.is_file());

    let handle = session::load(&path, ProviderChoice::Cpu).expect("the model should load");
    println!("{}", session::describe_io(&handle.session));

    let outputs: Vec<&str> = handle.session.outputs().iter().map(|o| o.name()).collect();
    for declared in &spec.output.tensors {
        assert!(
            outputs.contains(&declared.as_str()),
            "the manifest declares output {declared}, but the graph has {outputs:?}"
        );
    }

    let inputs: Vec<&str> = handle.session.inputs().iter().map(|i| i.name()).collect();
    assert!(
        inputs.contains(&spec.input.name.as_str()),
        "the manifest declares input {}, but the graph has {inputs:?}",
        spec.input.name
    );
}

/// Times the default pose model on this machine. Needs the model installed and
/// a working ONNX runtime, so it is ignored by default:
///
/// ```text
/// cargo test --release --test models -- --ignored benchmark --nocapture
/// ```
#[test]
#[ignore = "requires an installed model"]
fn the_default_pose_model_benchmarks() {
    let models = Manifest::load().unwrap();
    let spec = models
        .iter()
        .find(|spec| spec.id == optra::config::InferenceConfig::default().pose_model)
        .expect("the default pose model is in the catalogue");
    assert!(
        store::is_installed(spec),
        "install {} from the Models panel first",
        spec.id
    );

    let result = optra::infer::bench::run(spec, ProviderChoice::default()).unwrap();
    println!(
        "{}: {:.1} ms median, {:.1} ms worst, {} (built in {:.0} ms)",
        spec.id,
        result.median_ms,
        result.worst_ms,
        result.backend.label(),
        result.build_ms
    );

    assert!(result.median_ms > 0.0);
    assert!(result.worst_ms >= result.median_ms);
}
