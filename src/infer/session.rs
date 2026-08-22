//! ONNX Runtime session construction.
//!
//! DirectML is the only GPU path Optra uses: it works the same on AMD and
//! NVIDIA hardware, which is the whole point given the target machines. CPU
//! remains available so that everything else can be developed and tested on a
//! machine where the GPU path is unavailable.

use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use ort::ep::{CPU, DirectML};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::ValueType;
use serde::{Deserialize, Serialize};

/// Which execution provider the user asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderChoice {
    /// DirectML, falling back to CPU if it cannot be initialized.
    #[default]
    DirectMl,
    Cpu,
}

impl ProviderChoice {
    pub const ALL: [ProviderChoice; 2] = [ProviderChoice::DirectMl, ProviderChoice::Cpu];

    pub fn label(self) -> &'static str {
        match self {
            ProviderChoice::DirectMl => "DirectML (GPU)",
            ProviderChoice::Cpu => "CPU",
        }
    }
}

/// Which execution provider a session actually ended up using.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    DirectMl,
    Cpu,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::DirectMl => "DirectML",
            Backend::Cpu => "CPU",
        }
    }
}

/// A loaded model, together with what it ended up running on.
pub struct SessionHandle {
    pub session: Session,
    pub backend: Backend,
}

/// Initializes ONNX Runtime once per process.
fn init() {
    static STARTED: OnceLock<()> = OnceLock::new();

    STARTED.get_or_init(|| {
        // `commit` reports whether these options won the race to configure the
        // environment. Losing means one already exists, which is fine.
        ort::init().with_name("optra").commit();
    });
}

/// Loads a model.
///
/// A DirectML failure is reported and demoted to CPU rather than propagated:
/// tracking at a lower frame rate beats no tracking, and the camera panel shows
/// which backend each model actually got.
pub fn load(path: &Path, choice: ProviderChoice) -> Result<SessionHandle> {
    init();

    if choice == ProviderChoice::DirectMl {
        match build(path, ProviderChoice::DirectMl) {
            Ok(session) => {
                return Ok(SessionHandle {
                    session,
                    backend: Backend::DirectMl,
                });
            }
            Err(err) => {
                tracing::warn!(
                    model = %path.display(),
                    "DirectML is unavailable, falling back to CPU: {err:#}"
                );
            }
        }
    }

    let session = build(path, ProviderChoice::Cpu)?;
    Ok(SessionHandle {
        session,
        backend: Backend::Cpu,
    })
}

fn build(path: &Path, choice: ProviderChoice) -> Result<Session> {
    let providers = match choice {
        ProviderChoice::DirectMl => vec![DirectML::default().build()],
        ProviderChoice::Cpu => vec![CPU::default().build()],
    };

    // Builder errors carry the builder itself for recovery, which anyhow cannot
    // absorb, so they are flattened into a message here.
    let mut builder = Session::builder()
        .map_err(|err| anyhow::anyhow!("failed to create a session builder: {err}"))?
        .with_execution_providers(providers)
        .map_err(|err| anyhow::anyhow!("failed to select an execution provider: {err}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|err| anyhow::anyhow!("failed to set the optimization level: {err}"))?;

    builder
        .commit_from_file(path)
        .with_context(|| format!("failed to load {}", path.display()))
}

/// A human-readable summary of a model's inputs and outputs.
///
/// Manifest entries declare tensor names and shapes, and a mismatch has to be
/// reported against something; this is that something.
pub fn describe_io(session: &Session) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    for input in session.inputs() {
        let _ = writeln!(
            out,
            "  in  {} {}",
            input.name(),
            describe_type(input.dtype())
        );
    }
    for output in session.outputs() {
        let _ = writeln!(
            out,
            "  out {} {}",
            output.name(),
            describe_type(output.dtype())
        );
    }
    out
}

fn describe_type(dtype: &ValueType) -> String {
    match dtype {
        ValueType::Tensor { ty, shape, .. } => {
            let dims: Vec<String> = shape
                .iter()
                .map(|d| {
                    if *d < 0 {
                        "?".to_owned()
                    } else {
                        d.to_string()
                    }
                })
                .collect();
            format!("{ty:?}[{}]", dims.join(", "))
        }
        other => format!("{other:?}"),
    }
}
