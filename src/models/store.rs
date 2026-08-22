//! Downloading, verifying and unpacking models.
//!
//! Models are never vendored in the repository. They are fetched on demand,
//! checked against the digest in the manifest, and only then moved into place,
//! so a partial or tampered download can never be mistaken for a usable model.

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};

use super::manifest::{ModelSource, ModelSpec};
use crate::paths;

/// Where a model install has got to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Stage {
    Downloading { received: u64, total: Option<u64> },
    Verifying,
    Extracting,
    Done,
    Failed(String),
}

impl Stage {
    /// Progress from 0 to 1, where it is known.
    pub fn fraction(&self) -> Option<f32> {
        match self {
            Stage::Downloading {
                received,
                total: Some(total),
            } if *total > 0 => Some(*received as f32 / *total as f32),
            Stage::Done => Some(1.0),
            _ => None,
        }
    }

    pub fn label(&self) -> String {
        match self {
            Stage::Downloading { received, total } => match total {
                Some(total) => format!(
                    "downloading {} / {}",
                    human_bytes(*received),
                    human_bytes(*total)
                ),
                None => format!("downloading {}", human_bytes(*received)),
            },
            Stage::Verifying => "verifying".to_owned(),
            Stage::Extracting => "extracting".to_owned(),
            Stage::Done => "ready".to_owned(),
            Stage::Failed(err) => format!("failed: {err}"),
        }
    }
}

pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[0])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Where a model's `.onnx` lives once installed.
pub fn local_path(spec: &ModelSpec) -> Result<PathBuf> {
    if let ModelSource::Local { path } = &spec.source {
        return Ok(PathBuf::from(path));
    }
    Ok(paths::models_dir()?.join(format!("{}.onnx", spec.id)))
}

pub fn is_installed(spec: &ModelSpec) -> bool {
    local_path(spec).map(|path| path.is_file()).unwrap_or(false)
}

/// Removes an installed model.
pub fn remove(spec: &ModelSpec) -> Result<()> {
    if matches!(spec.source, ModelSource::Local { .. }) {
        bail!(
            "{} refers to a file you provided; Optra will not delete it",
            spec.id
        );
    }
    let path = local_path(spec)?;
    if path.exists() {
        std::fs::remove_file(&path)
            .with_context(|| format!("failed to remove {}", path.display()))?;
    }
    Ok(())
}

/// Fetches and installs a model, reporting progress through `report`.
///
/// Returns the path of the installed `.onnx`.
pub fn install(spec: &ModelSpec, report: &mut dyn FnMut(Stage)) -> Result<PathBuf> {
    let target = local_path(spec)?;

    if let ModelSource::Local { path } = &spec.source {
        let path = PathBuf::from(path);
        if !path.is_file() {
            bail!("{} does not exist", path.display());
        }
        report(Stage::Done);
        return Ok(path);
    }

    if target.is_file() {
        report(Stage::Done);
        return Ok(target);
    }

    let url = spec
        .source
        .url()
        .ok_or_else(|| anyhow!("{} has no download URL", spec.id))?;

    let scratch = paths::models_dir()?.join(".incoming");
    std::fs::create_dir_all(&scratch)
        .with_context(|| format!("failed to create {}", scratch.display()))?;
    let download = scratch.join(format!("{}.part", spec.id));

    let digest = download_to(url, &download, spec.source.size(), report)
        .with_context(|| format!("failed to download {url}"))?;

    report(Stage::Verifying);
    if let Some(expected) = spec.source.sha256()
        && !expected.eq_ignore_ascii_case(&digest)
    {
        let _ = std::fs::remove_file(&download);
        bail!("the download does not match the expected checksum (got {digest})");
    }

    report(Stage::Extracting);
    let staged = scratch.join(format!("{}.onnx", spec.id));
    extract(spec, &download, &staged)?;
    let _ = std::fs::remove_file(&download);

    std::fs::rename(&staged, &target)
        .with_context(|| format!("failed to move the model into {}", target.display()))?;

    report(Stage::Done);
    tracing::info!(model = %spec.id, "installed {}", target.display());
    Ok(target)
}

/// Streams a URL to a file, hashing as it goes.
fn download_to(
    url: &str,
    target: &Path,
    expected_size: Option<u64>,
    report: &mut dyn FnMut(Stage),
) -> Result<String> {
    let response = ureq::get(url).call().context("the request failed")?;

    let total = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .or(expected_size);

    let mut reader = response.into_body().into_reader();
    let mut file = BufWriter::new(
        File::create(target).with_context(|| format!("failed to create {}", target.display()))?,
    );

    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 128 * 1024];
    let mut received = 0u64;
    let mut last_reported = 0u64;

    loop {
        let read = reader.read(&mut buffer).context("the transfer failed")?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        file.write_all(&buffer[..read])
            .with_context(|| format!("failed to write {}", target.display()))?;
        received += read as u64;

        // Reporting every chunk would lock the shared progress state hundreds
        // of times a second for no visible benefit.
        if received - last_reported >= 1024 * 1024 {
            last_reported = received;
            report(Stage::Downloading { received, total });
        }
    }

    file.flush().context("failed to flush the download")?;
    report(Stage::Downloading {
        received,
        total: Some(received),
    });

    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::new(), |mut out, byte| {
        let _ = write!(out, "{byte:02x}");
        out
    })
}

/// Pulls the `.onnx` out of whatever the download turned out to be.
fn extract(spec: &ModelSpec, download: &Path, target: &Path) -> Result<()> {
    match &spec.source {
        ModelSource::File { .. } => {
            std::fs::rename(download, target).context("failed to move the downloaded model")?;
            Ok(())
        }
        ModelSource::Zip { entry, .. } => extract_zip(download, entry.as_deref(), target),
        ModelSource::TarGz { entry, .. } => extract_tar_gz(download, entry.as_deref(), target),
        ModelSource::Local { .. } => unreachable!("local sources are handled before extraction"),
    }
}

fn extract_zip(archive: &Path, entry: Option<&str>, target: &Path) -> Result<()> {
    let file = File::open(archive).context("failed to open the archive")?;
    let mut zip = zip::ZipArchive::new(file).context("failed to read the archive")?;

    let names: Vec<String> = (0..zip.len())
        .filter_map(|index| zip.by_index(index).ok().map(|f| f.name().to_owned()))
        .collect();
    let name = pick_entry(&names, entry)?;

    let mut source = zip
        .by_name(&name)
        .with_context(|| format!("{name} is not in the archive"))?;
    let mut out =
        File::create(target).with_context(|| format!("failed to create {}", target.display()))?;
    std::io::copy(&mut source, &mut out).context("failed to unpack the model")?;
    Ok(())
}

fn extract_tar_gz(archive: &Path, entry: Option<&str>, target: &Path) -> Result<()> {
    // A tar is sequential, so finding the wanted member means walking it. The
    // archive is read twice rather than buffered: these can be gigabytes.
    let names = {
        let file = File::open(archive).context("failed to open the archive")?;
        let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
        tar.entries()
            .context("failed to read the archive")?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| entry.path().ok().map(|p| p.display().to_string()))
            .collect::<Vec<_>>()
    };
    let name = pick_entry(&names, entry)?;

    let file = File::open(archive).context("failed to open the archive")?;
    let mut tar = tar::Archive::new(flate2::read::GzDecoder::new(file));
    for entry in tar.entries().context("failed to read the archive")? {
        let mut entry = entry.context("failed to read an archive entry")?;
        let path = entry
            .path()
            .context("an archive entry has an unreadable path")?
            .display()
            .to_string();
        if path == name {
            let mut out = File::create(target)
                .with_context(|| format!("failed to create {}", target.display()))?;
            std::io::copy(&mut entry, &mut out).context("failed to unpack the model")?;
            return Ok(());
        }
    }
    bail!("{name} vanished from the archive between passes")
}

/// Picks the model file out of an archive listing.
///
/// A manifest may name the entry, but most published bundles contain exactly
/// one `.onnx`, and requiring the path would make every entry brittle against
/// the date-stamped directories these archives use.
fn pick_entry(names: &[String], entry: Option<&str>) -> Result<String> {
    if let Some(entry) = entry {
        return names
            .iter()
            .find(|name| name.as_str() == entry)
            .cloned()
            .ok_or_else(|| anyhow!("{entry} is not in the archive"));
    }

    let mut onnx: Vec<&String> = names
        .iter()
        .filter(|name| name.to_ascii_lowercase().ends_with(".onnx"))
        .collect();

    match onnx.len() {
        0 => bail!("the archive contains no .onnx file"),
        1 => Ok(onnx.remove(0).clone()),
        _ => bail!(
            "the archive contains {} .onnx files; name one with `entry`",
            onnx.len()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_onnx_is_found_without_being_named() {
        let names = vec![
            "20230831/rtmpose_onnx/model/deploy.json".to_owned(),
            "20230831/rtmpose_onnx/model/end2end.onnx".to_owned(),
            "20230831/rtmpose_onnx/model/output.jpg".to_owned(),
        ];
        assert_eq!(
            pick_entry(&names, None).unwrap(),
            "20230831/rtmpose_onnx/model/end2end.onnx"
        );
    }

    #[test]
    fn an_ambiguous_archive_has_to_be_disambiguated() {
        let names = vec!["a.onnx".to_owned(), "b.onnx".to_owned()];
        assert!(pick_entry(&names, None).is_err());
        assert_eq!(pick_entry(&names, Some("b.onnx")).unwrap(), "b.onnx");
    }

    #[test]
    fn a_named_entry_that_is_absent_is_an_error() {
        let names = vec!["a.onnx".to_owned()];
        assert!(pick_entry(&names, Some("missing.onnx")).is_err());
    }

    #[test]
    fn byte_counts_read_sensibly() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(18_887_737), "18.0 MB");
    }
}
