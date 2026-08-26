//! The log file.
//!
//! The in-memory buffer behind the log panel holds the last few thousand
//! records, which at tracking rates is a matter of seconds. That is enough to
//! watch a problem happen and no use at all for one that already happened:
//! whatever the calibration wizard said has been pushed out of the buffer long
//! before a user notices their tracking is wrong. The file is what is left to
//! read afterwards.
//!
//! Rotation is by size rather than by run. One file per run reads better, but
//! an application that fails at startup is one a user restarts, and five
//! restarts would erase the run that explained it.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use parking_lot::Mutex;
use tracing_subscriber::fmt::MakeWriter;

/// Name of the file being written to now. The rolled ones sit beside it as
/// `optra.1.log` and upwards, oldest last.
const LIVE: &str = "optra.log";

/// Bytes written before the live file is rolled aside.
///
/// A tracking session logs little once it is running — the per-frame records
/// are below the level the file keeps — so this is sized for the startup,
/// calibration and model-download chatter that is actually worth reading, and
/// small enough that a user can be asked to send one.
pub const LIMIT: u64 = 4 * 1024 * 1024;

/// How many rolled files are kept behind the live one.
pub const KEEP: usize = 4;

/// A writer that rolls its file aside once it grows past a limit.
#[derive(Clone)]
pub struct FileLog(Arc<Mutex<Rotating>>);

impl FileLog {
    /// Opens (or creates) the live file in `dir`, appending to whatever is
    /// already there.
    pub fn open(dir: &Path, limit: u64, keep: usize) -> Result<Self> {
        fs::create_dir_all(dir)
            .with_context(|| format!("failed to create the log directory {}", dir.display()))?;

        let path = dir.join(LIVE);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let written = file.metadata().map(|meta| meta.len()).unwrap_or(0);

        Ok(Self(Arc::new(Mutex::new(Rotating {
            dir: dir.to_owned(),
            limit,
            keep,
            file,
            written,
        }))))
    }

    /// Path of the file currently being written to.
    pub fn path(&self) -> PathBuf {
        self.0.lock().dir.join(LIVE)
    }
}

impl<'a> MakeWriter<'a> for FileLog {
    type Writer = Handle;

    fn make_writer(&'a self) -> Self::Writer {
        Handle(self.0.clone())
    }
}

/// One event's worth of access to the file.
///
/// The formatting layer renders an event into a thread-local buffer and hands
/// it over in a single `write_all`, so taking the lock per call is what keeps
/// two threads from interleaving halves of a line.
pub struct Handle(Arc<Mutex<Rotating>>);

impl Write for Handle {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().record(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.0.lock().file.flush()
    }
}

struct Rotating {
    dir: PathBuf,
    limit: u64,
    keep: usize,
    file: File,
    written: u64,
}

impl Rotating {
    fn record(&mut self, buf: &[u8]) -> io::Result<()> {
        // Rolled on the record that would cross the limit rather than after
        // it, so that a record is never split across two files. A single
        // record longer than the whole limit is written anyway; the alternative
        // is an empty file and a lost line.
        if self.written > 0 && self.written + buf.len() as u64 > self.limit {
            self.roll()?;
        }

        self.file.write_all(buf)?;
        self.written += buf.len() as u64;
        Ok(())
    }

    /// `optra.log` becomes `optra.1.log`, `optra.1.log` becomes `optra.2.log`,
    /// and whatever was in the last slot is gone.
    fn roll(&mut self) -> io::Result<()> {
        for slot in (1..=self.keep).rev() {
            let from = if slot == 1 {
                self.dir.join(LIVE)
            } else {
                self.dir.join(rolled(slot - 1))
            };
            if from.exists() {
                // A rename onto an existing file replaces it on both platforms
                // this builds for, which is how the last slot is discarded.
                fs::rename(&from, self.dir.join(rolled(slot)))?;
            }
        }

        // Nothing to roll into means the user asked for no history at all, so
        // the live file simply starts again.
        if self.keep == 0 {
            fs::remove_file(self.dir.join(LIVE)).ok();
        }

        self.file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(self.dir.join(LIVE))?;
        self.written = 0;
        Ok(())
    }
}

fn rolled(slot: usize) -> String {
    format!("optra.{slot}.log")
}
