//! The log file: what it keeps, and what it is allowed to throw away.
//!
//! The file exists so that a problem can be diagnosed after the run that hit
//! it, which makes its rotation rules the thing worth asserting on: a run that
//! ends badly must still be readable afterwards, and a long session must not
//! grow a file nobody can send.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

use optra::logging::file::FileLog;
use tracing_subscriber::fmt::MakeWriter;

/// A directory of this test's own, removed if it was left behind by a previous
/// run so that a failure does not poison the next one.
fn scratch(name: &str) -> PathBuf {
    static COUNT: AtomicU32 = AtomicU32::new(0);
    let unique = COUNT.fetch_add(1, Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("optra-log-{name}-{}-{unique}", std::process::id()));
    fs::remove_dir_all(&dir).ok();
    dir
}

/// Writes one record the way the formatting layer does: rendered whole, handed
/// over in a single call.
fn record(log: &FileLog, line: &str) {
    let mut writer = log.make_writer();
    writer.write_all(line.as_bytes()).unwrap();
}

fn read(dir: &Path, name: &str) -> Option<String> {
    fs::read_to_string(dir.join(name)).ok()
}

#[test]
fn keeps_the_recent_past_and_drops_the_rest() {
    let dir = scratch("rotation");
    // Twenty bytes a record, so every fourth record rolls the file.
    let log = FileLog::open(&dir, 80, 2).unwrap();

    for index in 0..12 {
        record(&log, &format!("record {index:<11}\n"));
    }

    // The live file holds the newest records, and the two rolled files hold
    // the two batches before it. Anything older is gone, which is the point of
    // a limit.
    assert!(read(&dir, "optra.log").unwrap().contains("record 11"));
    assert!(read(&dir, "optra.1.log").unwrap().contains("record 7"));
    assert!(read(&dir, "optra.2.log").unwrap().contains("record 3"));
    assert!(
        read(&dir, "optra.3.log").is_none(),
        "the oldest file should have been discarded rather than kept"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_record_is_never_split_across_two_files() {
    let dir = scratch("whole");
    let log = FileLog::open(&dir, 64, 2).unwrap();

    record(&log, "short\n");
    // Longer than the whole limit. Splitting it would leave half a line at the
    // end of one file and half at the start of another, and half a stack trace
    // is worse than none.
    let long = format!("{}\n", "x".repeat(200));
    record(&log, &long);
    record(&log, "after\n");

    let live = read(&dir, "optra.log").unwrap();
    let rolled = read(&dir, "optra.1.log").unwrap();
    assert_eq!(rolled, long, "the long record should be whole in one file");
    assert_eq!(live, "after\n");

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_restart_does_not_erase_the_run_that_explained_it() {
    let dir = scratch("restart");

    let first = FileLog::open(&dir, 4096, 2).unwrap();
    record(&first, "the run that failed\n");
    drop(first);

    // An application that fails at startup is one a user restarts, usually
    // several times before asking anybody. Truncating on open would erase the
    // only evidence there is.
    let second = FileLog::open(&dir, 4096, 2).unwrap();
    record(&second, "the run after it\n");

    let live = read(&dir, "optra.log").unwrap();
    assert!(live.contains("the run that failed"));
    assert!(live.contains("the run after it"));

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_live_file_is_where_it_says_it_is() {
    let dir = scratch("path");
    let log = FileLog::open(&dir, 4096, 2).unwrap();
    record(&log, "hello\n");

    // The log panel hands this path to the file manager, so a wrong one is a
    // user staring at the wrong folder.
    assert_eq!(fs::read_to_string(log.path()).unwrap(), "hello\n");

    fs::remove_dir_all(&dir).ok();
}
