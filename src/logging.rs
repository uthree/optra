//! Logging setup.
//!
//! Records go to stderr as usual and, in parallel, into a bounded in-memory
//! buffer that the log panel renders. Diagnosing a tracking problem means
//! reading the log while the app is running, not after it exits.

use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use chrono::{DateTime, Local};
use parking_lot::Mutex;
use tracing::field::{Field, Visit};
use tracing::{Event, Level, Subscriber};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::prelude::*;

/// Maximum number of records kept in memory.
const CAPACITY: usize = 4096;

#[derive(Clone, Debug)]
pub struct LogRecord {
    pub at: DateTime<Local>,
    pub level: Level,
    pub target: String,
    pub message: String,
}

/// Shared ring buffer of recent log records.
#[derive(Clone, Default)]
pub struct LogBuffer(Arc<Mutex<VecDeque<LogRecord>>>);

impl LogBuffer {
    fn push(&self, record: LogRecord) {
        let mut records = self.0.lock();
        if records.len() == CAPACITY {
            records.pop_front();
        }
        records.push_back(record);
    }

    /// Runs `f` over the buffered records without cloning them.
    pub fn with_records<R>(&self, f: impl FnOnce(&VecDeque<LogRecord>) -> R) -> R {
        f(&self.0.lock())
    }

    pub fn clear(&self) {
        self.0.lock().clear();
    }
}

/// Installs the global subscriber and returns the buffer backing the log panel.
pub fn init() -> LogBuffer {
    let buffer = LogBuffer::default();

    let filter =
        EnvFilter::try_from_env("OPTRA_LOG").unwrap_or_else(|_| EnvFilter::new("optra=debug,warn"));

    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .with(BufferLayer(buffer.clone()))
        .init();

    buffer
}

struct BufferLayer(LogBuffer);

impl<S: Subscriber> Layer<S> for BufferLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);

        let metadata = event.metadata();
        self.0.push(LogRecord {
            at: Local::now(),
            level: *metadata.level(),
            target: metadata.target().to_owned(),
            message: visitor.finish(),
        });
    }
}

/// Collects the `message` field, appending any other fields as `key=value`.
#[derive(Default)]
struct MessageVisitor {
    message: String,
    fields: String,
}

impl MessageVisitor {
    fn finish(self) -> String {
        if self.fields.is_empty() {
            self.message
        } else {
            format!("{}{}", self.message, self.fields)
        }
    }

    fn record(&mut self, field: &Field, value: impl fmt::Display) {
        if field.name() == "message" {
            self.message = value.to_string();
        } else {
            use fmt::Write as _;
            let _ = write!(self.fields, " {}={}", field.name(), value);
        }
    }
}

impl Visit for MessageVisitor {
    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        self.record(field, format_args!("{value:?}"));
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.record(field, value);
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.record(field, value);
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.record(field, value);
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.record(field, value);
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.record(field, value);
    }
}
