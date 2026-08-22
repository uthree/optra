//! Named worker threads with cooperative shutdown and panic reporting.
//!
//! Optra's pipeline is a set of long-lived threads. When one of them dies the
//! user needs to know which one and why, rather than watching tracking silently
//! stop, so every worker is spawned through the supervisor.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, unbounded};
use parking_lot::{Condvar, Mutex};

/// Cooperative shutdown signal handed to every worker.
#[derive(Clone, Default)]
pub struct Shutdown(Arc<ShutdownInner>);

#[derive(Default)]
struct ShutdownInner {
    cancelled: AtomicBool,
    lock: Mutex<()>,
    changed: Condvar,
}

impl Shutdown {
    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::Relaxed)
    }

    /// Sleeps for `duration` unless shutdown is requested first.
    ///
    /// Returns `false` if the worker should stop.
    pub fn sleep(&self, duration: Duration) -> bool {
        if self.is_cancelled() {
            return false;
        }
        let mut guard = self.0.lock.lock();
        self.0.changed.wait_for(&mut guard, duration);
        !self.is_cancelled()
    }

    pub fn cancel(&self) {
        self.0.cancelled.store(true, Ordering::Relaxed);
        let _guard = self.0.lock.lock();
        self.0.changed.notify_all();
    }
}

/// Lifecycle events reported by workers, drained by the UI.
#[derive(Clone, Debug)]
pub enum WorkerEvent {
    Started(String),
    Finished(String),
    Panicked { name: String, message: String },
}

pub struct Supervisor {
    shutdown: Shutdown,
    events_tx: Sender<WorkerEvent>,
    events_rx: Receiver<WorkerEvent>,
    handles: Vec<(String, JoinHandle<()>)>,
}

impl Default for Supervisor {
    fn default() -> Self {
        Self::new()
    }
}

impl Supervisor {
    pub fn new() -> Self {
        let (events_tx, events_rx) = unbounded();
        Self {
            shutdown: Shutdown::default(),
            events_tx,
            events_rx,
            handles: Vec::new(),
        }
    }

    /// Spawns a worker. The closure receives the shutdown signal and is
    /// expected to poll it regularly.
    pub fn spawn<F>(&mut self, name: impl Into<String>, body: F)
    where
        F: FnOnce(Shutdown) + Send + 'static,
    {
        let name = name.into();
        let shutdown = self.shutdown.clone();
        let events = self.events_tx.clone();
        let thread_name = name.clone();

        let handle = std::thread::Builder::new()
            .name(name.clone())
            .spawn(move || {
                let _ = events.send(WorkerEvent::Started(thread_name.clone()));

                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    body(shutdown);
                }));

                let event = match result {
                    Ok(()) => WorkerEvent::Finished(thread_name),
                    Err(payload) => WorkerEvent::Panicked {
                        name: thread_name,
                        message: panic_message(&payload),
                    },
                };
                let _ = events.send(event);
            })
            .expect("failed to spawn a worker thread");

        self.handles.push((name, handle));
    }

    /// Lifecycle events, drained by the UI once per frame.
    pub fn events(&self) -> &Receiver<WorkerEvent> {
        &self.events_rx
    }

    /// Removes workers that have already exited.
    pub fn reap(&mut self) {
        self.handles.retain(|(_, handle)| !handle.is_finished());
    }

    pub fn running(&self) -> usize {
        self.handles
            .iter()
            .filter(|(_, h)| !h.is_finished())
            .count()
    }

    /// Requests shutdown and waits for every worker to exit.
    pub fn shutdown(&mut self) {
        self.shutdown.cancel();
        for (name, handle) in self.handles.drain(..) {
            if handle.join().is_err() {
                tracing::error!(worker = %name, "worker panicked during shutdown");
            }
        }
    }
}

impl Drop for Supervisor {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_owned()
    }
}
