//! Loading models without stalling the pipeline.
//!
//! Building a session takes about a second on DirectML. Doing that on the
//! inference thread would freeze tracking every time a model is swapped, so
//! builds run on their own thread and the loop keeps serving whatever it
//! already has until the replacement arrives.

use std::collections::HashMap;

use crossbeam_channel::{Receiver, TryRecvError, bounded};

use crate::infer::traits::{Detector, Pose2d};
use crate::infer::{ProviderChoice, arch};
use crate::models::manifest::Manifest;

/// What the pipeline can currently do with a model id.
pub enum Slot<'a, T: ?Sized> {
    Ready(&'a mut T),
    Loading,
    Failed(&'a str),
}

/// What the per-frame work asks of the models it is given.
///
/// Two lookups, either of which may answer "not yet" or "never". The trait
/// exists so that the frame loop can be run against models that are not ONNX
/// sessions: `ModelSet`'s only way to produce a `Detector` is to build one from
/// the catalogue on a background thread, so without a seam here the stride, the
/// carried box and the behaviour on a failed load are reachable only by
/// downloading a few hundred megabytes and owning a GPU. They are the parts
/// most likely to be got wrong and they were the parts nothing could reach.
pub trait Models {
    fn detector(&mut self, id: &str) -> Slot<'_, dyn Detector>;
    fn pose(&mut self, id: &str) -> Slot<'_, dyn Pose2d>;
}

impl Models for ModelSet {
    fn detector(&mut self, id: &str) -> Slot<'_, dyn Detector> {
        ModelSet::detector(self, id)
    }

    fn pose(&mut self, id: &str) -> Slot<'_, dyn Pose2d> {
        ModelSet::pose(self, id)
    }
}

enum Entry<T> {
    Ready(T),
    Loading(Receiver<Result<T, String>>),
    Failed(String),
}

/// The models the pipeline has, keyed by manifest id.
pub struct ModelSet {
    provider: ProviderChoice,
    detectors: HashMap<String, Entry<Box<dyn Detector>>>,
    poses: HashMap<String, Entry<Box<dyn Pose2d>>>,
}

impl ModelSet {
    pub fn new(provider: ProviderChoice) -> Self {
        Self {
            provider,
            detectors: HashMap::new(),
            poses: HashMap::new(),
        }
    }

    /// Moves finished builds into place. Call once per tick.
    pub fn poll(&mut self) {
        poll_map(&mut self.detectors, "detector");
        poll_map(&mut self.poses, "pose model");
    }

    pub fn detector(&mut self, id: &str) -> Slot<'_, dyn Detector> {
        let provider = self.provider;
        request(&mut self.detectors, id, provider, |spec, provider| {
            arch::build_detector(&spec, provider)
        });

        match self.detectors.get_mut(id) {
            Some(Entry::Ready(detector)) => Slot::Ready(detector.as_mut()),
            Some(Entry::Loading(_)) => Slot::Loading,
            Some(Entry::Failed(err)) => Slot::Failed(err),
            None => Slot::Failed("the model was never requested"),
        }
    }

    pub fn pose(&mut self, id: &str) -> Slot<'_, dyn Pose2d> {
        let provider = self.provider;
        request(&mut self.poses, id, provider, |spec, provider| {
            arch::build_pose2d(&spec, provider)
        });

        match self.poses.get_mut(id) {
            Some(Entry::Ready(pose)) => Slot::Ready(pose.as_mut()),
            Some(Entry::Loading(_)) => Slot::Loading,
            Some(Entry::Failed(err)) => Slot::Failed(err),
            None => Slot::Failed("the model was never requested"),
        }
    }

    /// Starts loading everything named here that is not loaded yet.
    ///
    /// Models are warmed as soon as the stage starts rather than when a person
    /// first appears, so the first person does not wait a second for a session
    /// to build.
    pub fn ensure(&mut self, detectors: &[String], poses: &[String]) {
        let provider = self.provider;
        for id in detectors {
            request(&mut self.detectors, id, provider, |spec, provider| {
                arch::build_detector(&spec, provider)
            });
        }
        for id in poses {
            request(&mut self.poses, id, provider, |spec, provider| {
                arch::build_pose2d(&spec, provider)
            });
        }
    }

    /// Drops everything not named here, so a swapped-out model releases its
    /// GPU memory.
    pub fn retain(&mut self, detectors: &[String], poses: &[String]) {
        self.detectors.retain(|id, _| detectors.contains(id));
        self.poses.retain(|id, _| poses.contains(id));
    }
}

fn poll_map<T>(map: &mut HashMap<String, Entry<T>>, what: &str) {
    let loading: Vec<String> = map
        .iter()
        .filter(|(_, entry)| matches!(entry, Entry::Loading(_)))
        .map(|(id, _)| id.clone())
        .collect();

    // Each channel is received from exactly once. Peeking first and receiving
    // afterwards would throw the model away between the two.
    for id in loading {
        let Some(Entry::Loading(rx)) = map.remove(&id) else {
            continue;
        };
        match rx.try_recv() {
            Ok(Ok(model)) => {
                tracing::info!("{what} {id} is ready");
                map.insert(id, Entry::Ready(model));
            }
            Ok(Err(err)) => {
                tracing::error!("failed to load {what} {id}: {err}");
                map.insert(id, Entry::Failed(err));
            }
            Err(TryRecvError::Empty) => {
                map.insert(id, Entry::Loading(rx));
            }
            Err(TryRecvError::Disconnected) => {
                let err = "the loader thread died".to_owned();
                tracing::error!("failed to load {what} {id}: {err}");
                map.insert(id, Entry::Failed(err));
            }
        }
    }
}

/// Starts a background build for `id` if there is nothing for it yet.
fn request<T, F>(map: &mut HashMap<String, Entry<T>>, id: &str, provider: ProviderChoice, build: F)
where
    T: Send + 'static,
    F: FnOnce(crate::models::ModelSpec, ProviderChoice) -> anyhow::Result<T> + Send + 'static,
{
    if map.contains_key(id) {
        return;
    }

    let spec = match Manifest::load()
        .ok()
        .and_then(|models| models.into_iter().find(|spec| spec.id == id))
    {
        Some(spec) => spec,
        None => {
            map.insert(
                id.to_owned(),
                Entry::Failed(format!("{id} is not in the catalogue")),
            );
            return;
        }
    };

    let (tx, rx) = bounded(1);
    let id_for_thread = id.to_owned();

    let spawned = std::thread::Builder::new()
        .name(format!("load:{id}"))
        .spawn(move || {
            tracing::info!("loading {id_for_thread}");
            let result = build(spec, provider).map_err(|err| format!("{err:#}"));
            let _ = tx.send(result);
        });

    match spawned {
        Ok(_) => {
            map.insert(id.to_owned(), Entry::Loading(rx));
        }
        Err(err) => {
            map.insert(
                id.to_owned(),
                Entry::Failed(format!("failed to start the loader: {err}")),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state<T>(map: &HashMap<String, Entry<T>>, id: &str) -> &'static str {
        match map.get(id) {
            Some(Entry::Ready(_)) => "ready",
            Some(Entry::Loading(_)) => "loading",
            Some(Entry::Failed(_)) => "failed",
            None => "absent",
        }
    }

    #[test]
    fn a_build_still_running_stays_loading() {
        let (_tx, rx) = bounded::<Result<u32, String>>(1);
        let mut map = HashMap::from([("m".to_owned(), Entry::Loading(rx))]);

        poll_map(&mut map, "model");
        assert_eq!(state(&map, "m"), "loading");
    }

    /// Polling has to receive the value exactly once. Peeking to decide whether
    /// a build finished and receiving afterwards drops the model on the floor,
    /// which shows up as a loader that "died" the moment it succeeded.
    #[test]
    fn a_finished_build_becomes_ready_and_survives_further_polls() {
        let (tx, rx) = bounded::<Result<u32, String>>(1);
        tx.send(Ok(7)).unwrap();
        drop(tx);

        let mut map = HashMap::from([("m".to_owned(), Entry::Loading(rx))]);

        poll_map(&mut map, "model");
        assert_eq!(state(&map, "m"), "ready");

        poll_map(&mut map, "model");
        assert_eq!(state(&map, "m"), "ready");

        match map.get("m") {
            Some(Entry::Ready(value)) => assert_eq!(*value, 7),
            _ => panic!("the built model should still be there"),
        }
    }

    #[test]
    fn a_failed_build_is_remembered_with_its_message() {
        let (tx, rx) = bounded::<Result<u32, String>>(1);
        tx.send(Err("no such file".to_owned())).unwrap();
        drop(tx);

        let mut map = HashMap::from([("m".to_owned(), Entry::Loading(rx))]);
        poll_map(&mut map, "model");

        match map.get("m") {
            Some(Entry::Failed(err)) => assert_eq!(err, "no such file"),
            _ => panic!("the failure should be recorded"),
        }
    }

    #[test]
    fn a_loader_that_vanished_is_reported_as_failed() {
        let (tx, rx) = bounded::<Result<u32, String>>(1);
        drop(tx);

        let mut map = HashMap::from([("m".to_owned(), Entry::Loading(rx))]);
        poll_map(&mut map, "model");
        assert_eq!(state(&map, "m"), "failed");
    }
}
