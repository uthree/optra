//! The OpenVR C interface, loaded at run time.
//!
//! The published bindings all build the OpenVR SDK from source with CMake and
//! link it statically. Neither half of that suits this application: it puts a
//! C++ toolchain in the way of anyone compiling it, and a statically linked
//! runtime means the program will not start on a machine without SteamVR, when
//! setting cameras up without a headset present is a thing people will do.
//!
//! So `openvr_api.dll` is found and loaded when it is needed, and its absence
//! is a message in the UI rather than a failure to launch.
//!
//! The layout below is transcribed from `openvr_capi.h` of the OpenVR SDK. Only
//! the leading part of the function table is described, up to the last entry
//! this application calls; the runtime's table continues past it and nothing
//! reads that far. Entries before that point which are never called are kept as
//! opaque pointers, because their position must be right even though their
//! signature does not matter.

use std::ffi::{CStr, CString, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use libloading::Library;

/// The interface version this table was transcribed from. A runtime older than
/// the one that introduced it will refuse the request, which is the correct
/// outcome: a mismatched table would be a crash instead.
const SYSTEM_INTERFACE: &str = "FnTable:IVRSystem_026";

/// Announce ourselves as a background application. It is what Optra is: it
/// reads poses, never draws, and must not keep SteamVR awake on its own.
const APPLICATION_BACKGROUND: i32 = 3;

pub const MAX_TRACKED_DEVICES: usize = 64;
pub const DEVICE_INDEX_HMD: u32 = 0;

pub const CLASS_HMD: i32 = 1;
pub const CLASS_CONTROLLER: i32 = 2;
pub const CLASS_TRACKER: i32 = 3;

pub const ROLE_LEFT_HAND: i32 = 1;
pub const ROLE_RIGHT_HAND: i32 = 2;

pub const UNIVERSE_STANDING: i32 = 1;

const PROP_SERIAL_NUMBER: i32 = 1002;
const PROP_MODEL_NUMBER: i32 = 1001;

/// Row-major 3x4: rotation in the first three columns, position in the last.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct HmdMatrix34 {
    pub m: [[f32; 4]; 3],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct TrackedDevicePose {
    pub device_to_absolute_tracking: HmdMatrix34,
    pub velocity: [f32; 3],
    pub angular_velocity: [f32; 3],
    pub tracking_result: i32,
    pub pose_is_valid: bool,
    pub device_is_connected: bool,
}

impl Default for TrackedDevicePose {
    fn default() -> Self {
        Self {
            device_to_absolute_tracking: HmdMatrix34::default(),
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            tracking_result: 0,
            pose_is_valid: false,
            device_is_connected: false,
        }
    }
}

/// The leading entries of `VR_IVRSystem_FnTable`, in the order the header
/// declares them. Do not reorder or remove anything: the position of a field is
/// the only thing identifying which function it is.
#[repr(C)]
struct SystemTable {
    get_recommended_render_target_size: *const c_void,
    get_projection_matrix: *const c_void,
    get_projection_raw: *const c_void,
    compute_distortion: *const c_void,
    compute_distortion_set: *const c_void,
    get_eye_to_head_transform: *const c_void,
    get_time_since_last_vsync: *const c_void,
    get_d3d9_adapter_index: *const c_void,
    get_dxgi_output_info: *const c_void,
    get_output_device: *const c_void,
    is_display_on_desktop: *const c_void,
    set_display_visibility: *const c_void,
    get_device_to_absolute_tracking_pose:
        unsafe extern "system" fn(i32, f32, *mut TrackedDevicePose, u32),
    get_seated_zero_pose_to_standing: *const c_void,
    get_raw_zero_pose_to_standing: *const c_void,
    get_sorted_tracked_device_indices_of_class: *const c_void,
    get_tracked_device_activity_level: *const c_void,
    apply_transform: *const c_void,
    get_tracked_device_index_for_controller_role: *const c_void,
    get_controller_role_for_tracked_device_index: unsafe extern "system" fn(u32) -> i32,
    get_tracked_device_class: unsafe extern "system" fn(u32) -> i32,
    is_tracked_device_connected: unsafe extern "system" fn(u32) -> bool,
    get_bool_tracked_device_property: *const c_void,
    get_float_tracked_device_property: *const c_void,
    get_int32_tracked_device_property: *const c_void,
    get_uint64_tracked_device_property: *const c_void,
    get_matrix34_tracked_device_property: *const c_void,
    get_array_tracked_device_property: *const c_void,
    get_string_tracked_device_property:
        unsafe extern "system" fn(u32, i32, *mut c_char, u32, *mut i32) -> u32,
}

type InitInternal = unsafe extern "C" fn(*mut i32, i32) -> isize;
type ShutdownInternal = unsafe extern "C" fn();
type GetGenericInterface = unsafe extern "C" fn(*const c_char, *mut i32) -> isize;
type InitErrorDescription = unsafe extern "C" fn(i32) -> *const c_char;

/// Whether this process already holds a connection.
///
/// `VR_InitInternal` and `VR_ShutdownInternal` act on the *process*, not on a
/// handle: a second connection shares the first one's state, and whichever is
/// dropped first invalidates the function table both of them are holding. The
/// survivor then calls through a dangling pointer, which is an access violation
/// rather than an error. Making a second connection a refusal keeps that
/// impossible.
static CONNECTED: AtomicBool = AtomicBool::new(false);

/// A live connection to the OpenVR runtime.
///
/// Not `Send`: OpenVR expects to be driven from the thread that initialized it,
/// and the raw pointers here enforce that at compile time.
pub struct Runtime {
    /// Kept alive for as long as the function pointers taken from it are used.
    library: Library,
    table: *const SystemTable,
    path: PathBuf,
}

impl Runtime {
    /// Loads `openvr_api.dll` and initializes it as a background application.
    ///
    /// Fails if this process is already connected; see [`CONNECTED`].
    pub fn connect() -> Result<Self> {
        if CONNECTED.swap(true, Ordering::SeqCst) {
            bail!("this process is already connected to SteamVR");
        }

        let mut attempts = Vec::new();
        for candidate in candidates() {
            match Self::open(&candidate) {
                Ok(runtime) => return Ok(runtime),
                Err(error) => attempts.push(format!("{}: {error:#}", candidate.display())),
            }
        }

        CONNECTED.store(false, Ordering::SeqCst);
        if attempts.is_empty() {
            bail!("openvr_api.dll was not found; is SteamVR installed?");
        }
        Err(anyhow!(attempts.join("; ")))
    }

    fn open(path: &Path) -> Result<Self> {
        // SAFETY: loading a library runs its initializers, which is the only
        // way to reach the runtime at all. The path comes from SteamVR's own
        // registration file or the system search path.
        let library = unsafe { Library::new(path) }.context("failed to load the library")?;

        // SAFETY: the symbols are the documented C exports of openvr_api, and
        // the signatures are transcribed from its header.
        unsafe {
            let init: libloading::Symbol<InitInternal> = library
                .get(b"VR_InitInternal\0")
                .context("VR_InitInternal is missing; this is not openvr_api")?;
            let describe: libloading::Symbol<InitErrorDescription> =
                library.get(b"VR_GetVRInitErrorAsEnglishDescription\0")?;
            let generic: libloading::Symbol<GetGenericInterface> =
                library.get(b"VR_GetGenericInterface\0")?;

            let mut error = 0i32;
            let token = init(&mut error, APPLICATION_BACKGROUND);
            if token == 0 || error != 0 {
                let message = CStr::from_ptr(describe(error))
                    .to_string_lossy()
                    .to_string();
                bail!("SteamVR refused the connection: {message}");
            }

            let name = CString::new(SYSTEM_INTERFACE).expect("a literal without a nul");
            let mut error = 0i32;
            let table = generic(name.as_ptr(), &mut error);
            if table == 0 || error != 0 {
                let message = CStr::from_ptr(describe(error))
                    .to_string_lossy()
                    .to_string();
                if let Ok(shutdown) = library.get::<ShutdownInternal>(b"VR_ShutdownInternal\0") {
                    shutdown();
                }
                bail!("{SYSTEM_INTERFACE} is not available: {message}");
            }

            Ok(Self {
                table: table as *const SystemTable,
                library,
                path: path.to_path_buf(),
            })
        }
    }

    /// Where the runtime was loaded from, for the UI to show.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Poses of every tracked device, in the standing universe.
    ///
    /// `prediction` is how far ahead of now to predict, in seconds. Zero asks
    /// for the pose as of this moment, which is what a recording wants; the
    /// tracking loop asks for its own latency instead.
    pub fn poses(&self, prediction: f32) -> Vec<TrackedDevicePose> {
        let mut poses = vec![TrackedDevicePose::default(); MAX_TRACKED_DEVICES];

        // SAFETY: the buffer is exactly the length being declared, and the
        // runtime is alive for as long as `self` is.
        unsafe {
            ((*self.table).get_device_to_absolute_tracking_pose)(
                UNIVERSE_STANDING,
                prediction,
                poses.as_mut_ptr(),
                MAX_TRACKED_DEVICES as u32,
            );
        }

        poses
    }

    pub fn device_class(&self, index: u32) -> i32 {
        // SAFETY: as above; the index is bounded by the caller.
        unsafe { ((*self.table).get_tracked_device_class)(index) }
    }

    pub fn controller_role(&self, index: u32) -> i32 {
        // SAFETY: as above.
        unsafe { ((*self.table).get_controller_role_for_tracked_device_index)(index) }
    }

    pub fn is_connected(&self, index: u32) -> bool {
        // SAFETY: as above.
        unsafe { ((*self.table).is_tracked_device_connected)(index) }
    }

    pub fn serial(&self, index: u32) -> String {
        self.string_property(index, PROP_SERIAL_NUMBER)
    }

    pub fn model(&self, index: u32) -> String {
        self.string_property(index, PROP_MODEL_NUMBER)
    }

    fn string_property(&self, index: u32, property: i32) -> String {
        // Serial and model numbers are short. The API can report far longer
        // strings, but nothing here asks for one.
        let mut buffer = [0u8; 256];
        let mut error = 0i32;

        // SAFETY: the length passed matches the buffer, and the runtime writes
        // a nul-terminated string within it or reports the size it needed.
        let written = unsafe {
            ((*self.table).get_string_tracked_device_property)(
                index,
                property,
                buffer.as_mut_ptr() as *mut c_char,
                buffer.len() as u32,
                &mut error,
            )
        };

        if error != 0 || written == 0 || written as usize > buffer.len() {
            return String::new();
        }

        // The count includes the terminating nul.
        String::from_utf8_lossy(&buffer[..written as usize - 1]).into_owned()
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: the library is still loaded, and this is the documented way
        // to release the runtime. Skipping it leaves SteamVR believing an
        // application is still attached.
        unsafe {
            if let Ok(shutdown) = self
                .library
                .get::<ShutdownInternal>(b"VR_ShutdownInternal\0")
            {
                shutdown();
            }
        }

        CONNECTED.store(false, Ordering::SeqCst);
    }
}

/// Whether a runtime is installed at all, without connecting to it.
///
/// Answering this without starting SteamVR is what lets the UI distinguish "no
/// SteamVR on this machine" from "SteamVR is not running", which are different
/// problems with different fixes.
pub fn is_installed() -> bool {
    candidates().iter().any(|path| path.is_file())
}

/// Where `openvr_api.dll` might be, best guess first.
fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();

    // An explicit override wins, as it does for every other OpenVR client.
    if let Ok(path) = std::env::var("VR_OVERRIDE") {
        out.push(PathBuf::from(path).join("bin/win64/openvr_api.dll"));
    }

    // SteamVR records its own location here when it installs.
    if let Some(runtime) = registered_runtime() {
        out.push(runtime.join("bin/win64/openvr_api.dll"));
    }

    if let Ok(steam) = std::env::var("ProgramFiles(x86)") {
        out.push(
            PathBuf::from(steam).join("Steam/steamapps/common/SteamVR/bin/win64/openvr_api.dll"),
        );
    }

    // Last, the ordinary search path, which covers a copy placed next to the
    // executable.
    out.push(PathBuf::from("openvr_api.dll"));

    // The registered path and the guess usually name the same directory with
    // different separators, and reporting the same refusal three times helps
    // nobody read the message.
    let mut seen = Vec::new();
    out.retain(|path| {
        let key = path.to_string_lossy().replace('\\', "/").to_lowercase();
        let fresh = !seen.contains(&key);
        seen.push(key);
        fresh
    });

    out
}

/// The runtime directory SteamVR registered, from `openvrpaths.vrpath`.
fn registered_runtime() -> Option<PathBuf> {
    let local = std::env::var("LOCALAPPDATA").ok()?;
    let file = PathBuf::from(local).join("openvr/openvrpaths.vrpath");
    let text = std::fs::read_to_string(file).ok()?;
    let parsed: serde_json::Value = serde_json::from_str(&text).ok()?;

    parsed
        .get("runtime")?
        .as_array()?
        .iter()
        .filter_map(|entry| entry.as_str())
        .map(PathBuf::from)
        .find(|path| path.is_dir())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pose struct is written by the runtime, so its size is not a detail:
    /// getting it wrong would have the runtime write past the end of the array.
    #[test]
    fn the_pose_struct_matches_the_c_layout() {
        assert_eq!(std::mem::size_of::<HmdMatrix34>(), 48);
        assert_eq!(std::mem::size_of::<TrackedDevicePose>(), 80);
        assert_eq!(std::mem::align_of::<TrackedDevicePose>(), 4);
    }

    /// Every entry is a function pointer, so the table's size is a direct check
    /// that none were dropped or duplicated while transcribing.
    #[test]
    fn the_function_table_has_the_transcribed_length() {
        assert_eq!(
            std::mem::size_of::<SystemTable>(),
            29 * std::mem::size_of::<*const c_void>()
        );
    }

    #[test]
    fn looking_for_the_runtime_does_not_panic() {
        let _ = is_installed();
        assert!(!candidates().is_empty());
    }

    /// A refused connection has to release the process-wide flag, or every
    /// later attempt reports "already connected" and the link never recovers
    /// once SteamVR does start.
    #[test]
    fn a_refused_connection_can_be_retried() {
        let Err(first) = Runtime::connect() else {
            // SteamVR is running here, so there is nothing to refuse. The
            // guard itself is covered by tests/vr.rs.
            return;
        };
        let Err(second) = Runtime::connect() else {
            return;
        };

        assert_eq!(
            first.to_string(),
            second.to_string(),
            "the second attempt should fail the same way, not for holding a flag"
        );
    }
}
