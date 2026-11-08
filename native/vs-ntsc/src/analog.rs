//! The analog filter: each frame encoded to the composite (or S-Video)
//! signal, given the tape, reception and deck effects of its frame number,
//! and decoded, by the simulator's `FrameProcessor`.

use std::ffi::{c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use ntsc_composite_simulator::effects::SignalEffects;
use ntsc_composite_simulator::params::ParamValues;
use ntsc_composite_simulator::pipeline::FrameProcessor;
use vapoursynth_sys::*;

use crate::vs::{self, Depth, Frames};

pub struct Analog {
    pub node: *mut VSNode,
    pub in_depth: Depth,
    pub out_format: VSVideoFormat,
    pub out_depth: Depth,
    pub values: ParamValues,
    /// The effects without the deck noise reduction, which runs in its own
    /// filter (dnr.rs).
    pub fx: SignalEffects,
    pub in_size: (usize, usize),
    pub out_size: (usize, usize),
    /// The 3D comb reads the two source frames before each frame.
    pub uses_previous: bool,
}

impl Analog {
    fn first_source(&self, n: u32) -> u32 {
        if self.uses_previous { n.saturating_sub(2) } else { n }
    }

    /// Frame `n` from its source frames: the two before it for the 3D
    /// comb, re-encoded with their own frame numbers and effects, as the
    /// CLI's roundtrip sees them.
    fn render(&self, n: u32, sources: &[Vec<u8>]) -> Vec<u8> {
        let (before, current) = sources.split_at(sources.len() - 1);
        let mut processor =
            FrameProcessor::with_effects(&self.values, self.fx.clone(), false, self.in_size, self.out_size, n);
        if self.uses_previous {
            processor.skip_to(n, before);
        }
        processor.process_batch(current).pop().expect("one frame in, one out")
    }
}

pub unsafe extern "C" fn get_frame(
    n: c_int,
    reason: c_int,
    data: *mut c_void,
    _frame_data: *mut *mut c_void,
    ctx: *mut VSFrameContext,
    core: *mut VSCore,
    api: *const VSAPI,
) -> *const VSFrame {
    let (f, api) = (&*(data as *const Analog), &*api);
    let n = n as u32;
    if reason == VSActivationReason_arInitial {
        vs::request_frames(api, f.node, f.first_source(n), n, ctx);
    } else if reason == VSActivationReason_arAllFramesReady {
        let frames = Frames::get(api, f.node, f.first_source(n), n, ctx);
        let sources: Vec<Vec<u8>> = frames.frames.iter().map(|&s| vs::read_rgb24(api, s, f.in_depth)).collect();
        return match catch_unwind(AssertUnwindSafe(|| f.render(n, &sources))) {
            Ok(rgb) => {
                let (w, h) = f.out_size;
                vs::new_frame(api, &f.out_format, f.out_depth, &rgb, w, h, frames.frame(n), core)
            }
            Err(_) => {
                vs::filter_error(api, ctx, &format!("the simulator failed on frame {n}"));
                ptr::null()
            }
        };
    }
    ptr::null()
}

pub unsafe extern "C" fn free(data: *mut c_void, _core: *mut VSCore, api: *const VSAPI) {
    let f = Box::from_raw(data as *mut Analog);
    (*api).freeNode.unwrap()(f.node);
}
