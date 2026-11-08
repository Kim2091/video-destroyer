//! The deck's recursive noise reduction, the one stage that carries state
//! from frame to frame, as a filter of its own over the analog filter.
//!
//! Each output frame is the recursion run from a fixed start frame
//! `chain_start(n)`, 16 to 31 frames before `n` (0 for the first 32
//! frames), so a frame's result does not depend on which frames were asked
//! for before it. The recursion forgets its past geometrically (by
//! 0.675 per frame in still areas at strength 0.5), so this matches an
//! unbroken run to well under one 8-bit level.
//!
//! In order, each frame costs two recursion steps: the chain that gives
//! its output, and the chain the next block will start from. Frames that
//! arrive out of order are taken from the outputs the chains passed on
//! the way, or else the chain is started again.

use std::collections::BTreeMap;
use std::ffi::{c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::Mutex;

use ntsc_composite_simulator::deck::TemporalDnr;
use vapoursynth_sys::*;

use crate::vs::{self, Depth, Frames};

/// Frames per block; a chain starts one block before the block of `n`.
const BLOCK: u32 = 16;

fn chain_start(n: u32) -> u32 {
    (n / BLOCK).saturating_sub(1) * BLOCK
}

struct Chain {
    start: u32,
    /// The frame the chain takes next.
    next: u32,
    dnr: TemporalDnr,
}

#[derive(Default)]
struct State {
    chains: Vec<Chain>,
    /// Outputs of frames the chains passed before they were asked for.
    passed: BTreeMap<u32, Vec<u8>>,
}

pub struct Dnr {
    /// The analog filter, RGB24.
    pub node: *mut VSNode,
    pub out_format: VSVideoFormat,
    pub out_depth: Depth,
    pub strength: f32,
    pub size: (usize, usize),
    state: Mutex<State>,
}

impl Dnr {
    pub fn new(node: *mut VSNode, out_format: VSVideoFormat, out_depth: Depth, strength: f32, size: (usize, usize)) -> Self {
        Dnr { node, out_format, out_depth, strength, size, state: Mutex::default() }
    }

    /// Output frame `n`; `frame(k)` is analog frame `k`, for `k` from
    /// `chain_start(n)` to `n`.
    fn render(&self, n: u32, frame: &dyn Fn(u32) -> Vec<u8>) -> Vec<u8> {
        let (w, h) = self.size;
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let State { chains, passed } = &mut *state;
        if let Some(out) = passed.remove(&n) {
            return out;
        }
        let mut take = |start: u32| match chains.iter().position(|c| c.start == start && c.next <= n) {
            Some(i) => chains.swap_remove(i),
            None => Chain { start, next: start, dnr: TemporalDnr::new(self.strength) },
        };
        let start = chain_start(n);
        let mut chain = take(start);
        let mut out = Vec::new();
        for k in chain.next..=n {
            let mut rgb = frame(k);
            chain.dnr.process(&mut rgb, w, h);
            if k == n {
                out = rgb;
            } else if chain_start(k) == start {
                passed.insert(k, rgb);
            }
        }
        chain.next = n + 1;
        let block = n / BLOCK * BLOCK;
        let mut kept = vec![chain];
        if block != start {
            let mut warm = take(block);
            for k in warm.next..=n {
                warm.dnr.process(&mut frame(k), w, h);
            }
            warm.next = n + 1;
            kept.push(warm);
        }
        *chains = kept;
        passed.retain(|&k, _| k + 2 * BLOCK > n);
        out
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
    let (f, api) = (&*(data as *const Dnr), &*api);
    let n = n as u32;
    if reason == VSActivationReason_arInitial {
        vs::request_frames(api, f.node, chain_start(n), n, ctx);
    } else if reason == VSActivationReason_arAllFramesReady {
        let frames = Frames::get(api, f.node, chain_start(n), n, ctx);
        let frame = |k: u32| vs::read_rgb24(api, frames.frame(k), Depth::Rgb24);
        return match catch_unwind(AssertUnwindSafe(|| f.render(n, &frame))) {
            Ok(rgb) => {
                let (w, h) = f.size;
                vs::new_frame(api, &f.out_format, f.out_depth, &rgb, w, h, frames.frame(n), core)
            }
            Err(_) => {
                vs::filter_error(api, ctx, &format!("the noise reduction failed on frame {n}"));
                ptr::null()
            }
        };
    }
    ptr::null()
}

pub unsafe extern "C" fn free(data: *mut c_void, _core: *mut VSCore, api: *const VSAPI) {
    let f = Box::from_raw(data as *mut Dnr);
    (*api).freeNode.unwrap()(f.node);
}
