//! VapourSynth plugin `ntsc`: ntsc-simulator's analog path (NTSC composite
//! encode, VHS tape and reception effects, playback deck, decoder) on each
//! frame of an RGB clip, with the simulator's presets and parameters.
//!
//! ```python
//! core.ntsc.Simulate(clip, preset="mq", deck="deck-basic", seed=7, tape_snr="30..33", dubs=1)
//! ```
//!
//! Every simulator parameter outside the DVD group is an optional string
//! argument named like its CLI flag with `_` for `-`, taking the CLI's
//! syntax (a number, a `min..max` range, true/false, or a choice). Values
//! given here win over the deck's, and the deck's over the preset's, as on
//! the CLI. Ranges are sampled once per clip from `seed`. `dubs` adds that
//! many tape generations to the sampled `generations`, so a dub of a tape
//! whose generations is a range copies whatever generation was drawn.
//!
//! A frame depends only on the settings, the seed, its number and the
//! source frames it reads, so frames may be asked for in any order and in
//! parallel: the effects of frame n come from the seed and n, as in the
//! CLI; the 3D comb reads source frames n - 2 and n - 1; the deck's
//! recursive noise reduction restarts at a fixed frame before n (dnr.rs).

mod analog;
mod dnr;
mod vs;

use std::ffi::{c_int, c_void, CStr};

use ntsc_composite_simulator::params::{Group, ParamValues, SPECS};
use ntsc_composite_simulator::pipeline::{resolve_effects, FrameProcessor};
use ntsc_composite_simulator::presets;
use vapoursynth_sys::*;

use analog::Analog;
use dnr::Dnr;
use vs::Depth;

const API_VERSION: c_int = (VAPOURSYNTH_API_MAJOR << 16 | VAPOURSYNTH_API_MINOR) as c_int;

/// The simulator parameters taken as arguments: all but the DVD path's and
/// the input colour conversion, which belongs to its ffmpeg reader.
fn plugin_params() -> impl Iterator<Item = &'static str> {
    SPECS.iter().filter(|s| s.group != Group::Dvd && !s.id.starts_with("input-")).map(|s| s.id)
}

fn arg_name(id: &str) -> String {
    id.replace('-', "_")
}

/// The settings from the arguments: explicit values, then the deck, then
/// the preset fill in what is not set.
unsafe fn settings(api: &VSAPI, args: *const VSMap) -> Result<ParamValues, String> {
    let mut values = ParamValues::default();
    for id in plugin_params() {
        if let Some(text) = vs::arg_str(api, args, &vs::cstr(&arg_name(id))) {
            values.set_from_str(id, &text)?;
        }
    }
    if let Some(name) = vs::arg_str(api, args, c"deck").filter(|s| !s.is_empty()) {
        let deck = presets::deck(&name)
            .or_else(|| presets::deck(&format!("deck-{name}")))
            .ok_or_else(|| format!("unknown deck '{name}'"))?;
        values.fill_from(&deck.to_values());
    }
    if let Some(name) = vs::arg_str(api, args, c"preset").filter(|s| !s.is_empty()) {
        let preset = presets::builtin(&name)
            .filter(|p| p.group != "DVD")
            .ok_or_else(|| format!("unknown preset '{name}' (the DVD presets need ffmpeg and are not available)"))?;
        values.fill_from(&preset.to_values());
    }
    values.seed = Some(vs::arg_int(api, args, c"seed").unwrap_or(0) as u64);
    Ok(values)
}

unsafe extern "C" fn create(
    args: *const VSMap,
    out: *mut VSMap,
    _user_data: *mut c_void,
    core: *mut VSCore,
    api: *const VSAPI,
) {
    let api = &*api;
    let node = api.mapGetNode.unwrap()(args, c"clip".as_ptr(), 0, std::ptr::null_mut());
    if let Err(msg) = build(api, node, args, out, core) {
        api.freeNode.unwrap()(node);
        vs::set_error(api, out, &msg);
    }
}

unsafe fn build(api: &VSAPI, node: *mut VSNode, args: *const VSMap, out: *mut VSMap, core: *mut VSCore) -> Result<(), String> {
    let vi = *api.getVideoInfo.unwrap()(node);
    let depth = Depth::of(&vi.format).ok_or("the clip must be RGB24, RGB48 or RGBS, with a constant format")?;
    if vi.width <= 0 || vi.height <= 0 {
        return Err("the clip must have a constant frame size".into());
    }
    let size = |key: &CStr, own: c_int| match vs::arg_int(api, args, key) {
        None | Some(0) => Ok(own as usize),
        Some(v) if v > 0 => Ok(v as usize),
        Some(_) => Err(format!("{} must be positive", key.to_string_lossy())),
    };
    let in_size = (vi.width as usize, vi.height as usize);
    let out_size = (size(c"width", vi.width)?, size(c"height", vi.height)?);
    let values = settings(api, args)?;

    let mut fx = resolve_effects(&values);
    match vs::arg_int(api, args, c"dubs") {
        None => {}
        Some(d) if d >= 0 => fx.generations += d as usize,
        Some(_) => return Err("dubs must be 0 or more".into()),
    }
    let dnr = fx.deck_dnr.take().filter(|s| *s > 0.0);
    let uses_previous = FrameProcessor::with_effects(&values, fx.clone(), false, in_size, out_size, 0).uses_previous_frames();
    let out_format = vi.format;
    let rgb24 = vs::format_by_id(api, VSPresetVideoFormat_pfRGB24, core);
    let analog = Box::new(Analog {
        node,
        in_depth: depth,
        out_format: if dnr.is_some() { rgb24 } else { out_format },
        out_depth: if dnr.is_some() { Depth::Rgb24 } else { depth },
        values,
        fx,
        in_size,
        out_size,
        uses_previous,
    });
    let out_vi = |format: VSVideoFormat| VSVideoInfo {
        format,
        width: out_size.0 as c_int,
        height: out_size.1 as c_int,
        ..vi
    };
    let deps = [VSFilterDependency { source: node, requestPattern: VSRequestPattern_rpGeneral as c_int }];
    let analog_vi = out_vi(analog.out_format);
    let analog_node = api.createVideoFilter2.unwrap()(
        c"Simulate".as_ptr(),
        &analog_vi,
        Some(analog::get_frame),
        Some(analog::free),
        VSFilterMode_fmParallel as c_int,
        deps.as_ptr(),
        1,
        Box::into_raw(analog) as *mut c_void,
        core,
    );
    let result = match dnr {
        None => analog_node,
        Some(strength) => {
            let deps = [VSFilterDependency { source: analog_node, requestPattern: VSRequestPattern_rpGeneral as c_int }];
            let data = Box::new(Dnr::new(analog_node, out_format, depth, strength, out_size));
            api.createVideoFilter2.unwrap()(
                c"SimulateDNR".as_ptr(),
                &out_vi(out_format),
                Some(dnr::get_frame),
                Some(dnr::free),
                VSFilterMode_fmParallelRequests as c_int,
                deps.as_ptr(),
                1,
                Box::into_raw(data) as *mut c_void,
                core,
            )
        }
    };
    api.mapConsumeNode.unwrap()(out, c"clip".as_ptr(), result, VSMapAppendMode_maReplace as c_int);
    Ok(())
}

/// # Safety
/// Called by VapourSynth when it loads the plugin.
#[no_mangle]
pub unsafe extern "C" fn VapourSynthPluginInit2(plugin: *mut VSPlugin, vspapi: *const VSPLUGINAPI) {
    let vspapi = &*vspapi;
    vspapi.configPlugin.unwrap()(
        c"com.kim2091.ntsc".as_ptr(),
        c"ntsc".as_ptr(),
        c"NTSC composite and VHS simulator (ntsc-simulator)".as_ptr(),
        1 << 16,
        API_VERSION,
        0,
        plugin,
    );
    let mut args = String::from("clip:vnode;preset:data:opt;deck:data:opt;seed:int:opt;dubs:int:opt;width:int:opt;height:int:opt;");
    for id in plugin_params() {
        args.push_str(&format!("{}:data:opt;", arg_name(id)));
    }
    let args = vs::cstr(&args);
    vspapi.registerFunction.unwrap()(c"Simulate".as_ptr(), args.as_ptr(), c"clip:vnode;".as_ptr(), Some(create), std::ptr::null_mut(), plugin);
}
