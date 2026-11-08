//! The few VapourSynth API calls the filters need, over the raw bindings.

use std::ffi::{c_int, CStr, CString};

use vapoursynth_sys::*;

/// The RGB formats the plugin reads and writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Depth {
    Rgb24,
    Rgb48,
    RgbS,
}

impl Depth {
    pub fn of(format: &VSVideoFormat) -> Option<Depth> {
        if format.colorFamily != VSColorFamily_cfRGB as c_int {
            return None;
        }
        let integer = format.sampleType as u32 == VSSampleType_stInteger;
        match format.bitsPerSample {
            8 if integer => Some(Depth::Rgb24),
            16 if integer => Some(Depth::Rgb48),
            32 if !integer => Some(Depth::RgbS),
            _ => None,
        }
    }
}

/// A string argument, or `None` when it is not given.
pub unsafe fn arg_str(api: &VSAPI, map: *const VSMap, key: &CStr) -> Option<String> {
    let mut err = 0;
    let data = api.mapGetData.unwrap()(map, key.as_ptr(), 0, &mut err);
    if err != 0 || data.is_null() {
        return None;
    }
    let size = api.mapGetDataSize.unwrap()(map, key.as_ptr(), 0, &mut err);
    let bytes = std::slice::from_raw_parts(data as *const u8, size.max(0) as usize);
    Some(String::from_utf8_lossy(bytes).into_owned())
}

/// An integer argument, or `None` when it is not given.
pub unsafe fn arg_int(api: &VSAPI, map: *const VSMap, key: &CStr) -> Option<i64> {
    let mut err = 0;
    let v = api.mapGetInt.unwrap()(map, key.as_ptr(), 0, &mut err);
    (err == 0).then_some(v)
}

pub unsafe fn set_error(api: &VSAPI, out: *mut VSMap, msg: &str) {
    let msg = CString::new(format!("Simulate: {msg}")).unwrap_or_default();
    api.mapSetError.unwrap()(out, msg.as_ptr());
}

pub unsafe fn filter_error(api: &VSAPI, ctx: *mut VSFrameContext, msg: &str) {
    let msg = CString::new(format!("Simulate: {msg}")).unwrap_or_default();
    api.setFilterError.unwrap()(msg.as_ptr(), ctx);
}

/// The preset format `id` (a `VSPresetVideoFormat`).
pub unsafe fn format_by_id(api: &VSAPI, id: u32, core: *mut VSCore) -> VSVideoFormat {
    let mut format = std::mem::zeroed();
    api.getVideoFormatByID.unwrap()(&mut format, id, core);
    format
}

/// A source frame as interleaved RGB24, the simulator's frame layout.
/// 16-bit and float samples are rounded to 8 bits; float is clamped to
/// [0, 1] first.
pub unsafe fn read_rgb24(api: &VSAPI, frame: *const VSFrame, depth: Depth) -> Vec<u8> {
    let w = api.getFrameWidth.unwrap()(frame, 0) as usize;
    let h = api.getFrameHeight.unwrap()(frame, 0) as usize;
    let mut out = vec![0u8; w * h * 3];
    for plane in 0..3 {
        let stride = api.getStride.unwrap()(frame, plane as c_int) as usize;
        let base = api.getReadPtr.unwrap()(frame, plane as c_int);
        for y in 0..h {
            let row = base.add(y * stride);
            let dst = out[y * w * 3..(y + 1) * w * 3].iter_mut().skip(plane).step_by(3);
            match depth {
                Depth::Rgb24 => {
                    let src = std::slice::from_raw_parts(row, w);
                    dst.zip(src).for_each(|(d, &s)| *d = s);
                }
                Depth::Rgb48 => {
                    let src = std::slice::from_raw_parts(row as *const u16, w);
                    dst.zip(src).for_each(|(d, &s)| *d = ((s as u32 * 255 + 32767) / 65535) as u8);
                }
                Depth::RgbS => {
                    let src = std::slice::from_raw_parts(row as *const f32, w);
                    dst.zip(src).for_each(|(d, &s)| *d = (s.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
        }
    }
    out
}

/// A new `w`×`h` frame of `format` holding interleaved RGB24 `rgb`, with
/// the frame properties of `props`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn new_frame(
    api: &VSAPI,
    format: &VSVideoFormat,
    depth: Depth,
    rgb: &[u8],
    w: usize,
    h: usize,
    props: *const VSFrame,
    core: *mut VSCore,
) -> *mut VSFrame {
    let frame = api.newVideoFrame.unwrap()(format, w as c_int, h as c_int, props, core);
    for plane in 0..3 {
        let stride = api.getStride.unwrap()(frame, plane as c_int) as usize;
        let base = api.getWritePtr.unwrap()(frame, plane as c_int);
        for y in 0..h {
            let row = base.add(y * stride);
            let src = rgb[y * w * 3..(y + 1) * w * 3].iter().skip(plane).step_by(3);
            match depth {
                Depth::Rgb24 => {
                    let dst = std::slice::from_raw_parts_mut(row, w);
                    dst.iter_mut().zip(src).for_each(|(d, &s)| *d = s);
                }
                Depth::Rgb48 => {
                    let dst = std::slice::from_raw_parts_mut(row as *mut u16, w);
                    dst.iter_mut().zip(src).for_each(|(d, &s)| *d = s as u16 * 257);
                }
                Depth::RgbS => {
                    let dst = std::slice::from_raw_parts_mut(row as *mut f32, w);
                    dst.iter_mut().zip(src).for_each(|(d, &s)| *d = s as f32 / 255.0);
                }
            }
        }
    }
    frame
}

/// Requests frames `first..=last` of `node` for the current frame.
pub unsafe fn request_frames(api: &VSAPI, node: *mut VSNode, first: u32, last: u32, ctx: *mut VSFrameContext) {
    for k in first..=last {
        api.requestFrameFilter.unwrap()(k as c_int, node, ctx);
    }
}

/// The requested frames `first..=last` of `node`; they are freed on drop.
pub struct Frames<'a> {
    api: &'a VSAPI,
    pub first: u32,
    pub frames: Vec<*const VSFrame>,
}

impl<'a> Frames<'a> {
    pub unsafe fn get(api: &'a VSAPI, node: *mut VSNode, first: u32, last: u32, ctx: *mut VSFrameContext) -> Self {
        let frames = (first..=last).map(|k| api.getFrameFilter.unwrap()(k as c_int, node, ctx)).collect();
        Frames { api, first, frames }
    }

    pub fn frame(&self, k: u32) -> *const VSFrame {
        self.frames[(k - self.first) as usize]
    }
}

impl Drop for Frames<'_> {
    fn drop(&mut self) {
        for &f in &self.frames {
            unsafe { self.api.freeFrame.unwrap()(f) };
        }
    }
}

pub fn cstr(s: &str) -> CString {
    CString::new(s).expect("no NUL in names")
}
