//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Minimal hand-rolled FFI to the CUDA Driver API (`libcuda`) and NVRTC
//! (`libnvrtc`) — only the symbols the Phase-1 NVRTC-compile + launch path
//! needs (CUDA_BACKEND_SPEC §4.4 / §4.5). `cuda-oxide`'s `cuda-core` is the
//! recommended longer-term host-runtime dep, but its `cuda-bindings` crate
//! is under the NVIDIA Software License (not Apache) and the stack is alpha
//! / Linux-only, so Phase 1 hand-rolls this small, stable surface for
//! control. Re-evaluate `cuda-core` adoption when it leaves alpha.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_uint, c_void};

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
/// `CUdeviceptr` is `unsigned long long` on 64-bit platforms.
pub type CUdeviceptr = u64;

pub type nvrtcResult = c_int;
pub type nvrtcProgram = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;
pub const NVRTC_SUCCESS: nvrtcResult = 0;

// Device attribute enum values (cuda.h: CUdevice_attribute).
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

// Function attribute: opt-in max dynamic shared memory (bytes) for >48KB.
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

// CUDA 13 ships the driver symbols under their `_v2` (and similar) names;
// the un-suffixed identifiers in cuda.h are preprocessor aliases. We bind
// the actual exported symbols directly.
#[link(name = "cuda")]
unsafe extern "C" {
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetAttribute(pi: *mut c_int, attrib: c_int, dev: CUdevice) -> CUresult;
    pub fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleUnload(module: CUmodule) -> CUresult;
    pub fn cuModuleGetFunction(
        func: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    pub fn cuFuncSetAttribute(func: CUfunction, attrib: c_int, value: c_int) -> CUresult;
    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, byte_count: usize) -> CUresult;
    pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, byte_count: usize) -> CUresult;
    #[allow(clippy::too_many_arguments)]
    pub fn cuLaunchKernel(
        f: CUfunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    pub fn cuGetErrorString(error: CUresult, p_str: *mut *const c_char) -> CUresult;
    // Event-based device timing (GPU-side wall clock, independent of host
    // scheduling jitter). `cuEventElapsedTime` returns milliseconds as f32.
    pub fn cuEventCreate(phEvent: *mut CUevent, flags: c_uint) -> CUresult;
    pub fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult;
    pub fn cuEventSynchronize(hEvent: CUevent) -> CUresult;
    pub fn cuEventElapsedTime(
        pMilliseconds: *mut f32,
        hStart: CUevent,
        hEnd: CUevent,
    ) -> CUresult;
    pub fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult;
}

/// Default event flag (`CU_EVENT_DEFAULT`) — blocking-sync timing event.
pub const CU_EVENT_DEFAULT: c_uint = 0;

#[link(name = "nvrtc")]
unsafe extern "C" {
    pub fn nvrtcCreateProgram(
        prog: *mut nvrtcProgram,
        src: *const c_char,
        name: *const c_char,
        num_headers: c_int,
        headers: *const *const c_char,
        include_names: *const *const c_char,
    ) -> nvrtcResult;
    pub fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;
    pub fn nvrtcCompileProgram(
        prog: nvrtcProgram,
        num_options: c_int,
        options: *const *const c_char,
    ) -> nvrtcResult;
    pub fn nvrtcGetPTXSize(prog: nvrtcProgram, ptx_size_ret: *mut usize) -> nvrtcResult;
    pub fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult;
    pub fn nvrtcGetProgramLogSize(prog: nvrtcProgram, log_size_ret: *mut usize) -> nvrtcResult;
    pub fn nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult;
    pub fn nvrtcGetErrorString(result: nvrtcResult) -> *const c_char;
}
