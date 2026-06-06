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
    // Pinned host memory + async H2D copy — lets per-token activation uploads
    // enqueue on the stream without a host-blocking GPU drain (pageable
    // cuMemcpyHtoD is always synchronous; pinned + Async is not).
    pub fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> CUresult;
    pub fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    pub fn cuMemcpyHtoDAsync_v2(dst: CUdeviceptr, src: *const c_void, byte_count: usize, stream: CUstream) -> CUresult;
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
    /// Cooperative kernel launch — all thread blocks can synchronize via
    /// `cg::grid_group::sync()`. Required for two-phase fused kernels that
    /// need a global barrier between phases (e.g. MoE up-proj → down-proj).
    #[allow(clippy::too_many_arguments)]
    pub fn cuLaunchCooperativeKernel(
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
    // ── Stream + CUDA-graph capture (replay a whole decode token as ONE graph
    // launch, eliminating the ~390 per-kernel launch/host-orchestration costs). ──
    pub fn cuStreamCreate(phStream: *mut CUstream, flags: c_uint) -> CUresult;
    pub fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult;
    pub fn cuStreamSynchronize(hStream: CUstream) -> CUresult;
    pub fn cuStreamBeginCapture_v2(hStream: CUstream, mode: c_int) -> CUresult;
    pub fn cuStreamEndCapture(hStream: CUstream, phGraph: *mut CUgraph) -> CUresult;
    pub fn cuGraphInstantiateWithFlags(
        phGraphExec: *mut CUgraphExec,
        hGraph: CUgraph,
        flags: u64,
    ) -> CUresult;
    pub fn cuGraphLaunch(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult;
    pub fn cuGraphExecDestroy(hGraphExec: CUgraphExec) -> CUresult;
    pub fn cuGraphDestroy(hGraph: CUgraph) -> CUresult;
}

pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;
/// `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` — capture scoped to this thread.
pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: c_int = 1;
/// `CU_STREAM_NON_BLOCKING` — stream does not implicitly sync with the null stream.
pub const CU_STREAM_NON_BLOCKING: c_uint = 1;

/// Default event flag (`CU_EVENT_DEFAULT`) — blocking-sync timing event.
pub const CU_EVENT_DEFAULT: c_uint = 0;

// ── cuBLAS (tensor-core GEMM escape hatch, Path A) ──────────────────────────
// Legacy cuBLAS API. `cublasGemmEx` exposes the mixed-precision tensor-core
// path: bf16/f16 A·B with f32 accumulate (CUBLAS_COMPUTE_32F) + the
// DEFAULT_TENSOR_OP algo selector drives the Tensor Cores on GB10.
pub type cublasHandle_t = *mut c_void;
pub type cublasStatus_t = c_int;
pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

// cublasOperation_t
pub const CUBLAS_OP_N: c_int = 0;
pub const CUBLAS_OP_T: c_int = 1;

// cudaDataType_t (library_types.h)
pub const CUDA_R_16F: c_int = 2;
pub const CUDA_R_32F: c_int = 0;
pub const CUDA_R_16BF: c_int = 14;

// cublasComputeType_t
pub const CUBLAS_COMPUTE_32F: c_int = 68;

// cublasGemmAlgo_t
pub const CUBLAS_GEMM_DEFAULT: c_int = -1;
pub const CUBLAS_GEMM_DEFAULT_TENSOR_OP: c_int = 99;

// cublasMath_t
pub const CUBLAS_DEFAULT_MATH: c_int = 0;
pub const CUBLAS_TENSOR_OP_MATH: c_int = 1;

#[link(name = "cublas")]
unsafe extern "C" {
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
    pub fn cublasSetStream_v2(handle: cublasHandle_t, stream: CUstream) -> cublasStatus_t;
    pub fn cublasSetMathMode(handle: cublasHandle_t, mode: c_int) -> cublasStatus_t;
    #[allow(clippy::too_many_arguments)]
    pub fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: CUdeviceptr,
        atype: c_int,
        lda: c_int,
        b: CUdeviceptr,
        btype: c_int,
        ldb: c_int,
        beta: *const c_void,
        c: CUdeviceptr,
        ctype: c_int,
        ldc: c_int,
        compute_type: c_int,
        algo: c_int,
    ) -> cublasStatus_t;
    /// Strided-batched GEMM — one call does `batch` independent GEMMs whose
    /// A/B/C each advance by a fixed stride. Used for the per-expert routed
    /// MoE grouped GEMM when all experts share the same (m,n,k).
    #[allow(clippy::too_many_arguments)]
    pub fn cublasGemmStridedBatchedEx(
        handle: cublasHandle_t,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: CUdeviceptr,
        atype: c_int,
        lda: c_int,
        stride_a: i64,
        b: CUdeviceptr,
        btype: c_int,
        ldb: c_int,
        stride_b: i64,
        beta: *const c_void,
        c: CUdeviceptr,
        ctype: c_int,
        ldc: c_int,
        stride_c: i64,
        batch_count: c_int,
        compute_type: c_int,
        algo: c_int,
    ) -> cublasStatus_t;
    /// Grouped-batched GEMM (CUDA 13+): one call over  independent
    /// GEMM groups, each with its own m/n/k and pointer arrays. Ideal for MoE
    /// prefill: each group is one active expert, eliminating the per-expert loop.
    /// All groups share the same dtype (f16 or bf16) and compute type (f32).
    #[allow(clippy::too_many_arguments)]
    pub fn cublasGemmGroupedBatchedEx(
        handle: cublasHandle_t,
        transa_array: *const c_int,
        transb_array: *const c_int,
        m_array: *const c_int,
        n_array: *const c_int,
        k_array: *const c_int,
        alpha_array: *const c_void,
        aarray: *const *const c_void,
        atype: c_int,
        lda_array: *const c_int,
        barray: *const *const c_void,
        btype: c_int,
        ldb_array: *const c_int,
        beta_array: *const c_void,
        carray: *mut *mut c_void,
        ctype: c_int,
        ldc_array: *const c_int,
        group_count: c_int,
        group_size: *const c_int,
        compute_type: c_int,
    ) -> cublasStatus_t;
}

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
