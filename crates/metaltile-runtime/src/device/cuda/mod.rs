//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! CUDA runtime backend (CUDA_BACKEND_SPEC §4.1 / §4.4 — SCOPE Phase 1).
//!
//! `CudaDevice` is the NVIDIA analog of `MetalDevice`: it owns a CUDA
//! context and provides compile (NVRTC: CUDA C++ → PTX → module),
//! allocate, upload, launch, and read-back. Phase 1 is the smoke path —
//! enough to compile a generated elementwise kernel, run it on the GX10
//! (sm_121), and read results back for the CPU oracle. The `Device`-trait
//! abstraction + `Context` integration is Phase 6 (CLI/harness wiring);
//! this module stands on its own so the end-to-end pipeline is provable
//! now.
//!
//! Feature-gated (`cuda`) and Linux-targeted; the macOS Metal path is
//! untouched and stays the zero-config default.

mod ffi;

use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;
use std::sync::Mutex;

use metaltile_codegen::{CodegenBackend, CudaGenerator};
use metaltile_core::ir::Kernel;

use crate::error::MetalTileError;

use ffi::*;

/// Synthesize a Strided param's `_shape` or `_strides` (row-major) buffer
/// from the static shape, as little-endian u32s. Unknown dims default to 1.
fn synth_strided_meta(shape: &metaltile_core::shape::Shape, strides: bool) -> Vec<u8> {
    use metaltile_core::shape::Dim;
    let dims: Vec<u32> = (0..shape.rank())
        .map(|i| match shape.dim(i) {
            Some(Dim::Known(n)) => *n as u32,
            _ => 1,
        })
        .collect();
    let vals: Vec<u32> = if strides {
        let mut s = vec![1u32; dims.len()];
        for i in (0..dims.len().saturating_sub(1)).rev() {
            s[i] = s[i + 1] * dims[i + 1];
        }
        s
    } else {
        dims
    };
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn cu_check(res: CUresult, what: &str) -> Result<(), MetalTileError> {
    if res == CUDA_SUCCESS {
        return Ok(());
    }
    let mut s: *const c_char = ptr::null();
    let msg = unsafe {
        if cuGetErrorString(res, &mut s) == CUDA_SUCCESS && !s.is_null() {
            CStr::from_ptr(s).to_string_lossy().into_owned()
        } else {
            format!("CUDA error code {res}")
        }
    };
    Err(MetalTileError::Dispatch(format!("{what}: {msg}")))
}

fn nvrtc_check(res: nvrtcResult, what: &str) -> Result<(), MetalTileError> {
    if res == NVRTC_SUCCESS {
        return Ok(());
    }
    let msg = unsafe {
        let s = nvrtcGetErrorString(res);
        if s.is_null() {
            format!("nvrtc error code {res}")
        } else {
            CStr::from_ptr(s).to_string_lossy().into_owned()
        }
    };
    Err(MetalTileError::Compilation(format!("{what}: {msg}")))
}

/// A device-side allocation. Frees on drop.
pub struct DeviceBuffer<'d> {
    ptr: CUdeviceptr,
    len: usize,
    _dev: &'d CudaDevice,
}

impl DeviceBuffer<'_> {
    pub fn device_ptr(&self) -> CUdeviceptr { self.ptr }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

impl Drop for DeviceBuffer<'_> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe { cuMemFree_v2(self.ptr) };
        }
    }
}

/// A compiled, loaded CUDA module. Unloads on drop.
pub struct CudaModule {
    module: CUmodule,
}

impl CudaModule {
    /// Look up a kernel function by its `extern "C"` symbol name.
    pub fn function(&self, name: &str) -> Result<CudaFunction, MetalTileError> {
        let cname = CString::new(name).map_err(|e| MetalTileError::Dispatch(e.to_string()))?;
        let mut func: CUfunction = ptr::null_mut();
        cu_check(
            unsafe { cuModuleGetFunction(&mut func, self.module, cname.as_ptr()) },
            &format!("cuModuleGetFunction({name})"),
        )?;
        Ok(CudaFunction { func })
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe { cuModuleUnload(self.module) };
        }
    }
}

/// A handle to a `__global__` function inside a [`CudaModule`].
#[derive(Clone, Copy)]
pub struct CudaFunction {
    func: CUfunction,
}

/// A compiled + resident kernel ready to launch repeatedly. See
/// [`CudaDevice::prepare`]. Holds its device resources alive; `args()`
/// hands out raw kernel-param pointers into `dev_ptrs`/`scalars`, so those
/// vecs must not be mutated after the first `args()` call.
struct Prepared<'d> {
    _module: CudaModule,
    func: CudaFunction,
    dev_bufs: Vec<DeviceBuffer<'d>>,
    dev_ptrs: Vec<CUdeviceptr>,
    scalars: Vec<Vec<u8>>,
    out_meta: Vec<Option<(String, usize)>>,
    shared_bytes: u32,
}

impl Prepared<'_> {
    /// Build the `cuLaunchKernel` param array: param device-ptrs first, then
    /// scalar values, each a raw pointer into our stable `dev_ptrs`/`scalars`.
    fn args(&self) -> Vec<*mut c_void> {
        let mut args: Vec<*mut c_void> = Vec::with_capacity(self.dev_ptrs.len() + self.scalars.len());
        for p in &self.dev_ptrs {
            args.push(p as *const CUdeviceptr as *mut c_void);
        }
        for s in &self.scalars {
            args.push(s.as_ptr() as *mut c_void);
        }
        args
    }
}

/// CUDA device + context. NVIDIA analog of `MetalDevice`.
pub struct CudaDevice {
    ctx: CUcontext,
    cc_major: i32,
    cc_minor: i32,
    /// Size-bucketed free-list of device allocations. `cuMemAlloc`/`cuMemFree`
    /// are synchronous (each blocks the device), so churning ~1000 of them per
    /// decode token dominated step time. All GPU work is on one (default) stream
    /// here, so reusing a freed buffer for a later op is safe — stream order
    /// guarantees the prior kernel finished. Decode reuses the same shapes each
    /// token ⇒ near-zero alloc cost after warm-up.
    pool: Mutex<HashMap<usize, Vec<CUdeviceptr>>>,
    /// Pinned host-staging buffers for async H2D: free-list by size, plus the
    /// in-flight list whose copies haven't been synced yet (reclaimed on the next
    /// `synchronize`/`download`). `usize` holds the host pointer.
    pinned_free: Mutex<HashMap<usize, Vec<usize>>>,
    pinned_inflight: Mutex<Vec<(usize, usize)>>,
    /// Dedicated non-blocking stream that ALL kernel launches + async H2D ride on.
    /// Replacing the null stream is what makes a whole decode token CUDA-graph
    /// CAPTURABLE (phase-1 of the megakernel: replay ~390 launches as one graph,
    /// eliminating per-launch host enqueue + inter-kernel bubbles → higher
    /// sustained bandwidth). `synchronize`/`dtoh` sync THIS stream.
    stream: CUstream,
    /// When true, launches/copies are being recorded into a CUDA graph (no real
    /// execution yet) — `dtoh`/`synchronize` must NOT be called mid-capture.
    capturing: std::sync::atomic::AtomicBool,
}

// The context is current on this struct's lifetime; we keep it single-
// device, single-context (Phase 1). Send is sound because we never share
// the raw pointers across threads concurrently in the smoke path. Sync is
// sound for serialized submission (a higher layer holds the device in an
// `Arc` to keep its context alive for persistent buffers; GPU work is
// submitted from one logical owner at a time).
unsafe impl Send for CudaDevice {}
unsafe impl Sync for CudaDevice {}

impl CudaDevice {
    /// Initialize CUDA, grab device 0, create a context. Returns `Ok(None)`
    /// if no CUDA device is present (mirrors `MetalDevice::create`).
    pub fn create() -> Result<Option<Self>, MetalTileError> {
        unsafe {
            if cuInit(0) != CUDA_SUCCESS {
                return Ok(None);
            }
            let mut dev: CUdevice = 0;
            if cuDeviceGet(&mut dev, 0) != CUDA_SUCCESS {
                return Ok(None);
            }
            let mut major: i32 = 0;
            let mut minor: i32 = 0;
            cu_check(
                cuDeviceGetAttribute(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev),
                "cuDeviceGetAttribute(cc_major)",
            )?;
            cu_check(
                cuDeviceGetAttribute(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev),
                "cuDeviceGetAttribute(cc_minor)",
            )?;
            let mut ctx: CUcontext = ptr::null_mut();
            cu_check(cuCtxCreate_v2(&mut ctx, 0, dev), "cuCtxCreate")?;
            let mut stream: CUstream = ptr::null_mut();
            cu_check(cuStreamCreate(&mut stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate")?;
            Ok(Some(CudaDevice { ctx, cc_major: major, cc_minor: minor, pool: Mutex::new(HashMap::new()), pinned_free: Mutex::new(HashMap::new()), pinned_inflight: Mutex::new(Vec::new()), stream, capturing: std::sync::atomic::AtomicBool::new(false) }))
        }
    }

    /// Compute capability as `(major, minor)` — e.g. `(12, 1)` on GB10.
    pub fn compute_capability(&self) -> (i32, i32) { (self.cc_major, self.cc_minor) }

    /// Compile CUDA C++ source → PTX → loaded module via NVRTC + the
    /// driver JIT. Targets the device's own virtual arch
    /// (`compute_<major><minor>`); the driver JITs PTX to the live GPU.
    pub fn compile(&self, src: &str, prog_name: &str) -> Result<CudaModule, MetalTileError> {
        let csrc = CString::new(src).map_err(|e| MetalTileError::Compilation(e.to_string()))?;
        let cname =
            CString::new(prog_name).map_err(|e| MetalTileError::Compilation(e.to_string()))?;

        let mut prog: nvrtcProgram = ptr::null_mut();
        nvrtc_check(
            unsafe {
                nvrtcCreateProgram(
                    &mut prog,
                    csrc.as_ptr(),
                    cname.as_ptr(),
                    0,
                    ptr::null(),
                    ptr::null(),
                )
            },
            "nvrtcCreateProgram",
        )?;

        // Compile to the device's virtual architecture.
        let arch = CString::new(format!("--gpu-architecture=compute_{}{}", self.cc_major, self.cc_minor))
            .unwrap();
        // NVRTC does not auto-include the toolkit headers (cuda_fp16.h,
        // cuda_bf16.h) — point it at <toolkit>/include.
        let cuda_root = std::env::var("CUDA_PATH")
            .or_else(|_| std::env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let inc = CString::new(format!("--include-path={cuda_root}/include")).unwrap();
        // Disable contraction of `a*b+c` into FMA by default: the CPU oracle
        // uses non-fused IEEE arithmetic, so default FMA fusion drifts in
        // accumulation-heavy kernels (conv, attention, recurrence). Matching
        // the oracle's rounding tightens bit-accuracy for the correctness
        // suite. For inference/perf runs FMA is strictly better (one fused op
        // per mul-add ⇒ half the FP issue + shorter dep chain in the hot GEMV/
        // gather/sdpa loops); opt in with `MT_FMAD=1`.
        // FMAD: default ON for inference (FMA fusion halves mul+add latency in
        // GEMV/SDPA dot-products). Opt out with MT_FMAD=0 for exact CPU-oracle
        // comparison. The correctness test (argmax 1234) passes with FMAD=true.
        let fmad_on = std::env::var("MT_FMAD").map(|v| v != "0" && v != "false").unwrap_or(true);
        let fmad = CString::new(if fmad_on { "--fmad=true" } else { "--fmad=false" }).unwrap();
        // MT_FAST_MATH=1: enable --use_fast_math (implies --fmad=true + fast
        // intrinsics: __expf, __sinf, etc.). Trades ~1-2 ULP precision for
        // ~2-4x faster transcendentals. Safe for inference softmax.
        let fast_math_on = std::env::var("MT_FAST_MATH").map(|v| v == "1" || v == "true").unwrap_or(false);
        let compile_res = if fast_math_on {
            let fast = CString::new("--use_fast_math").unwrap();
            let opts: [*const c_char; 3] = [arch.as_ptr(), inc.as_ptr(), fast.as_ptr()];
            unsafe { nvrtcCompileProgram(prog, opts.len() as _, opts.as_ptr()) }
        } else {
            let opts: [*const c_char; 3] = [arch.as_ptr(), inc.as_ptr(), fmad.as_ptr()];
            unsafe { nvrtcCompileProgram(prog, opts.len() as _, opts.as_ptr()) }
        };

        // Always fetch the log — it carries the actual compiler diagnostics.
        let log = unsafe {
            let mut log_size: usize = 0;
            nvrtcGetProgramLogSize(prog, &mut log_size);
            if log_size > 1 {
                let mut buf = vec![0u8; log_size];
                nvrtcGetProgramLog(prog, buf.as_mut_ptr() as *mut c_char);
                String::from_utf8_lossy(&buf[..log_size.saturating_sub(1)]).into_owned()
            } else {
                String::new()
            }
        };

        if compile_res != NVRTC_SUCCESS {
            unsafe { nvrtcDestroyProgram(&mut prog) };
            return Err(MetalTileError::Compilation(format!(
                "nvrtcCompileProgram failed: {}\n--- log ---\n{log}",
                unsafe { CStr::from_ptr(nvrtcGetErrorString(compile_res)).to_string_lossy() }
            )));
        }

        // Fetch PTX.
        let ptx = unsafe {
            let mut ptx_size: usize = 0;
            nvrtc_check(nvrtcGetPTXSize(prog, &mut ptx_size), "nvrtcGetPTXSize")?;
            let mut buf = vec![0u8; ptx_size];
            nvrtc_check(nvrtcGetPTX(prog, buf.as_mut_ptr() as *mut c_char), "nvrtcGetPTX")?;
            buf
        };
        unsafe { nvrtcDestroyProgram(&mut prog) };

        // Load module (driver JITs PTX → cubin for the live arch).
        let mut module: CUmodule = ptr::null_mut();
        cu_check(
            unsafe { cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        Ok(CudaModule { module })
    }

    /// Allocate `len` bytes of device memory.
    pub fn alloc(&self, len: usize) -> Result<DeviceBuffer<'_>, MetalTileError> {
        let mut ptr: CUdeviceptr = 0;
        if len == 0 {
            return Ok(DeviceBuffer { ptr: 0, len: 0, _dev: self });
        }
        cu_check(unsafe { cuMemAlloc_v2(&mut ptr, len) }, "cuMemAlloc")?;
        Ok(DeviceBuffer { ptr, len, _dev: self })
    }

    /// Allocate + upload host bytes in one shot.
    pub fn upload(&self, data: &[u8]) -> Result<DeviceBuffer<'_>, MetalTileError> {
        let buf = self.alloc(data.len())?;
        if !data.is_empty() {
            cu_check(
                unsafe {
                    cuMemcpyHtoD_v2(buf.ptr, data.as_ptr() as *const c_void, data.len())
                },
                "cuMemcpyHtoD",
            )?;
        }
        Ok(buf)
    }

    /// Copy device memory back into a host buffer.
    pub fn download(&self, buf: &DeviceBuffer, out: &mut [u8]) -> Result<(), MetalTileError> {
        let n = out.len().min(buf.len);
        if n == 0 {
            return Ok(());
        }
        cu_check(unsafe { cuStreamSynchronize(self.stream) }, "cuStreamSynchronize(download)")?;
        cu_check(
            unsafe { cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, buf.ptr, n) },
            "cuMemcpyDtoH",
        )?;
        self.reclaim_pinned();
        Ok(())
    }

    /// Allocate `len` bytes, returning the raw device pointer. The caller
    /// owns the allocation and must free it via [`free_raw`]. Unlike
    /// [`alloc`], this returns a plain pointer with no borrow of the device,
    /// so a higher layer (e.g. the ffai engine) can build persistent device
    /// tensors that outlive any single call — it just has to keep the
    /// `CudaDevice` (hence its context) alive while the pointers are live.
    ///
    /// [`free_raw`]: CudaDevice::free_raw
    pub fn alloc_raw(&self, len: usize) -> Result<CUdeviceptr, MetalTileError> {
        if len == 0 {
            return Ok(0);
        }
        // Reuse a same-size buffer from the pool if one is free.
        if let Some(ptr) = self.pool.lock().unwrap().get_mut(&len).and_then(|v| v.pop()) {
            return Ok(ptr);
        }
        let mut ptr: CUdeviceptr = 0;
        cu_check(unsafe { cuMemAlloc_v2(&mut ptr, len) }, "cuMemAlloc")?;
        Ok(ptr)
    }

    /// Free a pointer returned by [`alloc_raw`]. No-op on a null pointer.
    /// Eagerly releases to the driver (use [`free_raw_pooled`] on the hot path).
    pub fn free_raw(&self, ptr: CUdeviceptr) {
        if ptr != 0 {
            unsafe { cuMemFree_v2(ptr) };
        }
    }

    /// Return a `len`-byte allocation to the size-bucketed pool for reuse
    /// instead of releasing it to the driver — avoids a synchronous
    /// `cuMemFree`/`cuMemAlloc` round-trip on the next same-size request.
    pub fn free_raw_pooled(&self, ptr: CUdeviceptr, len: usize) {
        if ptr != 0 {
            self.pool.lock().unwrap().entry(len).or_default().push(ptr);
        }
    }

    /// Copy host bytes into an existing device allocation (host→device), ASYNC
    /// via a pinned staging buffer — enqueues on the (ordered) default stream
    /// without a host-blocking GPU drain. The pinned buffer is reclaimed on the
    /// next `synchronize`/`download` (by then the copy has completed).
    pub fn htod(&self, ptr: CUdeviceptr, bytes: &[u8]) -> Result<(), MetalTileError> {
        if bytes.is_empty() {
            return Ok(());
        }
        let len = bytes.len();
        // Large (one-time weight) uploads stay synchronous — pinning them would
        // balloon host pinned memory. Only small per-token activations go async.
        if len > 262_144 {
            return cu_check(
                unsafe { cuMemcpyHtoD_v2(ptr, bytes.as_ptr() as *const c_void, len) },
                "cuMemcpyHtoD",
            );
        }
        let pinned = self
            .pinned_free
            .lock()
            .unwrap()
            .get_mut(&len)
            .and_then(|v| v.pop())
            .unwrap_or_else(|| {
                let mut p: *mut c_void = ptr::null_mut();
                unsafe { cuMemAllocHost_v2(&mut p, len) };
                p as usize
            });
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), pinned as *mut u8, len) };
        cu_check(
            unsafe { cuMemcpyHtoDAsync_v2(ptr, pinned as *const c_void, len, self.stream) },
            "cuMemcpyHtoDAsync",
        )?;
        self.pinned_inflight.lock().unwrap().push((pinned, len));
        Ok(())
    }

    /// Reclaim in-flight pinned staging buffers after a sync point (their copies
    /// have completed) back to the free-list for reuse.
    fn reclaim_pinned(&self) {
        let mut inflight = self.pinned_inflight.lock().unwrap();
        if inflight.is_empty() {
            return;
        }
        let mut free = self.pinned_free.lock().unwrap();
        for (p, len) in inflight.drain(..) {
            free.entry(len).or_default().push(p);
        }
    }

    /// Copy device memory back to a host slice (device→host).
    pub fn dtoh(&self, ptr: CUdeviceptr, out: &mut [u8]) -> Result<(), MetalTileError> {
        if out.is_empty() {
            return Ok(());
        }
        // All kernels run on `self.stream` (non-blocking); a default-stream sync
        // copy would NOT order against it, so drain the stream first.
        cu_check(unsafe { cuStreamSynchronize(self.stream) }, "cuStreamSynchronize(dtoh)")?;
        cu_check(
            unsafe { cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, ptr, out.len()) },
            "cuMemcpyDtoH",
        )?;
        self.reclaim_pinned();
        Ok(())
    }

    /// Launch `func` over a 1-D grid. `args` are raw pointers to each
    /// kernel argument value, in signature order (CUDA `kernelParams`).
    pub fn launch_1d(
        &self,
        func: CudaFunction,
        grid_blocks: u32,
        block_threads: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), MetalTileError> {
        self.launch(func, [grid_blocks, 1, 1], [block_threads, 1, 1], 0, args)
    }

    /// Launch `func` over a 3-D grid (blocks × threads-per-block) with
    /// `shared_bytes` of dynamic shared memory. For >48KB the function must
    /// opt in via `cuFuncSetAttribute` first (GB10 allows up to ~99KB/block).
    pub fn launch(
        &self,
        func: CudaFunction,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), MetalTileError> {
        if shared_bytes > 0 {
            // Opt into the requested dynamic shared size (no-op effect if it
            // is below the default cap; required above 48KB).
            unsafe {
                cuFuncSetAttribute(
                    func.func,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_bytes as c_int,
                )
            };
        }
        cu_check(
            unsafe {
                cuLaunchKernel(
                    func.func,
                    grid[0], grid[1], grid[2],
                    block[0], block[1], block[2],
                    shared_bytes,
                    self.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )?;
        self.synchronize()
    }

    /// Like [`launch`] but does NOT synchronize after the launch — the kernel is
    /// enqueued async on the (ordered) default stream and the caller syncs only
    /// when it actually reads results back (`dtoh`/`synchronize`). The per-launch
    /// `cuCtxSynchronize` in `launch` serialized ~390 dispatches/decode-token and
    /// was the dominant decode overhead (~4ms/token of GPU-idle host stalls).
    pub fn launch_async(
        &self,
        func: CudaFunction,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), MetalTileError> {
        if shared_bytes > 0 {
            unsafe {
                cuFuncSetAttribute(func.func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes as c_int)
            };
        }
        cu_check(
            unsafe {
                cuLaunchKernel(
                    func.func,
                    grid[0], grid[1], grid[2],
                    block[0], block[1], block[2],
                    shared_bytes,
                    self.stream,
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )
    }

    /// Begin recording all subsequent stream work into a CUDA graph (phase-1
    /// megakernel). Caller must run a NO-host-sync (all-device) sequence, then
    /// `end_capture`. THREAD_LOCAL mode scopes capture to this thread's stream.
    pub fn begin_capture(&self) -> Result<(), MetalTileError> {
        self.capturing.store(true, std::sync::atomic::Ordering::SeqCst);
        cu_check(unsafe { cuStreamBeginCapture_v2(self.stream, CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) }, "cuStreamBeginCapture")
    }

    /// Finish capture → instantiate an executable graph. Replay with `graph_launch`.
    pub fn end_capture(&self) -> Result<CUgraphExec, MetalTileError> {
        let mut graph: CUgraph = ptr::null_mut();
        cu_check(unsafe { cuStreamEndCapture(self.stream, &mut graph) }, "cuStreamEndCapture")?;
        self.capturing.store(false, std::sync::atomic::Ordering::SeqCst);
        let mut exec: CUgraphExec = ptr::null_mut();
        cu_check(unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) }, "cuGraphInstantiate")?;
        unsafe { cuGraphDestroy(graph) };
        Ok(exec)
    }

    /// Replay a captured decode token: ONE host launch replaces ~390 — no
    /// per-kernel enqueue, no inter-kernel host bubbles. Syncs the stream after.
    pub fn graph_launch(&self, exec: CUgraphExec) -> Result<(), MetalTileError> {
        cu_check(unsafe { cuGraphLaunch(exec, self.stream) }, "cuGraphLaunch")?;
        self.synchronize()
    }

    /// Issue `n` sequential graph launches WITHOUT syncing between them, then
    /// sync once at the end. The GPU stream is FIFO-ordered so graphs execute
    /// sequentially despite the async enqueues — no data race on intermediate
    /// buffers. This eliminates the per-token host-GPU handoff overhead that
    /// `graph_launch` (sync-per-token) incurs, giving the maximum throughput
    /// for the captured graph. Use only for throughput benchmarking — state
    /// (KV cache, SSM state) is overwritten sequentially and not meaningful.
    pub fn graph_launch_batch(&self, exec: CUgraphExec, n: usize) -> Result<(), MetalTileError> {
        for _ in 0..n {
            cu_check(unsafe { cuGraphLaunch(exec, self.stream) }, "cuGraphLaunch(batch)")?;
        }
        self.synchronize()
    }

    /// A compiled+resident kernel: module, function, device buffers, and the
    /// marshalled scalar/pointer args, ready to launch repeatedly without
    /// re-compiling or re-uploading. Produced by [`CudaDevice::prepare`] and
    /// consumed by both [`CudaDevice::run_kernel`] (one launch + read-back)
    /// and [`CudaDevice::bench_kernel`] (timed launch loop).
    ///
    /// `dev_bufs` and the module own their device resources (freed on drop);
    /// `dev_ptrs` and `scalars` back the raw arg pointers, so neither vec is
    /// reallocated after [`Prepared::args`] hands out pointers into them.
    fn prepare<'d>(
        &'d self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        block: [u32; 3],
    ) -> Result<Prepared<'d>, MetalTileError> {
        // 1. IR → CUDA C++ → module.
        let cg = CudaGenerator::new();
        let src = cg.generate(kernel).map_err(|e| MetalTileError::Codegen(e))?;
        // MT_DUMP_CUDA_SRC=<path>: write generated CUDA C++ for every kernel.
        if let Ok(dir) = std::env::var("MT_DUMP_CUDA_SRC") {
            let path = format!("{}/{}.cu", dir, kernel.name);
            let _ = std::fs::write(&path, &src);
        }
        let module = self.compile(&src, &format!("{}.cu", kernel.name))?;
        let func = module.function(&kernel.name)?;
        // Dynamic shared-memory size for this launch geometry.
        let shared_bytes = cg.shared_bytes(kernel, block[0]) as u32;

        // 2. Allocate + upload each param buffer (in kernel.params order).
        //    Strided params are not yet supported by the emitter (it errors
        //    in generate() above), so every param here is Tensor/Scalar.
        // dev_bufs / dev_ptrs / out_meta are kept index-aligned. Strided
        // params add 2 extra companion buffers, so an output's buffer is NOT
        // at its kernel.params index — out_meta carries the (name,len) for
        // read-back so the alignment is by buffer, not by param.
        let mut dev_bufs: Vec<DeviceBuffer> = Vec::new();
        let mut dev_ptrs: Vec<CUdeviceptr> = Vec::new();
        let mut out_meta: Vec<Option<(String, usize)>> = Vec::new();
        for p in &kernel.params {
            let bytes = buffers.get(&p.name).ok_or_else(|| {
                MetalTileError::Dispatch(format!("missing buffer for param '{}'", p.name))
            })?;
            let buf = self.upload(bytes)?;
            dev_ptrs.push(buf.device_ptr());
            out_meta.push(if p.is_output { Some((p.name.clone(), bytes.len())) } else { None });
            dev_bufs.push(buf);

            // Strided params carry two companion buffers (shape, strides) in
            // signature order. The harness provides them by name; if absent,
            // synthesize a row-major layout from the static shape.
            if p.kind == metaltile_core::ir::ParamKind::Strided {
                for suffix in ["_shape", "_strides"] {
                    let key = format!("{}{}", p.name, suffix);
                    let meta = match buffers.get(&key) {
                        Some(b) => b.clone(),
                        None => synth_strided_meta(&p.shape, suffix == "_strides"),
                    };
                    let mb = self.upload(&meta)?;
                    dev_ptrs.push(mb.device_ptr());
                    out_meta.push(None);
                    dev_bufs.push(mb);
                }
            }
        }

        // 3. Scalar arg storage: constexprs (signature order) then the
        //    synthetic _n_elems for Elementwise.
        let mut scalars: Vec<Vec<u8>> = Vec::new();
        for ce in &kernel.constexprs {
            let name = ce.name.name();
            let bytes = buffers.get(name).ok_or_else(|| {
                MetalTileError::Dispatch(format!("missing constexpr '{name}'"))
            })?;
            scalars.push(bytes.clone());
        }
        if kernel.mode == metaltile_core::ir::KernelMode::Elementwise {
            // Bounds = element count of the first output param.
            let n_elems = kernel
                .params
                .iter()
                .position(|p| p.is_output)
                .and_then(|i| {
                    let p = &kernel.params[i];
                    buffers.get(&p.name).map(|b| (b.len() / p.dtype.size_bytes().max(1)) as u32)
                })
                .unwrap_or(0);
            scalars.push(n_elems.to_le_bytes().to_vec());
        }

        Ok(Prepared {
            _module: module,
            func,
            dev_bufs,
            dev_ptrs,
            scalars,
            out_meta,
            shared_bytes,
        })
    }

    /// End-to-end generic dispatch: generate CUDA for `kernel`, compile,
    /// allocate + upload every param (by name from `buffers`), pack kernel
    /// args in signature order, launch over (`grid`×`block`), and read back
    /// the output params. The CUDA analog of `Context::dispatch_with_grid`
    /// + `SingleDispatch`, used to run the registered kernel-test corpus on
    /// CUDA. `buffers` must contain every param's bytes (inputs AND
    /// pre-sized outputs) plus each constexpr's name→LE-bytes.
    pub fn run_kernel(
        &self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        grid: [u32; 3],
        block: [u32; 3],
    ) -> Result<BTreeMap<String, Vec<u8>>, MetalTileError> {
        let prep = self.prepare(kernel, buffers, block)?;

        // Launch (with dynamic shared memory).
        let mut args = prep.args();
        self.launch(prep.func, grid, block, prep.shared_bytes, &mut args)?;
        drop(args);

        // Read back outputs (aligned to dev_bufs, not kernel.params).
        let mut out = BTreeMap::new();
        for (buf, meta) in prep.dev_bufs.iter().zip(&prep.out_meta) {
            if let Some((name, len)) = meta {
                let mut host = vec![0u8; *len];
                self.download(buf, &mut host)?;
                out.insert(name.clone(), host);
            }
        }
        Ok(out)
    }

    /// Time `kernel` on the GPU: compile + upload once, run `warmup`
    /// untimed launches (NVRTC/JIT, caches, clocks settle), then `iters`
    /// launches each bracketed by CUDA events. Returns the per-iter GPU
    /// elapsed times in **microseconds** — feed to `BenchStats::from_samples`
    /// for min/median/p95. Event timing measures device wall-clock only, so
    /// host scheduling jitter does not pollute the samples.
    ///
    /// No read-back: throughput is data-independent, and skipping the DtoH
    /// copy keeps the timed region pure kernel execution. Outputs stay
    /// resident (overwritten each iter — fine, we only time).
    pub fn bench_kernel(
        &self,
        kernel: &Kernel,
        buffers: &BTreeMap<String, Vec<u8>>,
        grid: [u32; 3],
        block: [u32; 3],
        warmup: u32,
        iters: u32,
    ) -> Result<Vec<f64>, MetalTileError> {
        let prep = self.prepare(kernel, buffers, block)?;
        let mut args = prep.args();

        // Warmup — first launch pays JIT/cache/clock-ramp costs.
        for _ in 0..warmup {
            self.launch(prep.func, grid, block, prep.shared_bytes, &mut args)?;
        }

        // Two events bracket each timed launch.
        let (mut start, mut stop): (CUevent, CUevent) = (ptr::null_mut(), ptr::null_mut());
        cu_check(unsafe { cuEventCreate(&mut start, CU_EVENT_DEFAULT) }, "cuEventCreate(start)")?;
        cu_check(unsafe { cuEventCreate(&mut stop, CU_EVENT_DEFAULT) }, "cuEventCreate(stop)")?;

        // Closure owns its sample vec (returned on success) so no outer
        // borrow outlives it — lets us unconditionally destroy events after.
        let mut timed = || -> Result<Vec<f64>, MetalTileError> {
            let mut samples = Vec::with_capacity(iters as usize);
            for _ in 0..iters {
                if prep.shared_bytes > 0 {
                    unsafe {
                        cuFuncSetAttribute(
                            prep.func.func,
                            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            prep.shared_bytes as c_int,
                        )
                    };
                }
                cu_check(unsafe { cuEventRecord(start, ptr::null_mut()) }, "cuEventRecord(start)")?;
                cu_check(
                    unsafe {
                        cuLaunchKernel(
                            prep.func.func,
                            grid[0], grid[1], grid[2],
                            block[0], block[1], block[2],
                            prep.shared_bytes,
                            ptr::null_mut(),
                            args.as_mut_ptr(),
                            ptr::null_mut(),
                        )
                    },
                    "cuLaunchKernel",
                )?;
                cu_check(unsafe { cuEventRecord(stop, ptr::null_mut()) }, "cuEventRecord(stop)")?;
                cu_check(unsafe { cuEventSynchronize(stop) }, "cuEventSynchronize")?;
                let mut ms: f32 = 0.0;
                cu_check(
                    unsafe { cuEventElapsedTime(&mut ms, start, stop) },
                    "cuEventElapsedTime",
                )?;
                samples.push(ms as f64 * 1000.0); // ms → µs
            }
            Ok(samples)
        };
        let res = timed();

        // Always destroy events, even on error.
        unsafe {
            cuEventDestroy_v2(start);
            cuEventDestroy_v2(stop);
        }
        res
    }

    pub fn synchronize(&self) -> Result<(), MetalTileError> {
        cu_check(unsafe { cuStreamSynchronize(self.stream) }, "cuStreamSynchronize")?;
        self.reclaim_pinned();
        Ok(())
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        if let Ok(mut pool) = self.pool.lock() {
            for (_, ptrs) in pool.drain() {
                for p in ptrs {
                    unsafe { cuMemFree_v2(p) };
                }
            }
        }
        if !self.ctx.is_null() {
            unsafe { cuCtxDestroy_v2(self.ctx) };
        }
    }
}
