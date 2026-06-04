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

use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::ptr;

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

/// CUDA device + context. NVIDIA analog of `MetalDevice`.
pub struct CudaDevice {
    ctx: CUcontext,
    cc_major: i32,
    cc_minor: i32,
}

// The context is current on this struct's lifetime; we keep it single-
// device, single-context (Phase 1). Send is sound because we never share
// the raw pointers across threads concurrently in the smoke path.
unsafe impl Send for CudaDevice {}

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
            Ok(Some(CudaDevice { ctx, cc_major: major, cc_minor: minor }))
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
        // Disable contraction of `a*b+c` into FMA: the CPU oracle uses
        // non-fused IEEE arithmetic, so default FMA fusion drifts in
        // accumulation-heavy kernels (conv, attention, recurrence). Matching
        // the oracle's rounding tightens bit-accuracy.
        let fmad = CString::new("--fmad=false").unwrap();
        let opts: [*const c_char; 3] = [arch.as_ptr(), inc.as_ptr(), fmad.as_ptr()];
        let compile_res = unsafe { nvrtcCompileProgram(prog, opts.len() as _, opts.as_ptr()) };

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
        cu_check(
            unsafe { cuMemcpyDtoH_v2(out.as_mut_ptr() as *mut c_void, buf.ptr, n) },
            "cuMemcpyDtoH",
        )
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
                    ptr::null_mut(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )?;
        self.synchronize()
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
        // 1. IR → CUDA C++ → module.
        let cg = CudaGenerator::new();
        let src = cg.generate(kernel).map_err(|e| MetalTileError::Codegen(e))?;
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

        // 4. Build kernelParams: param device-ptrs, then scalar values.
        //    Pointers are taken after both vecs are fully built (no realloc).
        let mut args: Vec<*mut c_void> = Vec::with_capacity(dev_ptrs.len() + scalars.len());
        for p in &dev_ptrs {
            args.push(p as *const CUdeviceptr as *mut c_void);
        }
        for s in &scalars {
            args.push(s.as_ptr() as *mut c_void);
        }

        // 5. Launch (with dynamic shared memory).
        self.launch(func, grid, block, shared_bytes, &mut args)?;

        // 6. Read back outputs (aligned to dev_bufs, not kernel.params).
        let mut out = BTreeMap::new();
        for (buf, meta) in dev_bufs.iter().zip(&out_meta) {
            if let Some((name, len)) = meta {
                let mut host = vec![0u8; *len];
                self.download(buf, &mut host)?;
                out.insert(name.clone(), host);
            }
        }
        Ok(out)
    }

    pub fn synchronize(&self) -> Result<(), MetalTileError> {
        cu_check(unsafe { cuCtxSynchronize() }, "cuCtxSynchronize")
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { cuCtxDestroy_v2(self.ctx) };
        }
    }
}
