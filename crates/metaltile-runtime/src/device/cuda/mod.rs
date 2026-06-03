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

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::ptr;

use crate::error::MetalTileError;

use ffi::*;

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
        let opts: [*const c_char; 2] = [arch.as_ptr(), inc.as_ptr()];
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
        cu_check(
            unsafe {
                cuLaunchKernel(
                    func.func,
                    grid_blocks,
                    1,
                    1,
                    block_threads,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    args.as_mut_ptr(),
                    ptr::null_mut(),
                )
            },
            "cuLaunchKernel",
        )?;
        self.synchronize()
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
