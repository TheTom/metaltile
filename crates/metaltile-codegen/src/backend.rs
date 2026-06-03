//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Backend abstraction seam (CUDA_BACKEND_SPEC §4.1 / SCOPE Phase 0).
//!
//! Today MetalTile lowers IR → MSL only. This module introduces the
//! *codegen* half of the backend seam so the same IR can target a second
//! backend (CUDA) without forking the IR, the `#[kernel]` macro, or the
//! `quant::{codec,format}` layer.
//!
//! - [`Target`] selects the backend (`tile build --target {metal,cuda}`).
//! - [`CodegenBackend`] is the per-backend "IR → target-language text"
//!   trait. `MslGenerator` is the `Metal` impl (see `msl/mod.rs`); the
//!   CUDA impl lives in `cuda/mod.rs`.
//! - [`TargetProfile`] captures the *leaf* differences between backends —
//!   lane width, shared-memory keyword, math-intrinsic names, MMA strategy
//!   — so the op-walker structure can eventually be shared rather than
//!   duplicated. This is the encoding of spec §4.2's DSL→op mapping table.
//!
//! Phase 0 is intentionally non-breaking: Metal keeps working exactly as
//! before; the CUDA impl is a stub that carries a real [`TargetProfile`]
//! but does not yet emit (that is Phase 1).

use metaltile_core::ir::{Kernel, UnaryOpKind};

use crate::Result;

/// Which GPU backend to lower the IR to. Default stays `Metal` — the
/// zero-config macOS path. `cuda` is opt-in via `--target cuda`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Target {
    #[default]
    Metal,
    Cuda,
}

impl Target {
    pub fn as_str(self) -> &'static str {
        match self {
            Target::Metal => "metal",
            Target::Cuda => "cuda",
        }
    }

    /// File extension for emitted source (`emit.rs` writes `<name>.<ext>`).
    pub fn source_ext(self) -> &'static str {
        match self {
            Target::Metal => "metal",
            Target::Cuda => "cu",
        }
    }
}

/// Per-backend "IR → target-language text" generator. The MSL generator
/// (`msl::MslGenerator`) and the CUDA generator (`cuda::CudaGenerator`)
/// both implement this so `emit.rs`, the CLI, and the codegen-consistency
/// tests can be parameterized over the target.
pub trait CodegenBackend {
    /// Which target this backend emits for.
    fn target(&self) -> Target;

    /// The leaf-difference profile (intrinsics, lane width, MMA strategy).
    fn profile(&self) -> &TargetProfile;

    /// Lower `kernel`'s IR to a complete source string for this target.
    fn generate(&self, kernel: &Kernel) -> Result<String>;
}

/// MMA lowering strategy for cooperative matmul ops
/// (`SimdgroupMatMul`, `CoopTile*`). Metal uses fixed 8×8 simdgroup
/// matrices; CUDA has several escalating options (spec §4.2 / §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmaStrategy {
    /// Metal `simdgroup_matrix` 8×8 (today).
    Simdgroup8x8,
    /// CUDA `wmma` fragments, 16×16×16 — portable Ampere+. Phase 3.
    Wmma16x16x16,
    /// CUTLASS collective-MMA shim — cooperative/MPP/NAX reimpl. Phase 5.
    Cutlass,
    /// Blackwell `tcgen05` scaled tensor-core MMA, sm_100+. Phase 4.
    Tcgen05,
}

/// The leaf textual/structural differences between backends. Everything
/// the op-walker needs to differ on lives here, so the walker itself can
/// stay backend-agnostic. Encodes the spec §4.2 mapping table as data.
#[derive(Debug, Clone)]
pub struct TargetProfile {
    pub target: Target,
    /// SIMD/warp lane width. **32 on both Metal simdgroup and CUDA warp** —
    /// the structural match that lets reduction/shuffle kernels port.
    pub lane_width: u32,
    /// Threadgroup/shared-memory address-space keyword.
    /// Metal: `threadgroup`; CUDA: `__shared__`.
    pub shared_mem_kw: &'static str,
    /// Barrier intrinsic across the block/threadgroup.
    /// Metal: `threadgroup_barrier(mem_flags::mem_threadgroup)`; CUDA: `__syncthreads()`.
    pub block_barrier: &'static str,
    /// Warp/simd barrier. Metal: `simdgroup_barrier(...)`; CUDA: `__syncwarp()`.
    pub warp_barrier: &'static str,
    pub mma: MmaStrategy,
}

impl TargetProfile {
    pub fn metal() -> Self {
        TargetProfile {
            target: Target::Metal,
            lane_width: 32,
            shared_mem_kw: "threadgroup",
            block_barrier: "threadgroup_barrier(mem_flags::mem_threadgroup)",
            warp_barrier: "simdgroup_barrier(mem_flags::mem_threadgroup)",
            mma: MmaStrategy::Simdgroup8x8,
        }
    }

    /// Default CUDA profile (software-decode path; portable Ampere+).
    /// Phase 4 swaps `mma` to `Tcgen05` when compute capability ≥ sm_100.
    pub fn cuda() -> Self {
        TargetProfile {
            target: Target::Cuda,
            lane_width: 32,
            shared_mem_kw: "__shared__",
            block_barrier: "__syncthreads()",
            warp_barrier: "__syncwarp()",
            mma: MmaStrategy::Wmma16x16x16,
        }
    }

    /// `program_id(axis)` source for this target.
    /// Metal: threadgroup-position-in-grid; CUDA: `blockIdx.{x,y,z}`.
    pub fn block_idx(&self, axis: u32) -> String {
        let comp = ["x", "y", "z"].get(axis as usize).copied().unwrap_or("x");
        match self.target {
            // Metal injects tgid via attributes; the codegen references `tgid_<c>`.
            Target::Metal => format!("tgid_{comp}"),
            Target::Cuda => format!("blockIdx.{comp}"),
        }
    }

    /// Math-intrinsic name for a unary op. The decode intrinsics
    /// (`DecodeE2m1`/`E4m3`/`E5m2`/`Int8`) are NOT here — they lower to
    /// preamble `__device__`/inline helpers that port verbatim from
    /// `msl/features.rs` (pure arithmetic; spec §4.2 / §4.3).
    pub fn unary_intrinsic(&self, op: UnaryOpKind) -> &'static str {
        use UnaryOpKind::*;
        match self.target {
            Target::Metal => match op {
                Exp => "exp", Exp2 => "exp2", Expm1 => "expm1",
                Log => "log", Log2 => "log2", Log10 => "log10",
                Sqrt => "sqrt", Rsqrt => "rsqrt", Recip => "1.0/",
                Abs => "abs", Neg => "-", Ceil => "ceil", Floor => "floor",
                Round => "round", Trunc => "trunc", Sign => "sign",
                Sin => "sin", Cos => "cos", Tan => "tan",
                Asin => "asin", Acos => "acos", Atan => "atan",
                Sinh => "sinh", Cosh => "cosh",
                Asinh => "asinh", Acosh => "acosh", Atanh => "atanh",
                Erf => "erf", ErfInv => "erfinv",
                DecodeE2m1 | DecodeE4m3 | DecodeE5m2 | DecodeInt8 => "/*decode-helper*/",
            },
            // CUDA fast-math device intrinsics (single-precision `f` suffix).
            // bf16/half consumers cast to f32 around these (spec §6 arch gating).
            Target::Cuda => match op {
                Exp => "__expf", Exp2 => "exp2f", Expm1 => "expm1f",
                Log => "__logf", Log2 => "log2f", Log10 => "log10f",
                Sqrt => "sqrtf", Rsqrt => "rsqrtf", Recip => "__frcp_rn",
                Abs => "fabsf", Neg => "-", Ceil => "ceilf", Floor => "floorf",
                Round => "roundf", Trunc => "truncf", Sign => "copysignf",
                Sin => "__sinf", Cos => "__cosf", Tan => "tanf",
                Asin => "asinf", Acos => "acosf", Atan => "atanf",
                Sinh => "sinhf", Cosh => "coshf",
                Asinh => "asinhf", Acosh => "acoshf", Atanh => "atanhf",
                Erf => "erff", ErfInv => "erfinvf",
                DecodeE2m1 | DecodeE4m3 | DecodeE5m2 | DecodeInt8 => "/*decode-helper*/",
            },
        }
    }
}

// ─── Metal impl ──────────────────────────────────────────────────────
// `MslGenerator` is the `Metal` backend. The profile is a process-wide
// constant, so it lives in a `OnceLock` rather than on the struct (keeps
// `MslGenerator::new(config)` non-breaking).

impl CodegenBackend for crate::msl::MslGenerator {
    fn target(&self) -> Target { Target::Metal }

    fn profile(&self) -> &TargetProfile {
        static METAL: std::sync::OnceLock<TargetProfile> = std::sync::OnceLock::new();
        METAL.get_or_init(TargetProfile::metal)
    }

    fn generate(&self, kernel: &Kernel) -> Result<String> {
        // Delegate to the existing inherent method (unchanged behavior).
        crate::msl::MslGenerator::generate(self, kernel)
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::KernelMode;

    use super::*;

    #[test]
    fn msl_generator_impls_codegen_backend_as_metal() {
        // The Metal generator is the `Metal` CodegenBackend impl.
        let g = crate::generator_for_mode(KernelMode::Elementwise, None);
        let backend: &dyn CodegenBackend = &g;
        assert_eq!(backend.target(), Target::Metal);
        assert_eq!(backend.profile().shared_mem_kw, "threadgroup");
    }

    #[test]
    fn lane_widths_match_the_lucky_32() {
        // The whole reduction/shuffle port hinges on this equality.
        assert_eq!(TargetProfile::metal().lane_width, TargetProfile::cuda().lane_width);
        assert_eq!(TargetProfile::cuda().lane_width, 32);
    }

    #[test]
    fn cuda_profile_uses_cuda_idioms() {
        let p = TargetProfile::cuda();
        assert_eq!(p.shared_mem_kw, "__shared__");
        assert_eq!(p.block_barrier, "__syncthreads()");
        assert_eq!(p.block_idx(0), "blockIdx.x");
        assert_eq!(p.block_idx(2), "blockIdx.z");
        assert_eq!(p.unary_intrinsic(UnaryOpKind::Rsqrt), "rsqrtf");
        assert_eq!(p.unary_intrinsic(UnaryOpKind::Exp2), "exp2f");
        assert_eq!(p.mma, MmaStrategy::Wmma16x16x16);
    }

    #[test]
    fn metal_profile_unchanged() {
        let p = TargetProfile::metal();
        assert_eq!(p.shared_mem_kw, "threadgroup");
        assert_eq!(p.unary_intrinsic(UnaryOpKind::Rsqrt), "rsqrt");
        assert_eq!(p.mma, MmaStrategy::Simdgroup8x8);
    }

    #[test]
    fn source_ext_per_target() {
        assert_eq!(Target::Metal.source_ext(), "metal");
        assert_eq!(Target::Cuda.source_ext(), "cu");
    }
}
