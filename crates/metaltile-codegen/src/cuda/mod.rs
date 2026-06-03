//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! CUDA codegen backend (CUDA_BACKEND_SPEC §4.2 / SCOPE Phases 1–4).
//!
//! Lowers the backend-neutral algorithm IR to CUDA C++ (compiled at
//! runtime via NVRTC, or offline via `nvcc`). Shares the IR, the
//! `#[kernel]` macro, and the entire `quant::{codec,format}` layer with
//! the Metal path — only the leaf text differs, captured in
//! [`TargetProfile`](crate::backend::TargetProfile).
//!
//! **Status: Phase 0 — seam only.** `CudaGenerator` carries a real
//! `TargetProfile` (the §4.2 op-mapping as data) and implements
//! `CodegenBackend`, but `generate` is not wired to an op-walker yet.
//! Phase 1 brings the smoke kernel (`copy`/`binary`) online end-to-end
//! (NVRTC compile + launch on the GX10 / sm_121).
//!
//! Strategy note (SCOPE §6 — decided): do NOT fork `msl/emit_block.rs`
//! (61 KB) up front. Stand up a minimal CUDA op-walker covering the
//! Phase-1 subset (Const/BinOp/UnaryOp/Cast/Load/Store/ProgramId,
//! Elementwise mode), grow it per phase, and only extract a shared
//! backend-neutral walker once both emitters exist and the common
//! structure is empirically clear — premature extraction risks the wrong
//! abstraction over a hot path.

use metaltile_core::ir::Kernel;

use crate::{
    Result,
    backend::{CodegenBackend, Target, TargetProfile},
    error::Error,
};

/// CUDA C++ generator. Mirror of `msl::MslGenerator` for the NVIDIA target.
#[derive(Debug, Clone)]
pub struct CudaGenerator {
    profile: TargetProfile,
}

impl Default for CudaGenerator {
    fn default() -> Self { Self::new() }
}

impl CudaGenerator {
    pub fn new() -> Self {
        CudaGenerator { profile: TargetProfile::cuda() }
    }

    /// Build a generator pinned to a specific profile (e.g. a Blackwell
    /// profile with `MmaStrategy::Tcgen05` for Phase 4).
    pub fn with_profile(profile: TargetProfile) -> Self {
        debug_assert_eq!(profile.target, Target::Cuda);
        CudaGenerator { profile }
    }
}

impl CodegenBackend for CudaGenerator {
    fn target(&self) -> Target { Target::Cuda }

    fn profile(&self) -> &TargetProfile { &self.profile }

    fn generate(&self, kernel: &Kernel) -> Result<String> {
        // Phase 1 lands the op-walker here.
        Err(Error::UnsupportedOp(format!(
            "cuda codegen not yet implemented (Phase 1 WIP); kernel `{}`",
            kernel.name
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_generator_reports_cuda_target_and_profile() {
        let g = CudaGenerator::new();
        assert_eq!(g.target(), Target::Cuda);
        assert_eq!(g.profile().shared_mem_kw, "__shared__");
        assert_eq!(g.profile().lane_width, 32);
    }

    #[test]
    fn generate_is_honest_about_being_unimplemented() {
        let g = CudaGenerator::new();
        let k = Kernel::new("mt_copy_f32");
        let err = g.generate(&k).unwrap_err();
        assert!(err.to_string().contains("Phase 1"));
    }
}
