//! Steel tiled GEMM — metal/steel/gemm/kernels/steel_gemm_fused.metal
//!
//! High-performance tiled matrix multiply via simdgroup matrix ops:
//!   steel_gemm_fused_{nn|nt|tn|tt}_{dtype}  — A×B with transpose variants
//!   Block shapes: 64×64×16, 64×32×32, 32×64×16, 32×32×16, 64×32×8
//!   Dtypes: float16, bfloat16, float32, complex64
//!
//! NOT YET IMPLEMENTED in #[kernel] DSL:
//!   Requires simdgroup matrix multiply-accumulate primitives
//!   (`simdgroup_{float,half,bfloat}8x8`, `simdgroup_multiply_accumulate`),
//!   multi-level tiling (block → warp → simdgroup), shared-memory A/B tile
//!   staging, and register-file accumulation. The DSL `Op::Dot` currently
//!   runs scalar MACs; `use_simd_matrix` exists but defaults false with no
//!   chip detection. Requires M3+ (Apple9 GPU family).
