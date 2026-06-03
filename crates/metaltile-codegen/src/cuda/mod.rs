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
//! **Status: Phase 1 — elementwise smoke subset.** `generate` emits CUDA
//! C++ for `KernelMode::Elementwise` kernels using the ops
//! `ProgramId / Const / Load / Store / BinOp / UnaryOp / Cast / Fma`.
//! That covers the `copy` / `vector_add` smoke kernels end-to-end
//! (NVRTC compile + launch on the GX10 / sm_121). Reductions,
//! simdgroup/MMA, coop_tile, and `InlineMsl` return `UnsupportedOp`
//! pending Phases 2–5.
//!
//! Strategy note (SCOPE §6 — decided): do NOT fork `msl/emit_block.rs`
//! (61 KB) up front. This is a fresh, minimal op-walker for the Phase-1
//! subset; it grows per phase. A shared backend-neutral walker is only
//! extracted once both emitters exist and the common structure is
//! empirically clear — premature extraction risks the wrong abstraction
//! over a hot path.
//!
//! Phase-1 deliberate simplifications (tracked for Phase 2):
//! - No optimization passes run yet (the pipeline is backend-neutral but
//!   can introduce `FusedElementwise`/`Fma` ops the smoke walker doesn't
//!   handle). Smoke kernels emit correctly from raw IR.
//! - A trailing `unsigned int _n_elems` param + an `if (gtid >= _n_elems)
//!   return;` guard replace Metal's non-uniform-threadgroup bounds model.

use std::fmt::Write as _;

use metaltile_core::{
    dtype::DType,
    ir::{BinOpKind, Block, IndexExpr, Kernel, KernelMode, Op, Param, ParamKind, UnaryOpKind, ValueId},
};

use crate::{
    Result,
    backend::{CodegenBackend, Target, TargetProfile},
    error::Error,
};

/// Name of the synthetic element-count param appended to every emitted
/// CUDA kernel (drives the bounds guard). Chosen to not collide with DSL
/// param names.
pub const N_ELEMS_PARAM: &str = "_n_elems";

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

    // ── SSA value naming (mirrors msl helpers::vname) ──────────────────
    fn vname(&self, vid: Option<ValueId>, block: &Block) -> String {
        match vid {
            Some(v) => match block.names.get(&v) {
                Some(hint) => format!("v_{hint}"),
                None => format!("v{}", v.as_u32()),
            },
            None => "/*<no-value>*/".to_string(),
        }
    }

    // ── Flat index expression for Load/Store (Phase-1: single index) ───
    fn emit_idx(&self, indices: &[IndexExpr], block: &Block) -> Result<String> {
        match indices {
            [] => Ok(String::new()),
            [one] => Ok(self.idx_term(one, block)),
            // Multi-dim row-major flatten is a Phase-2 concern (needs the
            // shape-stride model the MSL path has). Smoke kernels are 1-D.
            _ => Err(Error::UnsupportedOp(
                "cuda Phase 1: multi-dimensional index (use 1-D elementwise)".into(),
            )),
        }
    }

    fn idx_term(&self, ix: &IndexExpr, block: &Block) -> String {
        match ix {
            IndexExpr::Value(v) => self.vname(Some(*v), block),
            IndexExpr::Const(c) => c.to_string(),
            IndexExpr::Range(v, off) => format!("({} + {off})", self.vname(Some(*v), block)),
        }
    }

    // ── Kernel signature ───────────────────────────────────────────────
    fn emit_signature(&self, kernel: &Kernel, out: &mut String) -> Result<()> {
        writeln!(out, "extern \"C\" __global__ void {}(", kernel.name).ok();
        let mut args: Vec<String> = Vec::new();
        for p in &kernel.params {
            args.push(self.emit_param(p)?);
        }
        for ce in &kernel.constexprs {
            // Scalars are passed by value as kernel arguments.
            args.push(format!("    {} {}", cuda_type_name(ce.dtype), ce.name.name()));
        }
        // Synthetic bounds param.
        args.push(format!("    unsigned int {N_ELEMS_PARAM}"));
        writeln!(out, "{}", args.join(",\n")).ok();
        writeln!(out, ") {{").ok();
        Ok(())
    }

    fn emit_param(&self, p: &Param) -> Result<String> {
        match p.kind {
            ParamKind::Tensor => {
                let ty = cuda_type_name(p.dtype);
                Ok(if p.is_output {
                    format!("    {ty}* {}", p.name)
                } else {
                    format!("    const {ty}* {}", p.name)
                })
            }
            ParamKind::Scalar => Ok(format!("    {} {}", cuda_type_name(p.dtype), p.name)),
            ParamKind::Strided => Err(Error::UnsupportedOp(
                "cuda Phase 1: Strided params (Phase 2 — shape/stride buffers)".into(),
            )),
        }
    }

    // ── Body ───────────────────────────────────────────────────────────
    fn emit_body(&self, kernel: &Kernel, out: &mut String) -> Result<()> {
        if kernel.mode != KernelMode::Elementwise {
            return Err(Error::UnsupportedOp(format!(
                "cuda Phase 1: only KernelMode::Elementwise (got {:?})",
                kernel.mode
            )));
        }
        // Global linear thread id + bounds guard.
        writeln!(
            out,
            "    const unsigned int _gtid = blockIdx.x * blockDim.x + threadIdx.x;"
        )
        .ok();
        writeln!(out, "    if (_gtid >= {N_ELEMS_PARAM}) return;").ok();

        let block = &kernel.body;
        for (i, op) in block.ops.iter().enumerate() {
            let vid = block.results.get(i).and_then(|x| *x);
            self.emit_op(op, vid, block, kernel, out)?;
        }
        writeln!(out, "}}").ok();
        Ok(())
    }

    fn emit_op(
        &self,
        op: &Op,
        vid: Option<ValueId>,
        block: &Block,
        _kernel: &Kernel,
        out: &mut String,
    ) -> Result<()> {
        let pad = "    ";
        match op {
            Op::ProgramId { axis } => {
                let v = self.vname(vid, block);
                match axis {
                    0 => writeln!(out, "{pad}unsigned int {v} = _gtid;").ok(),
                    _ => writeln!(out, "{pad}unsigned int {v} = 0;").ok(),
                };
            }
            Op::Const { value } => {
                let v = self.vname(vid, block);
                if *value >= 0 {
                    writeln!(out, "{pad}unsigned int {v} = {value}u;").ok();
                } else {
                    writeln!(out, "{pad}int {v} = {value};").ok();
                }
            }
            Op::Load { src, indices, .. } => {
                let v = self.vname(vid, block);
                if indices.is_empty() {
                    writeln!(out, "{pad}auto {v} = {src};").ok();
                } else {
                    let idx = self.emit_idx(indices, block)?;
                    writeln!(out, "{pad}auto {v} = {src}[{idx}];").ok();
                }
            }
            Op::Store { dst, indices, value, .. } => {
                let val = self.vname(Some(*value), block);
                let idx = self.emit_idx(indices, block)?;
                writeln!(out, "{pad}{dst}[{idx}] = {val};").ok();
            }
            Op::BinOp { op: bop, lhs, rhs } => {
                let v = self.vname(vid, block);
                let l = self.vname(Some(*lhs), block);
                let r = self.vname(Some(*rhs), block);
                writeln!(out, "{pad}auto {v} = {};", cuda_binop(*bop, &l, &r)).ok();
            }
            Op::Fma { a, b, c } => {
                let v = self.vname(vid, block);
                let av = self.vname(Some(*a), block);
                let bv = self.vname(Some(*b), block);
                let cv = self.vname(Some(*c), block);
                writeln!(out, "{pad}auto {v} = fmaf({av}, {bv}, {cv});").ok();
            }
            Op::UnaryOp { op: uop, value } => {
                let v = self.vname(vid, block);
                let rv = self.vname(Some(*value), block);
                writeln!(out, "{pad}auto {v} = {};", self.cuda_unary(*uop, &rv)).ok();
            }
            Op::Cast { value, dtype } => {
                let v = self.vname(vid, block);
                let rv = self.vname(Some(*value), block);
                let ty = cuda_type_name(*dtype);
                // CUDA C-style cast; source type comes from `auto` inference.
                writeln!(out, "{pad}{ty} {v} = ({ty})({rv});").ok();
            }
            other => {
                return Err(Error::UnsupportedOp(format!(
                    "cuda Phase 1: op {} not supported yet",
                    op_name(other)
                )));
            }
        }
        Ok(())
    }

    fn cuda_unary(&self, op: UnaryOpKind, arg: &str) -> String {
        use UnaryOpKind::*;
        match op {
            Neg => format!("(-{arg})"),
            Recip => format!("(1.0f / {arg})"),
            // Decode helpers port from msl/features.rs as __device__ fns (Phase 2/3).
            DecodeE2m1 | DecodeE4m3 | DecodeE5m2 | DecodeInt8 => {
                format!("/*decode TODO*/ {arg}")
            }
            _ => format!("{}({arg})", self.profile.unary_intrinsic(op)),
        }
    }
}

impl CodegenBackend for CudaGenerator {
    fn target(&self) -> Target { Target::Cuda }

    fn profile(&self) -> &TargetProfile { &self.profile }

    fn generate(&self, kernel: &Kernel) -> Result<String> {
        let mut out = String::new();
        out.push_str(
            "// Generated by MetalTile (CUDA backend). DO NOT EDIT.\n\
             #include <cuda_fp16.h>\n\
             #include <cuda_bf16.h>\n\n",
        );
        self.emit_signature(kernel, &mut out)?;
        self.emit_body(kernel, &mut out)?;
        Ok(out)
    }
}

/// DType → CUDA C++ scalar type name.
pub fn cuda_type_name(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "float",
        DType::F16 => "__half",
        DType::BF16 => "__nv_bfloat16",
        DType::I32 => "int",
        DType::U32 => "unsigned int",
        DType::I8 => "signed char",
        DType::U8 => "unsigned char",
        DType::I4 => "signed char", // packed; sub-byte handled in decode helpers
        DType::U16 => "unsigned short",
        DType::I64 => "long long",
        DType::U64 => "unsigned long long",
        DType::Bool => "bool",
    }
}

/// BinOp → CUDA expression (infix for arithmetic/compare/bitwise,
/// function-call for the few that need it). Mirrors `BinOpKind::msl_symbol`
/// semantics with CUDA intrinsic names.
fn cuda_binop(op: BinOpKind, l: &str, r: &str) -> String {
    use BinOpKind::*;
    match op {
        Add => format!("{l} + {r}"),
        Sub => format!("{l} - {r}"),
        Mul => format!("{l} * {r}"),
        Div => format!("{l} / {r}"),
        Max => format!("max({l}, {r})"),
        Min => format!("min({l}, {r})"),
        Pow => format!("powf({l}, {r})"),
        ATan2 => format!("atan2f({l}, {r})"),
        Rem => format!("fmodf({l}, {r})"),
        Mod => format!("({l} % {r})"),
        And => format!("({l} && {r})"),
        Or => format!("({l} || {r})"),
        Xor => format!("((bool){l} != (bool){r})"),
        BitAnd => format!("({l} & {r})"),
        BitOr => format!("({l} | {r})"),
        BitXor => format!("({l} ^ {r})"),
        Shl => format!("({l} << {r})"),
        Shr => format!("({l} >> {r})"),
        CmpLt => format!("({l} < {r})"),
        CmpGt => format!("({l} > {r})"),
        CmpLe => format!("({l} <= {r})"),
        CmpGe => format!("({l} >= {r})"),
        CmpEq => format!("({l} == {r})"),
        CmpNe => format!("({l} != {r})"),
    }
}

fn op_name(op: &Op) -> &'static str {
    match op {
        Op::Reduce { .. } => "Reduce",
        Op::StrideReduce { .. } => "StrideReduce",
        Op::SimdgroupMatMul { .. } => "SimdgroupMatMul",
        Op::InlineMsl { .. } => "InlineMsl",
        Op::Gather { .. } => "Gather",
        Op::Scatter { .. } => "Scatter",
        Op::Loop { .. } => "Loop",
        Op::If { .. } => "If",
        _ => "<unsupported>",
    }
}

#[cfg(test)]
mod tests {
    use metaltile_core::ir::{BinOpKind, IndexExpr, Op};
    use metaltile_core::shape::Shape;

    use super::*;

    fn vector_add_ir() -> Kernel {
        // Mirrors the codegen MSL test's `vector_add`: out[i] = a[i] + b[i].
        let mut k = Kernel::new("vector_add");
        for (name, is_out) in [("a", false), ("b", false), ("c", true)] {
            k.params.push(Param {
                name: name.into(),
                dtype: DType::F32,
                shape: Shape::scalar(),
                is_output: is_out,
                kind: ParamKind::Tensor,
            });
        }
        k.body.push_op(Op::ProgramId { axis: 0 }, ValueId::new(0));
        k.body.name_value(ValueId::new(0), "idx");
        k.body.push_op(
            Op::Load { src: "a".into(), indices: vec![IndexExpr::Value(ValueId::new(0))], mask: None, other: None },
            ValueId::new(1),
        );
        k.body.name_value(ValueId::new(1), "x");
        k.body.push_op(
            Op::Load { src: "b".into(), indices: vec![IndexExpr::Value(ValueId::new(0))], mask: None, other: None },
            ValueId::new(2),
        );
        k.body.name_value(ValueId::new(2), "y");
        k.body.push_op(
            Op::BinOp { op: BinOpKind::Add, lhs: ValueId::new(1), rhs: ValueId::new(2) },
            ValueId::new(3),
        );
        k.body.name_value(ValueId::new(3), "sum");
        k.body.push_op_no_result(Op::Store {
            dst: "c".into(),
            indices: vec![IndexExpr::Value(ValueId::new(0))],
            value: ValueId::new(3),
            mask: None,
        });
        k
    }

    #[test]
    fn emits_vector_add_cuda() {
        let g = CudaGenerator::new();
        let src = g.generate(&vector_add_ir()).unwrap();
        // Signature: pointers + synthetic bounds param.
        assert!(src.contains("extern \"C\" __global__ void vector_add("));
        assert!(src.contains("const float* a"));
        assert!(src.contains("const float* b"));
        assert!(src.contains("float* c"));
        assert!(src.contains(&format!("unsigned int {N_ELEMS_PARAM}")));
        // Body: global tid + guard + the add.
        assert!(src.contains("blockIdx.x * blockDim.x + threadIdx.x"));
        assert!(src.contains(&format!("if (_gtid >= {N_ELEMS_PARAM}) return;")));
        assert!(src.contains("unsigned int v_idx = _gtid;"));
        assert!(src.contains("auto v_x = a[v_idx];"));
        assert!(src.contains("auto v_y = b[v_idx];"));
        assert!(src.contains("auto v_sum = v_x + v_y;"));
        assert!(src.contains("c[v_idx] = v_sum;"));
    }

    #[test]
    fn rejects_non_elementwise() {
        let mut k = vector_add_ir();
        k.mode = KernelMode::Reduction;
        assert!(CudaGenerator::new().generate(&k).is_err());
    }

    #[test]
    fn cuda_generator_reports_cuda_target_and_profile() {
        let g = CudaGenerator::new();
        assert_eq!(g.target(), Target::Cuda);
        assert_eq!(g.profile().shared_mem_kw, "__shared__");
        assert_eq!(g.profile().lane_width, 32);
    }
}
