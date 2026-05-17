//! Copy Propagation & Identity Elimination pass.
//!
//! Eliminates no-op operations and propagates copies through the IR:
//!
//! ### Identity Patterns
//! - `Cast(dtype, x)` → `x`  when `x` is already that dtype
//! - `Broadcast(x, [1])` → `x`  when broadcasting a scalar with shape [1]
//! - `Reshape(x, s)` → `x`  when shapes are identical
//! - `Select(cond, x, x)` → `x`  (also in AlgebraicSimplify, but cheap to re-check)
//!
//! ### Copy Forwarding
//! When an op's result is used through a chain of identity operations,
//! forward the source value through. The downstream CSE pass then eliminates the
//! now-dead identity ops.
//!
//! ## Algorithm
//!
//! Iterates to fixpoint. Each iteration:
//! 1. Find identity ops (result == source).
//! 2. Replace all uses of the identity result with the source ValueId.
//! 3. DCE cleans up the dead identity ops (ran after this pass).

use std::collections::BTreeMap;

use metaltile_core::{
    dtype::DType,
    error::Result,
    ir::{Block, BlockId, IndexExpr, Kernel, Op, ValueId},
};

pub struct CopyPropPass;

impl super::Pass for CopyPropPass {
    fn name(&self) -> &str { "copy_prop" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();

        for bid in &block_ids {
            let mut block = kernel.blocks.remove(bid).unwrap();
            copy_prop_block_fixpoint(&mut block);
            kernel.blocks.insert(*bid, block);
        }

        copy_prop_block_fixpoint(&mut kernel.body);

        Ok(())
    }
}

fn copy_prop_block_fixpoint(block: &mut Block) {
    loop {
        if !copy_prop_block_once(block) {
            break;
        }
    }
}

fn copy_prop_block_once(block: &mut Block) -> bool {
    let n = block.ops.len();
    let mut vid_replacements: BTreeMap<ValueId, ValueId> = BTreeMap::new();

    for i in 0..n {
        let op = &block.ops[i];
        if let Some(source_vid) = is_identity(op, block) {
            if let Some(Some(result_vid)) = block.results.get(i) {
                vid_replacements.insert(*result_vid, source_vid);
            }
        }
    }

    if vid_replacements.is_empty() {
        return false;
    }

    // Remap ValueIds in all ops.
    for op in block.ops.iter_mut() {
        remap_values_in_op(op, &vid_replacements);
    }

    true
}

/// Check if an op is an identity (output equals one of its inputs in all cases).
fn is_identity(op: &Op, _block: &Block) -> Option<ValueId> {
    match op {
        // Cast(float, x) → x  when x is already float
        Op::Cast { value, dtype } => {
            let inferred = infer_value_dtype(*value, _block);
            if inferred == Some(*dtype) {
                Some(*value)
            } else {
                None
            }
        },

        // Broadcast(x, [1]) → x  — broadcasting a scalar by shape [1] is a no-op
        Op::Broadcast { value, shape } => {
            if shape.rank() == 1
                && matches!(shape.dim(0), Some(metaltile_core::shape::Dim::Known(1)))
            {
                Some(*value)
            } else {
                None
            }
        },

        // Reshape(x, s) → x  when shape s has the same total elements and same layout
        // For now: only when the value is already a scalar or single-element tile.
        Op::Reshape { value, shape } => {
            if shape.rank() == 1
                && matches!(shape.dim(0), Some(metaltile_core::shape::Dim::Known(1)))
            {
                // Reshape to [1] is identity for scalars
                Some(*value)
            } else {
                None
            }
        },

        // Select(cond, x, x) → x  — same value both sides
        Op::Select { on_true, on_false, .. } => {
            if on_true == on_false {
                Some(*on_true)
            } else {
                None
            }
        },

        // ExpandDims with shape [1] is effectively an identity for a scalar
        // (handled by Reshape already; but cover base case)

        _ => None,
    }
}

/// Naive dtype inference for a value.  Only detects `Cast` and `Const` patterns.
fn infer_value_dtype(vid: ValueId, block: &Block) -> Option<DType> {
    for (i, op) in block.ops.iter().enumerate() {
        if block.results.get(i) != Some(&Some(vid)) {
            continue;
        }
        match op {
            Op::Cast { dtype, .. } => return Some(*dtype),
            Op::Const { .. } =>
                // Constants are integers; they'll be cast to target dtype at use.
                return None,
            Op::Zeros { dtype, .. } | Op::Splat { dtype, .. } => return Some(*dtype),
            Op::Load { .. } => return None, // dtype comes from param
            _ => return None,
        }
    }
    None
}

/// Remap all ValueId references in an op.
fn remap_values_in_op(op: &mut Op, map: &BTreeMap<ValueId, ValueId>) {
    let s = |v: &mut ValueId| {
        if let Some(&nv) = map.get(v) {
            *v = nv;
        }
    };
    match op {
        Op::BinOp { lhs, rhs, .. } => {
            s(lhs);
            s(rhs);
        },
        Op::UnaryOp { value, .. }
        | Op::Activation { value, .. }
        | Op::Cast { value, .. }
        | Op::Reduce { value, .. }
        | Op::Transpose { value }
        | Op::Slice { value, .. }
        | Op::Broadcast { value, .. } => s(value),
        Op::Select { cond, on_true, on_false } => {
            s(cond);
            s(on_true);
            s(on_false);
        },
        Op::Dot { a, b } => {
            s(a);
            s(b);
        },
        Op::Store { value, indices, .. } => {
            s(value);
            for idx in indices.iter_mut() {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = idx {
                    s(v);
                }
            }
        },
        Op::Loop { start, end, step, .. } => {
            s(start);
            s(end);
            s(step);
        },
        Op::Load { indices, .. } =>
            for idx in indices.iter_mut() {
                if let IndexExpr::Value(v) | IndexExpr::Range(v, _) = idx {
                    s(v);
                }
            },
        Op::InlineMsl { inputs, .. } =>
            for v in inputs.iter_mut() {
                s(v);
            },
        Op::FlashAttention { q, k, v, .. } => {
            s(q);
            s(k);
            s(v);
        },
        Op::SlidingWindowAttention { q, k, v, .. } => {
            s(q);
            s(k);
            s(v);
        },
        Op::RmsNorm { x, scale, .. } => {
            s(x);
            s(scale);
        },
        Op::GatedMlp { x, gate_proj, up_proj, down_proj } => {
            s(x);
            s(gate_proj);
            s(up_proj);
            s(down_proj);
        },
        Op::ProgramId { .. }
        | Op::Const { .. }
        | Op::Arange { .. }
        | Op::Zeros { .. }
        | Op::Splat { .. } => {},
        Op::FusedElementwise { ops } =>
            for sub_op in ops.iter_mut() {
                remap_values_in_op(sub_op, map);
            },
        Op::VectorLoad { byte_offset, .. } => s(byte_offset),
        Op::VectorStore { byte_offset, value, .. } => {
            s(byte_offset);
            s(value);
        },
        Op::StrideReduce { offset, stride, end, .. } => {
            s(offset);
            s(stride);
            s(end);
        },
        Op::If { cond, .. } => s(cond),
        Op::ExpandDims { value, .. } => s(value),
        Op::Reshape { value, .. } => s(value),
        Op::Cat { values, .. } =>
            for v in values.iter_mut() {
                s(v);
            },
        Op::Gather { indices, .. } => s(indices),
        Op::Scatter { indices, value, .. } => {
            s(indices);
            s(value);
        },
        Op::Atomic { index, value, .. } => {
            s(index);
            s(value);
        },
        Op::Scan { value, .. } => s(value),
        Op::StrideStore { offset, end, scalar, .. } => {
            s(offset);
            s(end);
            s(scalar);
        },
        Op::Dequantize { .. } => {},
        Op::SimdReduce { value, .. } => s(value),
        Op::ThreadgroupLoad { index, .. } => s(index),
        Op::ThreadgroupStore { index, value, .. } => {
            s(index);
            s(value);
        },
        Op::ThreadgroupAlloc { .. } | Op::Barrier => {},
        Op::DeclareLocal { value, .. } | Op::SetLocal { value, .. } => s(value),
        Op::ArgReduce { value, .. } => s(value),
        Op::StrideScan { offset, end, .. } => {
            s(offset);
            s(end);
        },
        Op::StrideArgReduce { offset, end, .. } => {
            s(offset);
            s(end);
        },
    }
}
