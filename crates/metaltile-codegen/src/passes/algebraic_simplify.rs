//! Algebraic Simplification pass.
//!
//! A pattern-matching rewrite system that simplifies IR operations beyond what
//! ConstFold handles.  Each pattern is a function that tries to match an op and
//! returns either a new op or a ValueId replacement.
//!
//! ## Patterns
//!
//! ### Zero / One Absorption
//! - `x - x → Const(0)`
//! - `x / x → Const(1)`
//! - `x / Const(1) → x`
//! - `Const(0) - x → Neg(x)`
//! - `(-x) * (-y) → x * y`
//!
//! ### Select / Conditional Simplification
//! - `Select(true, a, b) → a`
//! - `Select(false, a, b) → b`
//! - `Select(cond, a, a) → a`
//! - `Select(Not(cond), a, b) → Select(cond, b, a)`
//!
//! ### Broadcast / Reshape Squashing
//! - `Broadcast(Broadcast(x)) → Broadcast(x)`
//! - `Reshape(Reshape(x)) → Reshape(x)`
//! - `Transpose(Transpose(x)) → x`
//! - `ExpandDims(Reshape(x)) → Reshape(x)`
//!
//! ### Min / Max Canonicalization
//! - `Max(x, x) → x`
//! - `Min(x, x) → x`
//!
//! ### Comparison Canonicalization
//! - `CmpLt(a, b) → CmpGt(b, a)`
//! - `CmpLe(a, b) → CmpGe(b, a)`
//! - `CmpEq(a, a) → Const(1)`
//! - `CmpNe(a, a) → Const(0)`
//!
//! ## Algorithm
//!
//! Iterates to fixpoint over each block.  Each iteration collects rewrites
//! (new ops or ValueId replacements), applies them, and stops when stable.

use std::collections::BTreeMap;

use metaltile_core::{
    error::Result,
    ir::{BinOpKind, Block, BlockId, Kernel, Op, UnaryOpKind, ValueId},
};

pub struct AlgebraicSimplifyPass;

impl super::Pass for AlgebraicSimplifyPass {
    fn name(&self) -> &str { "algebraic_simplify" }

    fn run(&self, kernel: &mut Kernel) -> Result<()> {
        let block_ids: Vec<BlockId> = kernel.blocks.keys().copied().collect();

        for bid in &block_ids {
            let mut block = kernel.blocks.remove(bid).unwrap();
            simplify_block_fixpoint(&mut block);
            kernel.blocks.insert(*bid, block);
        }

        simplify_block_fixpoint(&mut kernel.body);

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Block-level fixpoint
// ---------------------------------------------------------------------------

fn simplify_block_fixpoint(block: &mut Block) {
    loop {
        if !simplify_block_once(block) {
            break;
        }
    }
}

fn simplify_block_once(block: &mut Block) -> bool {
    let n = block.ops.len();
    let mut const_overwrites: Vec<(usize, i64)> = Vec::new();
    let mut op_replacements: Vec<(usize, Op)> = Vec::new();
    let mut vid_replacements: BTreeMap<ValueId, ValueId> = BTreeMap::new();

    // Build a map for peephole lookups.
    let vid_to_op_pos: BTreeMap<ValueId, usize> =
        block.results.iter().enumerate().filter_map(|(i, r)| r.and_then(|v| Some((v, i)))).collect();

    for i in 0..n {
        if let Some(result) = try_simplify(&block.ops[i], i, block, &vid_to_op_pos) {
            match result {
                SimpResult::Const(v) => const_overwrites.push((i, v)),
                SimpResult::ReplaceWith(op) => op_replacements.push((i, op)),
                SimpResult::ReplaceWithVid(new_vid) => {
                    if let Some(Some(old_vid)) = block.results.get(i) {
                        vid_replacements.insert(*old_vid, new_vid);
                    }
                },
            }
        }
    }

    if const_overwrites.is_empty() && op_replacements.is_empty() && vid_replacements.is_empty() {
        return false;
    }

    // Apply op replacements and const overwrites.
    for (idx, new_val) in const_overwrites {
        block.ops[idx] = Op::Const { value: new_val };
    }
    for (idx, new_op) in &op_replacements {
        block.ops[*idx] = new_op.clone();
    }

    // Remap ValueIds in all ops in the block.
    for op in block.ops.iter_mut() {
        remap_values_in_op(op, &vid_replacements);
    }

    // Also need to handle the case where a ReplaceWithVid is for an op
    // whose result gets DCE'd: the op stays but its uses are redirected.
    // ConstFold-style DCE will clean it up later.

    true
}

// ---------------------------------------------------------------------------
// Simplification result
// ---------------------------------------------------------------------------

enum SimpResult {
    Const(i64),
    ReplaceWith(Op),
    ReplaceWithVid(ValueId),
}

// ---------------------------------------------------------------------------
// Main pattern matcher
// ---------------------------------------------------------------------------

fn try_simplify(
    op: &Op,
    pos: usize,
    block: &Block,
    vid_to_pos: &BTreeMap<ValueId, usize>,
) -> Option<SimpResult> {
    match op {
        // ---- BinOp patterns ----
        Op::BinOp { op: kind, lhs, rhs } => simplify_binop(*kind, *lhs, *rhs, pos, block, vid_to_pos),

        // ---- Select patterns ----
        Op::Select { cond, on_true, on_false } => {
            simplify_select(*cond, *on_true, *on_false, block, vid_to_pos)
        },

        // ---- Broadcast squashing ----
        Op::Broadcast { value, shape } => {
            if shape.rank() == 1 && matches!(shape.dim(0), Some(metaltile_core::shape::Dim::Known(1))) {
                // Already a scalar broadcast — skip.
                return None;
            }
            if let Some((inner_pos, Op::Broadcast { .. })) =
                get_defining_op(*value, block, vid_to_pos)
            {
                // Broadcast(Broadcast(x)) → inner Broadcast
                let inner_op = &block.ops[inner_pos];
                Some(SimpResult::ReplaceWith(inner_op.clone()))
            } else {
                None
            }
        },

        // ---- Transpose squashing ----
        Op::Transpose { value } => {
            if let Some((_inner_pos, Op::Transpose { value: inner_val })) =
                get_defining_op(*value, block, vid_to_pos)
            {
                // Transpose(Transpose(x)) → x
                Some(SimpResult::ReplaceWithVid(*inner_val))
            } else {
                None
            }
        },

        // ---- Reshape squashing ----
        Op::Reshape { value, .. } => {
            if let Some((_inner_pos, Op::Reshape { value: inner_val, shape: _ })) =
                get_defining_op(*value, block, vid_to_pos)
            {
                // Reshape(Reshape(x)) → inner Reshape (shape already correct from outer)
                Some(SimpResult::ReplaceWithVid(*inner_val))
            } else {
                None
            }
        },

        // ---- ExpandDims(Reshape(x)) → Reshape(x) ----
        Op::ExpandDims { value, .. } => {
            if let Some((_inner_pos, Op::Reshape { value: inner_val, .. })) =
                get_defining_op(*value, block, vid_to_pos)
            {
                Some(SimpResult::ReplaceWithVid(*inner_val))
            } else {
                None
            }
        },

        _ => None,
    }
}

fn simplify_binop(
    kind: BinOpKind,
    lhs: ValueId,
    rhs: ValueId,
    _pos: usize,
    block: &Block,
    vid_to_pos: &BTreeMap<ValueId, usize>,
) -> Option<SimpResult> {
    let lv = find_const_in_block(block, lhs);
    let rv = find_const_in_block(block, rhs);

    match kind {
        BinOpKind::Sub => {
            if lhs == rhs {
                // x - x → 0
                Some(SimpResult::Const(0))
            } else if lv == Some(0) {
                // 0 - x → Neg(x)
                Some(SimpResult::ReplaceWith(Op::UnaryOp { op: UnaryOpKind::Neg, value: rhs }))
            } else {
                None
            }
        },

        BinOpKind::Div => {
            if lhs == rhs {
                // x / x → 1
                Some(SimpResult::Const(1))
            } else if rv == Some(1) {
                // x / 1 → x
                Some(SimpResult::ReplaceWithVid(lhs))
            } else {
                None
            }
        },

        BinOpKind::Mul => {
            // (-x) * (-y) → x * y  (sign cancellation)
            let neg_x = get_neg_arg(lhs, block, vid_to_pos);
            let neg_y = get_neg_arg(rhs, block, vid_to_pos);
            if let (Some(inner_x), Some(inner_y)) = (neg_x, neg_y) {
                Some(SimpResult::ReplaceWith(Op::BinOp { op: BinOpKind::Mul, lhs: inner_x, rhs: inner_y }))
            } else {
                None
            }
        },

        BinOpKind::Min | BinOpKind::Max => {
            if lhs == rhs {
                // Max(x, x) → x, Min(x, x) → x
                Some(SimpResult::ReplaceWithVid(lhs))
            } else {
                None
            }
        },

        BinOpKind::CmpLt => {
            // CmpLt(a, b) → CmpGt(b, a)
            Some(SimpResult::ReplaceWith(Op::BinOp { op: BinOpKind::CmpGt, lhs: rhs, rhs: lhs }))
        },

        BinOpKind::CmpLe => {
            // CmpLe(a, b) → CmpGe(b, a)
            Some(SimpResult::ReplaceWith(Op::BinOp { op: BinOpKind::CmpGe, lhs: rhs, rhs: lhs }))
        },

        BinOpKind::CmpEq => {
            if lhs == rhs {
                // CmpEq(a, a) → Const(1)
                Some(SimpResult::Const(1))
            } else {
                None
            }
        },

        BinOpKind::CmpNe => {
            if lhs == rhs {
                // CmpNe(a, a) → Const(0)
                Some(SimpResult::Const(0))
            } else {
                None
            }
        },

        _ => None,
    }
}

fn simplify_select(
    cond: ValueId,
    on_true: ValueId,
    on_false: ValueId,
    block: &Block,
    vid_to_pos: &BTreeMap<ValueId, usize>,
) -> Option<SimpResult> {
    // Select(true, a, b) → a
    if let Some(1) = find_const_in_block(block, cond) {
        return Some(SimpResult::ReplaceWithVid(on_true));
    }
    // Select(false, a, b) → b
    if let Some(0) = find_const_in_block(block, cond) {
        return Some(SimpResult::ReplaceWithVid(on_false));
    }
    // Select(cond, a, a) → a
    if on_true == on_false {
        return Some(SimpResult::ReplaceWithVid(on_true));
    }
    // Select(Not(cond), a, b) → Select(cond, b, a)
    if let Some(inner) = get_not_arg(cond, block, vid_to_pos) {
        return Some(SimpResult::ReplaceWith(Op::Select {
            cond: inner,
            on_true: on_false,
            on_false: on_true,
        }));
    }
    None
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_const_in_block(block: &Block, vid: ValueId) -> Option<i64> {
    for (i, op) in block.ops.iter().enumerate() {
        if block.results.get(i) == Some(&Some(vid))
            && let Op::Const { value } = op
        {
            return Some(*value);
        }
    }
    None
}

/// Get the defining op for a ValueId. Returns (position, op) if definition is in this block.
fn get_defining_op<'a>(
    vid: ValueId,
    block: &'a Block,
    vid_to_pos: &BTreeMap<ValueId, usize>,
) -> Option<(usize, &'a Op)> {
    let &pos = vid_to_pos.get(&vid)?;
    Some((pos, &block.ops[pos]))
}

/// If `vid` is defined by `UnaryOp(Neg, inner)`, return `Some(inner)`.
fn get_neg_arg(
    vid: ValueId,
    block: &Block,
    vid_to_pos: &BTreeMap<ValueId, usize>,
) -> Option<ValueId> {
    let (_pos, op) = get_defining_op(vid, block, vid_to_pos)?;
    if let Op::UnaryOp { op: UnaryOpKind::Neg, value } = op {
        Some(*value)
    } else {
        None
    }
}

/// If `vid` is defined by a logical NOT (CmpEq(x, 0) or Xor(x, 1)), return the inner condition.
fn get_not_arg(
    vid: ValueId,
    block: &Block,
    vid_to_pos: &BTreeMap<ValueId, usize>,
) -> Option<ValueId> {
    let (_pos, op) = get_defining_op(vid, block, vid_to_pos)?;
    match op {
        Op::BinOp { op: BinOpKind::CmpEq, lhs, rhs } => {
            // CmpEq(cond, 0) → logical NOT of cond
            if find_const_in_block(block, *rhs) == Some(0) {
                return Some(*lhs);
            }
            if find_const_in_block(block, *lhs) == Some(0) {
                return Some(*rhs);
            }
            None
        },
        Op::BinOp { op: BinOpKind::Xor, lhs, rhs } => {
            // Xor(cond, 1) → logical NOT of cond
            if find_const_in_block(block, *rhs) == Some(1) {
                return Some(*lhs);
            }
            if find_const_in_block(block, *lhs) == Some(1) {
                return Some(*rhs);
            }
            None
        },
        _ => None,
    }
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
                if let metaltile_core::ir::IndexExpr::Value(v)
                | metaltile_core::ir::IndexExpr::Range(v, _) = idx
                {
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
                if let metaltile_core::ir::IndexExpr::Value(v)
                | metaltile_core::ir::IndexExpr::Range(v, _) = idx
                {
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
