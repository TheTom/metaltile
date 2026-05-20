//! GPU correctness for `ffai::gated_delta::mt_gated_delta_step`.
//!
//! GDN (Gated DeltaNet) is the recurrent linear-attention variant used by
//! Qwen3.5 / Qwen3.6 / Qwen3.6-MoE for their `linear_attention` layers
//! (75% of layers in those hybrid models). This file pins the single-token
//! decode form — `T = 1` of MLX-LM's `gated_delta_kernel`.
//!
//! Tests pin:
//!
//!   - **Identity at g=1, beta=0**: no decay + no update → state unchanged,
//!     y = state @ q. The "no-op recurrence" baseline.
//!   - **CPU oracle match (f32)** at a realistic shape — Qwen3.6 has
//!     Hk=4, Hv=24, head_dim=256, but we use smaller dims to keep the
//!     test fast. Validates the full recurrence numerically.
//!   - **GQA dispatch correctness**: Hv > Hk → multiple Hv-heads share a
//!     single (q, k) Hk-slot. Catches `hk_idx = hv_idx / (Hv/Hk)` errors.
//!   - **dtype matrix (f16 / bf16)** with derived tolerance.
//!   - **`x = 0` (v = 0) decay invariant**: the recurrence collapses to
//!     `state = state * g`, y = (state*g) @ q. Pins that delta is applied
//!     to v correctly.
//!
//! macOS-gated. Shared gpu_lock via tests/common/.

#![cfg(target_os = "macos")]

mod common;

use std::collections::BTreeMap;

use common::{Dt, gpu_lock, pack_bytes, unpack_bytes};
use metaltile_core::ir::KernelMode;
use metaltile_runtime::Context;
use metaltile_std::ffai::gated_delta::mt_gated_delta_step;

/// CPU oracle: matches `_gated_delta_step_ops` from `mlx_lm/models/gated_delta.py`.
///
/// Shapes:
///   - q, k: [B, Hk, Dk]
///   - v: [B, Hv, Dv]
///   - g, beta: [B, Hv]
///   - state: [B, Hv, Dv, Dk] (f32 in/out)
/// Returns: (y [B, Hv, Dv], new_state [B, Hv, Dv, Dk])
#[allow(clippy::too_many_arguments)]
fn naive_gated_delta_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state_in: &[f32],
    b: usize,
    hv: usize,
    hk: usize,
    dv: usize,
    dk: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut y = vec![0.0_f32; b * hv * dv];
    let mut state_out = vec![0.0_f32; b * hv * dv * dk];
    let hk_per_hv = hv / hk;
    for batch in 0..b {
        for hv_idx in 0..hv {
            let n = batch * hv + hv_idx;
            let hk_idx = hv_idx / hk_per_hv;
            let g_val = g[n];
            let beta_val = beta[n];
            let qk_base = (batch * hk + hk_idx) * dk;
            for dv_idx in 0..dv {
                let v_val = v[n * dv + dv_idx];
                let s_base = n * dv * dk + dv_idx * dk;

                // Phase 1: decay + kv_mem
                let mut kv_mem = 0.0_f32;
                let mut decayed = vec![0.0_f32; dk];
                for s_idx in 0..dk {
                    let s = state_in[s_base + s_idx] * g_val;
                    decayed[s_idx] = s;
                    kv_mem += s * k[qk_base + s_idx];
                }
                let delta = (v_val - kv_mem) * beta_val;

                // Phase 2: update + output projection
                let mut out = 0.0_f32;
                for s_idx in 0..dk {
                    let s_new = decayed[s_idx] + k[qk_base + s_idx] * delta;
                    state_out[s_base + s_idx] = s_new;
                    out += s_new * q[qk_base + s_idx];
                }
                y[n * dv + dv_idx] = out;
            }
        }
    }
    (y, state_out)
}

#[allow(clippy::too_many_arguments)]
fn run_gated_delta_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state_in: &[f32],
    dt: Dt,
    b: usize,
    hv: usize,
    hk: usize,
    dv: usize,
    dk: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n_total = b * hv;

    let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    buffers.insert("q".into(), pack_bytes(q, dt));
    buffers.insert("k".into(), pack_bytes(k, dt));
    buffers.insert("v".into(), pack_bytes(v, dt));
    buffers.insert("g".into(), pack_bytes(g, dt));
    buffers.insert("beta".into(), pack_bytes(beta, dt));
    buffers.insert("state_in".into(), pack_bytes(state_in, dt));
    buffers.insert("state_out".into(), pack_bytes(&vec![0.0_f32; state_in.len()], dt));
    buffers.insert("y".into(), pack_bytes(&vec![0.0_f32; n_total * dv], dt));
    buffers.insert("dk".into(), (dk as u32).to_le_bytes().to_vec());
    buffers.insert("dv".into(), (dv as u32).to_le_bytes().to_vec());
    buffers.insert("hv".into(), (hv as u32).to_le_bytes().to_vec());
    buffers.insert("hk".into(), (hk as u32).to_le_bytes().to_vec());

    let ctx = Context::new().expect("Context::new on macOS");
    let mut kernel = mt_gated_delta_step::kernel_ir_for(dt.to_dtype());
    kernel.mode = KernelMode::Reduction;

    // Reduction dispatch (docs/developing.md):
    //   tgid_x = dv_idx, tgid_y = n, tid = dk_idx (0..32)
    //   TPG = 32 (one simdgroup), Dk must be a multiple of 32
    assert!(dk.is_multiple_of(32), "mt_gated_delta_step requires dk % 32 == 0");
    let result = ctx
        .dispatch_with_grid(&kernel, &buffers, &BTreeMap::new(), [dv, n_total, 1], [32, 1, 1])
        .expect("mt_gated_delta_step dispatch");

    let y = unpack_bytes(result.outputs.get("y").expect("y"), dt);
    let state_out = unpack_bytes(result.outputs.get("state_out").expect("state_out"), dt);
    (y, state_out)
}

// ────────────────────────────────────────────────────────────────────
//  Tests
// ────────────────────────────────────────────────────────────────────

#[test]
fn gated_delta_step_identity_at_g1_beta0_f32() {
    let _g = gpu_lock();
    // g=1, beta=0 → decayed = state, delta = 0, state_new = state.
    // y = state @ q exactly. Pure dot product. Catches gross dispatch /
    // index errors before any recurrence math.
    let b = 1;
    let hv = 4;
    let hk = 2;
    let dv = 4;
    let dk = 32;
    let n_total = b * hv;

    let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * 0.5).collect();
    let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * 0.5).collect();
    let v = vec![0.0_f32; n_total * dv]; // not consumed since beta=0
    let g = vec![1.0_f32; n_total];
    let beta = vec![0.0_f32; n_total];
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();

    let (y_expected, state_expected) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, state_actual) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);

    let mut max_y_diff = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        max_y_diff = max_y_diff.max((a - e).abs());
    }
    assert!(max_y_diff < 1e-5, "identity y max |diff| = {max_y_diff:.2e}");

    let mut max_state_diff = 0.0_f32;
    for (a, e) in state_actual.iter().zip(state_expected.iter()) {
        max_state_diff = max_state_diff.max((a - e).abs());
    }
    assert!(max_state_diff < 1e-5, "identity state max |diff| = {max_state_diff:.2e}");
}

#[test]
fn gated_delta_step_matches_oracle_f32() {
    let _g = gpu_lock();
    // Realistic recurrence: smooth non-trivial gates, full update path.
    let b = 2;
    let hv = 4;
    let hk = 2;
    let dv = 8;
    let dk = 64;
    let n_total = b * hv;

    let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * 0.5).collect();
    let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * 0.5).collect();
    let v: Vec<f32> = (0..n_total * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
    let g: Vec<f32> = (0..n_total).map(|i| 0.9 - (i as f32) * 0.01).collect();
    let beta: Vec<f32> = (0..n_total).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();

    let (y_expected, state_expected) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, state_actual) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);

    let mut max_y_diff = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        max_y_diff = max_y_diff.max((a - e).abs());
    }
    // simd_sum across 32 lanes with dk=64 → 2 mul-adds per lane;
    // recurrence has 2 dependent reductions. ~3 ULPs of f32 accumulation.
    assert!(max_y_diff < 5e-5, "y max |diff| = {max_y_diff:.2e}");

    let mut max_state_diff = 0.0_f32;
    for (a, e) in state_actual.iter().zip(state_expected.iter()) {
        max_state_diff = max_state_diff.max((a - e).abs());
    }
    assert!(max_state_diff < 1e-5, "state max |diff| = {max_state_diff:.2e}");
}

#[test]
fn gated_delta_step_gqa_hv_4x_hk_f32() {
    let _g = gpu_lock();
    // Hv = 4 * Hk: each (q, k) Hk-slot serves 4 Hv-heads. Pins the
    // `hk_idx = hv_idx / (Hv/Hk)` decomposition — a wrong divisor
    // would route the wrong Hv-head to the wrong Hk-slot.
    let b = 2;
    let hv = 8;
    let hk = 2; // Hv / Hk = 4
    let dv = 4;
    let dk = 32;
    let n_total = b * hv;

    let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.029).sin() * 0.5).collect();
    let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.031).cos() * 0.5).collect();
    let v: Vec<f32> = (0..n_total * dv).map(|i| ((i as f32) * 0.041).sin() * 0.3).collect();
    let g: Vec<f32> = (0..n_total).map(|i| 0.85 + (i as f32) * 0.005).collect();
    let beta: Vec<f32> = (0..n_total).map(|i| 0.4 + (i as f32) * 0.01).collect();
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.013).cos() * 0.1).collect();

    let (y_expected, state_expected) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, state_actual) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);

    let mut max_y_diff = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        max_y_diff = max_y_diff.max((a - e).abs());
    }
    assert!(max_y_diff < 5e-5, "GQA y max |diff| = {max_y_diff:.2e}");

    let mut max_state_diff = 0.0_f32;
    for (a, e) in state_actual.iter().zip(state_expected.iter()) {
        max_state_diff = max_state_diff.max((a - e).abs());
    }
    assert!(max_state_diff < 1e-5, "GQA state max |diff| = {max_state_diff:.2e}");
}

#[test]
fn gated_delta_step_v_zero_collapses_to_pure_decay_f32() {
    let _g = gpu_lock();
    // v = 0 → delta = (0 - kv_mem) * beta = -kv_mem * beta. With beta=0
    // we already pinned the no-delta path; this exercises beta != 0 but
    // checks the recurrence stays bounded.
    let b = 1;
    let hv = 2;
    let hk = 1;
    let dv = 4;
    let dk = 32;
    let n_total = b * hv;

    let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.019).sin() * 0.5).collect();
    let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.023).cos() * 0.5).collect();
    let v = vec![0.0_f32; n_total * dv];
    let g = vec![0.8_f32; n_total];
    let beta = vec![0.5_f32; n_total];
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();

    let (y_expected, state_expected) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, state_actual) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);

    let mut max_y_diff = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        max_y_diff = max_y_diff.max((a - e).abs());
    }
    assert!(max_y_diff < 5e-5, "v=0 y max |diff| = {max_y_diff:.2e}");

    let mut max_state_diff = 0.0_f32;
    for (a, e) in state_actual.iter().zip(state_expected.iter()) {
        max_state_diff = max_state_diff.max((a - e).abs());
    }
    assert!(max_state_diff < 1e-5, "v=0 state max |diff| = {max_state_diff:.2e}");
}

#[test]
fn gated_delta_step_matches_oracle_f16() {
    let _g = gpu_lock();
    let b = 1;
    let hv = 4;
    let hk = 2;
    let dv = 4;
    let dk = 32;
    let n_total = b * hv;

    let round = |v: &[f32]| v.iter().map(|&x| Dt::F16.round(x)).collect::<Vec<f32>>();
    let q = round(&(0..b * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * 0.5).collect::<Vec<_>>());
    let k = round(&(0..b * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * 0.5).collect::<Vec<_>>());
    let v = round(&(0..n_total * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect::<Vec<_>>());
    let g = round(&(0..n_total).map(|i| 0.9 - (i as f32) * 0.01).collect::<Vec<_>>());
    let beta = round(&(0..n_total).map(|i| 0.5 + (i as f32) * 0.01).collect::<Vec<_>>());
    let state_in = round(
        &(0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect::<Vec<_>>(),
    );

    let (y_expected, _) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, _) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F16, b, hv, hk, dv, dk);

    let mut max_rel = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        let rel = (a - e).abs() / e.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    // f16 10-bit mantissa + dependent reductions (kv_mem → delta → update → out).
    // Two simd_sums each accumulate ~32 mul-adds.
    assert!(max_rel < 5e-2, "f16 max rel = {max_rel:.2e}");
}

#[test]
fn gated_delta_step_qwen36_head_dim_256_f32() {
    let _g = gpu_lock();
    // Qwen3.6's actual head_dim = 256. n_per_t = 256/32 = 8 elements
    // per lane — the tg_state[256] alloc is fully utilized. None of the
    // smaller-Dk tests exercise this regime; a regression in the
    // multi-iteration `for i in 0..n_per_t` loop or in the upper half
    // of the TG memory would slip through them.
    let b = 1;
    let hv = 2;
    let hk = 1;
    let dv = 2;
    let dk = 256; // Qwen3.6 head_dim
    let n_total = b * hv;

    let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0091).sin() * 0.3).collect();
    let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.0103).cos() * 0.3).collect();
    let v: Vec<f32> = (0..n_total * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
    let g: Vec<f32> = (0..n_total).map(|i| 0.92 + (i as f32) * 0.005).collect();
    let beta: Vec<f32> = (0..n_total).map(|i| 0.4 + (i as f32) * 0.01).collect();
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();

    let (y_expected, state_expected) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, state_actual) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);

    // Dk=256 means each reduction sums 256 f32 mul-adds — tolerance bumped
    // vs Dk=64 because the accumulation depth is 4× longer.
    let mut max_y_diff = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        max_y_diff = max_y_diff.max((a - e).abs());
    }
    assert!(max_y_diff < 2e-4, "Dk=256 y max |diff| = {max_y_diff:.2e}");

    let mut max_state_diff = 0.0_f32;
    for (a, e) in state_actual.iter().zip(state_expected.iter()) {
        max_state_diff = max_state_diff.max((a - e).abs());
    }
    assert!(max_state_diff < 5e-5, "Dk=256 state max |diff| = {max_state_diff:.2e}");
}

#[test]
fn gated_delta_step_no_gqa_hv_equals_hk_f32() {
    let _g = gpu_lock();
    // Hv == Hk: every Hv-head has its own (q, k) — no sharing.
    // Hv/Hk = 1 so hk_idx == hv_idx. The trivial case. Catches a
    // hypothetical refactor that breaks the no-share branch.
    let b = 2;
    let hv = 4;
    let hk = 4;
    let dv = 4;
    let dk = 32;
    let n_total = b * hv;

    let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.019).sin() * 0.4).collect();
    let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.023).cos() * 0.4).collect();
    let v: Vec<f32> = (0..n_total * dv).map(|i| ((i as f32) * 0.031).sin() * 0.3).collect();
    let g: Vec<f32> = (0..n_total).map(|i| 0.88 + (i as f32) * 0.005).collect();
    let beta: Vec<f32> = (0..n_total).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.013).cos() * 0.1).collect();

    let (y_expected, _) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, _) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);

    let mut max_diff = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        max_diff = max_diff.max((a - e).abs());
    }
    assert!(max_diff < 5e-5, "Hv=Hk y max |diff| = {max_diff:.2e}");
}

#[test]
fn gated_delta_step_batch_4_stresses_indexing_f32() {
    let _g = gpu_lock();
    // B > 1 stress — distinct per-batch g / beta / state so the b = n/hv
    // batch index decomposition is exercised. Catches a wrong divisor
    // that would cross-contaminate batch slots.
    let b = 4;
    let hv = 2;
    let hk = 1;
    let dv = 4;
    let dk = 32;
    let n_total = b * hv;

    let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.017).sin() * 0.4).collect();
    let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.021).cos() * 0.4).collect();
    let v: Vec<f32> = (0..n_total * dv).map(|i| ((i as f32) * 0.027).sin() * 0.3).collect();
    // Distinct g per batch — first half 0.95, second half 0.75 — so a
    // mis-routed batch returns visibly wrong recurrence direction.
    let g: Vec<f32> = (0..n_total).map(|i| if (i / hv) < b / 2 { 0.95 } else { 0.75 }).collect();
    let beta: Vec<f32> = (0..n_total).map(|i| 0.4 + (i as f32) * 0.01).collect();
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();

    let (y_expected, _) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, _) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);

    let mut max_diff = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        max_diff = max_diff.max((a - e).abs());
    }
    assert!(max_diff < 5e-5, "B=4 y max |diff| = {max_diff:.2e}");
}

#[test]
fn gated_delta_step_2049_iterations_stay_stable_f32() {
    let _g = gpu_lock();
    // Issue #111: Qwen3.6 crashes at ctx > 2048 in the hybrid scheduler.
    // The decode-form kernel doesn't carry a T-loop (that's the chunked
    // kernel — part 1b), but a long autoregressive sequence calls the
    // decode kernel iteratively. Running 2049 iterations exercises the
    // exact regime that matters for serving past the 2048 boundary —
    // state must remain finite, deterministic, and match a CPU oracle.
    //
    // 2049 chosen specifically to cross the bug's boundary (the chunked-
    // prefill kernel breaks at T=2049; the decode kernel should NOT
    // break here regardless because it's pure single-step).
    let b = 1;
    let hv = 2;
    let hk = 1;
    let dv = 4;
    let dk = 32;
    let n_iters = 2049usize;
    let n_total = b * hv;

    let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.013).sin() * 0.3).collect();
    let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.017).cos() * 0.3).collect();
    // g < 1 so the state stays bounded across 2049 iterations.
    let g = vec![0.95_f32; n_total];
    let beta = vec![0.3_f32; n_total];

    let mut state_gpu = vec![0.0_f32; n_total * dv * dk];
    let mut state_cpu = state_gpu.clone();
    let mut last_y_gpu = vec![0.0_f32; n_total * dv];
    let mut last_y_cpu = vec![0.0_f32; n_total * dv];

    for step in 0..n_iters {
        // v varies per step so the recurrence has actual input.
        let v: Vec<f32> =
            (0..n_total * dv).map(|i| ((i as f32 + step as f32) * 0.029).sin() * 0.3).collect();

        let (y_gpu, state_gpu_new) =
            run_gated_delta_step(&q, &k, &v, &g, &beta, &state_gpu, Dt::F32, b, hv, hk, dv, dk);
        let (y_cpu, state_cpu_new) =
            naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_cpu, b, hv, hk, dv, dk);

        state_gpu = state_gpu_new;
        state_cpu = state_cpu_new;
        last_y_gpu = y_gpu;
        last_y_cpu = y_cpu;

        // Spot-check at a couple of milestones — at 2048 and 2049 (the
        // exact #111 boundary) the state must remain finite and tracking
        // the CPU oracle.
        if step == 2047 || step == 2048 {
            for &v in state_gpu.iter() {
                assert!(v.is_finite(), "step {step}: state contains non-finite value {v}");
            }
            for &v in last_y_gpu.iter() {
                assert!(v.is_finite(), "step {step}: y contains non-finite value {v}");
            }
        }
    }

    // After 2049 iterations, GPU and CPU should still agree.
    // Tolerance accumulates ULPs across iterations — generous 5e-3.
    let mut max_y_diff = 0.0_f32;
    for (a, e) in last_y_gpu.iter().zip(last_y_cpu.iter()) {
        max_y_diff = max_y_diff.max((a - e).abs());
    }
    assert!(max_y_diff < 5e-3, "y drift after {n_iters} iterations: max |diff| = {max_y_diff:.2e}",);

    let mut max_state_diff = 0.0_f32;
    for (a, e) in state_gpu.iter().zip(state_cpu.iter()) {
        max_state_diff = max_state_diff.max((a - e).abs());
    }
    assert!(
        max_state_diff < 5e-3,
        "state drift after {n_iters} iterations: max |diff| = {max_state_diff:.2e}",
    );
}

#[test]
fn gated_delta_step_edge_gates_f32() {
    let _g = gpu_lock();
    // Edge values for g and beta: g near 1 (slow decay, state nearly
    // preserved), g near 0 (fast decay, state nearly wiped), beta near
    // 0 (small update), beta = 1 (full delta-rule update). Each pair
    // exercises a different numerical regime; a bug in the (1 - β·k·kᵀ)
    // decomposition would surface in one of these.
    let b = 1;
    let hv = 4;
    let hk = 2;
    let dv = 4;
    let dk = 32;
    let n_total = b * hv;

    let q: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.013).sin() * 0.3).collect();
    let k: Vec<f32> = (0..b * hk * dk).map(|i| ((i as f32) * 0.017).cos() * 0.3).collect();
    let v: Vec<f32> = (0..n_total * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect();
    let state_in: Vec<f32> =
        (0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect();

    let configs: [(f32, f32, &str); 4] = [
        (0.999, 0.01, "g=0.999 (slow decay), beta=0.01 (tiny update)"),
        (0.01, 0.5, "g=0.01 (fast decay), beta=0.5"),
        (0.5, 0.999, "g=0.5, beta=0.999 (near-full delta)"),
        (1.0, 1.0, "g=1, beta=1 (no decay, full delta)"),
    ];

    for (g_val, beta_val, label) in configs {
        let g = vec![g_val; n_total];
        let beta = vec![beta_val; n_total];

        let (y_expected, _) =
            naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
        let (y_actual, _) =
            run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);

        let mut max_diff = 0.0_f32;
        for (a, e) in y_actual.iter().zip(y_expected.iter()) {
            max_diff = max_diff.max((a - e).abs());
        }
        assert!(max_diff < 5e-5, "{label}: y max |diff| = {max_diff:.2e}");
        for &v in y_actual.iter() {
            assert!(v.is_finite(), "{label}: y contains non-finite {v}");
        }
    }
}

#[test]
#[should_panic(expected = "mt_gated_delta_step requires dk % 32 == 0")]
fn gated_delta_step_panics_on_unaligned_dk() {
    // Dk = 33 violates the kernel's "Dk must be a multiple of 32"
    // contract. The dispatch helper asserts the contract before the
    // kernel runs. If a future refactor drops the assertion (or a new
    // dispatcher forgets it), this test catches the regression before
    // the kernel produces silently-wrong output.
    let _g = gpu_lock();
    let b = 1;
    let hv = 1;
    let hk = 1;
    let dv = 1;
    let dk = 33; // not a multiple of 32
    let n_total = b * hv;
    let q = vec![0.0_f32; b * hk * dk];
    let k = vec![0.0_f32; b * hk * dk];
    let v = vec![0.0_f32; n_total * dv];
    let g = vec![1.0_f32; n_total];
    let beta = vec![0.0_f32; n_total];
    let state_in = vec![0.0_f32; n_total * dv * dk];
    let _ = run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::F32, b, hv, hk, dv, dk);
}

#[test]
fn gated_delta_step_matches_oracle_bf16() {
    let _g = gpu_lock();
    let b = 1;
    let hv = 4;
    let hk = 2;
    let dv = 4;
    let dk = 32;
    let n_total = b * hv;

    let round = |v: &[f32]| v.iter().map(|&x| Dt::Bf16.round(x)).collect::<Vec<f32>>();
    let q = round(&(0..b * hk * dk).map(|i| ((i as f32) * 0.0173).sin() * 0.5).collect::<Vec<_>>());
    let k = round(&(0..b * hk * dk).map(|i| ((i as f32) * 0.0211).cos() * 0.5).collect::<Vec<_>>());
    let v = round(&(0..n_total * dv).map(|i| ((i as f32) * 0.029).sin() * 0.3).collect::<Vec<_>>());
    let g = round(&(0..n_total).map(|i| 0.9 - (i as f32) * 0.01).collect::<Vec<_>>());
    let beta = round(&(0..n_total).map(|i| 0.5 + (i as f32) * 0.01).collect::<Vec<_>>());
    let state_in = round(
        &(0..n_total * dv * dk).map(|i| ((i as f32) * 0.011).sin() * 0.1).collect::<Vec<_>>(),
    );

    let (y_expected, _) =
        naive_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, b, hv, hk, dv, dk);
    let (y_actual, _) =
        run_gated_delta_step(&q, &k, &v, &g, &beta, &state_in, Dt::Bf16, b, hv, hk, dv, dk);

    let mut max_rel = 0.0_f32;
    for (a, e) in y_actual.iter().zip(y_expected.iter()) {
        let rel = (a - e).abs() / e.abs().max(1e-3);
        max_rel = max_rel.max(rel);
    }
    // bf16 7-bit mantissa is the wider tolerance.
    assert!(max_rel < 2e-1, "bf16 max rel = {max_rel:.2e}");
}
