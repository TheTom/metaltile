//! Chunked-WY Gated DeltaNet prefill kernel — `mt_gated_delta_wy_chunk`.
//!
//! Spec 028 Phase 2-3 (mlx-swift-lm). Process a chunk of `C` tokens in
//! parallel via the compact Woodbury-Young representation of the delta-rule
//! product, then chain chunks sequentially across the prefill. Expected
//! 5-15× faster than the per-token sequential `mt_gated_delta_chunk` (which
//! is in turn ≈ MLX's `gated_delta_step` baseline) at long-context prefill.
//!
//! ## Recurrence (sequential reference)
//!
//! ```text
//! state    <- state * g_t                     // scalar decay
//! kv_mem    = state @ k_t                      // [Dv]
//! delta     = (v_t - kv_mem) * β_t            // [Dv]
//! state    <- state + outer(delta, k_t)       // rank-1 update
//! y_t       = state @ q_t                      // [Dv]
//! ```
//!
//! Matrix form (used by the chunked algorithm):
//!
//! ```text
//! S_t = g_t · S_{t-1} · (I − β_t k_t k_t^T) + β_t v_t k_t^T
//! ```
//!
//! ## Chunked-WY closed form (validated to fp64 ~1e-11 in
//! `/tmp/gdn_chunked_wy/gdn_wy_ref.py`)
//!
//! Per chunk of `C` tokens [t0..t0+C), inputs Q,K∈R^{C×Dk}, V∈R^{C×Dv},
//! g,β∈R^{C}, S_0∈R^{Dv×Dk}:
//!
//! 1. **Prefix gates**:  `G_t = Π_{i≤t} g_i`, `Γ[t,j] = G_t/G_j` (j≤t).
//!
//! 2. **Inner products**:  `KKT[i,j] = k_i · k_j`  ∈ R^{C×C}.
//!
//! 3. **Passthrough p-rows** (no gating in L by design — gates factor out of
//!    P-products): solve unit-lower-tri `(I + L) p = K`, with
//!    `L[j,i] = β_i · KKT[j,i]` for i<j. Output `p[1..C] ∈ R^{C×Dk}`.
//!
//! 4. **Chunk-local u^v rows**: solve `(I + A) u^v = β ⊙ V`, with
//!    `A[t,j] = β_t · Γ[t,j] · KKT[t,j]` for t>j. Output `u^v ∈ R^{C×Dv}`.
//!
//! 5. **y_local** (uses u^v, queries against own chunk):
//!    `weights = Γ ⊙ (Q @ K^T)`, masked lower-tri inclusive.
//!    `y_local = weights @ u^v`  ∈ R^{C×Dv}.
//!
//! 6. **y_pass** (uses S_0 through the passthrough):
//!    `S0Q[t,v] = (S_0 @ q_t)[v]`,
//!    `weight[t,i] = β_i · (k_i · q_t) · 1[i≤t]`,
//!    `S0_p[v,i] = (S_0 @ p_i)[v]`,
//!    `correction[t,v] = Σ_i weight[t,i] · S0_p[v,i]`,
//!    `y_pass[t,v] = G_t · (S0Q[t,v] − correction[t,v])`.
//!
//! 7. **Per-row output**: `y_t = y_pass_t + y_local_t`.
//!
//! 8. **End-of-chunk state**:
//!    `S_through = G_C · (S_0 − S_0 @ (β⊙p)^T @ K)`,
//!    `U_end = Σ_j (G_C/G_j) · u^v_j ⊗ k_j`,
//!    `S_end = S_through + U_end`.
//!
//! ## Dispatch contract
//!
//!   - **Mode**: Reduction (uses simdgroup + threadgroup ops)
//!   - **Grid**: `[1, B*Hv, 1]` — one TG per (batch, hv-head). The TG loops
//!     internally across chunks because chunk N's state feeds chunk N+1.
//!   - **TG**:   `[32 * SGS, 1, 1]` with SGS = 4 simdgroups (matches
//!     `mt_sdpa_prefill_mma`'s 4-SG geometry; SGS×8 = 32 covers an 8×8 tile
//!     row per SG with overlap).
//!   - Chunk size `C` must be a multiple of 8 to fit Apple 8×8 frag tiles.
//!     Likely sweet spot: C=64 (per spec #028 + FLA literature).
//!   - `Dk`, `Dv` must each be a multiple of 8 for the same reason.
//!
//! ## Layouts (matches `mt_gated_delta_chunk`)
//!
//!   - `q, k`:    [B, T, Hk, Dk]
//!   - `v, y`:    [B, T, Hv, Dv]
//!   - `g, beta`: [B, T, Hv]
//!   - `state_in/out`: [B, Hv, Dv, Dk]
//!
//! GQA: `hk_idx = hv_idx / (Hv / Hk)`.
//!
//! State accumulator runs in f32 — same bf16-drift reasoning as
//! `mt_gated_delta_step` and `mt_ssm_step`. The two triangular solves
//! (steps 3, 4) also use f32 to keep T-conditioning bounded.
//!
//! ## Implementation plan
//!
//! Multi-stage port from the validated Python reference at
//! `/tmp/gdn_chunked_wy/gdn_wy_ref.py`. The DSL kernel must:
//!
//! 1. **TG-shared chunk buffers**: load this chunk's K, V, Q, β into TG
//!    memory (small: C=64 × Dk=128 × bf16 = 16 KB per buffer).
//! 2. **Build KKT in TG**: SG-parallel C×Dk · Dk×C matmul via
//!    `simdgroup_matmul` over 8×8 tiles. Store KKT[C×C] in TG.
//! 3. **Triangular solves (T_WY, A)**: forward substitution. Scalar work
//!    over C iterations, one SG handles all C iterations; per-row work
//!    parallelized across the simdgroup's 32 lanes. p_rows and u^v stored
//!    in TG.
//! 4. **Compute Γ**: cumprod of g in TG; build Γ[C×C] in TG.
//! 5. **y_local matmul**: SG-parallel weighted-Γ⊙QKT × u^v.
//! 6. **y_pass**: matmul S_0 with Q^T, p^T; correction term; per-row scale
//!    by G_t.
//! 7. **End-of-chunk state**: matmul S_0 with (β⊙p)^T, then with K, plus
//!    Σ u^v ⊗ k_j outer-product accumulation.
//!
//! State write-out is the only inter-chunk dependency. We iterate chunks
//! inside one TG, keeping S in TG between iterations. At long context this
//! amortizes the per-chunk setup cost.

// Placeholder — kernel body lands in stages once the per-step DSL pattern
// is verified against the Python reference at /tmp/gdn_chunked_wy/.
//
// Skeleton-only until then to avoid landing untested DSL code in the
// public kernel registry.
//
// Next steps (in order):
//   1. Write a CPU-only Rust translation of `_process_chunk` (no DSL) and
//      check it matches `gated_delta_chunked_wy` on the same test vectors.
//   2. Port to DSL one step at a time:
//        a. KKT matmul (simdgroup_matmul tile pattern)
//        b. Triangular solves (small C, scalar in one SG)
//        c. Γ + y_local
//        d. y_pass
//        e. S_end accumulator + writeout
//   3. Add `tests/gated_delta_wy_gpu_correctness.rs` mirroring PR #112's
//      pattern: identity at g=1 β=0, vs sequential oracle f32/f16/bf16,
//      GQA, edge cases (T not divisible by C, T=1, T=C, T=2048).
//   4. Bench on M2 mini (per repo rule — never M5 for perf). Target ≥5×
//      speedup over MLX `gated_delta_step` at T≥1024.
