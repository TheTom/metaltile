//! Decode-form scaled dot-product attention — `mt_sdpa_vector`.
//!
//! Faithful port of MLX `sdpa_vector<T, D, V=D>` template instantiation
//! `sdpa_vector_{tname}_128_128`. One threadgroup per Q head, 1024
//! threads = `BN × BD = 32 simdgroups × 32 lanes`. Each simdgroup walks
//! a stride-`BN` slice of `n_kv` positions, then a two-step cross-
//! simdgroup reduction combines the partial online-softmax results.
//!
//! Differs from `mt_sdpa` (same file family) only by adding **GQA**
//! support: `kv_head = q_head / gqa_factor`. When `gqa_factor = 1`
//! this is exactly `mt_sdpa` semantically — but with the `mlx`-side
//! comparison wired through the `SdpaVector` dispatch, which handles
//! the parameterised K/V head count and the per-Q-head dispatch shape
//! the GQA case needs.
//!
//! `head_dim` is hardcoded to 128: each lane owns `head_dim / BD = 4`
//! consecutive Q/K/V quartiles, the dot-product across `head_dim`
//! reduces via `simd_sum`, and the V accumulator stays in 4 thread-
//! local f32 registers throughout the n_kv walk.

use metaltile::{bench_kernel, kernel};

#[bench_kernel(
    op="sdpa",
    subop="sdpa_vector",
    class=SdpaVector,
    h=128,        // head_dim
    n_kv=4096,
    n_heads=32,   // n_q_heads
    gqa_factor=4, // 32 Q heads grouped onto 8 KV heads
    batch=1,
    tpg=1024,     // BN × BD = 32 × 32
    tol=1e-3,
    metal_file="scaled_dot_product_attention.metal",
)]
#[kernel]
pub fn mt_sdpa_vector<T>(
    q: Tensor<T>,
    k: Tensor<T>,
    v: Tensor<T>,
    out: Tensor<T>,
    #[constexpr] head_dim: u32,
    #[constexpr] n_kv: u32,
    #[constexpr] gqa_factor: u32,
    #[constexpr] scale: f32,
) {
    let q_head = tgid_x;
    let kv_head = q_head / gqa_factor;
    let sg = simd_id;
    let lane = simd_lane;
    let ns = n_simd;

    // 32 slots per simdgroup-reduction array; 1024 slots per output
    // accumulator (one per thread). Matches mt_sdpa exactly.
    // Single 1024-float tg array reused 4× in the output reduction loop.
    // Mirrors MLX's pattern (~4 KB tg memory vs our previous 16 KB across
    // 4 separate arrays). On M2 with 32 KB tg memory per SM, this lifts
    // concurrent TGs/SM from 2 → 7 — the missing occupancy factor that
    // capped bf16 single-pass at 62% MT despite vectorized loads firing
    // correctly. f16 and f32 also benefit but less (cast cost wasn't the
    // limit there).
    threadgroup_alloc("tg_max", 32);
    threadgroup_alloc("tg_sum", 32);
    threadgroup_alloc("tg_out", 1024);

    let q_off = q_head * head_dim;
    let kv_base = kv_head * n_kv * head_dim;
    let d0 = lane * 4u32;

    // Each lane pre-scales its 4 query elements once. K/V are streamed.
    let q0 = load(q[q_off + d0]).cast::<f32>() * scale;
    let q1 = load(q[q_off + d0 + 1u32]).cast::<f32>() * scale;
    let q2 = load(q[q_off + d0 + 2u32]).cast::<f32>() * scale;
    let q3 = load(q[q_off + d0 + 3u32]).cast::<f32>() * scale;

    let mut run_max = neg_infinity();
    let mut run_sum = 0.0f32;
    let mut o0 = 0.0f32;
    let mut o1 = 0.0f32;
    let mut o2 = 0.0f32;
    let mut o3 = 0.0f32;

    // Each simdgroup walks every (n_simd)th KV position. Per iteration:
    // simd_sum reduces the per-lane quartile dot product into a full
    // score, then we apply the online-softmax update and accumulate
    // the V quartile.
    //
    // Pre-compute the 4 KV element indices BEFORE the loads so the four
    // Op::Load ops land consecutively in IR. Vectorize requires the loads
    // to be back-to-back (no BinOp/Const/Arith interleaved between them).
    // Cast the raw loads to f32 in a separate 4-op run after — same shape
    // as `sdpa_decode_2pass_pass1`. Without this, the K loads ended up
    // interleaved with `q_i * cast(...)` arithmetic, vectorize never fired,
    // and bf16 stayed at ~35% MT of MLX on M2 mini.
    for _t in range(sg, n_kv, ns) {
        let base = kv_base + _t * head_dim;
        let kv_idx = base + d0;
        let kv0 = kv_idx;
        let kv1 = kv_idx + 1u32;
        let kv2 = kv_idx + 2u32;
        let kv3 = kv_idx + 3u32;
        let k0_raw = load(k[kv0]);
        let k1_raw = load(k[kv1]);
        let k2_raw = load(k[kv2]);
        let k3_raw = load(k[kv3]);
        let k0 = k0_raw.cast::<f32>();
        let k1 = k1_raw.cast::<f32>();
        let k2 = k2_raw.cast::<f32>();
        let k3 = k3_raw.cast::<f32>();
        let partial = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        let score = simd_sum(partial);
        let new_max = select(score > run_max, score, run_max);
        let factor = exp(run_max - new_max);
        let weight = exp(score - new_max);
        run_sum = run_sum * factor + weight;
        run_max = new_max;
        let v0_raw = load(v[kv0]);
        let v1_raw = load(v[kv1]);
        let v2_raw = load(v[kv2]);
        let v3_raw = load(v[kv3]);
        let v0 = v0_raw.cast::<f32>();
        let v1 = v1_raw.cast::<f32>();
        let v2 = v2_raw.cast::<f32>();
        let v3 = v3_raw.cast::<f32>();
        o0 = o0 * factor + weight * v0;
        o1 = o1 * factor + weight * v1;
        o2 = o2 * factor + weight * v2;
        o3 = o3 * factor + weight * v3;
    }

    // ── Cross-simdgroup reduction: max + sum_exp ───────────────────
    if lane == 0 {
        threadgroup_store("tg_max", sg, run_max);
        threadgroup_store("tg_sum", sg, run_sum);
    }
    threadgroup_barrier();
    if sg == 0 {
        let g_max_in = select(lane < ns, threadgroup_load("tg_max", lane), neg_infinity());
        let g_max = simd_max(g_max_in);
        let g_sum_in =
            select(lane < ns, threadgroup_load("tg_sum", lane) * exp(g_max_in - g_max), 0.0f32);
        let g_sum = simd_sum(g_sum_in);
        if lane == 0 {
            threadgroup_store("tg_max", 0, g_max);
            threadgroup_store("tg_sum", 0, g_sum);
        }
    }
    threadgroup_barrier();

    // ── Cross-simdgroup reduction: outputs (one shared tg array, 4 iters) ─
    //
    // Mirrors MLX's `sdpa_vector` reduction. Each iteration handles one of
    // the 4 output elements per lane:
    //   1. Every (lane, sg) writes its partial to tg_out[lane*ns + sg]
    //   2. barrier
    //   3. transpose-load (read tg_out[sg*ns + lane]) × per-sg factor,
    //      then simd_sum across the 32 lanes of THIS simdgroup
    //   4. lane 0 of each sg holds the reduced value for output position
    //      sg*4 + i (i is the loop variable below) — same layout as the
    //      input load pattern (each lane covers head_dim/32 = 4 elements
    //      indexed by lane*4)
    //
    // Per-iter f32 store + 32-lane simd_sum reduction = 1 KB tg memory
    // touched, reused 4 times. vs the old layout which pre-allocated 4 KB ×
    // 4 arrays = 16 KB resident throughout the kernel.
    let g_max = threadgroup_load("tg_max", 0);
    let g_sum = threadgroup_load("tg_sum", 0);
    let factor_g = exp(run_max - g_max);
    let inv_sum = select(g_sum > 0.0f32, 1.0f32 / g_sum, 0.0f32);

    threadgroup_store("tg_out", lane * ns + sg, o0);
    threadgroup_barrier();
    let r0_in = threadgroup_load("tg_out", sg * ns + lane) * factor_g;
    let red0 = simd_sum(r0_in) * inv_sum;
    threadgroup_barrier();

    threadgroup_store("tg_out", lane * ns + sg, o1);
    threadgroup_barrier();
    let r1_in = threadgroup_load("tg_out", sg * ns + lane) * factor_g;
    let red1 = simd_sum(r1_in) * inv_sum;
    threadgroup_barrier();

    threadgroup_store("tg_out", lane * ns + sg, o2);
    threadgroup_barrier();
    let r2_in = threadgroup_load("tg_out", sg * ns + lane) * factor_g;
    let red2 = simd_sum(r2_in) * inv_sum;
    threadgroup_barrier();

    threadgroup_store("tg_out", lane * ns + sg, o3);
    threadgroup_barrier();
    let r3_in = threadgroup_load("tg_out", sg * ns + lane) * factor_g;
    let red3 = simd_sum(r3_in) * inv_sum;

    // lane 0 of each simdgroup writes its 4 elements. The output position
    // is sg-indexed now (was lane-indexed in the old layout), matching
    // MLX. f32→T narrowing is implicit at the MSL Store.
    if lane == 0u32 {
        let out_off = q_off + sg * 4u32;
        store(out[out_off], red0);
        store(out[out_off + 1u32], red1);
        store(out[out_off + 2u32], red2);
        store(out[out_off + 3u32], red3);
    }
}
