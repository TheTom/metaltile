//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! FFAI Q4 block dequant — expand a resident Q4 weight back to a dense
//! `[m, k]` f16/f32 slab for the compute-bound prefill GEMM path.
//!
//! The decode path keeps weights Q4-resident (1 nibble/weight) and runs
//! the inline-dequant `gemv_q4`, which is bandwidth-bound — perfect at
//! batch 1. Prefill is COMPUTE-bound: many tokens reuse each weight, so
//! the right move is to dequant the weight ONCE to f16 and feed the
//! tensor-core `ffai_gemm`. This kernel is that one-time expansion.
//!
//! ## Q4 block layout (matches `ffai_ops::quantize_q4`)
//!
//! Per row `r`, the `k` columns are grouped into `bpr = k/32` blocks of
//! 32 values. Each block is 4 packed u32 words (`word` 0..3), each word
//! holding 8 signed 4-bit nibbles (`i` 0..7) at bit `i*4`. The dequant
//! is `value = signed_nibble * scale`, `scale = scales[r*bpr + b]`
//! (`amax/7`, symmetric, no zero-point/bias).
//!
//! ```text
//!   qs      [m * (k/32) * 4]   u32   — 4 words/block, 8 nibbles/word
//!   scales  [m * (k/32)]       f32   — per-block scale
//!   out     [m * k]            T     — dense dequantized weight
//! ```
//!
//! ## Dispatch
//!
//! 1D grid, one thread per output value. Thread `i` → row `r = i/k`,
//! col `c = i%k`; block `b = c/32`, within-block `j = c%32`; word
//! `w = j/8`, nibble `n = j%8`. Reads one u32 word + one scale.
//!
//! Signed-nibble reconstruction without a bit_cast/arithmetic-shift
//! intrinsic: extract the 4-bit field `nib = (word >> (n*4)) & 0xF`,
//! then `select(nib >= 8, nib - 16, nib)` (the `gguf_dequant_q8_0`
//! sign trick at 4-bit width).

use metaltile::kernel;

// Scales are f16 (the resident projection/expert weights store amax/7 as f16
// via the loader's `tb_f16`). The kernel widens to f32 for the multiply.
#[kernel]
pub fn ffai_dequant_q4<T>(
    qs: Tensor<u32>,
    scales: Tensor<f16>,
    out: Tensor<T>,
    #[constexpr] k_in: u32,
    #[constexpr] n_values: u32,
    // Block offset into qs/scales (in 32-value Q4 blocks). Lets a caller dequant
    // a contiguous sub-slab of a larger pool (e.g. one MoE expert's rows) without
    // a separate tensor view: qs offset = blk_off*4 words, scales = blk_off.
    #[constexpr] blk_off: u32,
) {
    // Grid3D 1-D launch: program_id::<0>() = global linear thread id (gid_x).
    let i = program_id::<0>();
    if i < n_values {
        let r = i / k_in;
        let c = i - r * k_in;
        let bpr = k_in / 32u32;
        let b = c / 32u32;
        let j = c - b * 32u32; // within-block 0..31
        let w = j / 8u32; // word 0..3
        let n = j - w * 8u32; // nibble 0..7
        let blk = blk_off + r * bpr + b;
        let word = load(qs[blk * 4u32 + w]);
        let nib = (word >> (n * 4u32)) & 0xfu32;
        // Sign-extend 4-bit: values 8..15 represent -8..-1 (nib - 16).
        let q_signed = select(nib >= 8u32, nib - 16u32, nib);
        let q = q_signed.cast::<i32>().cast::<f32>();
        let d = load(scales[blk]).cast::<f32>();
        store(out[i], q * d);
    }
}

pub mod kernel_tests {
    use metaltile::{test::*, test_kernel};

    use super::ffai_dequant_q4;
    use crate::utils::pack_f32;

    /// Reference Q4 quantizer — mirrors `ffai_ops::quantize_q4` exactly:
    /// signed 4-bit, per-32-block scale = amax/7, 4 u32 words/block.
    fn quantize_q4(w: &[f32], m: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
        let bpr = k / 32;
        let mut qs = vec![0u32; m * bpr * 4];
        let mut scales = vec![0f32; m * bpr];
        for r in 0..m {
            for b in 0..bpr {
                let base = r * k + b * 32;
                let amax = (0..32).fold(0f32, |a, i| a.max(w[base + i].abs()));
                let d = amax / 7.0;
                scales[r * bpr + b] = d;
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                for word in 0..4 {
                    let mut packed = 0u32;
                    for i in 0..8 {
                        let q = (w[base + word * 8 + i] * inv).round().clamp(-7.0, 7.0) as i32;
                        packed |= ((q as u32) & 0xf) << (i * 4);
                    }
                    qs[r * bpr * 4 + b * 4 + word] = packed;
                }
            }
        }
        (qs, scales)
    }

    fn cpu_dequant(qs: &[u32], scales: &[f32], m: usize, k: usize) -> Vec<f32> {
        let bpr = k / 32;
        let mut out = vec![0f32; m * k];
        for r in 0..m {
            for b in 0..bpr {
                let d = scales[r * bpr + b];
                for word in 0..4 {
                    let packed = qs[r * bpr * 4 + b * 4 + word];
                    for i in 0..8 {
                        let nib = (packed >> (i * 4)) & 0xf;
                        let q = if nib >= 8 { nib as i32 - 16 } else { nib as i32 };
                        out[r * k + b * 32 + word * 8 + i] = q as f32 * d;
                    }
                }
            }
        }
        out
    }

    fn setup(m: usize, k: usize, dt: DType) -> TestSetup {
        let n = m * k;
        let values: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017 - 0.3).sin() * 2.1).collect();
        let (qs, scales) = quantize_q4(&values, m, k);
        // Scales are stored f16 (matches the resident loader). Round the oracle
        // through f16 so expected == GPU.
        let scales_f16: Vec<f32> = scales.iter().map(|&s| half::f16::from_f32(s).to_f32()).collect();
        let dequantized = cpu_dequant(&qs, &scales_f16, m, k);
        let qs_bytes: Vec<u8> = qs.iter().flat_map(|x| x.to_le_bytes()).collect();
        TestSetup::new(ffai_dequant_q4::kernel_ir_for(dt))
            .mode(KernelMode::Grid3D)
            .input(TestBuffer::from_vec("qs", qs_bytes, DType::U32))
            .input(TestBuffer::from_vec("scales", pack_f32(&scales, DType::F16), DType::F16))
            .input(TestBuffer::zeros("out", n, dt))
            .constexpr("k_in", k as u32)
            .constexpr("n_values", n as u32)
            .constexpr("blk_off", 0u32)
            .expect(TestBuffer::from_vec("out", pack_f32(&dequantized, dt), dt))
            .grid_3d((n as u32).div_ceil(256), 1, 1, [256, 1, 1])
    }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_ffai_dequant_q4_single_row(dt: DType) -> TestSetup { setup(1, 64, dt) }

    #[test_kernel(dtypes = [f32, f16, bf16], tol = [1e-4, 5e-3, 5e-2])]
    fn test_ffai_dequant_q4_slab(dt: DType) -> TestSetup { setup(48, 128, dt) }
}
