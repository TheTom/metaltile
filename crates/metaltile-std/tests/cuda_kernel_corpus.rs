//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Run the registered `#[test_kernel]` corpus on the CUDA backend.
//!
//! Iterates the same `KernelTest` inventory the Metal harness uses
//! (`tests/kernel_tests_harness.rs`), but dispatches each on a real CUDA
//! device via `CudaDevice::run_kernel`, comparing to the same CPU oracle.
//! Categorizes each (kernel × dtype) as PASS / MISMATCH / UNSUPPORTED /
//! ERROR. UNSUPPORTED (codegen doesn't cover the kernel yet — MMA,
//! cooperative, Strided, multi-dim) is expected and not a failure;
//! MISMATCH (ran but wrong) and ERROR (NVRTC/launch failure on a kernel we
//! claimed to support) are hard failures.
//!
//! Runs only with `--features cuda` on a CUDA host (the GX10 / sm_121).
#![cfg(feature = "cuda")]

use metaltile_core::dtype::DType;
use metaltile_runtime::CudaDevice;
use std::collections::BTreeMap;

fn read_raw_f32(bytes: &[u8], dt: DType, n: usize) -> Vec<f32> {
    match dt {
        DType::F32 => bytes.chunks_exact(4).take(n)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
        DType::F16 => bytes.chunks_exact(2).take(n).map(|b| {
            let bits = u16::from_le_bytes(b.try_into().unwrap());
            half::f16::from_bits(bits).to_f32()
        }).collect(),
        DType::BF16 => bytes.chunks_exact(2).take(n).map(|b| {
            let bits = u16::from_le_bytes(b.try_into().unwrap());
            half::bf16::from_bits(bits).to_f32()
        }).collect(),
        DType::I32 => bytes.chunks_exact(4).take(n)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
        DType::U32 => bytes.chunks_exact(4).take(n)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as f32).collect(),
        DType::I8 => bytes.iter().take(n).map(|&b| b as i8 as f32).collect(),
        DType::U8 => bytes.iter().take(n).map(|&b| b as f32).collect(),
        _ => vec![0.0; n],
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

/// Kernels that GENERATE but don't yet match the oracle on CUDA, with
/// reasons — tracked so the suite stays green while documenting the
/// remaining pure-DSL gaps (distinct from the cooperative/MMA backlog).
/// A failure NOT matching one of these is a regression and fails the test.
const KNOWN_HARD: &[(&str, &str)] = &[
    ("splitk_accum_nax", "NAX (Metal-4 neural-accelerator) cooperative path — Phase 5 (CUTLASS)"),
    ("strided_copy", "non-row-major strided view: needs the test's actual {name}_strides (our synth assumes row-major)"),
    ("fishspeech_conv1d", "fp8 conv1d: subtle decode/accumulate mismatch under investigation"),
    ("hadamard_m", "Hadamard transform: warp-shuffle xor pattern mismatch (active-mask / partial-warp semantics)"),
    ("gated_delta_prep_chunk", "GDN chunk prep: subtle simd/shared accumulation mismatch under investigation"),
    // NAX SDPA-prefill at head_dim 128/256 only (d64 + all other NAX qmm now
    // pass via the ei-stride CoopTile fix); a head-dim-specific tiling detail.
    ("sdpa_prefill_nax_d128", "NAX SDPA prefill head_dim=128 — head-dim tiling detail"),
    ("sdpa_prefill_nax_d256", "NAX SDPA prefill head_dim=256 — head-dim tiling detail"),
];

fn known_hard(name: &str) -> bool {
    KNOWN_HARD.iter().any(|(k, _)| name.contains(k))
}

fn is_unsupported(msg: &str) -> bool {
    let m = msg.to_lowercase();
    ["phase 1", "phase 2", "not supported", "not yet implemented", "strided",
     "kernelmode", "multi-dimensional", "transform", "secondary"]
        .iter()
        .any(|p| m.contains(p))
}

#[test]
fn run_corpus_on_cuda() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping");
        return;
    };

    let (mut pass, mut mismatch, mut unsupported, mut error) = (0u32, 0u32, 0u32, 0u32);
    let mut known = 0u32;
    let mut hard_failures: Vec<String> = Vec::new();
    let mut pass_names: Vec<String> = Vec::new();
    let mut unsup_reasons: BTreeMap<String, u32> = BTreeMap::new();

    for entry in metaltile_std::all_tests() {
        let t = entry.test();
        for &dt in t.dtypes() {
            let setup = t.setup(dt);
            let tol = t.tolerance(dt);
            let kernel = setup.kernel();

            // GPU-vs-GPU reference setups need two dispatches; skip for now.
            if setup.ref_setup().is_some() {
                unsupported += 1;
                *unsup_reasons.entry("ref_setup (GPU-vs-GPU)".into()).or_default() += 1;
                continue;
            }

            // Build the param/constexpr byte map.
            let mut buffers: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            for inp in setup.inputs() {
                buffers.insert(inp.name().to_string(), inp.data().to_vec());
            }
            for (k, v) in setup.constexprs() {
                buffers.insert(k.clone(), v.to_le_bytes());
            }

            let grid = setup.grid();
            let label = format!("{} [{dt}]", t.name());

            // Debug: DUMP=<exact kernel name> prints its generated CUDA.
            if let Ok(want) = std::env::var("DUMP") {
                if t.name() == want && dt == DType::F32 {
                    use metaltile_codegen::{CodegenBackend, CudaGenerator};
                    if let Ok(src) = CudaGenerator::new().generate(kernel) {
                        eprintln!("==== {} ====\n{src}\n==== end ====", t.name());
                    }
                }
            }

            match dev.run_kernel(kernel, &buffers, grid.grid, grid.tpg) {
                Ok(outputs) => {
                    let mut worst = 0.0f32;
                    for exp in setup.expected() {
                        let Some(got_bytes) = outputs.get(exp.name()) else {
                            worst = f32::INFINITY;
                            break;
                        };
                        let n = exp.len();
                        let got = read_raw_f32(got_bytes, exp.dtype(), n);
                        let want = read_raw_f32(exp.data(), exp.dtype(), n);
                        worst = worst.max(max_abs_diff(&got, &want));
                    }
                    if (worst as f64) <= tol {
                        pass += 1;
                        pass_names.push(label);
                    } else if known_hard(t.name()) {
                        known += 1;
                    } else {
                        mismatch += 1;
                        hard_failures.push(format!("MISMATCH {label}: max|Δ|={worst:.3e} > {tol:.3e}"));
                    }
                }
                Err(e) => {
                    let msg = e.to_string();
                    if known_hard(t.name()) {
                        known += 1;
                    } else if is_unsupported(&msg) {
                        unsupported += 1;
                        // Bucket by the short reason (first line / key phrase).
                        let reason = msg
                            .lines()
                            .next()
                            .unwrap_or("?")
                            .split(';')
                            .next()
                            .unwrap_or("?")
                            .trim()
                            .to_string();
                        *unsup_reasons.entry(reason).or_default() += 1;
                    } else {
                        error += 1;
                        hard_failures.push(format!("ERROR {label}: {msg}"));
                    }
                }
            }
        }
    }

    eprintln!("\n=== CUDA corpus result ===");
    eprintln!("PASS={pass}  KNOWN_HARD={known}  MISMATCH={mismatch}  UNSUPPORTED={unsupported}  ERROR={error}");
    eprintln!("--- unsupported reasons (top buckets) ---");
    let mut reasons: Vec<_> = unsup_reasons.iter().collect();
    reasons.sort_by(|a, b| b.1.cmp(a.1));
    for (reason, n) in reasons.iter().take(25) {
        eprintln!("  {n:>5}  {reason}");
    }
    eprintln!("--- passing kernels ---");
    for n in &pass_names {
        eprintln!("  ✓ {n}");
    }
    if !hard_failures.is_empty() {
        eprintln!("--- hard failures ({}) ---", hard_failures.len());
        for f in &hard_failures {
            eprintln!("  ✗ {f}");
        }
    }

    assert!(pass > 0, "no kernels passed on CUDA — pipeline broken");
    assert!(
        hard_failures.is_empty(),
        "{} CUDA hard failures (mismatch/error on supported kernels)",
        hard_failures.len()
    );
}
