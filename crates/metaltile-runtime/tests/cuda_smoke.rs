//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Phase-1 CUDA smoke test (CUDA_BACKEND_SPEC §5.1): prove the pipeline
//! end-to-end on a real NVIDIA device —
//!   IR → CudaGenerator (CUDA C++) → NVRTC (PTX) → module → launch →
//!   read-back → compare against the CPU oracle.
//!
//! Runs only with `--features cuda` on a CUDA host (the GX10 / sm_121).
//! When no device is present, it skips (no failure) so CI without a GPU
//! is unaffected.
#![cfg(feature = "cuda")]

use std::ffi::c_void;

use metaltile_codegen::{CodegenBackend, CudaGenerator};
use metaltile_core::{
    dtype::DType,
    ir::{BinOpKind, IndexExpr, Kernel, Param, ParamKind, Op, ValueId},
    shape::Shape,
};
use metaltile_runtime::CudaDevice;

/// out[i] = a[i] + b[i]  (KernelMode::Elementwise, f32). Mirrors the
/// codegen `vector_add` reference kernel.
fn vector_add_ir() -> Kernel {
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

fn f32s_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}
fn bytes_to_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn vector_add_cuda_end_to_end() {
    let Some(dev) = CudaDevice::create().expect("CUDA init") else {
        eprintln!("no CUDA device — skipping CUDA smoke test");
        return;
    };
    let (maj, min) = dev.compute_capability();
    eprintln!("CUDA device compute capability: sm_{maj}{min}");

    // 1. IR → CUDA C++.
    let src = CudaGenerator::new().generate(&vector_add_ir()).expect("cuda codegen");
    eprintln!("--- generated CUDA ---\n{src}\n----------------------");

    // 2. NVRTC compile + load.
    let module = dev.compile(&src, "vector_add.cu").expect("nvrtc compile");
    let func = module.function("vector_add").expect("get function");

    // 3. Host data + CPU oracle.
    const N: usize = 4096;
    let a: Vec<f32> = (0..N).map(|i| (i as f32 * 0.013 - 0.4).sin() * 1.2).collect();
    let b: Vec<f32> = (0..N).map(|i| (i as f32 * 0.017 + 0.1).cos() * 0.8).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    // 4. Upload, allocate output, launch.
    let da = dev.upload(&f32s_to_bytes(&a)).expect("upload a");
    let db = dev.upload(&f32s_to_bytes(&b)).expect("upload b");
    let dc = dev.alloc(N * 4).expect("alloc c");

    let mut pa = da.device_ptr();
    let mut pb = db.device_ptr();
    let mut pc = dc.device_ptr();
    let mut n: u32 = N as u32;
    let mut args: [*mut c_void; 4] = [
        &mut pa as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut pc as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (N as u32).div_ceil(block);
    dev.launch_1d(func, grid, block, &mut args).expect("launch");

    // 5. Read back + compare to oracle.
    let mut out_bytes = vec![0u8; N * 4];
    dev.download(&dc, &mut out_bytes).expect("download c");
    let got = bytes_to_f32s(&out_bytes);

    let mut max_err = 0.0f32;
    for (g, e) in got.iter().zip(&expected) {
        max_err = max_err.max((g - e).abs());
    }
    eprintln!("max|Δ| = {max_err:.3e} over {N} elements");
    assert!(max_err <= 1e-6, "CUDA vector_add mismatch: max|Δ|={max_err:.3e}");
}
