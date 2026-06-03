//! Copyright 2026 0xClandestine, Ekryski, TheTom, Ambisphaeric
//! SPDX-License-Identifier: Apache-2.0
//! Link config for the optional `cuda` feature: point the linker at the
//! CUDA toolkit's `libnvrtc` and the driver's `libcuda` (the stub under
//! the toolkit suffices for link time; the real driver loads at runtime).
//! No-op unless the `cuda` feature is enabled, so the macOS Metal build
//! is unaffected.

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return; // Metal-only / default build — nothing to link.
    }

    let cuda_root = std::env::var("CUDA_PATH")
        .or_else(|_| std::env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // `libnvrtc` lives in the toolkit lib dir; `libcuda` (driver) link stub
    // lives under `lib64/stubs`. Both 64-bit dir names are tried.
    for sub in ["lib64", "lib64/stubs", "lib", "lib/stubs"] {
        println!("cargo:rustc-link-search=native={cuda_root}/{sub}");
    }
    // Common driver locations (real libcuda.so on a GPU host).
    for p in ["/usr/lib/aarch64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"] {
        println!("cargo:rustc-link-search=native={p}");
    }

    println!("cargo:rustc-link-lib=dylib=nvrtc");
    println!("cargo:rustc-link-lib=dylib=cuda");
}
