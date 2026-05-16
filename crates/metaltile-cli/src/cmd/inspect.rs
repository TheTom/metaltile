//! `tile inspect` — Print IR and/or MSL for kernels.
//!
//! Auto-discovers kernels via inventory — no hardcoded list.
//!
//! Usage:
//!   tile inspect                           # list all registered kernels
//!   tile inspect <kernel>                  # print final MSL
//!   tile inspect <kernel> --ir             # print raw IR
//!   tile inspect <kernel> -o /tmp/out      # write .metal file
//!   tile inspect --all -o /tmp/out         # dump every kernel to disk

use std::collections::BTreeMap;

use metaltile_bench::{
    ops::DType,
    spec::BenchSpec,
    term::{Color, Style, paint_stdout},
};
use metaltile_codegen::{
    msl::{MslConfig, MslGenerator},
    TileSchedule,
};
use metaltile_core::ir::KernelMode;

use crate::{flag_present, flag_val, matches_filter, positional};

pub fn run(args: &[String]) {
    let dir = flag_val(args, "--dir").or_else(|| flag_val(args, "-o"));
    let filter = flag_val(args, "--filter")
        .or_else(|| positional(args));
    let all_flag = flag_present(args, "--all");

    // Collect all specs and group by kernel_name (dedup).
    let mut kernels: BTreeMap<&str, (&BenchSpec, Vec<DType>)> = BTreeMap::new();
    for spec in inventory::iter::<BenchSpec> {
        let entry = kernels.entry(spec.kernel_name).or_insert_with(|| (spec, Vec::new()));
        for &dt in spec.dtypes {
            if !entry.1.contains(&dt) {
                entry.1.push(dt);
            }
        }
    }

    if kernels.is_empty() {
        eprintln!("No kernels registered.");
        return;
    }

    // Sort kernel names for stable output.
    let mut sorted: Vec<(&str, (&BenchSpec, Vec<DType>))> =
        kernels.into_iter().collect();
    sorted.sort_unstable_by_key(|(name, _)| *name);

    // --all flag: dump every kernel (to stdout or dir)
    if all_flag {
        for (name, (spec, dtypes)) in &sorted {
            let msl = generate_msl_for_all_dtypes(spec, dtypes);
            if let Some(ref d) = dir {
                let path = format!("{}/{}.metal", d, name);
                std::fs::create_dir_all(d).expect("failed to create output directory");
                std::fs::write(&path, &msl).expect("write failed");
                println!("wrote {path}");
            } else {
                let mode_str = mode_label(first_mode(spec, dtypes));
                println!(
                    "// ═══════════════════════════════════════════════════════"
                );
                println!("// kernel: {}  mode: {}", name, mode_str);
                println!(
                    "// ═══════════════════════════════════════════════════════"
                );
                println!("{}", msl);
            }
        }
        return;
    }

    // No filter: list all kernels
    let Some(filter) = &filter else {
        println!(
            "{}",
            paint_stdout(
                "Kernels registered:",
                Style::new().fg(Color::BrightBlack).bold(),
            ),
        );
        println!();
        for (name, (spec, dtypes)) in &sorted {
            let dtype_str = dtypes
                .iter()
                .map(|dt| dtype_label(*dt))
                .collect::<Vec<_>>()
                .join("/");
            let mode_str = mode_label(first_mode(spec, dtypes));
            println!(
                "  {}   {}   {dtype_str}",
                paint_stdout(format!("{name:<20}"), Style::new().fg(Color::Cyan).bold()),
                paint_stdout(mode_str, Style::new().fg(Color::BrightBlack)),
            );
        }
        println!(
            "\n  {}",
            paint_stdout(
                format!("{} kernels", sorted.len()),
                Style::new().fg(Color::BrightBlack),
            ),
        );
        println!(
            "  {}",
            paint_stdout(
                "Run 'tile inspect <kernel>' to see MSL.",
                Style::new().fg(Color::BrightBlack),
            ),
        );
        return;
    };

    // Filter by kernel name.
    let matched: Vec<_> = sorted
        .iter()
        .filter(|(name, _)| matches_filter(Some(filter), name))
        .collect();

    if matched.is_empty() {
        eprintln!(
            "{} {}",
            paint_stdout("error:", Style::new().fg(Color::Red).bold()),
            paint_stdout(
                format!("no kernel matched '{filter}'"),
                Style::new().fg(Color::BrightWhite),
            ),
        );
        eprintln!(
            "\n{} {}",
            paint_stdout("Available:", Style::new().fg(Color::BrightBlack)),
            paint_stdout(
                sorted.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", "),
                Style::new().fg(Color::BrightWhite),
            ),
        );
        std::process::exit(1);
    }

    for (name, (spec, dtypes)) in &matched {
        let msl = generate_msl_for_all_dtypes(spec, dtypes);
        if let Some(ref d) = dir {
            let path = format!("{}/{}.metal", d, name);
            std::fs::create_dir_all(d).expect("failed to create output directory");
            std::fs::write(&path, &msl).expect("write failed");
            println!("wrote {path}");
        } else {
            let mode_str = mode_label(first_mode(spec, dtypes));
            println!(
                "// ═══════════════════════════════════════════════════════"
            );
            println!("// kernel: {}  mode: {}", name, mode_str);
            println!(
                "// ═══════════════════════════════════════════════════════"
            );
            println!("{}", msl);
        }
    }
}

/// Determine the primary kernel mode for display purposes.
fn first_mode(spec: &BenchSpec, _dtypes: &[DType]) -> KernelMode {
    // Determine mode from the BenchDispatch.
    // Generic dispatch → check the first shape's mode.
    match &spec.dispatch {
        metaltile_bench::spec::BenchDispatch::Generic => {
            spec.shapes.first().map(|s| s.mode).unwrap_or(KernelMode::Elementwise)
        },
        metaltile_bench::spec::BenchDispatch::Sort { .. }
        | metaltile_bench::spec::BenchDispatch::Scan { .. }
        | metaltile_bench::spec::BenchDispatch::ArgReduce { .. }
        | metaltile_bench::spec::BenchDispatch::QuantizedMatVec { .. }
        | metaltile_bench::spec::BenchDispatch::Attention { .. } => KernelMode::Reduction,
        metaltile_bench::spec::BenchDispatch::Random { .. }
        | metaltile_bench::spec::BenchDispatch::FpQuantized { .. } => KernelMode::Elementwise,
        metaltile_bench::spec::BenchDispatch::Rope { .. }
        | metaltile_bench::spec::BenchDispatch::StridedCopy { .. } => KernelMode::Grid3D,
    }
}

fn mode_label(mode: KernelMode) -> &'static str {
    match mode {
        KernelMode::Elementwise => "Elementwise",
        KernelMode::Reduction => "Reduction",
        KernelMode::Tile2D => "Tile2D",
        KernelMode::Grid3D => "Grid3D",
    }
}

fn dtype_label(dt: DType) -> &'static str {
    match dt {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::I32 => "i32",
        DType::U32 => "u32",
        DType::I8 => "i8",
        DType::U8 => "u8",
        _ => "?",
    }
}

/// Generate MSL for a kernel, trying the first supported dtype.
fn generate_msl_for_all_dtypes(spec: &BenchSpec, dtypes: &[DType]) -> String {
    let dt = dtypes.first().copied().unwrap_or(DType::F32);
    let mut k = (spec.kernel_ir)(dt);
    let mode = first_mode(spec, dtypes);
    k.mode = mode;

    // Use simd_matrix config for Tile2D kernels (matmul).
    let generator: MslGenerator = if matches!(mode, KernelMode::Tile2D) {
        MslGenerator::new(MslConfig {
            tile_schedule: TileSchedule::default(),
            use_simd_matrix: true,
            ..MslConfig::default()
        })
    } else {
        MslGenerator::default()
    };

    generator.generate(&k).unwrap_or_else(|e| format!("// ERROR: {e}\n"))
}
