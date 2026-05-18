//! Codegen pass-pipeline timing.
//!
//! Runs the standard pass pipeline over every registered `BenchSpec` × dtype
//! `ITERS` times and prints per-pass median wall_us aggregated across the
//! whole corpus. Used as before/after proof for codegen-perf changes:
//!
//! ```text
//! cargo run --release --example pass_perf -p metaltile-std
//! ```
//!
//! Output schema: `pass_name  median_total_us  median_per_kernel_us`. Compare
//! the same pass row across before/after runs at the same git tree state of
//! everything else.

use std::time::Instant;

use metaltile_codegen::passes::{PassStats, PipelineBuilder, run_passes_with_stats};
use metaltile_std::{bench_types::DType, spec::BenchSpec};

const ITERS: usize = 25;
const DTYPES: &[DType] = &[DType::F32, DType::F16, DType::BF16];

fn main() {
    let specs: Vec<&BenchSpec> = inventory::iter::<BenchSpec>().collect();
    let kernels: Vec<_> = specs
        .iter()
        .flat_map(|s| {
            DTYPES
                .iter()
                .filter(|dt| s.dtypes.contains(dt))
                .map(|dt| (s.subop, *dt, (s.kernel_ir)(*dt)))
        })
        .collect();
    eprintln!("{} kernels × {} iters", kernels.len(), ITERS);

    // Per-iter, per-pass total wall_us across the whole corpus.
    let mut iter_totals: Vec<Vec<u64>> = Vec::with_capacity(ITERS);
    let overall_start = Instant::now();
    for _ in 0..ITERS {
        let mut pass_totals: Vec<u64> = Vec::new();
        let mut pass_names: Vec<String> = Vec::new();
        for (_, _, k) in &kernels {
            let mut kc = k.clone();
            let pipeline = PipelineBuilder::standard().build();
            let stats: Vec<PassStats> = match run_passes_with_stats(&mut kc, &pipeline) {
                Ok(s) => s,
                Err(_) => continue,
            };
            if pass_totals.is_empty() {
                pass_totals = vec![0u64; stats.len()];
                pass_names = stats.iter().map(|s| s.name.clone()).collect();
            }
            for (i, s) in stats.iter().enumerate() {
                pass_totals[i] += s.wall_us;
            }
        }
        if iter_totals.is_empty() {
            iter_totals = (0..pass_names.len()).map(|_| Vec::with_capacity(ITERS)).collect();
            // Stash names in the first slot's header line.
            println!("# pass_name  median_total_us  median_per_kernel_us");
            for n in &pass_names {
                print!("{n}\t");
            }
            println!();
        }
        for (i, t) in pass_totals.iter().enumerate() {
            iter_totals[i].push(*t);
        }
    }
    let overall = overall_start.elapsed();

    let mut pass_names: Vec<String> = Vec::new();
    // Re-derive names from a single dry run (cheap).
    if let Some((_, _, k)) = kernels.first() {
        let mut kc = k.clone();
        let pipeline = PipelineBuilder::standard().build();
        if let Ok(stats) = run_passes_with_stats(&mut kc, &pipeline) {
            pass_names = stats.iter().map(|s| s.name.clone()).collect();
        }
    }

    let n_kernels = kernels.len() as f64;
    println!("\n# results (median across {ITERS} iters)");
    println!("{:<24}  {:>14}  {:>16}", "pass", "median_us", "median_us/kernel");
    for (i, name) in pass_names.iter().enumerate() {
        let mut samples = iter_totals[i].clone();
        samples.sort_unstable();
        let median = samples[samples.len() / 2];
        let per_kernel = median as f64 / n_kernels;
        println!("{name:<24}  {median:>14}  {per_kernel:>16.1}");
    }
    eprintln!("\nwall elapsed: {:.2}s", overall.as_secs_f64());
}
