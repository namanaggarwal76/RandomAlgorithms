"""Advanced analysis for Miller-Rabin benchmark results.

Generates:
- Heatmap of accuracy vs bits vs rounds
- Heatmap of false positive rate vs bits vs rounds
- Runtime scaling plots (avg, median, p95)
- Distribution (hist) of core times for selected bit size & rounds
- Accuracy vs theoretical error probability scatter
- False-positive rate vs rounds per strategy (log scale)
- Carmichael detection rate per strategy
- Recommended k vs bits per strategy
- Base collision fraction violin per strategy
- Output label distribution stacked bar
- Pseudoprime strength vs bits
- Deterministic vs probabilistic timing comparison

Usage:
    python analyze_miller_rabin.py --summary miller_rabin/results/miller_rabin_summary.csv --raw miller_rabin/results/miller_rabin_raw.csv --outdir miller_rabin/results/analysis_plots
"""
from __future__ import annotations
import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def heatmap_per_strategy(df, value_col, title_base, outdir, filename_prefix):
    if 'strategy' in df.columns:
        strategies = sorted(df['strategy'].unique())
    else:
        strategies = [None]
    for strat in strategies:
        sub = df if strat is None else df[df['strategy'] == strat]
        if sub.empty:
            continue
        # average across duplicates if any
        df2 = sub.groupby(['bits', 'rounds'], as_index=False)[value_col].mean()
        pivot = df2.pivot(index='bits', columns='rounds', values=value_col)
        plt.figure(figsize=(7,5))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
        title = title_base if strat is None else f"{title_base} (strategy={strat})"
        plt.title(title)
        plt.ylabel('Bits')
        plt.xlabel('Rounds (k)')
        plt.tight_layout()
        suffix = 'overall' if strat is None else strat
        plt.savefig(os.path.join(outdir, f"{filename_prefix}_{suffix}.png"), dpi=160)
        plt.close()


def runtime_plot(df, outdir):
    plt.figure(figsize=(7,5))
    sns.lineplot(data=df, x='rounds', y='avg_time_core', hue='bits', style='strategy', marker='o')
    plt.title('Avg Core Time vs Rounds (by strategy)')
    plt.ylabel('Avg Core Time (s)')
    plt.xlabel('Rounds (k)')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(outdir, 'avg_core_time_vs_rounds_by_strategy.png'), dpi=160)
    plt.close()

    if 'p95_time_core' in df.columns:
        plt.figure(figsize=(7,5))
        sns.lineplot(data=df, x='rounds', y='p95_time_core', hue='bits', style='strategy', marker='o')
        plt.title('P95 Core Time vs Rounds (by strategy)')
        plt.ylabel('P95 Core Time (s)')
        plt.xlabel('Rounds (k)')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(outdir, 'p95_core_time_vs_rounds_by_strategy.png'), dpi=160)
        plt.close()


def fp_rate_plot(df, outdir):
    plt.figure(figsize=(7,5))
    sns.lineplot(data=df, x='rounds', y='composite_only_fp_rate', hue='strategy', style='bits', marker='o')
    plt.yscale('log')
    plt.title('False-Positive Rate (composites only) vs Rounds')
    plt.ylabel('FP Rate (log)')
    plt.xlabel('Rounds (k)')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(outdir, 'fp_rate_vs_rounds.png'), dpi=160)
    plt.close()


def carmichael_plot(df, outdir):
    sub = df[df.carmichael_count > 0]
    if sub.empty:
        return
    plt.figure(figsize=(7,5))
    sns.lineplot(data=sub, x='rounds', y='carmichael_detection_rate', hue='strategy', style='bits', marker='o')
    plt.title('Carmichael Detection Rate vs Rounds')
    plt.ylabel('Detection Rate')
    plt.xlabel('Rounds (k)')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(outdir, 'carmichael_detection_rate.png'), dpi=160)
    plt.close()


def recommended_k_plot(summary_df, outdir, rec_csv_path):
    from benchmark_miller_rabin import recommend_k
    rec = recommend_k(summary_df)
    rec.to_csv(rec_csv_path, index=False)
    plt.figure(figsize=(7,5))
    sns.lineplot(data=rec, x='bits', y='recommended_k', hue='strategy', marker='o')
    plt.title('Recommended k vs Bits (FP<=1e-6, Acc>=0.999)')
    plt.ylabel('Recommended k')
    plt.xlabel('Bits')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(outdir, 'recommended_k_vs_bits.png'), dpi=160)
    plt.close()


def collision_violin(raw_df, outdir):
    sub = raw_df.dropna(subset=['base_collision_fraction'])
    if sub.empty:
        return
    plt.figure(figsize=(7,5))
    sns.violinplot(data=sub, x='strategy', y='base_collision_fraction')
    plt.title('Base Collision Fraction by Strategy')
    plt.ylabel('Collision Fraction')
    plt.xlabel('Strategy')
    plt.savefig(os.path.join(outdir, 'base_collision_violin.png'), dpi=160)
    plt.close()


def output_label_distribution(raw_df, outdir):
    df = raw_df.copy()
    df['label'] = df['probable_prime'].map({True: 'prime', False: 'composite'})
    agg = df.groupby(['bits', 'strategy', 'label']).size().reset_index(name='count')
    plt.figure(figsize=(8,5))
    sns.barplot(data=agg, x='bits', y='count', hue='label')
    plt.title('Output Label Distribution (Prime vs Composite)')
    plt.xlabel('Bits')
    plt.ylabel('Count')
    plt.savefig(os.path.join(outdir, 'output_label_distribution.png'), dpi=160)
    plt.close()


def pseudoprime_strength(raw_df, outdir):
    comp = raw_df[~raw_df.ground_truth]
    if comp.empty:
        return
    agg = comp.groupby(['bits', 'strategy']).pseudoprime_flag.mean().reset_index(name='pseudoprime_rate')
    plt.figure(figsize=(7,5))
    sns.lineplot(data=agg, x='bits', y='pseudoprime_rate', hue='strategy', marker='o')
    plt.title('Pseudoprime Rate (among composites)')
    plt.xlabel('Bits')
    plt.ylabel('Rate')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(outdir, 'pseudoprime_rate_vs_bits.png'), dpi=160)
    plt.close()


def deterministic_vs_prob_time(summary_df, outdir):
    sub = summary_df.copy()
    keep_cols = ['bits', 'strategy', 'rounds', 'avg_time_core', 'impl_sympy_time', 'impl_det64_time']
    sub = sub[keep_cols]
    long = sub.melt(id_vars=['bits', 'strategy', 'rounds'], value_vars=['avg_time_core', 'impl_sympy_time', 'impl_det64_time'], var_name='impl', value_name='time')
    plt.figure(figsize=(9,5))
    sns.lineplot(data=long, x='rounds', y='time', hue='impl', style='bits', marker='o')
    plt.title('Implementation Timing Comparison')
    plt.xlabel('Rounds (k)')
    plt.ylabel('Time (s)')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(outdir, 'implementation_timing_comparison.png'), dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--summary', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--outdir', default='miller_rabin/results/analysis_plots')
    ap.add_argument('--dist-bits', type=int, default=32, help='Bit size for time distribution plot')
    ap.add_argument('--dist-rounds', type=int, default=10, help='Rounds for time distribution plot')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    summary_df = pd.read_csv(args.summary)
    raw_df = pd.read_csv(args.raw)

    # Heatmaps per strategy to avoid duplicates across strategies
    heatmap_per_strategy(summary_df, 'accuracy', 'Accuracy Heatmap', args.outdir, 'heatmap_accuracy')
    if 'composite_only_fp_rate' in summary_df.columns:
        heatmap_per_strategy(summary_df, 'composite_only_fp_rate', 'FP Rate (Composites Only) Heatmap', args.outdir, 'heatmap_fp_rate_composites')
    heatmap_per_strategy(summary_df, 'stability_variance', 'Stability Variance Heatmap', args.outdir, 'heatmap_stability_variance')

    runtime_plot(summary_df, args.outdir)
    fp_rate_plot(summary_df, args.outdir)
    carmichael_plot(summary_df, args.outdir)
    recommended_k_plot(summary_df, args.outdir, os.path.join(args.outdir, 'miller_rabin_recommended_k.csv'))
    collision_violin(raw_df, args.outdir)
    output_label_distribution(raw_df, args.outdir)
    pseudoprime_strength(raw_df, args.outdir)
    deterministic_vs_prob_time(summary_df, args.outdir)

    print(f"Analysis plots written to {args.outdir}")

if __name__ == '__main__':
    main()
