#!/usr/bin/env python3
"""
generate_figures.py — Generate paper figures and LaTeX tables from results.

Outputs:
  - fig1a_scale_degradation.pdf — Line plot: accuracy vs DB count
  - fig1b_ambiguity_degradation.pdf — Bar chart: accuracy vs ambiguity bin
  - table_multi_model.tex — LaTeX table comparing all models
  - table_ablation_cross_model.tex — LaTeX ablation matrix

Usage:
    python eval/scripts/generate_figures.py
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available. Will generate tables only.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
FIGURES_DIR = PROJECT_ROOT / "eval" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

MODEL_DISPLAY_NAMES = {
    "nim_gptoss": "GPT-OSS-120B (NIM)",
    "bedrock_sonnet46": "Claude Sonnet 4.6",
    "bedrock_opus46": "Claude Opus 4.6",
    "bedrock_qwen3_80b": "Qwen 3 Next 80B",
    "bedrock_llama4_maverick": "Llama 4 Maverick 17B",
}

def load_results(subdir: str, pattern: str = "*.json") -> list[dict]:
    """Load all JSON result files from a subdirectory."""
    results = []
    result_dir = RESULTS_DIR / subdir
    if not result_dir.exists():
        return results
    for f in sorted(result_dir.glob(pattern)):
        try:
            data = json.loads(f.read_text())
            data["_filename"] = f.name
            results.append(data)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [WARN] Could not load {f}: {e}")
    return results


def display_name(model_short: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_short, model_short)


# ── Table 1: Multi-Model Pipeline Comparison ────────────────────────────────

def generate_multi_model_table():
    """Generate LaTeX table comparing all models on the full pipeline."""
    results = load_results("multi_model", "pipeline_*_full_*.json")
    if not results:
        print("[SKIP] No multi-model results found")
        return

    print("\n=== Table: Multi-Model Pipeline Evaluation ===\n")

    rows = []
    for r in results:
        model = display_name(r.get("model_short", "?"))
        strict = r.get("strict_accuracy", 0) * 100
        relaxed = r.get("relaxed_accuracy", 0) * 100
        discovery = r.get("discovery_rate", 0) * 100
        ast = (r.get("ast_score", 0) or 0) * 100
        savings = r.get("token_efficiency", {}).get("savings_pct", 0)
        n = r.get("n", 0)
        rows.append((model, strict, relaxed, discovery, ast, savings, n))

    # Print table
    header = f"{'Model':<30s} {'N':>5s} {'Strict':>8s} {'Relaxed':>8s} {'Disc.':>7s} {'AST':>6s} {'Savings':>8s}"
    print(header)
    print("-" * len(header))
    for model, strict, relaxed, disc, ast, savings, n in sorted(rows, key=lambda x: -x[1]):
        print(f"{model:<30s} {n:>5d} {strict:>7.1f}% {relaxed:>7.1f}% {disc:>6.1f}% {ast:>5.1f}% {savings:>7.1f}%")

    # LaTeX
    tex_path = FIGURES_DIR / "table_multi_model.tex"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Multi-model pipeline evaluation (N=1,584, test split).}",
        r"\label{tab:multi-model}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Strict & Relaxed & Discovery & AST & Token Savings \\",
        r"\midrule",
    ]
    for model, strict, relaxed, disc, ast, savings, n in sorted(rows, key=lambda x: -x[1]):
        lines.append(f"{model} & {strict:.1f}\\% & {relaxed:.1f}\\% & {disc:.1f}\\% & {ast:.1f}\\% & {savings:.1f}\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path.write_text("\n".join(lines))
    print(f"\nSaved to {tex_path}")


# ── Table 2: Cross-Model Ablation ───────────────────────────────────────────

def generate_ablation_table():
    """Generate LaTeX ablation table across model backbones."""
    results = load_results("ablation_models")
    if not results:
        print("[SKIP] No ablation results found")
        return

    print("\n=== Table: Cross-Model Ablation ===\n")

    # Group by model and config
    matrix = defaultdict(dict)  # matrix[model][config] = relaxed_accuracy
    for r in results:
        model = display_name(r.get("model_short", "?"))
        config = r.get("config", {})
        config_parts = [k for k, v in config.items() if v]
        config_name = "_".join(config_parts) if config_parts else "full"
        relaxed = r.get("relaxed_accuracy", 0) * 100
        matrix[model][config_name] = relaxed

    configs = ["full", "no_kg", "no_embedding", "no_correction", "no_adaptive"]
    config_labels = ["None (full)", "No KG", "No Embedding", "No Correction", "No Adaptive"]

    models = sorted(matrix.keys())
    header = f"{'Component Removed':<20s}" + "".join(f" {m:>20s}" for m in models)
    print(header)
    print("-" * len(header))

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-model ablation study (N=200, test split, relaxed accuracy).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{l" + "r" * len(models) + "}",
        r"\toprule",
        "Component Removed & " + " & ".join(models) + r" \\",
        r"\midrule",
    ]

    for config, label in zip(configs, config_labels):
        vals = []
        for m in models:
            v = matrix[m].get(config, None)
            if v is not None:
                full_v = matrix[m].get("full", v)
                if config == "full":
                    vals.append(f"{v:.1f}%")
                else:
                    diff = v - full_v
                    vals.append(f"{diff:+.1f}%")
            else:
                vals.append("—")
        print(f"{label:<20s}" + "".join(f" {v:>20s}" for v in vals))
        tex_lines.append(f"{label} & " + " & ".join(v.replace('%', r'\%') for v in vals) + r" \\")

    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_path = FIGURES_DIR / "table_ablation_cross_model.tex"
    tex_path.write_text("\n".join(tex_lines))
    print(f"\nSaved to {tex_path}")


# ── Figure 1(a): Scale Degradation ──────────────────────────────────────────

def generate_scale_figure():
    """Generate line plot: accuracy vs DB count."""
    results = load_results("scale_exp")
    if not results:
        print("[SKIP] No scale experiment results found")
        return

    print("\n=== Figure 1(a): Scale Degradation ===\n")

    # Group by method+model and db count
    series = defaultdict(dict)  # series["pipeline_sonnet"][4] = accuracy
    for r in results:
        model = r.get("model_short", "?")
        experiment = r.get("experiment", "pipeline")
        max_dbs = r.get("max_dbs")
        if max_dbs is None:
            continue
        config = r.get("config", {})
        config_parts = [k for k, v in config.items() if v]
        method = "embed_only" if config_parts else experiment
        key = f"{method}_{model}"
        strict = r.get("strict_accuracy", 0) * 100
        relaxed = r.get("relaxed_accuracy", 0) * 100
        series[key][max_dbs] = {"strict": strict, "relaxed": relaxed}

    # Print table
    all_dbs = sorted(set(d for s in series.values() for d in s.keys()))
    header = f"{'Method':<40s}" + "".join(f" {d:>6d} DBs" for d in all_dbs)
    print(header)
    print("-" * len(header))
    for key in sorted(series.keys()):
        vals = []
        for d in all_dbs:
            v = series[key].get(d, {}).get("relaxed")
            vals.append(f"{v:>8.1f}%" if v is not None else "      —")
        print(f"{key:<40s}" + "".join(vals))

    if not HAS_MPL:
        print("\n[SKIP] matplotlib not available, cannot generate PDF")
        return

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    styles = {
        "pipeline": {"linestyle": "-", "marker": "o"},
        "embed_only": {"linestyle": "--", "marker": "s"},
        "blind_llm_baseline": {"linestyle": ":", "marker": "^"},
    }

    for key in sorted(series.keys()):
        parts = key.split("_", 1)
        method = parts[0] if parts[0] in styles else "pipeline"
        model_part = parts[1] if len(parts) > 1 else key
        style = styles.get(method, {"linestyle": "-", "marker": "o"})
        dbs = sorted(series[key].keys())
        strict_vals = [series[key][d]["strict"] for d in dbs]
        relaxed_vals = [series[key][d]["relaxed"] for d in dbs]

        label = f"{method} ({display_name(model_part)})"
        ax1.plot(dbs, strict_vals, label=label, **style)
        ax2.plot(dbs, relaxed_vals, label=label, **style)

    for ax, title in [(ax1, "Strict Accuracy"), (ax2, "Relaxed Accuracy")]:
        ax.set_xlabel("Number of Databases")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig1a_scale_degradation.pdf"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved to {fig_path}")


# ── Figure 1(b): Ambiguity Degradation ──────────────────────────────────────

def generate_ambiguity_figure():
    """Generate bar chart: accuracy vs ambiguity bin."""
    amb_file = RESULTS_DIR / "ambiguity" / "per_bin_accuracy.json"
    if not amb_file.exists():
        print("[SKIP] No ambiguity data found. Run analyze_ambiguity.py first.")
        return

    data = json.loads(amb_file.read_text())
    bins_order = ["low", "medium", "high"]
    methods = data.get("methods", {})

    print("\n=== Figure 1(b): Ambiguity Degradation ===\n")
    header = f"{'Method':<40s}" + "".join(f" {b:>10s}" for b in bins_order)
    print(header)
    print("-" * len(header))
    for method, bin_data in sorted(methods.items()):
        vals = [f"{bin_data[b]['strict_accuracy']:>9.1f}%" for b in bins_order]
        print(f"{method:<40s}" + "".join(vals))

    if not HAS_MPL:
        print("\n[SKIP] matplotlib not available, cannot generate PDF")
        return

    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(bins_order))
    width = 0.8 / max(len(methods), 1)

    for i, (method, bin_data) in enumerate(sorted(methods.items())):
        vals = [bin_data[b]["strict_accuracy"] for b in bins_order]
        ax.bar(x + i * width, vals, width, label=method)

    ax.set_xlabel("Schema Ambiguity")
    ax.set_ylabel("Strict Accuracy (%)")
    ax.set_title("Performance vs Schema Ambiguity")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(["Low\n(gap < 10%)", "Medium\n(10-30%)", "High\n(gap ≥ 30%)"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig1b_ambiguity_degradation.pdf"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved to {fig_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("GENERATING PAPER FIGURES AND TABLES")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Figures dir: {FIGURES_DIR}")
    print("=" * 70)

    generate_multi_model_table()
    generate_ablation_table()
    generate_scale_figure()
    generate_ambiguity_figure()

    print(f"\n{'=' * 70}")
    print("Done. Generated files:")
    for f in sorted(FIGURES_DIR.glob("*")):
        print(f"  {f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
