import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("/Users/simonekudia/Desktop/cs185/final_project/homework_spring2026/final_project_llm_rl")
OUT = BASE / "report_assets" / "part2_graphs"

RUNS = {
    "GRPO+rm445":           BASE / "part2_runs" / "grpo_rm445_metrics.jsonl",
    "GRPO+GOPO":            BASE / "part2_runs" / "grpo_gopo_rm445_metrics.jsonl",
    "GSPO+rm445":           BASE / "part2_runs" / "gspo_rm_445_metrics.jsonl",
    "GSPO+GOPO":            BASE / "part2_runs" / "gspo_gopo_rm_445_metrics.jsonl",
    "GSPO+ens(mean)":       BASE / "part2_runs" / "gspo_rm100_rm445_ensemble_mean_metrics.jsonl",
    "GSPO+ens(min)":        BASE / "part2_runs" / "gspo_rm100_rm445_ensemble_min_metrics.jsonl",
    "GSPO+ens(pess)":       BASE / "part2_runs" / "gspo_rm100_rm445_ensemble_pess_metrics.jsonl",
    "GSPO+GOPO+ens(mean)":  BASE / "part2_runs" / "gspo_gopo_rm100_rm445_ensemble_mean_metrics.jsonl",
    "GSPO+GOPO+ens(min)":   BASE / "part2_runs" / "gspo_gopo_rm100_rm445_ensemble_min_metrics.jsonl",
    "GSPO+GOPO+ens(pess)":  BASE / "part2_runs" / "gspo_gopo_rm100_rm445_ensemble_pess_metrics.jsonl",
}

# Color groups: GRPO baselines, GSPO baselines, ensemble-only, GOPO+ensemble
STYLES = {
    "GRPO+rm445":           {"color": "#1f77b4", "linestyle": "-",  "marker": "o"},
    "GRPO+GOPO":            {"color": "#aec7e8", "linestyle": "--", "marker": "o"},
    "GSPO+rm445":           {"color": "#ff7f0e", "linestyle": "-.",  "marker": "s"},
    "GSPO+GOPO":            {"color": "#ffbb78", "linestyle": "--", "marker": "s"},
    "GSPO+ens(mean)":       {"color": "#2ca02c", "linestyle": "-",  "marker": "^"},
    "GSPO+ens(min)":        {"color": "#98df8a", "linestyle": "--", "marker": "^"},
    "GSPO+ens(pess)":       {"color": "#d62728", "linestyle": ":",  "marker": "^"},
    "GSPO+GOPO+ens(mean)":  {"color": "#9467bd", "linestyle": "-",  "marker": "D"},
    "GSPO+GOPO+ens(min)":   {"color": "#c5b0d5", "linestyle": "--", "marker": "D"},
    "GSPO+GOPO+ens(pess)":  {"color": "#8c564b", "linestyle": ":",  "marker": "D"},
}

LW = 1.0
MS = 3

TRAIN_METRICS = [
    ("train/loss",       "Train Loss",  "Loss"),
    ("train/kl",         "Train KL",    "KL divergence"),
]

EVAL_METRICS = [
    ("eval/rm_fraction_policy_scores_above_reference",    "Eval Fraction Policy > Reference", "Fraction"),
    ("eval/rm_score_mean_on_policy_generations",          "Eval RM Score Mean",               "RM score"),
    ("eval/generation_mean_unique_token_ratio",           "Eval Unique Token Ratio",         "Unique token ratio"),
    ("eval/rm_margin_policy_minus_reference_mean",        "Eval RM Margin vs Reference",     "Margin"),
]


def load_metrics(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        row = {"step": obj.get("step")}
        row.update(obj.get("metrics", obj))
        rows.append(row)
    return pd.DataFrame(rows)


def save_plot(fig, name: str):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}")


def plot_group(run_keys, title, filename, eval_metric="eval/rm_fraction_policy_scores_above_reference"):
    """Plot a subset of runs as a 2x3 grid."""
    dfs = {}
    for k in run_keys:
        try:
            dfs[k] = load_metrics(RUNS[k])
        except FileNotFoundError:
            print(f"WARNING: missing {RUNS[k]}, skipping {k}")

    if not dfs:
        print(f"No data for {filename}, skipping.")
        return

    fig, axes_grid = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes_grid.flatten()
    fig.suptitle(title, fontsize=14)

    for ax_idx, (metric, title_str, ylabel) in enumerate(EVAL_METRICS):
        for label, df in dfs.items():
            sub = df.dropna(subset=[metric]) if metric in df.columns else pd.DataFrame()
            if sub.empty:
                continue
            style = STYLES[label]
            axes[ax_idx].plot(
                sub["step"], sub[metric],
                label=label, lw=LW, ms=MS, **style,
            )
        axes[ax_idx].set_title(title_str)
        axes[ax_idx].set_xlabel("Step")
        axes[ax_idx].set_ylabel(ylabel)
        axes[ax_idx].legend(fontsize=7)

    # Train loss in axes[4]
    for ax_idx, (metric, title_str, ylabel) in enumerate(TRAIN_METRICS):
        idx = ax_idx + 4
        for label, df in dfs.items():
            sub = df.dropna(subset=[metric]) if metric in df.columns else pd.DataFrame()
            if sub.empty:
                continue
            style = STYLES[label]
            axes[idx].plot(
                sub["step"], sub[metric],
                label=label, lw=LW, ms=MS, **style,
            )
        axes[idx].set_title(title_str)
        axes[idx].set_xlabel("Step")
        axes[idx].set_ylabel(ylabel)
        axes[idx].legend(fontsize=7)
    save_plot(fig, filename)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Plot 1: GRPO baseline vs GRPO+GOPO — isolates GOPO effect
    plot_group(
        ["GRPO+rm445", "GRPO+GOPO", "GSPO+rm445", "GSPO+GOPO"],
        "GRPO & GSPO Baselines vs GOPO",
        "part2_grpo_gopo.png",
    )

    # Plot 2: GSPO variants — baseline, GOPO, ensembles, GOPO+ensembles
    plot_group(
        ["GSPO+rm445", "GSPO+GOPO", "GSPO+ens(mean)", "GSPO+ens(min)", "GSPO+ens(pess)"],
        "GSPO: Baseline vs GOPO vs Ensemble Variants",
        "part2_gspo_variants.png",
    )

    # Plot 3: Best combinations — GSPO+GOPO+ensemble variants
    plot_group(
        ["GSPO+rm445","GSPO+GOPO", "GSPO+GOPO+ens(mean)", "GSPO+GOPO+ens(min)", "GSPO+GOPO+ens(pess)"],
        "GSPO+GOPO: Ensemble Aggregation Comparison",
        "part2_gspo_gopo_ensemble.png",
    )

    # Plot 4: All runs together for overview
    plot_group(
        list(RUNS.keys()),
        "Part 2 All Runs Overview",
        "part2_all_runs.png",
    )

    print(f"\nAll graphs saved to: {OUT}")
    for p in sorted(OUT.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()