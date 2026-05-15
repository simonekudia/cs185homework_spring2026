import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE = Path("/Users/simonekudia/Desktop/cs185/final_project/homework_spring2026/final_project_llm_rl")
OUT = BASE / "report_assets" / "part1_graphs"

RUN_FILES = {
    "dpo":    BASE / "part1_runs" / "part1_dpo_metrics.jsonl",
    "ipo":    BASE / "part1_runs" / "part1_ipo_metrics.jsonl",
    "aot":    BASE / "part1_runs" / "part1_aot_metrics.jsonl",
    "grpo":   BASE / "part1_runs" / "part1_grpo_metrics.jsonl",
    "drgrpo": BASE / "part1_runs" / "part1_drgrpo_metrics.jsonl",
    "gspo":   BASE / "part1_runs" / "part1_gspo_metrics.jsonl",
}

OFFLINE_STYLES = {
    "DPO":    {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "IPO":    {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "AOT":    {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
}

ONLINE_STYLES = {
    "GRPO":   {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
    "DrGRPO": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    "GSPO":   {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
    "GRPO-RM445": {"color": "#d62728", "marker": "D", "linestyle": ":"},
}

LW = 1.2
MS = 4


def load_metrics_jsonl(path: Path) -> pd.DataFrame:
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


def plot_offline_comparison():
    dpo_df  = load_metrics_jsonl(RUN_FILES["dpo"])
    ipo_df  = load_metrics_jsonl(RUN_FILES["ipo"])
    aot_df  = load_metrics_jsonl(RUN_FILES["aot"])

    dpo_eval  = dpo_df.dropna(subset=["eval/pref_accuracy_sum_logp"])
    ipo_eval  = ipo_df.dropna(subset=["eval/pref_accuracy_sum_logp"])
    aot_eval  = aot_df.dropna(subset=["eval/pref_accuracy_sum_logp"])

    dpo_train = dpo_df.dropna(subset=["train/preference/loss"])
    ipo_train = ipo_df.dropna(subset=["train/preference/loss"])
    aot_train = aot_df.dropna(subset=["train/preference/loss"])

    fig, axes_grid = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes_grid.flatten()
    axes[-1].set_visible(False)  # hide unused 6th cell
    fig.suptitle("Offline Methods Comparison: DPO vs IPO vs AOT", fontsize=14)

    for label, df, style in [
        ("DPO", dpo_eval, OFFLINE_STYLES["DPO"]),
        ("IPO", ipo_eval, OFFLINE_STYLES["IPO"]),
        ("AOT", aot_eval, OFFLINE_STYLES["AOT"]),
    ]:
        axes[0].plot(
            df["step"], df["eval/pref_accuracy_sum_logp"],
            label=label, lw=LW, ms=MS,
            **style,
        )
        axes[1].plot(
            df["step"], df["eval/reference_corrected_pref_accuracy"],
            label=label, lw=LW, ms=MS,
            **style,
        )
        axes[2].plot(
            df["step"], df["eval/generation_mean_unique_token_ratio"],
            label=label, lw=LW, ms=MS, **style,
        )
        axes[3].plot(
            df["step"], df["eval/generation_mean_dominant_token_fraction"],
            label=label, lw=LW, ms=MS, **style,
        )

    for label, df, style in [
        ("DPO", dpo_train, OFFLINE_STYLES["DPO"]),
        ("IPO", ipo_train, OFFLINE_STYLES["IPO"]),
        ("AOT", aot_train, OFFLINE_STYLES["AOT"]),
    ]:
        axes[4].plot(
            df["step"], df["train/preference/loss"],
            label=label, lw=LW, alpha=0.85,
            color=style["color"], linestyle=style["linestyle"],
        )

    axes[0].set_title("Eval Preference Accuracy")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].set_title("Eval Reference-Corrected Accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    axes[2].set_title("Eval Generation Unique Token Ratio")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Unique token ratio")
    axes[2].legend()

    axes[3].set_title("Eval Generation Dominant Token Fraction")
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("Dominant token fraction")
    axes[3].legend()

    axes[4].set_title("Train Loss")
    axes[4].set_xlabel("Step")
    axes[4].set_ylabel("Loss")
    axes[4].legend()

    save_plot(fig, "offline_comparison.png")


def plot_online_comparison():
    grpo_df   = load_metrics_jsonl(RUN_FILES["grpo"])
    drgrpo_df = load_metrics_jsonl(RUN_FILES["drgrpo"])
    gspo_df   = load_metrics_jsonl(RUN_FILES["gspo"])

    grpo_eval   = grpo_df.dropna(subset=["eval/rm_fraction_policy_scores_above_reference"])
    drgrpo_eval = drgrpo_df.dropna(subset=["eval/rm_fraction_policy_scores_above_reference"])
    gspo_eval   = gspo_df.dropna(subset=["eval/rm_fraction_policy_scores_above_reference"])

    grpo_train   = grpo_df.dropna(subset=["train/loss"])
    drgrpo_train = drgrpo_df.dropna(subset=["train/loss"])
    gspo_train   = gspo_df.dropna(subset=["train/loss"])

    eval_runs = [
        ("GRPO",   grpo_eval,   ONLINE_STYLES["GRPO"]),
        ("DrGRPO", drgrpo_eval, ONLINE_STYLES["DrGRPO"]),
        ("GSPO",   gspo_eval,   ONLINE_STYLES["GSPO"]),
    ]

    train_runs = [
        ("GRPO",   grpo_train,   ONLINE_STYLES["GRPO"]),
        ("DrGRPO", drgrpo_train, ONLINE_STYLES["DrGRPO"]),
        ("GSPO",   gspo_train,   ONLINE_STYLES["GSPO"]),
    ]

    fig, axes_grid = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes_grid.flatten()
    axes[-1].set_visible(False)  # hide unused 6th cell
    fig.suptitle("Online Methods Comparison: GRPO vs DrGRPO vs GSPO", fontsize=14)

    for label, df, style in eval_runs:
        axes[0].plot(
            df["step"], df["eval/rm_fraction_policy_scores_above_reference"],
            label=label, lw=LW, ms=MS, **style,
        )
        axes[1].plot(
            df["step"], df["eval/rm_score_mean_on_policy_generations"],
            label=label, lw=LW, ms=MS, **style,
        )
        axes[2].plot(
            df["step"], df["eval/generation_mean_unique_token_ratio"],
            label=label, lw=LW, ms=MS, **style,
        )
        axes[3].plot(
            df["step"], df["eval/generation_mean_dominant_token_fraction"],
            label=label, lw=LW, ms=MS, **style,
        )

    for label, df, style in train_runs:
        axes[4].plot(
            df["step"], df["train/loss"],
            label=label, lw=LW, alpha=0.85,
            color=style["color"], linestyle=style["linestyle"],
        )

    axes[0].set_title("Eval Fraction Policy > Reference")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Fraction")
    axes[0].legend()

    axes[1].set_title("Eval RM Score Mean")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("RM score")
    axes[1].legend()

    axes[2].set_title("Eval Generation Unique Token Ratio")
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Unique token ratio")
    axes[2].legend()

    axes[3].set_title("Eval Generation Dominant Token Fraction")
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("Dominant token fraction")
    axes[3].legend()

    axes[4].set_title("Train Loss")
    axes[4].set_xlabel("Step")
    axes[4].set_ylabel("Loss")
    axes[4].legend()

    save_plot(fig, "online_comparison.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_offline_comparison()
    plot_online_comparison()
    print(f"\nAll graphs saved to: {OUT}")
    for p in sorted(OUT.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
    