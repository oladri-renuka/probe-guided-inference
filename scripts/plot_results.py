"""
Generates the analysis figures referenced in README.md from
results/report_summary.json (produced by benchmark/report.py).

Run after benchmark/report.py:
    python scripts/plot_results.py
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

COLORS = {
    "baseline": "#475569",
    "probe_terminate": "#B45309",
    "probe_deprioritize": "#1D4ED8",
    "false": "#B91C1C",
    "true": "#15803D",
    "convergent": "#1D4ED8",
    "divergent": "#B45309",
}
LABELS = {
    "baseline": "baseline",
    "probe_terminate": "probe_terminate",
    "probe_deprioritize": "probe_deprioritize",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.edgecolor": "#CBD5E1",
    "axes.labelcolor": "#1E293B",
    "text.color": "#1E293B",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "axes.grid": True,
    "grid.color": "#E2E8F0",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_summary() -> dict:
    path = RESULTS_DIR / "report_summary.json"
    if not path.exists():
        sys.exit(f"{path} not found -- run benchmark/report.py first.")
    return json.loads(path.read_text())


def plot_latency_comparison(summary: dict) -> None:
    strategies = ["baseline", "probe_terminate", "probe_deprioritize"]
    p50 = [summary[s]["latency_p50_s"] / 3600 for s in strategies]
    p95 = [summary[s]["latency_p95_s"] / 3600 for s in strategies]

    x = np.arange(len(strategies))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars1 = ax.bar(x - width / 2, p50, width, label="p50", color=[COLORS[s] for s in strategies], alpha=1.0)
    bars2 = ax.bar(x + width / 2, p95, width, label="p95", color=[COLORS[s] for s in strategies], alpha=0.45)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}h", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_ylabel("Latency (hours)")
    ax.set_title("Per-request Latency by Routing Strategy\n(200 AIME problems, max_batch_size=8)")
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[s] for s in strategies])
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "latency_comparison.png", dpi=180)
    plt.close(fig)


def plot_throughput_and_tokens(summary: dict) -> None:
    strategies = ["baseline", "probe_terminate", "probe_deprioritize"]
    throughput = [summary[s]["throughput_req_per_s"] * 3600 for s in strategies]  # req/hr, more readable
    tokens = [summary[s]["avg_tokens_per_request"] for s in strategies]
    accuracy = [summary[s]["accuracy_on_completed"] * 100 for s in strategies]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    for ax, values, title, ylabel, fmt in [
        (axes[0], throughput, "Throughput", "requests / hour", "{:.2f}"),
        (axes[1], tokens, "Avg. Tokens / Request", "tokens", "{:.0f}"),
        (axes[2], accuracy, "Accuracy (completed requests)", "%", "{:.1f}%"),
    ]:
        bars = ax.bar([LABELS[s] for s in strategies], values, color=[COLORS[s] for s in strategies])
        for bar, v in zip(bars, values, strict=False):
            ax.annotate(fmt.format(v), xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Scheduler Throughput, Cost, and Accuracy by Strategy", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "throughput_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_false_termination_breakdown(summary: dict) -> None:
    ftr = summary["_false_termination_rate"]
    n_false = ftr["n_false_terminations"]
    n_true = ftr["n_terminated"] - n_false

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [1, 1.3]})

    ax1.pie(
        [n_true, n_false],
        labels=[f"Correctly terminated\n(n={n_true})", f"False termination\n(n={n_false})"],
        colors=[COLORS["true"], COLORS["false"]],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 10},
    )
    ax1.set_title(f"probe_terminate: outcome of {ftr['n_terminated']}\nrequests killed at token 150")

    categories = ["All requests\n(baseline)", "Terminated requests\n(probe_terminate)"]
    conv_rate_all = summary["baseline"]["convergence_rate"] * 100
    conv_rate_terminated = (n_false / ftr["n_terminated"]) * 100
    bars = ax2.bar(categories, [conv_rate_all, conv_rate_terminated],
                    color=[COLORS["baseline"], COLORS["false"]])
    for bar, v in zip(bars, [conv_rate_all, conv_rate_terminated], strict=False):
        ax2.annotate(f"{v:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
    ax2.set_ylabel("Would-have-converged rate")
    ax2.set_title("Convergence rate: population vs.\nthe subset probe_terminate killed")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "false_termination_breakdown.png", dpi=180)
    plt.close(fig)


def plot_deprioritize_latency_breakdown(summary: dict) -> None:
    d = summary["_deprioritize_latency_breakdown"]

    categories = [
        "baseline\n(convergent subset)",
        "probe_deprioritize\n(convergent)",
        "probe_deprioritize\n(divergent)",
    ]
    values_hr = [d["baseline_convergent_p50_s"] / 3600, d["convergent_p50_s"] / 3600, d["divergent_p50_s"] / 3600]
    colors = [COLORS["baseline"], COLORS["convergent"], COLORS["divergent"]]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_ylim(0, max(values_hr) * 1.25)
    bars = ax.bar(categories, values_hr, color=colors, width=0.55)
    for bar, v in zip(bars, values_hr, strict=False):
        ax.annotate(f"{v:.1f}h", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=10)

    # Improvement callout in the clear space above the two convergent bars --
    # deliberately no arrow/bracket (those are fragile to get right without
    # rendering feedback); a plain, well-placed label reads just as clearly.
    ax.annotate(
        f"convergent p50: {d['convergent_p50_improvement_pct']:+.1f}%",
        xy=(0.5, max(values_hr[0], values_hr[1]) + max(values_hr) * 0.14),
        ha="center", fontsize=12, fontweight="bold", color=COLORS["convergent"],
    )

    ax.set_ylabel("p50 latency (hours)")
    ax.set_title("probe_deprioritize: p50 Latency by Predicted Class\nvs. baseline's latency for the same requests")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "deprioritize_latency_breakdown.png", dpi=180)
    plt.close(fig)


def main() -> None:
    summary = load_summary()
    plot_latency_comparison(summary)
    plot_throughput_and_tokens(summary)
    plot_false_termination_breakdown(summary)
    plot_deprioritize_latency_breakdown(summary)
    print(f"Wrote 4 figures to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
