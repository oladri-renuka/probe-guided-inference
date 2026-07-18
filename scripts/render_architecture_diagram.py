"""
Renders docs/architecture.png -- the system diagram referenced in
README.md. Static (not data-driven); re-run after any structural change
to src/ to keep it in sync.

Run:
    python scripts/render_architecture_diagram.py
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "docs" / "architecture.png"

INK = "#1E293B"
SCHED_FILL, SCHED_EDGE = "#EFF6FF", "#1D4ED8"
PROBE_FILL, PROBE_EDGE = "#FFF7ED", "#B45309"
DATA_FILL, DATA_EDGE = "#F1F5F9", "#475569"
STRAT_FILL, STRAT_EDGE = "#F0FDF4", "#15803D"


def box(ax, xy, w, h, text, fill, edge, fontsize=11, weight="bold", subtext=None):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.6, edgecolor=edge, facecolor=fill,
    )
    ax.add_patch(patch)
    if subtext:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center",
                 fontsize=fontsize, fontweight=weight, color=INK)
        ax.text(x + w / 2, y + h * 0.28, subtext, ha="center", va="center",
                 fontsize=fontsize - 2.5, color="#475569", family="monospace")
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                 fontsize=fontsize, fontweight=weight, color=INK)
    return (x, y, w, h)


def arrow(ax, start, end, color=INK, style="-|>", lw=1.6, connectionstyle="arc3,rad=0.0", ls="-"):
    patch = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=16,
        linewidth=lw, color=color, connectionstyle=connectionstyle, linestyle=ls,
        shrinkA=2, shrinkB=2, zorder=1,
    )
    ax.add_patch(patch)


def center_bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def center_top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def side(b, which):
    x, y, w, h = b
    if which == "left":
        return (x, y + h / 2)
    if which == "right":
        return (x + w, y + h / 2)


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(6.25, 8.15, "Probe-Guided Inference Scheduler", ha="center", fontsize=17, fontweight="bold", color=INK)
    ax.text(6.25, 7.75, "src/scheduler.py  ·  src/model_runner.py  ·  src/gate.py  ·  src/probe.py  ·  src/hf_cache_bridge.py",
            ha="center", fontsize=9.5, color="#64748B", family="monospace")

    # Request queue
    q = box(ax, (4.9, 6.5), 2.7, 0.75, "RequestQueue", DATA_FILL, DATA_EDGE, subtext="FIFO + priority-aware requeue")

    # Continuous batcher (central)
    sched = box(ax, (3.9, 5.0), 4.7, 1.0, "ContinuousBatcher", SCHED_FILL, SCHED_EDGE, fontsize=13,
                subtext="admit  →  decode  →  route  →  evict")

    arrow(ax, center_bottom(q), center_top(sched))

    # model_runner (left-bottom), probe hook (mid-bottom), gate (right-bottom)
    runner = box(ax, (0.6, 3.0), 3.1, 0.95, "model_runner.py", PROBE_FILL, PROBE_EDGE,
                 subtext="HF forward pass bridge")
    capture = box(ax, (4.55, 3.0), 3.0, 0.95, "probe.py", PROBE_FILL, PROBE_EDGE,
                   subtext="layer-16 forward hook")
    gate = box(ax, (8.4, 3.0), 3.3, 0.95, "gate.py", PROBE_FILL, PROBE_EDGE,
               subtext="hidden state → RoutingDecision")

    sched_x, sched_y, sched_w, _ = sched
    arrow(ax, (sched_x + 0.4, sched_y), center_top(runner), connectionstyle="arc3,rad=-0.2")
    arrow(ax, center_bottom(sched), center_top(capture))
    arrow(ax, (sched_x + sched_w - 0.4, sched_y), center_top(gate), connectionstyle="arc3,rad=0.2")

    # gate reads capture's hidden state
    arrow(ax, side(capture, "right"), side(gate, "left"), color=PROBE_EDGE, lw=1.3, ls="--")

    # gate's decision informs the batcher's routing step (feedback edge)
    arrow(ax, (10.05, 3.95), (7.6, 5.0), color=SCHED_EDGE, lw=1.4,
          connectionstyle="arc3,rad=0.35", ls="--")
    ax.text(9.85, 4.55, "RoutingDecision", fontsize=8.5, color=SCHED_EDGE, style="italic",
            rotation=28, ha="center")

    # hf_cache_bridge under model_runner
    cache = box(ax, (0.6, 1.55), 3.1, 0.95, "hf_cache_bridge.py", DATA_FILL, DATA_EDGE,
                subtext="batched KV-cache, padded/masked")
    arrow(ax, center_bottom(runner), center_top(cache))

    # HF model at the very bottom, fed by both model_runner and cache bridge
    model = box(ax, (0.6, 0.25), 3.1, 0.85, "transformers.AutoModelForCausalLM", "#FAFAF9", "#78716C",
                fontsize=9.5)
    arrow(ax, center_bottom(cache), center_top(model))

    # Strategy legend, bottom right
    leg_x, leg_y = 8.4, 1.7
    box(ax, (leg_x, leg_y), 3.3, 1.35, "", STRAT_FILL, STRAT_EDGE)
    ax.text(leg_x + 1.65, leg_y + 1.13, "Routing strategies", fontsize=10, fontweight="bold",
            ha="center", color=INK)
    for i, label in enumerate(["baseline", "probe_terminate", "probe_deprioritize"]):
        ax.text(leg_x + 0.2, leg_y + 0.82 - i * 0.31, f"• {label}", fontsize=9, color="#166534",
                family="monospace", va="center")

    fig.tight_layout()
    OUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
