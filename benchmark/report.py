"""
Aggregates benchmark/routing_eval.py's raw JSON into the comparison
tables: throughput, accuracy, and token savings across baseline /
probe_terminate / probe_deprioritize, plus two strategy-specific
diagnostics -- false termination rate, and convergent-vs-divergent
latency -- both of which need a same-run baseline to compute against (see
docstrings below for why).

Usage:
    python -m benchmark.report --in-path results/routing_eval.json --out results/report.md
"""

import argparse
import json
import statistics
from pathlib import Path


def _median(values: list) -> float:
    return statistics.median(values) if values else float("nan")


def _mean(values: list) -> float:
    return statistics.mean(values) if values else float("nan")


def _percentile(values: list, pct: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    k = (len(values) - 1) * pct
    f, c = int(k), min(int(k) + 1, len(values) - 1)
    return values[f] if f == c else values[f] + (values[c] - values[f]) * (k - f)


def summarize_strategy(result: dict) -> dict:
    records = result["records"]
    n = len(records)
    completed = [r for r in records if not r["terminated_by_probe"]]
    n_correct_completed = sum(1 for r in completed if r["correct"])
    n_conv = sum(1 for r in records if r["converged"])
    latencies = [r["latency_s"] for r in records if r["latency_s"] is not None]

    return {
        "strategy": result["strategy"],
        "n_requests": n,
        "wall_clock_s": result["wall_clock_s"],
        "throughput_req_per_s": n / result["wall_clock_s"] if result["wall_clock_s"] > 0 else 0.0,
        "avg_tokens_per_request": result["total_tokens_generated"] / n if n else 0.0,
        "accuracy_on_completed": n_correct_completed / len(completed) if completed else float("nan"),
        "n_completed": len(completed),
        "n_terminated": n - len(completed),
        "convergence_rate": n_conv / n if n else 0.0,
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
        "batcher_stats": result["batcher_stats"],
    }


def false_termination_rate(all_results: dict) -> dict | None:
    """False termination rate: of the requests probe_terminate cut short,
    what fraction would ACTUALLY have converged had they been allowed to
    continue?

    We can't observe that counterfactual directly from the terminated run
    itself -- the whole point of terminating is that generation stops
    before its natural outcome is known. Decoding is greedy (do_sample=
    False) and deterministic, and the probe's decision doesn't alter the
    model's forward pass, only the scheduler's continue/stop decision --
    so baseline's untouched run of the SAME aime_idx, under the SAME
    model and seed, produced the identical token sequence up to the
    termination point and its true convergence outcome is a valid stand-in
    ground truth. This requires baseline to have been run in the same
    routing_eval.json invocation; returns None if it wasn't.
    """
    if "baseline" not in all_results or "probe_terminate" not in all_results:
        return None
    baseline_by_idx = {r["aime_idx"]: r for r in all_results["baseline"]["records"]}
    terminated = [r for r in all_results["probe_terminate"]["records"] if r["terminated_by_probe"]]
    if not terminated:
        return {"n_terminated": 0, "n_false_terminations": 0, "false_termination_rate": float("nan")}

    false_terms = [r for r in terminated if baseline_by_idx.get(r["aime_idx"], {}).get("converged")]
    return {
        "n_terminated": len(terminated),
        "n_false_terminations": len(false_terms),
        "false_termination_rate": len(false_terms) / len(terminated),
    }


def deprioritize_latency_breakdown(all_results: dict) -> dict | None:
    """Average latency for convergent vs divergent requests under
    probe_deprioritize -- the project doc's "positive result" criterion is
    specifically about convergent-request latency improving, so the
    breakdown (not just the pooled p50) is the number that actually
    confirms or refutes the hypothesis."""
    if "probe_deprioritize" not in all_results:
        return None
    records = all_results["probe_deprioritize"]["records"]
    conv_latencies = [r["latency_s"] for r in records if r["routing_label"] == "convergent" and r["latency_s"]]
    div_latencies = [r["latency_s"] for r in records if r["routing_label"] == "divergent" and r["latency_s"]]
    baseline_conv_latencies = None
    if "baseline" in all_results:
        baseline_by_idx = {r["aime_idx"]: r for r in all_results["baseline"]["records"]}
        depri_by_idx = {r["aime_idx"]: r for r in records}
        baseline_conv_latencies = [
            baseline_by_idx[idx]["latency_s"]
            for idx, r in depri_by_idx.items()
            if r["routing_label"] == "convergent" and idx in baseline_by_idx and baseline_by_idx[idx]["latency_s"]
        ]

    result = {
        "convergent_p50_s": _percentile(conv_latencies, 0.50),
        "convergent_mean_s": _mean(conv_latencies),
        "divergent_p50_s": _percentile(div_latencies, 0.50),
        "divergent_mean_s": _mean(div_latencies),
        "n_convergent": len(conv_latencies),
        "n_divergent": len(div_latencies),
    }
    if baseline_conv_latencies:
        depri_p50 = result["convergent_p50_s"]
        base_p50 = _percentile(baseline_conv_latencies, 0.50)
        result["baseline_convergent_p50_s"] = base_p50
        result["convergent_p50_improvement_pct"] = (
            100.0 * (base_p50 - depri_p50) / base_p50 if base_p50 else float("nan")
        )
    return result


def format_markdown(all_results: dict) -> str:
    order = ["baseline", "probe_terminate", "probe_deprioritize"]
    labels = [label for label in order if label in all_results]
    labels += [label for label in all_results if label not in order]
    summaries = {label: summarize_strategy(all_results[label]) for label in labels}

    lines = ["# Probe-Guided Routing: Benchmark Results\n"]
    lines.append("| Strategy | Throughput (req/s) | Avg tokens/req | Accuracy (completed) | "
                  "Convergence rate | p50 latency (s) | p95 latency (s) |")
    lines.append("|---|---|---|---|---|---|---|")
    for label in labels:
        s = summaries[label]
        lines.append(
            f"| {label} | {s['throughput_req_per_s']:.3f} | {s['avg_tokens_per_request']:.0f} | "
            f"{s['accuracy_on_completed']:.1%} | {s['convergence_rate']:.1%} | "
            f"{s['latency_p50_s']:.1f} | {s['latency_p95_s']:.1f} |"
        )

    lines.append("\n## Scheduler Activity\n")
    lines.append("| Strategy | Admitted | Finished | Probe-terminated | Preemptions |")
    lines.append("|---|---|---|---|---|")
    for label in labels:
        bs = summaries[label]["batcher_stats"]
        lines.append(f"| {label} | {bs['total_admitted']} | {bs['total_finished']} | "
                      f"{bs['total_probe_terminated']} | {bs['total_preemptions']} |")

    ftr = false_termination_rate(all_results)
    if ftr is not None:
        lines.append("\n## probe_terminate: False Termination Rate\n")
        lines.append(
            "Of the requests probe_terminate cut short at token 150, this is the fraction "
            "that baseline's untouched run of the same problem shows WOULD have converged "
            "(see benchmark/report.py `false_termination_rate` docstring for the ground-truth "
            "methodology).\n"
        )
        lines.append(f"- Terminated: {ftr['n_terminated']}")
        lines.append(f"- False terminations (would have converged): {ftr['n_false_terminations']}")
        lines.append(f"- False termination rate: {ftr['false_termination_rate']:.1%}"
                      if ftr["n_terminated"] else "- False termination rate: n/a (nothing terminated)")

    depri = deprioritize_latency_breakdown(all_results)
    if depri is not None:
        lines.append("\n## probe_deprioritize: Convergent vs Divergent Latency\n")
        lines.append("| Group | n | p50 latency (s) | mean latency (s) |")
        lines.append("|---|---|---|---|")
        lines.append(f"| Convergent (predicted) | {depri['n_convergent']} | "
                      f"{depri['convergent_p50_s']:.1f} | {depri['convergent_mean_s']:.1f} |")
        lines.append(f"| Divergent (predicted) | {depri['n_divergent']} | "
                      f"{depri['divergent_p50_s']:.1f} | {depri['divergent_mean_s']:.1f} |")
        if "baseline_convergent_p50_s" in depri:
            lines.append(f"\nBaseline p50 latency for the SAME (predicted-convergent) requests: "
                          f"{depri['baseline_convergent_p50_s']:.1f}s. "
                          f"probe_deprioritize change: {depri['convergent_p50_improvement_pct']:+.1f}%.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-path", type=Path, default=Path("results/routing_eval.json"))
    parser.add_argument("--out", type=Path, default=Path("results/report.md"))
    args = parser.parse_args()

    all_results = json.loads(args.in_path.read_text())
    markdown = format_markdown(all_results)
    print(markdown)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown)

    summary_json = {
        label: summarize_strategy(result) for label, result in all_results.items()
    }
    summary_json["_false_termination_rate"] = false_termination_rate(all_results)
    summary_json["_deprioritize_latency_breakdown"] = deprioritize_latency_breakdown(all_results)
    (args.out.parent / "report_summary.json").write_text(json.dumps(summary_json, indent=2))

    print(f"\nWritten to {args.out} and {args.out.parent / 'report_summary.json'}")


if __name__ == "__main__":
    main()
