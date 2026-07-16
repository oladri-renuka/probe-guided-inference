"""
Runs the same 200 AIME problems through all three routing strategies
(baseline, probe_terminate, probe_deprioritize) and records per-request
outcomes for benchmark/report.py to aggregate.

Unlike inference-server's benchmark/load_test.py, this drives the
scheduler IN-PROCESS rather than over HTTP: the research question here is
routing quality under a fixed model + fixed KV-cache budget, not HTTP
serving overhead, and spinning up a fresh 7B-parameter server process
three times (one per strategy) would add minutes of model-load noise to
every comparison for no benefit to that question. The model and tokenizer
are loaded ONCE and reused across all three strategy runs; only the
queue/scheduler/gate are rebuilt per run.

All 200 requests are submitted UP FRONT (bursty arrival), not trickled in,
and MAX_BATCH_SIZE is deliberately smaller than 200 -- this is what
creates the queueing contention probe_deprioritize's preemption path (see
src/scheduler.py) needs in order to ever do anything. Run the SAME
--max-batch-size across strategies for a fair comparison, exactly as
inference-server's run.py insists on identical load-test flags across
naive/static/continuous.

Usage:
    python -m benchmark.routing_eval --out results/routing_eval.json
    python -m benchmark.routing_eval --strategies baseline probe_deprioritize --n-samples 50
"""

import argparse
import asyncio
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.aime_loader import extract_answer, load_aime_samples, normalize_answer  # noqa: E402
from src.config import settings  # noqa: E402
from src.gate import ClassificationGate  # noqa: E402
from src.probe import load_probe  # noqa: E402
from src.queue import RequestQueue  # noqa: E402
from src.scheduler import ContinuousBatcher, RoutingStrategy  # noqa: E402

ALL_STRATEGIES = [s.value for s in RoutingStrategy]


async def run_strategy(
    strategy: RoutingStrategy, model, tokenizer, samples: list[dict],
    max_batch_size: int, max_tokens: int,
) -> dict:
    gate = None
    if strategy != RoutingStrategy.BASELINE:
        probe = load_probe(settings.probe_weights_path, settings.probe_layer, settings.checkpoint_position)
        gate = ClassificationGate(probe)

    queue = RequestQueue()
    batcher = ContinuousBatcher(
        model=model, tokenizer=tokenizer, queue=queue, strategy=strategy, gate=gate,
        max_batch_size=max_batch_size, max_tokens_cap=max_tokens,
        terminate_confidence_threshold=settings.terminate_confidence_threshold,
    )
    batcher_task = asyncio.create_task(batcher.run_forever())

    reqs = []
    for i, sample in enumerate(samples):
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": sample["question"]}], add_generation_prompt=True, tokenize=False,
        )
        prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].tolist()
        req = queue.submit(
            prompt_text, max_tokens, prompt_ids,
            metadata={"aime_idx": i, "expected_answer": sample["answer"]},
        )
        reqs.append(req)

    print(f"  [{strategy.value}] {len(reqs)} requests submitted, max_batch_size={max_batch_size} ...")
    t0 = time.time()
    await asyncio.gather(*(r.result for r in reqs))
    wall_clock_s = time.time() - t0

    batcher.stop()
    batcher.close()
    batcher_task.cancel()

    records = []
    for req in reqs:
        generated = req.generated_ids
        decoded = tokenizer.decode(generated, skip_special_tokens=False)
        converged = settings.think_end_token_id in generated
        extracted = extract_answer(decoded)
        correct = normalize_answer(extracted) == normalize_answer(req.metadata["expected_answer"])
        records.append({
            "aime_idx": req.metadata["aime_idx"],
            "n_generated": req.n_generated,
            "converged": converged,
            "correct": correct,
            "finish_reason": req.finish_reason,
            "terminated_by_probe": req.terminated_by_probe,
            "routing_label": req.routing_decision.label if req.routing_decision else None,
            "routing_confidence": req.routing_decision.confidence if req.routing_decision else None,
            "checkpoint_hit_time": req.checkpoint_hit_time,
            "arrival_time": req.arrival_time,
            "completion_time": req.completion_time,
            "latency_s": (req.completion_time - req.arrival_time) if req.completion_time else None,
        })

    result = {
        "strategy": strategy.value,
        "n_requests": len(reqs),
        "max_batch_size": max_batch_size,
        "wall_clock_s": wall_clock_s,
        "total_tokens_generated": sum(r["n_generated"] for r in records),
        "batcher_stats": {
            "total_admitted": batcher.total_admitted,
            "total_finished": batcher.total_finished,
            "total_probe_terminated": batcher.total_probe_terminated,
            "total_preemptions": batcher.total_preemptions,
        },
        "records": records,
    }

    del batcher, queue, reqs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


async def main_async(args: argparse.Namespace) -> None:
    print(f"Loading model: {settings.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name, torch_dtype=getattr(torch, settings.torch_dtype), device_map=device,
    )
    model.eval()
    print(f"Model loaded on {device}.")

    print(f"\nLoading {args.n_samples} AIME problems (seed={settings.aime_seed})...")
    samples = load_aime_samples()[: args.n_samples]
    print(f"Loaded {len(samples)} samples.")

    strategies = [RoutingStrategy(s) for s in args.strategies]
    all_results = {}
    for strategy in strategies:
        print(f"\n{'=' * 60}\nStrategy: {strategy.value}\n{'=' * 60}")
        result = await run_strategy(
            strategy, model, tokenizer, samples, args.max_batch_size, args.max_tokens,
        )
        all_results[strategy.value] = result
        n_correct = sum(1 for r in result["records"] if r["correct"])
        n_conv = sum(1 for r in result["records"] if r["converged"])
        print(f"  wall_clock={result['wall_clock_s']:.1f}s  "
              f"accuracy={n_correct}/{len(result['records'])} ({n_correct/len(result['records']):.1%})  "
              f"converged={n_conv}/{len(result['records'])}  "
              f"probe_terminated={result['batcher_stats']['total_probe_terminated']}  "
              f"preemptions={result['batcher_stats']['total_preemptions']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_results, indent=2))
    print(f"\nRaw results written to {args.out}")
    print("Next: python -m benchmark.report --in-path", args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES, choices=ALL_STRATEGIES)
    parser.add_argument("--n-samples", type=int, default=settings.aime_n_samples)
    parser.add_argument("--max-batch-size", type=int, default=settings.max_batch_size)
    parser.add_argument("--max-tokens", type=int, default=settings.max_new_tokens)
    parser.add_argument("--out", type=Path, default=Path("results/routing_eval.json"))
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
