# Probe-Guided Inference Scheduling

**Using a mechanistic interpretability probe as an inference-scheduling signal, not just a research metric.**

Renuka Oladri · MS Applied Machine Learning, University of Maryland

**STATUS: novel combination — no published paper, codebase, or blog post wires a mechanistic probe into a serving scheduler's routing decision.**

---

## The Core Idea

[early_detection](../early_detection) showed that a linear probe on
DeepSeek-R1-Distill-Qwen-7B's layer-16 hidden state, read at token 150 of
a reasoning chain, predicts whether that generation will converge cleanly
or run away into a non-converging loop — **AUC 0.612 vs. 0.445 for a
behavioral-only baseline, p=0.001** — days before the behavioral evidence
(repetition, token count) becomes informative on its own.

That result sat in a research repo as a number in a table. This project
asks the obvious next question: **is that signal good enough to make a
scheduling decision with?** `src/gate.py` wires the probe into
[inference-server](https://github.com/oladri-renuka/inference-server)'s
continuous-batching scheduler (adapted here for a real 7B HF model — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)). At token 150 of every
active generation, the gate classifies it `convergent` / `divergent` with
a confidence score, and the scheduler routes on that classification under
one of three strategies:

| Strategy | Behavior | Hypothesis |
|---|---|---|
| `baseline` | Standard continuous batching, routing_decision computed but ignored | Control condition |
| `probe_terminate` | Divergent generations above 0.7 confidence are killed at token 150 | Saves compute on doomed generations, at the cost of some false terminations |
| `probe_deprioritize` | Divergent generations are preempted (recompute-based) to free batch slots for waiting normal-priority requests, but never killed | Improves latency for likely-convergent requests without discarding any generation |

**Hypothesis risk is real and stated up front.** The probe might not move
throughput enough to matter, or the gate's own overhead (one forward hook
read + one `sklearn` inference call per request, at exactly one tick each)
might cancel out any scheduling win. Both outcomes are reported the same
way — see "What a Null or Negative Result Looks Like" below.

| | |
|---|---|
| **Target companies** | OpenAI (inference efficiency), Anthropic, Scale AI ML Systems |
| **Hardware** | RunPod A5000 (24GB) — same as early_detection |
| **Model** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` — same as early_detection |
| **Dependencies** | Reuses early_detection's probe methodology (retrained/exported here — see `scripts/train_probe.py`) and inference-server's continuous-batching scheduler design |
| **Repo name** | `probe-guided-inference` |

---

## Architecture

```
probe-guided-inference/
├── src/
│   ├── config.py            # single source of truth: model, layer, checkpoint, thresholds, batch size
│   ├── probe.py              # loads the fitted probe; registers the layer-16 forward hook
│   ├── gate.py                # hidden state -> RoutingDecision(label, confidence)
│   ├── queue.py               # GenerationRequest + RequestQueue (priority-aware requeue)
│   ├── hf_cache_bridge.py     # batched KV-cache bridge for heterogeneous-length decode steps
│   ├── model_runner.py        # bridges GenerationRequest <-> HF forward calls; fires the gate
│   ├── scheduler.py           # ContinuousBatcher: admit / decode / route / evict
│   └── server.py              # FastAPI: /generate, /stream, /health, /metrics
├── scripts/
│   ├── train_probe.py         # fits + exports the probe artifact from early_detection's activations
│   └── verify_setup.py        # pre-flight check: model, probe, hook, AND cache-bridge parity
├── benchmark/
│   ├── aime_loader.py         # same 200 AIME problems, same seed, as early_detection
│   ├── routing_eval.py        # runs all 3 strategies against identical traffic
│   └── report.py              # throughput / accuracy / token-savings / false-termination tables
├── tests/                     # CPU-only, offline — see "Testing" below
├── docs/ARCHITECTURE.md       # design decisions and why (start here for the "how")
└── results/
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the reasoning behind
every non-obvious decision here — most importantly, why this project
serves the model through unmodified HuggingFace `transformers` rather
than hand-rolling Qwen2's forward pass the way inference-server hand-rolls
GPT-2's (ADR 1), and how a batched KV-cache bridge was built on top of
that constraint (ADR 2).

## How Routing Actually Happens

1. A request is admitted into the active batch (`ContinuousBatcher._try_admit`) and decoded one token at a time in a shared batched forward pass (`model_runner.run_decode_step`).
2. The moment a request has generated exactly `checkpoint_position` (150) tokens, `src/gate.py`'s `ClassificationGate` reads that token's layer-16 hidden state (captured via a forward hook, `src/probe.py`) and classifies it. This happens **once per request**, not once per tick — the probe's runtime cost is a single `sklearn` inference call per generation, not a recurring per-token cost.
3. `ContinuousBatcher._apply_routing` reacts to that decision according to the active strategy:
   - `probe_terminate`: confidence-thresholded early termination.
   - `probe_deprioritize`: sets `priority=1`; the request is only actually affected once the batch is at capacity and a normal-priority request is waiting, at which point the least-progressed deprioritized request is preempted (recompute-based — its progress is discarded and it restarts after requeueing) to free the slot. See ADR 4.
4. `benchmark/routing_eval.py` runs all three strategies against the identical 200 AIME problems (same seed, same ordering as early_detection) and `benchmark/report.py` aggregates the comparison, including the two diagnostics the project explicitly calls for: probe_terminate's false-termination rate (measured against `baseline`'s untouched run of the same problems, since decoding is greedy/deterministic and gives a valid counterfactual — see that module's docstring) and probe_deprioritize's convergent-vs-divergent latency breakdown.

## Definition of Done

- [x] Probe artifact loads and the gate fires at exactly token 150 during live generation (`tests/test_gate.py`, `tests/test_scheduler_smoke.py`)
- [x] Batched cache bridge is numerically verified against plain `model.generate()`, both single-sequence and multi-sequence with staggered admission/eviction (`tests/test_hf_cache_bridge.py`, `scripts/verify_setup.py`)
- [x] All three strategies implemented against a shared scheduler core, unit- and integration-tested (`tests/test_scheduler_smoke.py`)
- [ ] Three strategies benchmarked on identical 200 AIME problems on the target GPU — **requires the RunPod A5000 run; not executed in this environment.** See "Reproduction."
- [ ] Results table (throughput, accuracy, token savings, false-termination rate) populated with real numbers
- [ ] README's Results section replaced with the actual outcome — positive, negative, or null — and honest interpretation

This repo ships everything short of the last three boxes: those require
GPU time this environment doesn't have. Running `make verify && make
train-probe && make benchmark && make report` on the RunPod box completes
them; `results/report.md` is what to paste back into this section
afterward.

## What a Positive Result Looks Like

`probe_deprioritize` improves p50 latency for predicted-convergent
requests by 15–25% relative to `baseline`, with no accuracy loss and no
drop in the divergent group's completion rate. This is the best case:
probing buys faster responses for the requests most likely to succeed,
without sacrificing anything on the harder ones.

## What a Null or Negative Result Looks Like

The gate's overhead (however small — one hook read plus one sklearn call
per request) doesn't pay for itself if the probe's AUC isn't high enough
to make routing decisions that matter at this batch size and contention
level; or `probe_terminate`'s false-termination rate is too high to be
worth the compute saved; or `probe_deprioritize`'s recompute-based
preemption (ADR 4) costs the divergent group more wall-clock time than
routing saves the convergent group. **Any of these is a complete,
reportable finding** — early_detection's own AUC (0.612) was already
flagged as "statistically significant, not obviously operationally
decisive," and this project's job is to find out exactly where that
line falls, not to manufacture a win.

## Results

**Not yet run.** This environment has no GPU and no access to
`early_detection`'s raw activation checkpoints (they live on the RunPod
volume from that project's original 8–12 hour generation run, not in this
repo — see `scripts/train_probe.py`'s docstring). Everything up to the
GPU run is built, tested, and verified; see "Reproduction" below for the
exact commands to produce this section's real content.

## Reproduction

### Hardware Requirements

Same as early_detection: ≥24GB VRAM (A5000 or better), ~20GB disk for
model weights + checkpoints.

### Steps

```bash
# 1. Setup
git clone <this repo>
cd probe-guided-inference
bash setup_runpod.sh
source venv/bin/activate

# 2. Verify BEFORE spending GPU time on anything else -- catches config
#    issues AND validates the batched cache bridge against model.generate()
python scripts/verify_setup.py

# 3. Train + export the probe artifact (early_detection must have already
#    run its generate.py -- see that project's README)
python scripts/train_probe.py --early-detection-dir ../early_detection

# 4. Run all three strategies against the same 200 AIME problems
python -m benchmark.routing_eval --out results/routing_eval.json

# 5. Aggregate into the comparison table + false-termination /
#    deprioritize-latency diagnostics
python -m benchmark.report --in-path results/routing_eval.json --out results/report.md
```

`Makefile` wraps each of these (`make verify`, `make train-probe`, `make
benchmark`, `make report`).

### Testing

```bash
pytest tests/ -v      # CPU-only, offline, ~0.5s -- no GPU, no model download
ruff check .
```

The test suite builds a real, tiny, randomly-initialized
`Qwen2ForCausalLM` in-process (see `tests/conftest.py`) instead of
downloading a checkpoint or hand-rolling a fake model — this exercises the
exact `transformers` Cache API and attention code path the production
model uses, so a bug in `src/hf_cache_bridge.py`'s padding/masking logic
shows up here too, not just on the GPU with the real 7B model. It cannot
and does not validate probe AUC or throughput numbers, which require the
real model on a GPU (see "Reproduction").

---

## Interview Talking Point

> "I combined my mechanistic interpretability work with an inference
> server to build a probe-guided scheduler. At token 150 of each
> generation, a linear probe on layer-16 hidden states predicts whether
> the generation will converge or loop. I route based on that prediction:
> deprioritize likely-divergent requests to improve latency for requests
> most likely to succeed. No existing system does this, because
> mechanistic probes and inference schedulers live in separate research
> communities. I can tell you exactly what the throughput improvement
> was, whether the probe's overhead was worth it, and — just as
> importantly — I built the numerical-parity tests that prove the serving
> path itself is correct before trusting any of those numbers."

## Limitations

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#known-limitations) for
the full list — no paged attention (memory scales with batch size ×
sequence length, not a fixed pool), greedy decoding only, recompute-based
(not swap-based) preemption, and a probe AUC that early_detection itself
already flagged as "significant, not decisive."

## Extends

This project builds directly on:
- [early_detection](../early_detection) — the probe methodology and its validated AUC results
- [inference-server](https://github.com/oladri-renuka/inference-server) — the continuous-batching scheduler design (admit/decode/evict tick loop, request queue, FastAPI serving pattern) this project adapts for a real HF model
- [token-efficiency-math-reasoning](https://github.com/oladri-renuka/token-efficiency-math-reasoning) — the AIME dataset loader and bimodal-convergence framing both of the above build on
