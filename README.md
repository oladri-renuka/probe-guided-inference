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
- [x] Three strategies benchmarked on identical 200 AIME problems on an A40 (RunPod) — see "Results"
- [x] Results table (throughput, accuracy, token savings, false-termination rate) populated with real numbers
- [x] README's Results section replaced with the actual outcome — positive, negative, or null — and honest interpretation

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

Run on an A40 (48GB), 200 AIME problems, `max_batch_size=8`, seed=42 —
identical traffic across all three strategies. Probe: layer 16, checkpoint
150, fit fresh on this run's own `early_detection/generate.py` output
(5-fold CV AUC **0.567 ± 0.051** — lower than early_detection's originally
reported 0.612, within the noise band of a fresh 200-sample generation;
see `scripts/train_probe.py`'s output for that run).

| Strategy | Wall-clock (200 req) | Throughput (req/s) | Avg tokens/req | Accuracy (completed) | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| `baseline` | 36,636.6s (10.2h) | 0.005 | 6,939 | 59.0% | 17,249.9s | 35,225.4s |
| `probe_terminate` | 24,460.2s (6.8h) | 0.008 | 4,657 | 66.0% | 10,937.2s | 23,385.4s |
| `probe_deprioritize` | 36,964.5s (10.3h) | 0.005 | 6,987 | 59.5% | 16,385.8s | 35,077.4s |

Full table, scheduler activity, and both diagnostics: `results/report.md`.

### The finding: destructive vs. non-destructive routing decisions tolerate probe noise very differently

**`probe_deprioritize` cleared its own positive-result bar, by a wide margin.**
The pre-registered bar (see "What a Positive Result Looks Like" above) was
+15–25% p50 improvement for predicted-convergent requests, no accuracy
loss, divergent requests still completing at the same rate. The actual
result:

- Convergent-predicted p50: **10,112.6s vs. baseline's 18,323.9s for the
  identical subset of requests — a 44.8% improvement**, nearly double the
  upper end of the target.
- Accuracy: 59.5% vs. baseline's 59.0% — no loss.
- Completion rate: 200/200 either way — nothing is ever discarded, only
  delayed.

The cost is concentrated entirely in the divergent group's latency
(p50 28,482.7s, from 75 recompute-based preemptions — ADR 4's predicted
mechanism, showing up exactly where expected) and in aggregate wall-clock,
which is a wash against `baseline` (36,964.5s vs. 36,636.6s). This is not
a system-throughput win — it's a **prioritization** win: latency gets
reallocated from likely-successful requests onto likely-doomed ones,
which is precisely what "deprioritize" was designed to do, and it did it
well past the bar this project set for itself before looking at the data.

**`probe_terminate` did not clear a comparable bar, because its failure
mode is destructive.** False termination rate: **47.5%** (28 of 59
terminated requests would have converged, per `baseline`'s untouched run
of the same problems — see `benchmark/report.py`'s ground-truth
methodology). At a probe AUC of 0.567, that's close to a coin flip.
Interestingly, `accuracy_on_completed` still rose (66.0% vs. baseline's
59.0%) — even a noisy filter removes more true non-convergent cases
(~10% conditional accuracy, per early_detection) than it wrongly removes
convergent ones in absolute count — but that headline number obscures the
real cost: 28 of 200 problems permanently lost an answer that would have
been correct, with no way to recover it.

**The honest conclusion isn't "does the probe work" — it's "which class
of decision can a probe at this AUC level actually support."** A signal
too noisy to justify discarding a generation outright (`probe_terminate`)
is still reliable enough to justify de-prioritizing it (`probe_deprioritize`),
because the two strategies have asymmetric failure costs: a wrong
`probe_deprioritize` call costs time; a wrong `probe_terminate` call costs
a correct answer, permanently. This is a mechanistically-grounded,
statistically measured line between two failure regimes, not a single
up-or-down verdict on "mechanistic probes for inference scheduling" —
and it's a genuinely useful one for deciding how to deploy a probe at a
given AUC in a real serving system.

### Caveats on these numbers

- **Single run, single seed.** No confidence intervals on the benchmark
  metrics themselves (only the probe's own 5-fold CV AUC has one). The
  false-termination rate (28/59) and the deprioritize latency delta are
  point estimates from one 200-problem pass each.
- **This run's probe (AUC 0.567) is weaker than early_detection's original
  (0.612).** Both are draws from the same underlying methodology on a
  fresh 200-sample generation; the gap is within plausible run-to-run
  noise given early_detection's own documented variance across runs (see
  that project's README "Sanity Check Against Original Project"), but it
  means `probe_terminate`'s 47.5% false-termination rate is likely close
  to a worst-case reading for this methodology, not a best case.
- **A known performance inefficiency inflated wall-clock time across all
  three strategies** (`src/hf_cache_bridge.py` rebuilds the full padded KV
  cache every decode tick rather than only on admission/eviction — see
  ADR 2 in `docs/ARCHITECTURE.md`). This affects the *absolute* wall-clock
  numbers but not the *relative* comparison between strategies, since all
  three pay the same per-tick tax under identical `max_batch_size`.

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
> generation, a linear probe on layer-16 hidden states — AUC ~0.57-0.61,
> not a strong signal — predicts whether a DeepSeek-R1 generation will
> converge or loop. Routing on that signal, I found the same probe
> supports two very different decisions very differently: deprioritizing
> likely-divergent requests improved p50 latency for likely-convergent
> ones by 44.8%, almost double my own pre-registered target, with zero
> accuracy loss and zero requests dropped — because a wrong call there
> just costs time. But outright terminating likely-divergent requests had
> a 47.5% false-termination rate — because a wrong call there permanently
> destroys a correct answer, and this probe isn't accurate enough to
> justify that. So the finding isn't 'does the probe work' — it's that
> the same signal's reliability threshold depends entirely on whether the
> downstream decision is reversible. No existing system reasons about
> mechanistic probes this way, because interpretability and inference
> scheduling live in separate research communities. And before I trusted
> any of these numbers, I built parity tests proving my custom batched
> KV-cache bridge produces token-identical output to native `generate()`
> — both alone and when multiple different-length sequences are decoded
> together in the same batch."

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
