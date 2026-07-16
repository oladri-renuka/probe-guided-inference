# Architecture

This project wires [early_detection](../../early_detection)'s validated
mechanistic probe (AUC 0.612 vs 0.445 behavioral baseline at layer 16,
token 150, p=0.001) into a continuous-batching inference scheduler adapted
from [inference-server](https://github.com/oladri-renuka/inference-server),
so that the probe's early-convergence prediction becomes a routing signal
instead of a research artifact that only ever produced a number in a
README table.

```
                    ┌─────────────────────────────────────────┐
                    │              RequestQueue                │
                    │  (FIFO submit; priority-aware requeue)   │
                    └───────────────┬───────────────────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │      ContinuousBatcher         │
                     │  admit → decode → route → evict│
                     └──┬──────────────┬─────────────┬┘
                        │              │             │
              ┌─────────▼───┐  ┌───────▼──────┐  ┌───▼──────────────┐
              │ model_runner │  │ probe.py hook│  │ gate.py           │
              │ (HF forward  │  │ layer 16     │  │ RoutingDecision    │
              │  pass bridge)│  │ activation   │  │ (label, confidence)│
              └──────┬───────┘  └──────────────┘  └────────────────────┘
                     │
              ┌──────▼────────────┐
              │ hf_cache_bridge.py │
              │ per-seq DynamicCache
              │ padded batched decode
              └────────────────────┘
```

Three routing strategies (`src/scheduler.py` `RoutingStrategy`) share this
same pipeline and differ only in what they do with `req.routing_decision`:
`baseline` ignores it, `probe_terminate` ends divergent-above-threshold
requests immediately, `probe_deprioritize` preempts active deprioritized
requests to admit waiting normal-priority ones under contention.

This document records the design decisions that make this pipeline
different from a naive reading of the project brief, and why.

---

## ADR 1: HuggingFace's forward pass, not a hand-rolled Qwen2

inference-server's `gpt2_model.py` reimplements GPT-2's forward pass by
hand specifically so its paged-attention cache can plug directly into a
`write`/`read` interface it controls end to end. That project's README is
explicit that this is a **reference implementation, not a production
one** -- readable and benchmarkable, not fast, with a Python-loop attention
gather it names as a known limitation.

This project does not repeat that choice, for a reason specific to what
it's serving: **the probe's validity depends on activation numerics it
was never tested against being replicated exactly.** early_detection
measured AUC 0.612 using native HuggingFace forward hooks on the
unmodified `transformers` implementation of DeepSeek-R1-Distill-Qwen-7B
(Qwen2 architecture -- GQA, RoPE, RMSNorm, SwiGLU). A hand-rolled
reimplementation of that stack is materially higher-risk to get bit-exact
than GPT-2's plain multi-head attention (more moving parts: rotary
position embeddings, grouped-query KV head repetition, RMSNorm epsilon
placement), and any numerical drift -- even one that produces fluent,
plausible-looking text -- would shift the layer-16 hidden state
distribution away from the one the probe's `LogisticRegression`
coefficients were fit on. That failure mode wouldn't crash; it would
silently degrade routing decisions to noise while looking like it works.

**Decision:** serve DeepSeek-R1-Distill-Qwen-7B through
`transformers.AutoModelForCausalLM`, unmodified. All batching/scheduling
logic operates around that model as a black box that takes `input_ids` +
`attention_mask` + `past_key_values` and returns `logits` +
`past_key_values` + (via a forward hook) hidden states. The cost of this
choice is that this project cannot demonstrate a paged-attention-style
memory efficiency win the way inference-server does -- see ADR 2.

## ADR 2: A padded/masked batched cache, not paged blocks

Given ADR 1, `src/hf_cache_bridge.py` needed to solve a problem
inference-server's hand-rolled model never had to: batching a decode step
across N sequences of different lengths using only HuggingFace's public
`Cache` API (`cache.layers[i].keys` / `.values`, and the
`DynamicCache(ddp_cache_data=...)` constructor -- see that module's
"Version note" for why this project targets transformers>=5.0
specifically).

The mechanism: left-pad every active sequence's cache up to the tick's
max active length, concatenate along the batch dimension, run one forward
pass, then immediately right-trim the result back down to each sequence's
true (now +1) length before the next tick. This is the **naive/contiguous
end** of the paging spectrum inference-server's README describes -- full
re-pad-and-copy every tick, not incremental block allocation with a fixed
pool. That's an intentional scope cut, not an oversight: this project's
AIME workload runs at most `max_batch_size` sequences at a time to natural
convergence (up to 10K tokens), not inference-server's bursty
short/long-mixed HTTP traffic, so the paged allocator's core benefit --
not reserving memory for tokens a sequence never generates -- has far less
to offer here. Building a paged allocator on top of ADR 1's constraint
(can't touch the attention kernel) would also buy nothing: paging changes
memory layout, not the numerics HF's attention actually computes over it.

**Correctness verification:** `tests/test_hf_cache_bridge.py` and
`scripts/verify_setup.py` both assert token-for-token parity between this
bridge and plain sequential `model.generate()` calls, for both a single
sequence and multiple different-length sequences decoded together. This
is the single most load-bearing test in the project -- see that module's
docstring.

## ADR 3: Single-position probe features, not mean-pooled

The project brief's Step 2 pseudocode sketches
`hidden_state[0, :token_position, :].mean(dim=0)` -- a mean pool over
every generated token's hidden state up to the checkpoint.
`early_detection/generate.py`'s actual instrumentation
(`GenerationInstrumenter._make_layer_hook` / `_sweep_layer_hook`) does
something different: it captures `hidden_state[:, -1, :]` **at exactly**
token 150 -- the hidden state of the single token generated at that
position, not an aggregate over the ones before it. That's also what
`analyze.py` trained and measured AUC 0.612 against.

**Decision:** `src/probe.py` and `src/gate.py` follow early_detection's
actual, validated methodology (single-position vector) rather than the
brief's illustrative pseudocode. A probe's fitted coefficients are
meaningless against a differently-aggregated feature vector, and there is
no early_detection result establishing what a mean-pooled variant's AUC
would even be -- shipping the mean-pool version would mean serving an
untested probe while claiming the tested one's numbers. If mean-pooling
turns out to be worth exploring, that's a new early_detection experiment
(retrain + re-report AUC), not a silent substitution here.

## ADR 4: probe_deprioritize's preemption is recompute-based

The brief describes `probe_deprioritize` as "divergent generations are
moved to lowest batch priority but allowed to continue" -- allowed to
continue, but from where, and at whose expense? `src/scheduler.py`
resolves this the same way inference-server resolves cache-exhaustion
preemption: when the active batch is full (`max_batch_size`) and a
normal-priority request is waiting, the least-progressed active
deprioritized request is evicted back to the pending queue and **restarts
from scratch** (its KV cache is discarded, not swapped to host memory).
This is the simpler of the two preemption strategies the paged-attention
literature describes (recompute vs. swap) -- inference-server made the
same choice for the same reason, and names the same cost: whatever that
victim had generated so far is thrown away. `RequestQueue
.push_front_priority_aware` (see its docstring) ensures a re-queued
deprioritized request can't cut in front of requests that were already
waiting normal-priority, while still preserving FIFO order among
multiple deprioritized requests queued in the same tick.

This is also, honestly, the mechanism most likely to produce a **null or
negative result** for `probe_deprioritize` under high contention: a
divergent request that gets preempted repeatedly could spend more total
wall-clock time than it would have under `baseline`, even while
successfully improving convergent-request latency. `benchmark/report.py`
reports both sides of this trade explicitly rather than only the
convergent-side win.

## ADR 5: In-process benchmark harness, not HTTP load testing

inference-server's `benchmark/load_test.py` drives traffic over real
HTTP against a running server process, which is the right choice when
the question is "how does this behave as a network service." This
project's `benchmark/routing_eval.py` instead drives the scheduler
in-process, loading the model once and reusing it across all three
strategy runs. The research question here is routing quality under a
fixed model and fixed batch capacity, not HTTP serving overhead --
spinning up a fresh 7B-parameter server process three times would add
minutes of model-load noise per comparison for no benefit to that
question. `src/server.py` still exists and is a real, runnable FastAPI
service (one strategy per process, matching inference-server's
one-backend-per-process convention) for anyone who wants to point load
at it over HTTP; it is just not how this project's own headline
numbers are produced.

---

## Known Limitations

1. **No paged attention / fixed memory pool** -- see ADR 2. Peak memory
   during the benchmark scales with `max_batch_size × max sequence
   length`, not a bounded pool. Fine for this project's single-model,
   bounded-concurrency workload; would need real paging to serve
   production multi-tenant traffic at this model size.
2. **Attention math is HF's, unmodified** -- see ADR 1. This means no
   custom kernel fusion, no speculative decoding, no quantization; this
   project's contribution is the routing logic and the batched-cache
   bridge around an otherwise-stock model.
3. **Greedy decoding only**, matching both early_detection (so serving
   activations match the training distribution) and inference-server
   (deterministic, comparable runs, no sampling-seed noise).
4. **Preemption is recompute-based, not swap-based** -- see ADR 4. Under
   sustained high contention with many divergent requests,
   `probe_deprioritize` could show elevated tail latency or wasted total
   compute from repeated restarts; `benchmark/report.py`'s preemption
   count is the diagnostic for this.
5. **The probe's AUC (0.612) is not close to the ceiling.**
   early_detection's own limitations section already says this: 0.612 is
   statistically significant but not obviously "operationally decisive."
   This project's Definition of Done does not require
   `probe_deprioritize` or `probe_terminate` to beat `baseline` -- a
   negative result (probe signal too weak, or gate overhead cancels the
   scheduling benefit) is a complete, reportable, and equally honest
   outcome. See README "What a Null or Negative Result Looks Like."
6. **Single model, single hardware target.** Same scope constraint
   early_detection names for itself: whether any of this generalizes past
   DeepSeek-R1-Distill-Qwen-7B on an A5000-class GPU is untested.
