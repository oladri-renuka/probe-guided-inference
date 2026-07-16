"""
Continuous batching scheduler with probe-guided routing.

Structurally this is inference-server's `src/batcher.py` admit / decode /
evict tick loop (see that project's docstring for the base algorithm,
reused here near-verbatim), extended with a third phase specific to this
project: ROUTE, which reacts to `req.routing_decision` right after the
gate fires and applies one of three strategies:

  baseline            -- routing_decision is read but never acted on.
  probe_terminate      -- a divergent decision above
                          TERMINATE_CONFIDENCE_THRESHOLD ends the request
                          immediately with finish_reason="probe_terminated"
                          instead of letting it run to its natural
                          EOS/max_tokens.
  probe_deprioritize   -- a divergent decision sets req.priority = 1. That
                          alone does nothing until the active batch is at
                          capacity AND a priority-0 request is waiting --
                          only then does a priority-1 active request get
                          preempted (recompute-based, like
                          inference-server's cache-exhaustion preemption)
                          to free its slot for the higher-priority waiter.

There is no fixed KV-cache block pool here (contrast inference-server's
PagedKVCache) -- see docs/ARCHITECTURE.md for why paging was out of scope
for this project's research question. Capacity is governed purely by
`max_batch_size`, which is why deprioritize's preemption trigger is
"batch is at max_batch_size", not "cache is out of blocks".
"""

import asyncio
import logging
import time
from enum import Enum

from src import model_runner
from src.gate import ClassificationGate
from src.hf_cache_bridge import BatchedSequenceCache
from src.probe import LayerActivationCapture
from src.queue import GenerationRequest, RequestQueue

logger = logging.getLogger("scheduler")


class RoutingStrategy(str, Enum):
    BASELINE = "baseline"
    PROBE_TERMINATE = "probe_terminate"
    PROBE_DEPRIORITIZE = "probe_deprioritize"


class ContinuousBatcher:
    def __init__(
        self,
        model,
        tokenizer,
        queue: RequestQueue,
        strategy: RoutingStrategy = RoutingStrategy.BASELINE,
        gate: ClassificationGate | None = None,
        max_batch_size: int = 8,
        max_tokens_cap: int = 10_000,
        terminate_confidence_threshold: float = 0.7,
        tick_sleep: float = 0.0,
    ):
        if strategy != RoutingStrategy.BASELINE and gate is None:
            raise ValueError(f"strategy={strategy} requires a ClassificationGate; got gate=None")

        self.model = model
        self.tokenizer = tokenizer
        self.queue = queue
        self.strategy = strategy
        self.gate = gate
        self.max_batch_size = max_batch_size
        self.max_tokens_cap = max_tokens_cap
        self.terminate_confidence_threshold = terminate_confidence_threshold
        self.tick_sleep = tick_sleep

        self.eos_token_id = tokenizer.eos_token_id
        self.cache = BatchedSequenceCache()
        self.capture: LayerActivationCapture | None = (
            LayerActivationCapture(model, gate.probe.trained_layer) if gate is not None else None
        )

        self.active: dict[int, GenerationRequest] = {}
        self._stop = False

        # counters surfaced on /health and consumed by benchmark/report.py
        self.total_admitted = 0
        self.total_finished = 0
        self.total_probe_terminated = 0
        self.total_preemptions = 0

    # ── ADMIT ────────────────────────────────────────────────────────────────
    def _admit_one(self) -> None:
        req = self.queue.popleft()
        already_finished = model_runner.run_prefill(
            self.model, self.cache, self.capture, req, self.eos_token_id, self.max_tokens_cap
        )
        self.total_admitted += 1
        if already_finished:
            self._finish(req)
        else:
            self.active[req.request_id] = req

    def _find_deprioritize_victim(self) -> GenerationRequest | None:
        """Among currently active, deprioritized (priority=1) requests,
        picks the one with the LEAST generated progress -- minimizing the
        recompute cost of preempting it, since recompute-based preemption
        discards everything generated so far (see _preempt docstring)."""
        candidates = [r for r in self.active.values() if r.priority == 1]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.n_generated)

    def _preempt(self, victim: GenerationRequest) -> None:
        """Evicts an active deprioritized request back to the pending
        queue, discarding its progress so far. This is recompute-based
        preemption -- the same strategy (and the same cost) as
        inference-server's cache-exhaustion preemption, just triggered by
        routing priority instead of KV-cache exhaustion."""
        self.cache.evict(victim.request_id)
        self.active.pop(victim.request_id, None)
        victim.generated_ids = []
        victim.position = 0
        victim.routing_decision = None  # will be reclassified fresh after resuming
        victim.routed = False
        self.total_preemptions += 1
        self.queue.push_front_priority_aware(victim)

    def _try_admit(self) -> None:
        while len(self.active) < self.max_batch_size and len(self.queue) > 0:
            self._admit_one()

        if self.strategy != RoutingStrategy.PROBE_DEPRIORITIZE:
            return

        # Deprioritize's preemption path: only triggers when the batch is
        # genuinely full AND a normal-priority request is waiting behind a
        # deprioritized one that's currently occupying a slot.
        attempts = 0
        while (
            len(self.active) >= self.max_batch_size
            and len(self.queue) > 0
            and self.queue.peek().priority == 0
            and attempts < self.max_batch_size
        ):
            victim = self._find_deprioritize_victim()
            if victim is None:
                break
            self._preempt(victim)
            self._admit_one()
            attempts += 1

    # ── ROUTE (probe-guided reaction, runs right after decode) ──────────────
    def _apply_routing(self) -> list:
        """Reads routing_decision on every active request that just got one
        this tick and applies the active strategy. Returns request_ids that
        should be evicted as a DIRECT result of routing (probe_terminate
        only) -- separate from the natural EOS/max_tokens finished_ids
        run_decode_step already returns."""
        if self.gate is None:
            return []

        newly_routed = [r for r in self.active.values() if r.routing_decision is not None and not r.routed]
        terminated_ids = []
        for req in newly_routed:
            req.routed = True
            if not req.routing_decision.is_divergent:
                continue

            if self.strategy == RoutingStrategy.PROBE_TERMINATE:
                if req.routing_decision.confidence > self.terminate_confidence_threshold:
                    req.terminated_by_probe = True
                    req.finish_reason = "probe_terminated"
                    terminated_ids.append(req.request_id)

            elif self.strategy == RoutingStrategy.PROBE_DEPRIORITIZE:
                req.priority = 1

        return terminated_ids

    # ── EVICT ────────────────────────────────────────────────────────────────
    def _finish(self, req: GenerationRequest) -> None:
        self.cache.evict(req.request_id)
        self.active.pop(req.request_id, None)
        self.total_finished += 1
        req.completion_time = time.time()
        if req.terminated_by_probe:
            self.total_probe_terminated += 1
        text = self.tokenizer.decode(req.generated_ids, skip_special_tokens=False)
        if req.stream and req.token_queue is not None:
            req.token_queue.put_nowait(None)
        if not req.result.done():
            req.result.set_result(text)

    # ── Main tick loop ───────────────────────────────────────────────────────
    async def run_forever(self) -> None:
        while not self._stop:
            self._try_admit()

            if self.active:
                active_reqs = list(self.active.values())
                finished_ids = model_runner.run_decode_step(
                    self.model, self.cache, self.capture, self.gate,
                    active_reqs, self.eos_token_id, self.max_tokens_cap,
                )
                terminated_ids = self._apply_routing()

                for rid in set(finished_ids) | set(terminated_ids):
                    self._finish(self.active[rid])

            await asyncio.sleep(self.tick_sleep)

    def stop(self) -> None:
        self._stop = True

    def close(self) -> None:
        if self.capture is not None:
            self.capture.remove()
