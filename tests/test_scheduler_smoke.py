"""
End-to-end scheduler smoke tests: real tiny Qwen2 model, real
ContinuousBatcher.run_forever() tick loop, all three routing strategies.
Uses stub probes (deterministic predict_proba, no real sklearn fit needed)
so routing OUTCOMES are controllable and assertable -- these tests verify
scheduling MECHANICS (admission, eviction, termination, preemption
bookkeeping), not probe classification quality, which is a completely
separate, already-published question (early_detection/README.md).
"""

import asyncio

import numpy as np
import pytest

from src.gate import ClassificationGate
from src.probe import ProbeArtifact
from src.queue import RequestQueue
from src.scheduler import ContinuousBatcher, RoutingStrategy
from tests.conftest import TINY_EOS_TOKEN_ID, TINY_HIDDEN_DIM

CHECKPOINT = 3        # small, so the gate fires quickly in a CPU test
MAX_TOKENS = 8         # > CHECKPOINT, so requests survive long enough to be routed
N_REQUESTS = 6
MAX_BATCH_SIZE = 2     # << N_REQUESTS, guarantees queueing contention


class _ConstantPipeline:
    def __init__(self, prob_convergent: float):
        self.prob_convergent = prob_convergent

    def predict_proba(self, X):
        return np.tile([1 - self.prob_convergent, self.prob_convergent], (X.shape[0], 1))


class _AlternatingPipeline:
    """First call -> divergent, second -> convergent, third -> divergent, ...
    Deterministic and order-dependent, so tests can assert a MIX of both
    routing outcomes rather than only ever hitting one branch."""

    def __init__(self):
        self.calls = 0

    def predict_proba(self, X):
        self.calls += 1
        prob_convergent = 0.9 if self.calls % 2 == 0 else 0.1
        return np.tile([1 - prob_convergent, prob_convergent], (X.shape[0], 1))


def make_gate(pipeline, checkpoint=CHECKPOINT) -> ClassificationGate:
    artifact = ProbeArtifact(
        pipeline=pipeline, trained_layer=1, trained_checkpoint=checkpoint,
        expected_hidden_dim=TINY_HIDDEN_DIM, n_train_samples=200, train_auc=0.61,
    )
    return ClassificationGate(artifact)


async def _run_batcher_to_completion(batcher: ContinuousBatcher, reqs: list, timeout_s: float = 30.0) -> None:
    task = asyncio.create_task(batcher.run_forever())
    try:
        await asyncio.wait_for(asyncio.gather(*(r.result for r in reqs)), timeout=timeout_s)
    finally:
        batcher.stop()
        batcher.close()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_baseline_strategy_finishes_every_request_with_no_gate(tiny_model):
    queue = RequestQueue()
    batcher = ContinuousBatcher(
        model=tiny_model, tokenizer=_FakeTokenizer(), queue=queue,
        strategy=RoutingStrategy.BASELINE, gate=None, max_batch_size=MAX_BATCH_SIZE,
        max_tokens_cap=MAX_TOKENS,
    )
    reqs = [queue.submit(f"p{i}", MAX_TOKENS, [i + 1, i + 2, i + 3]) for i in range(N_REQUESTS)]

    await _run_batcher_to_completion(batcher, reqs)

    assert batcher.total_finished == N_REQUESTS
    assert batcher.total_probe_terminated == 0
    assert batcher.total_preemptions == 0
    for req in reqs:
        assert req.routing_decision is None
        assert req.n_generated == MAX_TOKENS
        assert req.finish_reason in ("max_tokens", "eos")


@pytest.mark.asyncio
async def test_probe_terminate_ends_requests_early_when_always_divergent(tiny_model):
    gate = make_gate(_ConstantPipeline(prob_convergent=0.05))  # confidence 0.95, always "divergent"
    queue = RequestQueue()
    batcher = ContinuousBatcher(
        model=tiny_model, tokenizer=_FakeTokenizer(), queue=queue,
        strategy=RoutingStrategy.PROBE_TERMINATE, gate=gate, max_batch_size=MAX_BATCH_SIZE,
        max_tokens_cap=MAX_TOKENS, terminate_confidence_threshold=0.7,
    )
    reqs = [queue.submit(f"p{i}", MAX_TOKENS, [i + 1, i + 2, i + 3]) for i in range(N_REQUESTS)]

    await _run_batcher_to_completion(batcher, reqs)

    assert batcher.total_finished == N_REQUESTS
    assert batcher.total_probe_terminated == N_REQUESTS
    for req in reqs:
        assert req.terminated_by_probe is True
        assert req.finish_reason == "probe_terminated"
        assert req.n_generated == CHECKPOINT  # cut short well before MAX_TOKENS
        assert req.routing_decision.is_divergent is True


@pytest.mark.asyncio
async def test_probe_terminate_never_terminates_when_always_convergent(tiny_model):
    gate = make_gate(_ConstantPipeline(prob_convergent=0.95))
    queue = RequestQueue()
    batcher = ContinuousBatcher(
        model=tiny_model, tokenizer=_FakeTokenizer(), queue=queue,
        strategy=RoutingStrategy.PROBE_TERMINATE, gate=gate, max_batch_size=MAX_BATCH_SIZE,
        max_tokens_cap=MAX_TOKENS, terminate_confidence_threshold=0.7,
    )
    reqs = [queue.submit(f"p{i}", MAX_TOKENS, [i + 1, i + 2, i + 3]) for i in range(N_REQUESTS)]

    await _run_batcher_to_completion(batcher, reqs)

    assert batcher.total_probe_terminated == 0
    for req in reqs:
        assert req.terminated_by_probe is False
        assert req.n_generated == MAX_TOKENS
        assert req.routing_decision.is_divergent is False


@pytest.mark.asyncio
async def test_probe_deprioritize_preempts_active_divergent_requests_under_contention(tiny_model):
    gate = make_gate(_ConstantPipeline(prob_convergent=0.05))  # everyone is "divergent" -> priority 1
    queue = RequestQueue()
    batcher = ContinuousBatcher(
        model=tiny_model, tokenizer=_FakeTokenizer(), queue=queue,
        strategy=RoutingStrategy.PROBE_DEPRIORITIZE, gate=gate, max_batch_size=MAX_BATCH_SIZE,
        max_tokens_cap=MAX_TOKENS,
    )
    reqs = [queue.submit(f"p{i}", MAX_TOKENS, [i + 1, i + 2, i + 3]) for i in range(N_REQUESTS)]

    await _run_batcher_to_completion(batcher, reqs)

    # Every request eventually finishes -- preemption requeues, it never drops.
    assert batcher.total_finished == N_REQUESTS
    assert sum(r.n_generated for r in reqs) == N_REQUESTS * MAX_TOKENS
    # With MAX_BATCH_SIZE=2 << N_REQUESTS=6 and everyone divergent, later
    # arrivals (still priority=0, unclassified) must have preempted at
    # least one already-active, now-deprioritized request to get a slot.
    assert batcher.total_preemptions > 0


@pytest.mark.asyncio
async def test_probe_deprioritize_with_mixed_outcomes_never_loses_a_request(tiny_model):
    """Alternating convergent/divergent classifications -- the more
    realistic case where SOME requests get deprioritized and others don't.
    The invariant under test is conservation: every submitted request
    completes exactly once with the tokens it was entitled to, regardless
    of how much preemption churn happened in between."""
    gate = make_gate(_AlternatingPipeline())
    queue = RequestQueue()
    batcher = ContinuousBatcher(
        model=tiny_model, tokenizer=_FakeTokenizer(), queue=queue,
        strategy=RoutingStrategy.PROBE_DEPRIORITIZE, gate=gate, max_batch_size=MAX_BATCH_SIZE,
        max_tokens_cap=MAX_TOKENS,
    )
    reqs = [queue.submit(f"p{i}", MAX_TOKENS, [i + 1, i + 2, i + 3]) for i in range(N_REQUESTS)]

    await _run_batcher_to_completion(batcher, reqs)

    assert batcher.total_finished == N_REQUESTS
    labels = {req.routing_decision.label for req in reqs}
    assert labels == {"convergent", "divergent"}, "expected a genuine mix, not all-one-label"
    for req in reqs:
        assert req.n_generated == MAX_TOKENS
        assert req.finish_reason in ("max_tokens", "eos")


class _FakeTokenizer:
    """Only what the scheduler/model_runner actually touch on a tokenizer:
    eos_token_id (for the finish check) and decode (for _finish's result
    text). Avoids loading a real HF tokenizer for a test that never
    constructs prompts through a chat template."""

    eos_token_id = TINY_EOS_TOKEN_ID

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)
