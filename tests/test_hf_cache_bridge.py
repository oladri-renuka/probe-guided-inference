"""
Numerical parity tests for src/hf_cache_bridge.py -- THE highest-risk piece
of this project's own code (everything else either reuses
inference-server's established design or calls straight into HF/sklearn).
A bug in the pad/mask/position_ids bookkeeping here would not crash; it
would silently produce a different token sequence than a correct
implementation, and by extension feed the probe hidden states it was never
trained on. These tests are the CPU-runnable version of
scripts/verify_setup.py's parity checks, run here against the tiny
synthetic Qwen2 model (see tests/conftest.py) so they execute in CI
without a GPU or the real 7B model.
"""

import torch

from src import model_runner
from src.hf_cache_bridge import BatchedSequenceCache
from src.probe import LayerActivationCapture
from src.queue import GenerationRequest
from tests.conftest import TINY_EOS_TOKEN_ID


@torch.no_grad()
def _reference_generate(model, prompt_ids: list[int], n_tokens: int) -> list[int]:
    """Plain sequential HF generation (batch size 1, one call per token) --
    the ground truth every bridge-produced sequence must match exactly
    under greedy decoding."""
    input_ids = torch.tensor([prompt_ids])
    out = model(input_ids=input_ids, use_cache=True)
    generated = [int(torch.argmax(out.logits[0, -1, :]).item())]
    past = out.past_key_values
    for _ in range(n_tokens - 1):
        out = model(input_ids=torch.tensor([[generated[-1]]]), past_key_values=past, use_cache=True)
        past = out.past_key_values
        generated.append(int(torch.argmax(out.logits[0, -1, :]).item()))
    return generated


def _run_via_bridge(model, cache, capture, req, n_tokens) -> None:
    model_runner.run_prefill(model, cache, capture, req, TINY_EOS_TOKEN_ID, n_tokens)
    for _ in range(n_tokens - 1):
        if req.n_generated >= n_tokens:
            break
        model_runner.run_decode_step(model, cache, capture, None, [req], TINY_EOS_TOKEN_ID, n_tokens)


PROMPT_A = [1, 2, 3, 4, 5, 6, 7]
PROMPT_B = [10, 11, 12]
PROMPT_C = [20, 21, 22, 23, 24, 25, 26, 27, 28]
N_TOKENS = 12


async def test_single_sequence_matches_reference_generation(tiny_model):
    ref = _reference_generate(tiny_model, PROMPT_A, N_TOKENS)

    cache = BatchedSequenceCache()
    capture = LayerActivationCapture(tiny_model, layer_idx=1)
    req = GenerationRequest(prompt="a", max_tokens=N_TOKENS, prompt_ids=PROMPT_A)
    _run_via_bridge(tiny_model, cache, capture, req, N_TOKENS)
    capture.remove()

    assert req.generated_ids == ref


async def test_two_different_length_sequences_batched_together_match_reference(tiny_model):
    """The core claim of this module: co-batching sequences of DIFFERENT
    lengths through one padded/masked forward pass must not perturb either
    sequence's output relative to decoding it alone. This is what a
    left-padding or attention-mask bug would break first."""
    ref_a = _reference_generate(tiny_model, PROMPT_A, N_TOKENS)
    ref_b = _reference_generate(tiny_model, PROMPT_B, N_TOKENS)

    cache = BatchedSequenceCache()
    capture = LayerActivationCapture(tiny_model, layer_idx=1)
    req_a = GenerationRequest(prompt="a", max_tokens=N_TOKENS, prompt_ids=PROMPT_A)
    req_b = GenerationRequest(prompt="b", max_tokens=N_TOKENS, prompt_ids=PROMPT_B)
    model_runner.run_prefill(tiny_model, cache, capture, req_a, TINY_EOS_TOKEN_ID, N_TOKENS)
    model_runner.run_prefill(tiny_model, cache, capture, req_b, TINY_EOS_TOKEN_ID, N_TOKENS)
    for _ in range(N_TOKENS - 1):
        active = [r for r in (req_a, req_b) if r.n_generated < N_TOKENS]
        if not active:
            break
        model_runner.run_decode_step(tiny_model, cache, capture, None, active, TINY_EOS_TOKEN_ID, N_TOKENS)
    capture.remove()

    assert req_a.generated_ids == ref_a
    assert req_b.generated_ids == ref_b


async def test_three_sequences_with_staggered_admission_match_reference(tiny_model):
    """Admits req_a and req_b first, runs a few ticks, THEN admits req_c
    mid-stream -- exercising build_batch()/scatter_updated_batch() at a
    changing max_len across ticks, not just a fixed batch formed once."""
    ref_a = _reference_generate(tiny_model, PROMPT_A, N_TOKENS)
    ref_b = _reference_generate(tiny_model, PROMPT_B, N_TOKENS)
    ref_c = _reference_generate(tiny_model, PROMPT_C, N_TOKENS)

    cache = BatchedSequenceCache()
    capture = LayerActivationCapture(tiny_model, layer_idx=1)
    req_a = GenerationRequest(prompt="a", max_tokens=N_TOKENS, prompt_ids=PROMPT_A)
    req_b = GenerationRequest(prompt="b", max_tokens=N_TOKENS, prompt_ids=PROMPT_B)
    req_c = GenerationRequest(prompt="c", max_tokens=N_TOKENS, prompt_ids=PROMPT_C)

    model_runner.run_prefill(tiny_model, cache, capture, req_a, TINY_EOS_TOKEN_ID, N_TOKENS)
    model_runner.run_prefill(tiny_model, cache, capture, req_b, TINY_EOS_TOKEN_ID, N_TOKENS)

    active = [req_a, req_b]
    tick = 0
    c_admitted = False
    while any(r.n_generated < N_TOKENS for r in (req_a, req_b, req_c)):
        if tick == 3 and not c_admitted:
            model_runner.run_prefill(tiny_model, cache, capture, req_c, TINY_EOS_TOKEN_ID, N_TOKENS)
            active.append(req_c)
            c_admitted = True
        active = [r for r in active if r.n_generated < N_TOKENS]
        if active:
            model_runner.run_decode_step(tiny_model, cache, capture, None, active, TINY_EOS_TOKEN_ID, N_TOKENS)
        tick += 1
    capture.remove()

    assert req_a.generated_ids == ref_a
    assert req_b.generated_ids == ref_b
    assert req_c.generated_ids == ref_c


async def test_evicting_a_sequence_does_not_perturb_the_survivor(tiny_model):
    """Admits two sequences, finishes/evicts one early (simulating natural
    completion), and confirms the survivor's continuation still matches
    solo reference generation -- i.e. eviction actually removes the freed
    row rather than leaving a stale/zeroed row that could leak into the
    survivor's attention via the shared batch tensor."""
    ref_a = _reference_generate(tiny_model, PROMPT_A, N_TOKENS)

    cache = BatchedSequenceCache()
    capture = LayerActivationCapture(tiny_model, layer_idx=1)
    req_a = GenerationRequest(prompt="a", max_tokens=N_TOKENS, prompt_ids=PROMPT_A)
    req_b = GenerationRequest(prompt="b", max_tokens=4, prompt_ids=PROMPT_B)  # finishes after 4 tokens

    model_runner.run_prefill(tiny_model, cache, capture, req_a, TINY_EOS_TOKEN_ID, N_TOKENS)
    model_runner.run_prefill(tiny_model, cache, capture, req_b, TINY_EOS_TOKEN_ID, 4)

    active = [req_a, req_b]
    for _ in range(N_TOKENS - 1):
        active = [r for r in active if r.n_generated < r.max_tokens]
        if not active:
            break
        finished_ids = model_runner.run_decode_step(
            tiny_model, cache, capture, None, active, TINY_EOS_TOKEN_ID, N_TOKENS
        )
        for rid in finished_ids:
            cache.evict(rid)
            active = [r for r in active if r.request_id != rid]
    capture.remove()

    assert req_a.generated_ids == ref_a
    assert req_b.n_generated == 4
