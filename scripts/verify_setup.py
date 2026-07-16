"""
Pre-flight verification -- run BEFORE the expensive 200-problem x 3-strategy
benchmark. Adapted from early_detection/verify_hooks.py's "catch everything
before spending GPU hours" philosophy, extended with the one check that
project never needed: numerical parity between this project's hand-built
BATCHED KV-cache bridge (src/hf_cache_bridge.py) and plain
`model.generate()`.

That parity check is the single most important thing to verify here. The
padded/masked batched decode path is new code with real correctness risk
(a wrong attention-mask offset or an off-by-one in the left-pad/right-trim
logic would produce plausible-looking but WRONG token soup, or worse,
silently-wrong hidden states that get fed to the probe) -- and unlike a bug
in, say, the FastAPI layer, this class of bug would not crash, it would
just quietly serve incorrect generations and incorrect routing decisions.
Comparing against `model.generate()` on identical inputs, token-for-token,
is the check that catches it.

Usage:
    python scripts/verify_setup.py
"""

import asyncio
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import model_runner  # noqa: E402
from src.config import settings  # noqa: E402
from src.hf_cache_bridge import BatchedSequenceCache  # noqa: E402
from src.probe import LayerActivationCapture, load_probe  # noqa: E402
from src.queue import GenerationRequest  # noqa: E402


def main():
    errors = []
    print("=" * 60)
    print("VERIFICATION: probe-guided-inference")
    print("=" * 60)

    # 1. Load model
    print(f"\n[1/6] Loading model {settings.model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name, torch_dtype=getattr(torch, settings.torch_dtype), device_map=device,
    )
    model.eval()
    if device == "cuda":
        vram = torch.cuda.memory_allocated(0) / 1e9
        print(f"  Model loaded. VRAM: {vram:.2f} GB")
        if vram > 20:
            errors.append(f"VRAM usage {vram:.1f}GB is dangerously high for a 24GB A5000")
    else:
        print("  Model loaded on CPU (no CUDA device found -- fine for this check, "
              "not representative of serving performance).")

    # 2. Token IDs
    print("\n[2/6] Verifying think_start/think_end token IDs...")
    added = tokenizer.get_added_vocab()
    inv = {v: k for k, v in added.items()}
    start_tok = inv.get(settings.think_start_token_id, "NOT FOUND")
    end_tok = inv.get(settings.think_end_token_id, "NOT FOUND")
    print(f"  think_start_token_id {settings.think_start_token_id} -> '{start_tok}'")
    print(f"  think_end_token_id   {settings.think_end_token_id} -> '{end_tok}'")
    if "<think>" not in start_tok:
        errors.append(f"think_start_token_id maps to '{start_tok}', expected '<think>'")
    if "</think>" not in end_tok:
        errors.append(f"think_end_token_id maps to '{end_tok}', expected '</think>'")

    # 3. Probe (optional at this stage -- warn, don't fail, if untrained)
    print("\n[3/6] Checking probe artifact...")
    gate = None
    if settings.probe_weights_path.exists():
        try:
            probe = load_probe(settings.probe_weights_path, settings.probe_layer, settings.checkpoint_position)
            from src.gate import ClassificationGate
            gate = ClassificationGate(probe)
            print(f"  Probe loaded: layer={probe.trained_layer} checkpoint={probe.trained_checkpoint} "
                  f"cv_auc={probe.train_auc:.3f}")
            if probe.expected_hidden_dim != model.config.hidden_size:
                errors.append(
                    f"Probe expects hidden_dim={probe.expected_hidden_dim}, model has "
                    f"hidden_size={model.config.hidden_size} -- wrong model for this probe?"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Probe exists but failed to load: {exc!r}")
    else:
        print(f"  No probe at {settings.probe_weights_path} yet -- run scripts/train_probe.py "
              f"before benchmarking probe_terminate/probe_deprioritize. baseline strategy "
              f"doesn't need one, so this is not a hard failure here.")

    # 4. Chat template
    print("\n[4/6] Verifying chat template...")
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}], add_generation_prompt=True, tokenize=False,
    )
    if "<think>" not in prompt_text:
        errors.append("Chat template does not include <think> -- thinking-mode assumption is wrong")
    else:
        print("  CONFIRMED: template includes <think>")

    # 5. Single-sequence parity: hand-built cache bridge vs model.generate()
    print("\n[5/6] Verifying batched cache bridge parity (single sequence, 15 tokens)...")
    # GenerationRequest.result is an asyncio.Future created via
    # asyncio.get_running_loop() (see src/queue.py) -- constructing one
    # outside a running loop raises RuntimeError, so these checks need
    # asyncio.run() even though neither actually awaits anything.
    asyncio.run(_check_single_sequence_parity(model, tokenizer, errors))

    # 6. Multi-sequence parity: TWO different-length prompts decoded together
    print("\n[6/6] Verifying batched cache bridge parity (two sequences, different lengths)...")
    asyncio.run(_check_multi_sequence_parity(model, tokenizer, errors))

    print("\n" + "=" * 60)
    if errors:
        print(f"FAILED -- {len(errors)} error(s):")
        for e in errors:
            print(f"  x {e}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        print(f"  Model: {settings.model_name}")
        print(f"  Probe layer: {settings.probe_layer}  Checkpoint: {settings.checkpoint_position}")
        print(f"  Gate ready: {gate is not None}")
        print("\nSafe to proceed with: python -m benchmark.routing_eval")
    print("=" * 60)


async def _check_single_sequence_parity(model, tokenizer, errors: list) -> None:
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 17 * 24? Answer directly."}],
        add_generation_prompt=True, tokenize=False,
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"][0].tolist()
    n_tokens = 15

    with torch.no_grad():
        ref_input = torch.tensor([prompt_ids], device=model.device)
        ref_output = model.generate(
            ref_input, max_new_tokens=n_tokens, do_sample=False, temperature=None, top_p=None,
        )
        ref_tokens = ref_output[0, len(prompt_ids):].tolist()

        cache = BatchedSequenceCache()
        capture = LayerActivationCapture(model, settings.probe_layer)
        req = GenerationRequest(prompt=prompt_text, max_tokens=n_tokens, prompt_ids=prompt_ids)
        model_runner.run_prefill(model, cache, capture, req, tokenizer.eos_token_id, n_tokens)
        for _ in range(n_tokens - 1):
            if req.n_generated >= n_tokens:
                break
            model_runner.run_decode_step(model, cache, capture, None, [req], tokenizer.eos_token_id, n_tokens)
        capture.remove()

    bridge_tokens = req.generated_ids[:n_tokens]
    if bridge_tokens == ref_tokens[:len(bridge_tokens)]:
        print(f"  MATCH: {len(bridge_tokens)} tokens identical to model.generate()")
    else:
        errors.append(
            f"Single-sequence parity FAILED: bridge produced {bridge_tokens}, "
            f"model.generate() produced {ref_tokens[:len(bridge_tokens)]} -- "
            f"the batched cache bridge is not numerically equivalent to native generation. "
            f"Do not trust probe routing decisions until this is fixed."
        )


async def _check_multi_sequence_parity(model, tokenizer, errors: list) -> None:
    prompts = [
        "What is 2+2? Answer directly.",
        "Write the first five prime numbers, separated by commas, and nothing else.",
    ]
    n_tokens = 12
    prompt_ids_list = [
        tokenizer(
            tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False),
            return_tensors="pt",
        )["input_ids"][0].tolist()
        for p in prompts
    ]

    with torch.no_grad():
        ref_tokens_list = []
        for prompt_ids in prompt_ids_list:
            ref_input = torch.tensor([prompt_ids], device=model.device)
            ref_output = model.generate(ref_input, max_new_tokens=n_tokens, do_sample=False, temperature=None, top_p=None)
            ref_tokens_list.append(ref_output[0, len(prompt_ids):].tolist())

        cache = BatchedSequenceCache()
        capture = LayerActivationCapture(model, settings.probe_layer)
        reqs = [
            GenerationRequest(prompt=p, max_tokens=n_tokens, prompt_ids=ids)
            for p, ids in zip(prompts, prompt_ids_list, strict=False)
        ]
        for req in reqs:
            model_runner.run_prefill(model, cache, capture, req, tokenizer.eos_token_id, n_tokens)
        for _ in range(n_tokens - 1):
            active = [r for r in reqs if r.n_generated < n_tokens]
            if not active:
                break
            model_runner.run_decode_step(model, cache, capture, None, active, tokenizer.eos_token_id, n_tokens)
        capture.remove()

    all_match = True
    for i, (req, ref_tokens) in enumerate(zip(reqs, ref_tokens_list, strict=False)):
        bridge_tokens = req.generated_ids[:n_tokens]
        if bridge_tokens != ref_tokens[:len(bridge_tokens)]:
            all_match = False
            errors.append(
                f"Multi-sequence parity FAILED for prompt {i} ({prompts[i]!r}): "
                f"bridge produced {bridge_tokens}, model.generate() produced "
                f"{ref_tokens[:len(bridge_tokens)]} -- co-batching different-length sequences "
                f"through the padded cache bridge is corrupting at least one of them "
                f"(likely a left-padding / attention-mask / position_ids bug in "
                f"src/hf_cache_bridge.py)."
            )
    if all_match:
        print(f"  MATCH: both sequences identical to model.generate() when decoded together "
              f"in one batched cache (different prompt lengths: "
              f"{[len(p) for p in prompt_ids_list]})")


if __name__ == "__main__":
    main()
