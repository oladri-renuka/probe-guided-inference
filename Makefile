.PHONY: install test lint verify train-probe benchmark report serve-baseline serve-terminate serve-deprioritize clean

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	ruff check .

# GPU-required steps (RunPod / any CUDA box with the real model). Run in
# this order -- each one assumes the previous has succeeded.
verify:
	python scripts/verify_setup.py

train-probe:
	python scripts/train_probe.py --early-detection-dir ../early_detection

benchmark:
	python -m benchmark.routing_eval --out results/routing_eval.json

report:
	python -m benchmark.report --in-path results/routing_eval.json --out results/report.md

serve-baseline:
	PGI_STRATEGY=baseline uvicorn src.server:app --host 0.0.0.0 --port 8000

serve-terminate:
	PGI_STRATEGY=probe_terminate uvicorn src.server:app --host 0.0.0.0 --port 8000

serve-deprioritize:
	PGI_STRATEGY=probe_deprioritize uvicorn src.server:app --host 0.0.0.0 --port 8000

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
