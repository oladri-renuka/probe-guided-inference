#!/bin/bash
# One-time RunPod setup, mirroring early_detection/setup_runpod.sh so both
# projects' pods are provisioned the same way.
set -euo pipefail

echo "=== probe-guided-inference: RunPod setup ==="

python3 -m venv --system-site-packages venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# --system-site-packages above matters: a plain `python3 -m venv venv` would
# shadow RunPod's preinstalled system torch (built to match the pod's actual
# driver) with a fresh `pip install torch` that grabs whatever the latest
# PyPI wheel is -- which is very often built against a newer CUDA runtime
# than the pod's NVIDIA driver supports, and fails with "The NVIDIA driver
# on your system is too old" even though the GPU and driver are both fine.
# --system-site-packages inherits the working system torch instead; pip
# then only installs what's still missing (transformers>=5.0, etc.) without
# touching torch, since its `torch>=2.1.0` constraint is already satisfied.

echo ""
echo "Setup complete. Next steps:"
echo "  1. python scripts/verify_setup.py                 # < 1 min, catches config issues"
echo "  2. python scripts/train_probe.py --early-detection-dir ../early_detection"
echo "  3. python -m benchmark.routing_eval                # the 200-problem x 3-strategy run"
echo "  4. python -m benchmark.report                      # aggregate into results/report.md"
