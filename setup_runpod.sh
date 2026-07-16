#!/bin/bash
# One-time RunPod setup, mirroring early_detection/setup_runpod.sh so both
# projects' pods are provisioned the same way.
set -euo pipefail

echo "=== probe-guided-inference: RunPod setup ==="

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete. Next steps:"
echo "  1. python scripts/verify_setup.py                 # < 1 min, catches config issues"
echo "  2. python scripts/train_probe.py --early-detection-dir ../early_detection"
echo "  3. python -m benchmark.routing_eval                # the 200-problem x 3-strategy run"
echo "  4. python -m benchmark.report                      # aggregate into results/report.md"
