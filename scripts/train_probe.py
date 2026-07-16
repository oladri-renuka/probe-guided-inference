"""
Fits and exports the probe artifact this project routes on.

early_detection/analyze.py deliberately never saves a fitted model -- it
only ever reports 5-fold CV metrics (see its `run_cv_probe`, which fits a
fresh StandardScaler+LogisticRegression per fold and discards it). That's
the right call for a research script whose output is a number in a paper,
but it means there is no `probe_layer16.pkl` sitting in early_detection/
for this project to load -- the projB build-order doc's Step 1
(`with open("../early_detection/results/probe_layer16.pkl", "rb")`)
assumes an artifact that has to be produced first. This script produces it:

  1. Load the same records.json + checkpoint_acts.pt[150] early_detection's
     analyze.py loads.
  2. Refit the SAME preprocessing (StandardScaler) + model (LogisticRegression,
     C=1.0, lbfgs) on ALL available samples (not one CV fold) -- this is the
     artifact that actually gets served.
  3. Separately, re-run 5-fold CV (identical to analyze.py) to get an honest
     out-of-fold AUC estimate to stamp into the artifact's metadata. This
     number, NOT training-set accuracy, is what should be quoted anywhere
     this probe's quality is discussed -- a probe fit and evaluated on the
     same 200 points would silently overstate itself.
  4. Save via joblib to probe_weights/probe_layer16_cp150.pkl.

Usage:
    python scripts/train_probe.py \\
        --early-detection-dir ../early_detection \\
        --out probe_weights/probe_layer16_cp150.pkl
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import settings  # noqa: E402


def load_activations(early_detection_dir: Path, checkpoint_position: int):
    records_path = early_detection_dir / "checkpoints" / "records.json"
    acts_path = early_detection_dir / "checkpoints" / "checkpoint_acts.pt"
    if not records_path.exists() or not acts_path.exists():
        raise FileNotFoundError(
            f"Expected {records_path} and {acts_path}. These are produced by "
            f"`python early_detection/generate.py`, which takes ~8-12 hours on a GPU -- "
            f"run that first (see early_detection/README.md). This script only trains "
            f"the probe on activations that already exist; it does not generate them."
        )

    with open(records_path) as f:
        records = json.load(f)
    all_acts = torch.load(acts_path, map_location="cpu", weights_only=True)
    cp_acts = all_acts.get(checkpoint_position, {})

    valid_idx = [i for i in range(len(records)) if cp_acts.get(i) is not None]
    if len(valid_idx) < 20:
        raise ValueError(
            f"Only {len(valid_idx)} samples have an activation at checkpoint "
            f"{checkpoint_position}. Need at least 20 for a meaningful fit -- most "
            f"likely `early_detection/generate.py` hasn't finished enough samples yet, "
            f"or --checkpoint-position doesn't match one of its CHECKPOINT_POSITIONS."
        )

    X = np.stack([cp_acts[i].squeeze(0).numpy() for i in valid_idx])
    y = np.array([1 if records[i]["converged"] else 0 for i in valid_idx])
    return X, y


def cross_validated_auc(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> tuple[float, float]:
    """Identical procedure to early_detection/analyze.py's run_cv_probe --
    kept in lockstep deliberately so this script's reported AUC is
    comparable to the number in early_detection/README.md, not a
    differently-measured lookalike."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, random_state=42)
        clf.fit(X_train_s, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1])
        aucs.append(max(auc, 1 - auc))  # correct for label-flip, matching analyze.py
    aucs = np.array(aucs)
    return float(aucs.mean()), float(aucs.std())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--early-detection-dir", type=Path, default=Path("../early_detection"))
    parser.add_argument("--checkpoint-position", type=int, default=settings.checkpoint_position)
    parser.add_argument("--layer", type=int, default=settings.probe_layer)
    parser.add_argument("--out", type=Path, default=settings.probe_weights_path)
    args = parser.parse_args()

    print(f"Loading activations from {args.early_detection_dir} at checkpoint {args.checkpoint_position}...")
    X, y = load_activations(args.early_detection_dir, args.checkpoint_position)
    print(f"  {X.shape[0]} samples, hidden_dim={X.shape[1]}, "
          f"convergence rate={y.mean():.1%}")

    print("\nEstimating out-of-fold AUC (5-fold stratified CV, same procedure as "
          "early_detection/analyze.py)...")
    cv_auc_mean, cv_auc_std = cross_validated_auc(X, y)
    print(f"  CV AUC: {cv_auc_mean:.3f} +/- {cv_auc_std:.3f}")
    if X.shape[0] < 200:
        print(f"  NOTE: early_detection's headline result (AUC 0.612 at cp=150) was measured "
              f"on 200 samples. This fit uses {X.shape[0]} -- expect a noisier estimate.")

    print("\nFitting final pipeline on ALL available samples (this is what gets served)...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, random_state=42)),
    ])
    pipeline.fit(X, y)

    bundle = {
        "pipeline": pipeline,
        "trained_layer": args.layer,
        "trained_checkpoint": args.checkpoint_position,
        "expected_hidden_dim": X.shape[1],
        "n_train_samples": int(X.shape[0]),
        "train_auc": cv_auc_mean,  # out-of-fold estimate, NOT resubstitution accuracy -- see module docstring
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(bundle, args.out)
    print(f"\nSaved probe artifact to {args.out}")
    print(f"  layer={args.layer} checkpoint={args.checkpoint_position} "
          f"hidden_dim={X.shape[1]} cv_auc={cv_auc_mean:.3f}")


if __name__ == "__main__":
    main()
