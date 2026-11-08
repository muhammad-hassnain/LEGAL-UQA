"""Compare JEV (noul-mode) verdicts to the human ground truth.

Reads every ``<model>_jev_verdicts.csv`` under ``data/automated_evaluations_qa/jev/``
and reports, per model:
  - counts (n, human_pos, jev_pos)
  - accuracy of JEV binary verdict vs human binary verdict
  - Cohen's kappa
  - 2x2 confusion matrix
  - Pearson/Spearman between JEV probability and human Score (and Strict when
    present)
  - mean absolute error between JEV probability and human Score

Writes a tidy ``jev_vs_human_agreement.csv`` summary next to the per-model
files, and prints the same table.

Usage:
    python -m scripts.jev.agreement                   # all models
    python -m scripts.jev.agreement --model gpt-5.4   # one model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from scripts.jev.pipeline import HUMAN_EVAL_FILES, OUT_DIR, ROOT


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    if a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.corr(b, method=method))


def safe_kappa(human, prediction) -> float:
    """Kappa is undefined only when chance agreement is one."""
    human, prediction = np.asarray(human), np.asarray(prediction)
    if len(human) == 0 or (np.all(human == human[0]) and np.all(prediction == human[0])):
        return float('nan')
    return float(cohen_kappa_score(human, prediction))


def summarise(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    # Drop rows the human did not fully grade.
    df = df.dropna(subset=["human_score", "noul"])

    human_bin = (df["human_score"] >= 0.5).astype(int)
    jev_bin = (df["noul"] >= 0.5).astype(int)

    cm = confusion_matrix(human_bin, jev_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    out = {
        "model": csv_path.stem.removesuffix("_jev_verdicts"),
        "n": int(len(df)),
        "human_pos_rate": float(human_bin.mean()),
        "jev_pos_rate": float(jev_bin.mean()),
        "accuracy": float((human_bin == jev_bin).mean()),
        "cohens_kappa": safe_kappa(human_bin, jev_bin),
        "always_positive_accuracy": float(human_bin.mean()),
        "positive_recall": float(tp / (tp + fn)) if tp + fn else float('nan'),
        "specificity": float(tn / (tn + fp)) if tn + fp else float('nan'),
        "mae_prob_vs_score": float((df["noul"] - df["human_score"]).abs().mean()),
        "pearson_prob_vs_score": _safe_corr(df["noul"], df["human_score"], "pearson"),
        "spearman_prob_vs_score": _safe_corr(df["noul"], df["human_score"], "spearman"),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    if "Strict" in df.columns:
        strict = df["Strict"].dropna()
        aligned = df.loc[strict.index, "noul"]
        out["pearson_prob_vs_strict"] = _safe_corr(aligned, strict, "pearson")
        out["spearman_prob_vs_strict"] = _safe_corr(aligned, strict, "spearman")
        out["mae_prob_vs_strict"] = float((aligned - strict).abs().mean())
    return out


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=sorted(HUMAN_EVAL_FILES),
                   help="Only one model. Omit for all available JEV CSVs.")
    p.add_argument("--out", type=Path,
                   default=OUT_DIR / "jev_vs_human_agreement.csv",
                   help="Where to write the summary CSV.")
    return p.parse_args()


def main() -> None:
    args = _parse()
    if args.model:
        candidates = [OUT_DIR / f"{args.model}_jev_verdicts.csv"]
    else:
        candidates = sorted(OUT_DIR.glob("*_jev_verdicts.csv"))

    rows = []
    for p in candidates:
        if not p.exists():
            print(f"skip (missing): {p.relative_to(ROOT)}")
            continue
        try:
            rows.append(summarise(p))
        except Exception as exc:  # noqa: BLE001
            print(f"skip (error): {p.relative_to(ROOT)}  -> {exc!r}")

    if not rows:
        print("No JEV verdict files found. Run scripts/jev/run.py first.")
        return

    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)

    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", lambda v: f"{v:0.3f}"):
        print(summary.to_string(index=False))
    print(f"\nWrote summary -> {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
