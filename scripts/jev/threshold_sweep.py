"""Sweep JEV's decision threshold and plot its effect on human agreement.

For every threshold ``t`` in ``np.arange(0.05, 1.0, 0.025)``, we treat JEV's
noul probability as "correct" when ``noul >= t`` and recompute Cohen's kappa
and accuracy against the human binary label on each per-model verdicts CSV,
plus the pooled set of all 1,116 verdicts.

Outputs
-------
- data/automated_evaluations_qa/jev/jev_threshold_sweep.csv
    tidy frame with columns (model, threshold, n, accuracy, kappa, jev_pos_rate)
- data/automated_evaluations_qa/jev/jev_threshold_sweep.png
    two-panel figure: per-model kappa (left), pooled accuracy + kappa (right)
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from scripts.jev.agreement import safe_kappa

OUT = Path(__file__).resolve().parents[2] / "data" / "automated_evaluations_qa" / "jev"
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.025), 3)


def _load_all() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for p in sorted(OUT.glob("*_jev_verdicts.csv")):
        key = p.stem.removesuffix("_jev_verdicts")
        df = pd.read_csv(p).dropna(subset=["human_score", "noul"])
        df["human_bin"] = (df["human_score"] >= 0.5).astype(int)
        frames[key] = df[["noul", "human_bin"]].reset_index(drop=True)
    return frames


def _sweep(df: pd.DataFrame, label: str) -> list[dict]:
    rows: list[dict] = []
    human = df["human_bin"].to_numpy()
    noul = df["noul"].to_numpy()
    for t in THRESHOLDS:
        pred = (noul >= t).astype(int)
        kappa = safe_kappa(human, pred)
        rows.append({
            "model": label,
            "threshold": float(t),
            "n": int(len(df)),
            "accuracy": float((pred == human).mean()),
            "kappa": kappa,
            "jev_pos_rate": float(pred.mean()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-plot', action='store_true', help='Offline CSV analysis without matplotlib')
    args = parser.parse_args()
    frames = _load_all()
    pool = pd.concat(frames.values(), ignore_index=True)
    pool.name = "pooled"  # just for logging

    tidy: list[dict] = []
    for key, df in frames.items():
        tidy.extend(_sweep(df, key))
    tidy.extend(_sweep(pool, "POOLED"))
    sweep = pd.DataFrame(tidy)
    sweep.to_csv(OUT / "jev_threshold_sweep.csv", index=False)
    if args.no_plot:
        print(f"wrote {OUT / 'jev_threshold_sweep.csv'}")
        return
    import matplotlib.pyplot as plt

    # --- plot ------------------------------------------------------------
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.2),
                                            constrained_layout=True)

    # Left panel: per-model kappa vs threshold.
    cmap = plt.get_cmap("tab10")
    model_order = [k for k in frames.keys()]
    for i, key in enumerate(model_order):
        sub = sweep[sweep["model"] == key]
        ax_left.plot(sub["threshold"], sub["kappa"], color=cmap(i % 10),
                     label=key, lw=1.5, alpha=0.9)
    ax_left.axvline(0.5, color="grey", lw=0.8, ls="--",
                    label="default (0.5)")
    ax_left.set_xlabel("JEV decision threshold")
    ax_left.set_ylabel("Cohen's κ (JEV vs human)")
    ax_left.set_title("Per-model κ vs threshold")
    ax_left.grid(alpha=0.3)
    ax_left.legend(fontsize=8, loc="lower center", ncol=2)
    ax_left.set_xlim(0.05, 0.975)

    # Right panel: pooled kappa + accuracy, with best-threshold marker.
    pooled = sweep[sweep["model"] == "POOLED"]
    ax_right.plot(pooled["threshold"], pooled["kappa"], color="C3",
                  lw=2.2, label="pooled κ")
    ax_right.plot(pooled["threshold"], pooled["accuracy"], color="C0",
                  lw=2.2, label="pooled accuracy")
    best_k_idx = pooled["kappa"].idxmax()
    best_k = pooled.loc[best_k_idx]
    ax_right.axvline(best_k["threshold"], color="C3", lw=0.8, ls=":",
                     label=f"best κ @ t={best_k['threshold']:.3f}")
    ax_right.axvline(0.5, color="grey", lw=0.8, ls="--",
                     label="default (0.5)")
    ax_right.scatter([best_k["threshold"]], [best_k["kappa"]],
                     color="C3", zorder=5)
    ax_right.annotate(f"κ={best_k['kappa']:.3f}",
                      (best_k["threshold"], best_k["kappa"]),
                      textcoords="offset points", xytext=(8, 8),
                      color="C3")
    ax_right.set_xlabel("JEV decision threshold")
    ax_right.set_ylabel("value")
    ax_right.set_title("Pooled across all 9 models (n=1,116)")
    ax_right.grid(alpha=0.3)
    ax_right.legend(fontsize=9, loc="lower center")
    ax_right.set_xlim(0.05, 0.975)

    fig.suptitle("JEV decision threshold vs human agreement", fontsize=13)
    fig.savefig(OUT / "jev_threshold_sweep.png", dpi=160)
    print(f"wrote {OUT / 'jev_threshold_sweep.csv'}")
    print(f"wrote {OUT / 'jev_threshold_sweep.png'}")

    # Quick console summary: argmax threshold per model + pooled.
    best_rows = (sweep.sort_values("kappa", ascending=False)
                      .groupby("model", as_index=False).first())
    with pd.option_context("display.float_format", lambda v: f"{v:0.3f}"):
        print("\nBest threshold per model (by κ):")
        print(best_rows[["model", "threshold", "kappa", "accuracy", "jev_pos_rate"]]
              .sort_values("kappa", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
