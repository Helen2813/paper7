"""Deterministic-to-seed synthetic experiments. NEVER report as GEO results."""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import Reference, compare_models, fit_delayed, onset_basis, design_identifiability


def synthetic_reference() -> Reference:
    days = np.array([0., 2., 4., 6., 8.])
    values = 1.5 / (1 + np.exp(-(days - 3.4) / 0.9))
    return Reference(days, values)


def simulate_one_mechanism(ref: Reference, mechanism: str, rng: np.random.Generator,
                           noise: float = 0.06) -> dict:
    days = ref.days
    if mechanism == "normal":
        delta, amplitude = 0.0, 1.0
    elif mechanism == "amplitude":
        delta, amplitude = 0.0, 0.58
    elif mechanism == "delay":
        delta, amplitude = 1.25, 1.0
    elif mechanism == "both":
        delta, amplitude = 1.1, 0.65
    elif mechanism == "shape":
        delta, amplitude = 0.0, 1.0
    else:
        raise ValueError("Unknown mechanism")
    y = 0.2 + amplitude * onset_basis(ref, days, delta)
    if mechanism == "shape":
        y = y + 0.4 * np.exp(-0.5 * ((days - 5) / 0.8) ** 2)
    y = y + rng.normal(0, noise, len(days))
    fit = compare_models(ref, days, y)
    return {"mechanism": mechanism, "true_delay": delta, "true_amplitude": amplitude,
            **fit}


def run_simulation(out: Path, reps: int = 40, seed: int = 726) -> pd.DataFrame:
    if reps < 1:
        raise ValueError("reps must be >=1")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    ref = synthetic_reference()
    rows = [dict(rep=k, **simulate_one_mechanism(ref, mechanism, rng))
            for k in range(reps) for mechanism in ("normal", "amplitude", "delay", "both", "shape")]
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "SYNTHETIC_individual_fits.csv", index=False)
    agg = raw.groupby("mechanism").agg(n=("rep", "size"),
        median_estimated_delay=("fit_delay_days", "median"),
        median_estimated_amplitude=("fit_both_amp", "median"),
        median_loto_improvement=("loto_delta_rmse", "median"),
        fraction_loto_improves=("loto_delta_rmse", lambda z: float((z > 0).mean()))).reset_index()
    agg.to_csv(out / "SYNTHETIC_summary.csv", index=False)
    # Illustration only: true noiseless response curves for three mechanisms.
    dense = np.linspace(0, float(ref.days[-1]), 301)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(dense, onset_basis(ref, dense, 0), label="Normal response (synthetic)")
    ax.plot(dense, 0.58 * onset_basis(ref, dense, 0), label="Amplitude attenuation (synthetic)")
    ax.plot(dense, onset_basis(ref, dense, 1.25), label="1.25-day onset lag (synthetic)")
    ax.scatter(ref.days, onset_basis(ref, ref.days, 0), marker="o", s=22)
    ax.set(xlabel="Days after stimulus", ylabel="Simulated molecular program response",
           title="SIMULATED ONLY — no real GEO data")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "SYNTHETIC_trajectory_illustration.png", dpi=170)
    plt.close(fig)
    (out / "SYNTHETIC_NOT_GEO.txt").write_text(
        "THIS IS SIMULATED DATA ONLY. No claims about endometriosis or real GEO cohorts.\n"
        "The reference curve is known (oracle): these results can overstate real-data performance.\n"
        "Shape residual is not equivalent to molecular rewiring.\n", encoding="utf-8")
    return agg
