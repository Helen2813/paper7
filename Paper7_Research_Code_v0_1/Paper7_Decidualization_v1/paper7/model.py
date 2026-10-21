"""Small-sample trajectory modeling and *local* identifiability diagnostics.

Scientific target: a response-onset delay vs changed response magnitude, NOT
identification of a patient's true menstrual phase or causal disease genes.

The anchored trajectory is y_i(t) = b_i + a_i [f(max(0,t-delta_i))-f(0)] + eps_i(t).
This is a *delayed onset*, not a general time translation. f is an untreated,
training-control-only program trajectory. No extrapolation is permitted.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, UnivariateSpline


@dataclass(frozen=True)
class Reference:
    days: np.ndarray
    values: np.ndarray

    def __post_init__(self):
        t = np.asarray(self.days, dtype=float)
        y = np.asarray(self.values, dtype=float)
        if t.ndim != 1 or y.ndim != 1 or len(t) != len(y):
            raise ValueError("Reference must be equal-length 1D vectors")
        if len(t) < 4 or not np.isfinite(t).all() or not np.isfinite(y).all() or not np.all(np.diff(t) > 0):
            raise ValueError("Reference needs >=4 strictly increasing finite time points")
        if abs(t[0]) > 1e-9:
            raise ValueError("Reference needs a day-0 baseline for onset-delay model")

    @cached_property
    def _curve(self):
        return PchipInterpolator(self.days, self.values, extrapolate=False)

    def interpolator(self):
        return self._curve


def make_reference(scores: pd.DataFrame, min_donors: int = 3) -> Reference:
    """Estimate reference with equal donor weight (not pseudoreplicated samples)."""
    required = {"donor_id", "days", "score"}
    if not required.issubset(scores.columns):
        raise ValueError(f"scores missing {sorted(required - set(scores.columns))}")
    df = scores.dropna(subset=list(required)).copy()
    if df.donor_id.nunique() < min_donors:
        raise ValueError(f"Need >={min_donors} different control donors")
    donor_time = df.groupby(["donor_id", "days"], as_index=False).score.mean()
    ref = donor_time.groupby("days").agg(value=("score", "mean"), n_donors=("donor_id", "nunique"))
    ref = ref[ref.n_donors >= min_donors].sort_index()
    if len(ref) < 4:
        raise ValueError("Need >=4 times represented by >=3 control donors")
    return Reference(ref.index.to_numpy(dtype=float), ref.value.to_numpy(dtype=float))


def onset_basis(ref: Reference, days: Sequence[float], delay_days: float = 0) -> np.ndarray:
    """Trajectory above reference day-0, with a nonnegative response-onset delay."""
    tt = np.asarray(days, dtype=float)
    if delay_days < 0:
        raise ValueError("The primary model permits delays >=0 only")
    if not np.isfinite(tt).all() or tt.min() < ref.days.min() - 1e-8 or tt.max() > ref.days.max() + 1e-8:
        raise ValueError("Evaluation outside the control trajectory support is forbidden")
    x = np.maximum(float(ref.days.min()), tt - float(delay_days))
    return np.asarray(ref.interpolator()(x) - ref.values[0], dtype=float)


def fit_amplitude(ref: Reference, days: Sequence[float], y: Sequence[float], delay_days: float = 0,
                  amplitude_bounds: tuple[float, float] = (0.0, 2.5)) -> dict:
    t = np.asarray(days, dtype=float)
    yy = np.asarray(y, dtype=float)
    if len(t) != len(yy) or len(t) < 3 or not np.isfinite(yy).all():
        raise ValueError("At least 3 paired finite observations are needed")
    x = onset_basis(ref, t, delay_days)
    dx = x - x.mean()
    dy = yy - yy.mean()
    denom = float(dx @ dx)
    a = float(np.clip((dx @ dy) / denom, *amplitude_bounds)) if denom > 1e-10 else 0.0
    b = float(yy.mean() - a * x.mean())
    predictions = b + a * x
    return {"amplitude": a, "intercept": b, "delay_days": float(delay_days),
            "sse": float(np.sum((predictions - yy) ** 2)), "predictions": predictions}


def fit_delayed(ref: Reference, days: Sequence[float], y: Sequence[float],
                max_delay_days: float = 2, n_grid: int = 81) -> dict:
    """Profile the lag; choose smaller delay when RSS ties (conservative rule)."""
    if max_delay_days <= 0 or n_grid < 3:
        raise ValueError("Positive delay bound and >=3 profiling grid points are required")
    tt = np.asarray(days, dtype=float)
    yy = np.asarray(y, dtype=float)
    if len(yy) != len(tt) or len(tt) < 3 or not np.isfinite(yy).all():
        raise ValueError("At least 3 paired finite observations needed")
    candidates = np.linspace(0.0, min(max_delay_days, float(tt.max() - tt.min())), n_grid)
    # Vectorized profile: 81 lag candidates x 3-5 observations, one PCHIP evaluation.
    eval_t = np.maximum(float(ref.days[0]), tt[None, :] - candidates[:, None])
    basis = np.asarray(ref.interpolator()(eval_t) - ref.values[0], dtype=float)
    centered = basis - basis.mean(axis=1, keepdims=True)
    cov = centered @ (yy - yy.mean())
    denom = np.sum(centered ** 2, axis=1)
    amplitude = np.clip(np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-10), 0., 2.5)
    intercept = yy.mean() - amplitude * basis.mean(axis=1)
    preds = intercept[:, None] + amplitude[:, None] * basis
    sse = np.sum((preds - yy[None,:]) ** 2, axis=1)
    # Conservative tie-break when two nearby lags are practically equivalent.
    minimum = int(np.flatnonzero(sse <= np.min(sse) + 1e-12)[0])
    return {"amplitude": float(amplitude[minimum]), "intercept": float(intercept[minimum]),
            "delay_days": float(candidates[minimum]), "sse": float(sse[minimum]),
            "predictions": preds[minimum], "profile_delay": candidates, "profile_sse": sse}


def compare_models(ref: Reference, days: Sequence[float], observations: Sequence[float],
                   max_delay_days: float = 2.0) -> dict:
    """Leave-one-*timepoint*-out prediction within one donor.

    This is NOT independent donor-level external validation: results are aggregated
    by donor, and held-out timepoints from the same donor are correlated.
    """
    t = np.asarray(days, dtype=float)
    y = np.asarray(observations, dtype=float)
    if len(t) < 5 or len(set(t)) != len(t):
        raise ValueError("Require >=5 distinct measured times for within-donor held-out-timepoint comparison")
    full_a = fit_amplitude(ref, t, y)
    full_d = fit_delayed(ref, t, y, max_delay_days=max_delay_days)
    preds_a, preds_d = np.empty(len(t)), np.empty(len(t))
    for j in range(len(t)):
        mask = np.arange(len(t)) != j
        ma = fit_amplitude(ref, t[mask], y[mask])
        md = fit_delayed(ref, t[mask], y[mask], max_delay_days=max_delay_days)
        preds_a[j] = ma["intercept"] + ma["amplitude"] * onset_basis(ref, [t[j]], 0)[0]
        preds_d[j] = md["intercept"] + md["amplitude"] * onset_basis(ref, [t[j]], md["delay_days"])[0]
    ea = preds_a - y
    ed = preds_d - y
    return {
        "fit_amp": full_a["amplitude"], "fit_amp_sse": full_a["sse"],
        "fit_both_amp": full_d["amplitude"], "fit_delay_days": full_d["delay_days"],
        "fit_both_sse": full_d["sse"], "fit_both_rmse": float(np.sqrt(full_d["sse"] / len(t))),
        "loto_amp_rmse": float(np.sqrt(np.mean(ea**2))),
        "loto_both_rmse": float(np.sqrt(np.mean(ed**2))),
        "loto_delta_rmse": float(np.sqrt(np.mean(ea**2)) - np.sqrt(np.mean(ed**2))),
        "n_timepoints": int(len(t)),
        "shape_residual_rmse": float(np.sqrt(full_d["sse"] / len(t))),
        "profile_min_at_boundary": bool(np.isclose(full_d["delay_days"], 0, atol=1e-8) or
                                        np.isclose(full_d["delay_days"], max_delay_days, atol=1e-8)),
    }


def design_identifiability(ref: Reference, days: Sequence[float], derivative: str = "pchip",
                           smoothing: float = 0.03) -> dict:
    """Scaled local Jacobian of intercept, amplitude, and response-lag.

    Linearization is evaluated at delay 0 away from t=0. This is a diagnostic,
    not a structural/global identifiability proof or an unbiased SE estimator.
    """
    t = np.asarray(days, dtype=float)
    if len(t) < 4 or len(np.unique(t)) != len(t):
        raise ValueError("Need >=4 distinct time points")
    v = onset_basis(ref, t, 0)
    if derivative == "pchip":
        f = ref.interpolator()
    elif derivative == "spline":
        f = UnivariateSpline(ref.days, ref.values, k=3, s=smoothing * len(ref.days))
    else:
        raise ValueError("derivative must be pchip or spline")
    g = -np.asarray(f.derivative()(t), dtype=float)
    g[np.isclose(t, ref.days[0])] = 0  # max(0,t-delta) has a non-smooth boundary
    x = np.column_stack([v, g])
    x = x - x.mean(axis=0)  # nuisance intercept b profiled out
    norms = np.linalg.norm(x, axis=0)
    if (norms <= 1e-10).any():
        return {"abs_corr": np.nan, "scaled_condition": float("inf"),
                "min_singular": 0.0, "d_opt_det": 0.0, "rank_two": False,
                "amplitude_column_norm": float(norms[0]), "lag_column_norm": float(norms[1])}
    scaled = x / norms
    singular = np.linalg.svd(scaled, compute_uv=False)
    return {"abs_corr": float(abs(scaled[:, 0] @ scaled[:, 1])),
            "scaled_condition": float(singular[0]/singular[-1]) if singular[-1] > 1e-12 else float("inf"),
            "min_singular": float(singular[-1]), "d_opt_det": float(np.linalg.det(scaled.T @ scaled)),
            "rank_two": bool(singular[-1] > 1e-8),
            "amplitude_column_norm": float(norms[0]), "lag_column_norm": float(norms[1])}


def suggest_sampling_times(ref: Reference, candidate_days: Sequence[float], n_samples: int = 5,
                           top_k: int = 10) -> pd.DataFrame:
    """Exploratory conditioned-Jacobian designs; assume equal homoscedastic noise.

    Fix day 0 and final reference day; optimize intermediate times. The D-opt
    score here uses *unit-normalized columns*: a separability diagnostic, not a
    rigorous cost-aware Fisher-information design for a laboratory protocol.
    """
    allowed = sorted(set(float(t) for t in candidate_days if ref.days[0] <= t <= ref.days[-1]))
    if ref.days[0] not in allowed or ref.days[-1] not in allowed:
        raise ValueError("Candidate set must contain first and last control sampling times")
    if n_samples < 4 or len(allowed) < n_samples:
        raise ValueError("Insufficient candidate times")
    interior = [t for t in allowed if t not in {ref.days[0], ref.days[-1]}]
    rows = []
    for middle in combinations(interior, n_samples - 2):
        chosen = (float(ref.days[0]), *middle, float(ref.days[-1]))
        p = design_identifiability(ref, chosen)
        rows.append({"days": ",".join(f"{x:g}" for x in chosen), "scaled_condition": p["scaled_condition"],
                     "min_singular": p["min_singular"], "abs_corr": p["abs_corr"]})
    return pd.DataFrame(rows).sort_values(["min_singular", "scaled_condition"], ascending=[False, True]).head(top_k).reset_index(drop=True)
