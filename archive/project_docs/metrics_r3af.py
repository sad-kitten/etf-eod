"""metrics_r3af.py

R3AF — Regime Resilience + Convexity + Tail Orthogonality.

See docs/metrics_r3af.md for the canonical definition.

Self-contained (numpy/pandas only) and robust to column naming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class R3AFConfig:
    # Regime split by benchmark weighted quantiles
    q_low: float = 0.20
    q_high: float = 0.80

    # Final weighting
    w_tail_alpha: float = 0.55
    w_convexity: float = 0.25
    w_tail_ortho: float = 0.20

    # TailOrtho bounding
    tail_beta_cap: float = 2.0


def _pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    s = np.sum(w)
    if s <= 0 or len(x) == 0:
        return float("nan")
    return float(np.sum(w * x) / s)


def weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted quantile for 0<=q<=1 using linear interpolation on the weighted CDF."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x = x[mask]
    w = w[mask]
    if len(x) == 0:
        return float("nan")
    if np.sum(w) == 0:
        return float(np.quantile(x, q))
    order = np.argsort(x)
    x = x[order]
    w = w[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return float(np.interp(q, cw, x))


def weighted_tail_beta(r: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    """Weighted regression slope of r on b. Returns beta. Robust to constant benchmark."""
    r = np.asarray(r, dtype=float)
    b = np.asarray(b, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(r) & np.isfinite(b) & np.isfinite(w) & (w >= 0)
    r = r[mask]
    b = b[mask]
    w = w[mask]
    if len(r) < 2 or np.sum(w) == 0:
        return float("nan")
    bw = np.sum(w * b) / np.sum(w)
    rw = np.sum(w * r) / np.sum(w)
    vb = np.sum(w * (b - bw) ** 2)
    if vb <= 1e-18:
        return float("nan")
    cov = np.sum(w * (b - bw) * (r - rw))
    return float(cov / vb)


def compute_r3af_from_df(
    df: pd.DataFrame,
    *,
    scenario_col: str = "Scenario",
    weight_col: Optional[str] = None,
    port_col: Optional[str] = None,
    bench_col: Optional[str] = None,
    cfg: R3AFConfig = R3AFConfig(),
) -> Tuple[float, dict]:
    """Compute R3AF from a dataframe containing scenario returns for portfolio and benchmark.

    Returns (r3af_value, components_dict).
    """
    if weight_col is None:
        weight_col = _pick_col(df, ["Weight_Normalized", "Weight", "w", "ScenarioWeight"])
    if port_col is None:
        port_col = _pick_col(df, ["PortCAGR", "PortfolioCAGR", "CAGR", "r", "PortReturn"])
    if bench_col is None:
        bench_col = _pick_col(df, ["BenchCAGR", "BenchmarkCAGR", "Bench", "b", "BenchReturn"])

    missing = [c for c in [scenario_col, weight_col, port_col, bench_col] if c is None or c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    w = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(df[port_col], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(df[bench_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(w) & np.isfinite(r) & np.isfinite(b) & (w >= 0)
    w = w[mask]
    r = r[mask]
    b = b[mask]
    if len(w) == 0 or np.sum(w) <= 0:
        return float("nan"), {"TailAlpha": np.nan, "Convexity": np.nan, "TailBeta": np.nan, "TailOrtho": np.nan}

    ql = weighted_quantile(b, w, cfg.q_low)
    qh = weighted_quantile(b, w, cfg.q_high)

    bad = b <= ql
    good = b >= qh
    mid = ~(bad | good)

    mu_bad_r = weighted_mean(r[bad], w[bad])
    mu_mid_r = weighted_mean(r[mid], w[mid])
    mu_good_r = weighted_mean(r[good], w[good])

    mu_bad_b = weighted_mean(b[bad], w[bad])

    tail_alpha = mu_bad_r - mu_bad_b
    convexity = (mu_good_r - mu_mid_r) - (mu_mid_r - mu_bad_r)

    tail_beta = weighted_tail_beta(r[bad], b[bad], w[bad])
    tail_ortho = 1.0 - min(cfg.tail_beta_cap, abs(tail_beta)) if np.isfinite(tail_beta) else np.nan

    r3af = cfg.w_tail_alpha * tail_alpha + cfg.w_convexity * convexity + cfg.w_tail_ortho * tail_ortho

    components = {
        "TailAlpha": float(tail_alpha),
        "Convexity": float(convexity),
        "TailBeta": float(tail_beta) if np.isfinite(tail_beta) else np.nan,
        "TailOrtho": float(tail_ortho) if np.isfinite(tail_ortho) else np.nan,
        "q_low": float(ql),
        "q_high": float(qh),
    }
    return float(r3af), components


def merge_portfolio_and_benchmark(
    port_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    *,
    scenario_col: str = "Scenario",
    weight_col: Optional[str] = None,
    port_col: Optional[str] = None,
    bench_col: Optional[str] = None,
    port_name: str = "PortCAGR",
    bench_name: str = "BenchCAGR",
) -> pd.DataFrame:
    """Merge portfolio scenario returns with benchmark scenario returns on Scenario."""
    if weight_col is None:
        weight_col = _pick_col(port_df, ["Weight_Normalized", "Weight", "w", "ScenarioWeight"])
    if port_col is None:
        port_col = _pick_col(port_df, ["PortCAGR", "PortfolioCAGR", "CAGR", "r", "PortReturn"])
    if bench_col is None:
        bench_col = _pick_col(bench_df, ["BenchCAGR", "BenchmarkCAGR", "CAGR", "b", "BenchReturn"])

    if scenario_col not in port_df.columns or scenario_col not in bench_df.columns:
        raise ValueError("Both port_df and bench_df must contain the scenario_col.")
    if weight_col is None or weight_col not in port_df.columns:
        raise ValueError("port_df must contain scenario weights (e.g., Weight_Normalized).")
    if port_col is None or port_col not in port_df.columns:
        raise ValueError("port_df must contain portfolio scenario returns (e.g., CAGR).")
    if bench_col is None or bench_col not in bench_df.columns:
        raise ValueError("bench_df must contain benchmark scenario returns (e.g., CAGR).")

    a = port_df[[scenario_col, weight_col, port_col]].copy().rename(columns={port_col: port_name})
    b = bench_df[[scenario_col, bench_col]].copy().rename(columns={bench_col: bench_name})
    return a.merge(b, on=scenario_col, how="inner")
