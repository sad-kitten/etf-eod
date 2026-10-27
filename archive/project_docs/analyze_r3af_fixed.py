# analyze.py
# Equities project — multi-window scoring with SWC + R3AF
#
# Updates in this version:
#   - Latest UPDATED: QTUM -> XSD (+5%) AND QUAL -> SPMO (SPMO back). Latest name unchanged.
#   - LatestC included (your pre-Current version).
#   - Current included: LatestC with 5% XAR -> SHLD.
#   - Frontier portfolios filtered to: P1, P2, P4, P11 only.
#
# Inputs (expected in same folder):
#   - yfinance.zip
#   - KEEP_Scenarios_Canonical_2026Q1.xlsx  (sheet: Scenario_Weights)
# Optional:
#   - inception_overrides.csv               (Ticker, InceptionDate)
#   - scenario_shocks.csv                   (Scenario, Ticker, CAGR_Shock)
#   - metrics_r3af.py                       (optional; fallback used if missing)
#   - KEEP_Frontier_Weights_2025Q4.csv      (to load P1/P2/P4/P11)
#   - KEEP_Portfolios_Canonical_2025Q4.xlsx (optional; loads additional ports if present)
#
# Run:
#   python analyze.py
#
# Output:
#   out/scoreboard_r3_swc_multiwindow.csv
#   out/proxy_weights.csv
#   out/inception_overrides_INFERRED.csv  (if you don't provide inception_overrides.csv)
#   out/scenario_shocks_AUTO.csv          (if you don't provide scenario_shocks.csv)

from __future__ import annotations

import sys
import zipfile
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# Optional SciPy NNLS (recommended)
try:
    from scipy.optimize import nnls  # type: ignore
except Exception:
    nnls = None

# ----------------- FILES -----------------
ZIP_PATH = "yfinance.zip"
SCENARIOS_XLSX = "KEEP_Scenarios_Canonical_2026Q1.xlsx"

R3AF_PY = "metrics_r3af.py"  # optional

KEEP_PORT_XLSX = "KEEP_Portfolios_Canonical_2025Q4.xlsx"    # optional
FRONTIER_W_CSV = "KEEP_Frontier_Weights_2025Q4.csv"         # optional

INCEPTION_OVERRIDES_CSV = "inception_overrides.csv"  # optional; inferred if missing
SCENARIO_SHOCKS_CSV = "scenario_shocks.csv"          # optional; auto-generated if missing

OUTDIR = Path("out")
OUTDIR.mkdir(exist_ok=True)

# ----------------- SCORING WINDOWS -----------------
WINDOWS = [
    ("MAX", None),
    ("ETF_ERA_2010", "2010-01-04"),
    ("POST_GFC_2012", "2012-01-03"),
    ("POST_COVID_2020", "2020-01-02"),
]

KEEP_FRONTIER_PORTS = {"P1", "P2", "P4", "P11"}
BENCHMARK = "VT"

MIN_FIT_OBS = 252

# Proxies: extend pre-inception history for these ETFs using components
PROXY_SPECS = {
    "AVGE": ["AVUS", "AVDE", "AVDV", "AVES", "VT", "VTI", "VXUS", "VWO", "IEMG"],
    "SETM": ["GRID", "URNM", "URA", "XLE", "PAVE", "AIRR"],
    "NUKZ": ["NLR", "XAR", "XLE", "URA", "URNM"],
    # SHLD is new-ish; proxy with A&D mix + tech/cyber flavor if components exist
    "SHLD": ["XAR", "CIBR", "QQEW", "ROKT", "LMT", "GD", "RTX"],
}


# ----------------- IO / DATA -----------------
def clean_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" not in df.columns:
        raise ValueError("CSV missing Date column.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")

    if "Adj Close" in df.columns:
        px = pd.to_numeric(df["Adj Close"], errors="coerce")
    elif "Close" in df.columns:
        px = pd.to_numeric(df["Close"], errors="coerce")
    else:
        raise ValueError("CSV missing Adj Close/Close columns.")

    vol = pd.to_numeric(df["Volume"], errors="coerce") if "Volume" in df.columns else pd.Series(index=df.index, dtype=float)

    out = pd.DataFrame({"px": px, "vol": vol})
    out = out.dropna(subset=["px"])
    out = out[out["px"] > 0]
    out = out[~out.index.duplicated(keep="last")]
    return out


def load_zip_ohlcv(zip_path: str) -> dict[str, pd.DataFrame]:
    z = zipfile.ZipFile(zip_path)
    out: dict[str, pd.DataFrame] = {}
    for name in z.namelist():
        if not name.lower().endswith(".csv"):
            continue
        t = Path(name).stem.upper()
        df = pd.read_csv(z.open(name))
        out[t] = clean_ohlcv_df(df)
    return out


def log_returns(px: pd.Series) -> pd.Series:
    r = np.log(px).diff()
    return r.replace([np.inf, -np.inf], np.nan)


# ----------------- PROXY FITTING -----------------
def fit_proxy_weights(target_r: pd.Series, comps: dict[str, pd.Series], min_obs=MIN_FIT_OBS) -> dict[str, float] | None:
    df = pd.concat([target_r.rename("y")] + [comps[k].rename(k) for k in comps], axis=1).dropna()
    if len(df) < min_obs:
        return None
    y = df["y"].values
    X = df[list(comps.keys())].values
    if nnls is None:
        # Fallback: equal weights if SciPy is missing
        w = np.ones(X.shape[1], dtype=float)
    else:
        w, _ = nnls(X, y)
    if w.sum() <= 0:
        return None
    w = w / w.sum()
    return dict(zip(comps.keys(), w))


def proxy_series(weights: dict[str, float], comps: dict[str, pd.Series], idx: pd.DatetimeIndex) -> pd.Series:
    cols = list(weights.keys())
    df = pd.concat([comps[c].reindex(idx).rename(c) for c in cols], axis=1)
    W = np.array([weights[c] for c in cols], dtype=float)

    def row_val(row):
        m = ~row.isna()
        if not m.any():
            return np.nan
        ww = W[m.values].copy()
        s = float(ww.sum())
        if not np.isfinite(s) or s <= 0:
            return float(np.mean(row[m].values))
        ww /= s
        return float(np.dot(row[m].values, ww))

    return df.apply(row_val, axis=1)


def stitch(actual: pd.Series, proxy: pd.Series) -> pd.Series:
    return actual.fillna(proxy)


# ----------------- INCEPTION OVERRIDES -----------------
def load_inception_overrides(path: str) -> dict[str, pd.Timestamp]:
    df = pd.read_csv(path)
    need = {"Ticker", "InceptionDate"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must have columns: {sorted(need)}")
    out: dict[str, pd.Timestamp] = {}
    for _, r in df.iterrows():
        t = str(r["Ticker"]).strip().upper()
        d = pd.to_datetime(r["InceptionDate"], errors="coerce")
        if t and pd.notna(d):
            out[t] = d
    return out


def infer_inceptions_from_ohlcv(
    ohlcv: dict[str, pd.DataFrame],
    min_vol_pos_days: int = 30,
    lookahead_days: int = 90,
    max_flat_run: int = 15,
    min_unique_px: int = 20,
) -> dict[str, pd.Timestamp]:
    def longest_flat_run(px: pd.Series) -> int:
        v = px.dropna().values
        if len(v) == 0:
            return 999
        best = run = 1
        for i in range(1, len(v)):
            if np.isfinite(v[i]) and np.isfinite(v[i - 1]) and v[i] == v[i - 1]:
                run += 1
                best = max(best, run)
            else:
                run = 1
        return best

    inferred: dict[str, pd.Timestamp] = {}
    for t, df in ohlcv.items():
        t = t.upper()
        if df.empty:
            continue
        px = df["px"]
        vol = df["vol"] if "vol" in df.columns else pd.Series(index=df.index, dtype=float)
        idx = df.index

        if not vol.notna().any():
            inferred[t] = idx.min()
            continue

        vpos = (vol.fillna(0.0) > 0).astype(int)
        start = idx.min()
        found = False

        for i in range(0, max(1, len(idx) - lookahead_days)):
            win = slice(i, min(len(idx), i + lookahead_days))
            if int(vpos.iloc[win].sum()) < min_vol_pos_days:
                continue
            px_win = px.iloc[win]
            if px_win.nunique(dropna=True) < min_unique_px:
                continue
            if longest_flat_run(px_win) > max_flat_run:
                continue
            start = idx[i]
            found = True
            break

        if not found:
            pos = vpos[vpos > 0]
            start = pos.index.min() if len(pos) else idx.min()

        inferred[t] = pd.to_datetime(start)

    return inferred


# ----------------- SCENARIOS -----------------
def load_scenario_weights(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name="Scenario_Weights")
    if "Scenario" not in df.columns:
        raise ValueError("Scenario_Weights sheet must contain 'Scenario'.")
    if "Weight_Normalized" not in df.columns:
        if "Weight_%" in df.columns:
            w = pd.to_numeric(df["Weight_%"], errors="coerce").fillna(0.0)
            df["Weight_Normalized"] = w / w.sum() if w.sum() else 0.0
        else:
            raise ValueError("Scenario_Weights needs Weight_Normalized or Weight_%")
    df = df[["Scenario", "Weight_Normalized"]].copy()
    df["Weight_Normalized"] = pd.to_numeric(df["Weight_Normalized"], errors="coerce").fillna(0.0)
    s = df["Weight_Normalized"].sum()
    if s > 0:
        df["Weight_Normalized"] /= s
    return df


def load_scenario_shocks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"Scenario", "Ticker", "CAGR_Shock"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must have columns: {sorted(need)}")
    df["Scenario"] = df["Scenario"].astype(str)
    df["Ticker"] = df["Ticker"].astype(str).str.upper()
    df["CAGR_Shock"] = pd.to_numeric(df["CAGR_Shock"], errors="coerce").fillna(0.0)
    return df


def _norm_s(s: str) -> str:
    return str(s).strip().lower()


def classify_scenario(s: str) -> str:
    t = _norm_s(s)

    if "base case" in t or "managed re-industrialization" in t:
        return "base"
    if "reflation" in t or "industrial up-cycle" in t:
        return "reflation"
    if "stagflation" in t or "fiscal dominance" in t:
        return "stagflation"
    if "debt overhang" in t or "credit stress" in t:
        return "debt_crisis"
    if "ai breakout" in t or "tech concentration" in t:
        return "ai_boom"
    if "geopolitical fragmentation" in t or "multipolar" in t:
        return "geo_shock"
    if "cronyism" in t or "policy distortion" in t:
        return "cronyism"
    if "china hard landing" in t:
        return "china_hard"
    if "taiwan hostilities" in t or "kinetic shock" in t:
        return "geo_shock"
    if "credit accident" in t or "u.s. regional" in t:
        return "debt_crisis"
    if "space race" in t or "tech-industrial rivalry" in t:
        return "ai_boom"
    if "pandemic" in t or "bio disruption" in t:
        return "pandemic"
    if "climate transition" in t:
        return "climate"
    if "energy shock" in t or "supply constraint" in t:
        return "energy_shock"
    if "political polarization" in t or "governance shock" in t:
        return "risk_off"
    if "liquidity:" in t or ("jgb" in t and "fiscal dominance" in t):
        return "debt_crisis"

    return "base"


def classify_ticker_bucket(t: str) -> str:
    t = t.upper()
    if t in {"VT", "VTI", "VXUS", "IEMG", "VWO", "AVGE"}:
        return "broad"
    if t in {"QUAL", "SPMO"}:
        return "quality_factor"
    if t in {"XSD", "SOXX"}:
        return "semis"
    if t in {"XAR", "SHLD"}:
        return "defense"
    if t in {"GRID"}:
        return "grid"
    if t in {"PAVE", "AIRR"}:
        return "infra"
    if t in {"XLE"}:
        return "energy"
    if t in {"NLR", "URA", "URNM", "NUKZ"}:
        return "nuclear"
    if t in {"SETM"}:
        return "materials"
    if t in {"ROKT", "QTUM", "CIBR", "QQEW"}:
        return "theme_growth"
    return "broad"


def default_bucket_shocks() -> dict[str, dict[str, float]]:
    # shocks are annualized CAGR deltas applied per scenario (heuristic)
    base = {"broad": 0.00, "quality_factor": 0.005, "semis": 0.005, "defense": 0.003,
            "grid": 0.004, "infra": 0.004, "energy": 0.00, "nuclear": 0.004, "materials": 0.00, "theme_growth": 0.003}
    risk_off = {"broad": -0.020, "quality_factor": -0.010, "semis": -0.030, "defense": -0.010,
                "grid": -0.015, "infra": -0.020, "energy": 0.000, "nuclear": 0.010, "materials": 0.005, "theme_growth": -0.030}
    reflation = {"broad": 0.010, "quality_factor": 0.005, "semis": 0.010, "defense": 0.005,
                 "grid": 0.010, "infra": 0.020, "energy": 0.015, "nuclear": 0.010, "materials": 0.015, "theme_growth": 0.010}
    stagflation = {"broad": -0.020, "quality_factor": -0.015, "semis": -0.030, "defense": -0.005,
                   "grid": -0.010, "infra": -0.020, "energy": 0.030, "nuclear": 0.020, "materials": 0.020, "theme_growth": -0.025}
    debt_crisis = {"broad": -0.030, "quality_factor": -0.020, "semis": -0.040, "defense": -0.010,
                   "grid": -0.020, "infra": -0.030, "energy": 0.000, "nuclear": 0.015, "materials": 0.010, "theme_growth": -0.040}
    energy_shock = {"broad": -0.010, "quality_factor": -0.005, "semis": -0.010, "defense": 0.000,
                    "grid": -0.005, "infra": 0.000, "energy": 0.040, "nuclear": 0.010, "materials": 0.020, "theme_growth": -0.005}
    ai_boom = {"broad": 0.010, "quality_factor": 0.010, "semis": 0.040, "defense": 0.000,
               "grid": 0.005, "infra": 0.000, "energy": -0.005, "nuclear": 0.000, "materials": 0.000, "theme_growth": 0.020}
    geo_shock = {"broad": -0.020, "quality_factor": -0.015, "semis": -0.030, "defense": 0.030,
                 "grid": -0.010, "infra": -0.010, "energy": 0.010, "nuclear": 0.020, "materials": 0.010, "theme_growth": -0.010}
    cronyism = {"broad": -0.010, "quality_factor": -0.008, "semis": -0.010, "defense": 0.000,
                "grid": 0.000, "infra": 0.000, "energy": 0.010, "nuclear": 0.000, "materials": 0.000, "theme_growth": 0.000}
    china_hard = {"broad": -0.020, "quality_factor": -0.015, "semis": -0.020, "defense": 0.010,
                  "grid": -0.010, "infra": -0.010, "energy": -0.005, "nuclear": 0.005, "materials": -0.005, "theme_growth": -0.010}
    pandemic = {"broad": -0.020, "quality_factor": -0.010, "semis": 0.000, "defense": -0.010,
                "grid": -0.010, "infra": -0.020, "energy": -0.020, "nuclear": -0.005, "materials": -0.010, "theme_growth": -0.005}
    climate = {"broad": -0.010, "quality_factor": 0.000, "semis": 0.005, "defense": 0.000,
               "grid": 0.020, "infra": 0.010, "energy": -0.020, "nuclear": 0.010, "materials": 0.010, "theme_growth": 0.010}
    return {"base": base, "risk_off": risk_off, "reflation": reflation, "stagflation": stagflation,
            "debt_crisis": debt_crisis, "energy_shock": energy_shock, "ai_boom": ai_boom,
            "geo_shock": geo_shock, "cronyism": cronyism, "china_hard": china_hard,
            "pandemic": pandemic, "climate": climate}


def build_auto_scenario_shocks(scenario_weights_df: pd.DataFrame, tickers: list[str], bench: str = "VT") -> pd.DataFrame:
    regime_map = default_bucket_shocks()
    all_tickers = sorted(set([t.upper() for t in tickers] + [bench.upper()]))
    rows = []
    for scen in scenario_weights_df["Scenario"].tolist():
        regime = classify_scenario(scen)
        bucket_shocks = regime_map.get(regime, regime_map["base"])
        for t in all_tickers:
            bucket = classify_ticker_bucket(t)
            shock = float(bucket_shocks.get(bucket, 0.0))
            rows.append({"Scenario": scen, "Ticker": t, "CAGR_Shock": shock})
    return pd.DataFrame(rows)


# ----------------- PORTFOLIOS (LOAD) -----------------
def try_load_canonical() -> dict[str, dict[str, float]]:
    ports: dict[str, dict[str, float]] = {}
    p = Path(KEEP_PORT_XLSX)
    if not p.exists():
        return ports
    try:
        df = pd.read_excel(KEEP_PORT_XLSX, sheet_name="Portfolios")
        for name in df["Portfolio"].unique():
            d = df[df["Portfolio"] == name].set_index("Ticker")["Weight_%"].to_dict()
            ports[str(name)] = {k.upper(): float(v) / 100.0 for k, v in d.items() if float(v) > 0}
        print(f"Loaded canonical portfolios: {len(ports)}")
    except Exception as e:
        print(f"Canonical load skipped (ok): {e}")
    return ports


def try_load_frontier() -> dict[str, dict[str, float]]:
    ports: dict[str, dict[str, float]] = {}
    p = Path(FRONTIER_W_CSV)
    if not p.exists():
        return ports
    try:
        df = pd.read_csv(FRONTIER_W_CSV).rename(columns={"Unnamed: 0": "Ticker"})
        frontier_cols = [c for c in df.columns if str(c) in KEEP_FRONTIER_PORTS]
        for col in frontier_cols:
            d = df[["Ticker", col]].set_index("Ticker")[col].to_dict()
            ports[str(col)] = {k.upper(): float(v) / 100.0 for k, v in d.items() if v and float(v) > 0}
        print(f"Loaded frontier portfolios (filtered): {len(ports)} -> {sorted(ports.keys())}")
    except Exception as e:
        print(f"Frontier load skipped (ok): {e}")
    return ports


# ----------------- METRICS -----------------
def port_returns(wts: dict[str, float], ret_map: dict[str, pd.Series]) -> pd.Series:
    parts = []
    for t, w in wts.items():
        t = t.upper()
        if t not in ret_map:
            raise KeyError(f"Missing returns for ticker {t}")
        parts.append(ret_map[t] * w)
    return pd.concat(parts, axis=1).sum(axis=1).dropna()


def max_dd(r: pd.Series) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1
    return float(dd.min())


def cagr(r: pd.Series, pp=252) -> float:
    eq = (1 + r.fillna(0)).cumprod()
    n = len(eq)
    if n < 2:
        return np.nan
    years = n / pp
    return float(eq.iloc[-1] ** (1 / years) - 1)


def sortino(r: pd.Series, pp=252, mar=0.0) -> float:
    downside = (r - mar / pp).clip(upper=0)
    dd = downside.std() * np.sqrt(pp)
    if dd == 0 or np.isnan(dd):
        return np.nan
    return float((r.mean() * pp - mar) / dd)


def calmar(r: pd.Series) -> float:
    m = abs(max_dd(r))
    if m == 0 or np.isnan(m):
        return np.nan
    return float(cagr(r) / m)


def slice_window(r: pd.Series, start_date: str | None) -> pd.Series:
    if start_date is None:
        return r.dropna()
    sd = pd.to_datetime(start_date)
    return r[r.index >= sd].dropna()


# ----------------- R3AF LOADING + FALLBACK -----------------
def _load_r3af_module_if_present(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("metrics_r3af", str(p))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["metrics_r3af"] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        if not hasattr(mod, "compute_r3af_from_df"):
            return None
        return mod
    except Exception:
        return None


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -12, 12))
    return 1.0 / (1.0 + np.exp(-x))


def compute_r3af_fallback(df: pd.DataFrame,
                          weight_col="Weight_Normalized",
                          port_col="PortCAGR",
                          bench_col="BenchCAGR"):
    d = df.copy()
    for c in [weight_col, port_col, bench_col]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=[weight_col, port_col, bench_col])
    if d.empty or d[weight_col].sum() <= 0:
        return np.nan, {"asymmetry": np.nan, "convexity": np.nan, "recovery": np.nan}

    w = d[weight_col].values.astype(float)
    w = w / w.sum()

    excess = (d[port_col].values - d[bench_col].values).astype(float)
    bench = d[bench_col].values.astype(float)
    bench_med = np.median(bench)
    stress = bench < bench_med
    risk_on = ~stress

    def wavg(x, mask=None):
        if mask is None:
            return float(np.sum(w * x))
        ww = w[mask]
        if ww.sum() <= 0:
            return float(np.sum(w * x))
        ww = ww / ww.sum()
        return float(np.sum(ww * x[mask]))

    ex_stress = wavg(excess, stress) if stress.any() else wavg(excess)
    ex_riskon = wavg(excess, risk_on) if risk_on.any() else wavg(excess)

    asym = _sigmoid(10.0 * (ex_riskon - ex_stress))

    q25 = np.quantile(bench, 0.25)
    q75 = np.quantile(bench, 0.75)
    tail_bad = bench <= q25
    tail_good = bench >= q75

    ex_tail_bad = wavg(excess, tail_bad) if tail_bad.any() else ex_stress
    ex_tail_good = wavg(excess, tail_good) if tail_good.any() else ex_riskon

    convexity = _sigmoid(8.0 * (ex_tail_good - ex_tail_bad))

    worst_ex = float(np.min(excess))
    recovery = _sigmoid(6.0 * (worst_ex + 0.02))

    comps = {"asymmetry": float(asym), "convexity": float(convexity), "recovery": float(recovery)}
    r3af = 0.4 * comps["asymmetry"] + 0.3 * comps["convexity"] + 0.3 * comps["recovery"]
    return float(r3af), comps


# ----------------- SWC + SHOCKS -----------------
def shock_for_portfolio(shocks_df: pd.DataFrame, scenario: str, wts: dict[str, float]) -> float:
    sub = shocks_df[shocks_df["Scenario"] == scenario]
    if sub.empty:
        return 0.0
    m = dict(zip(sub["Ticker"].values, sub["CAGR_Shock"].values))
    return float(sum(w * float(m.get(t.upper(), 0.0)) for t, w in wts.items()))


def shock_for_benchmark(shocks_df: pd.DataFrame, scenario: str, bench: str) -> float:
    sub = shocks_df[(shocks_df["Scenario"] == scenario) & (shocks_df["Ticker"] == bench.upper())]
    if sub.empty:
        return 0.0
    return float(sub["CAGR_Shock"].iloc[0])


def compute_swc_and_r3af(
    r3mod_or_none,
    scenario_weights: pd.DataFrame,
    shocks_df: pd.DataFrame,
    base_cagr_port: float,
    base_cagr_bench: float,
    wts: dict[str, float],
    bench: str,
):
    rows = []
    for _, row in scenario_weights.iterrows():
        scen = str(row["Scenario"])
        w = float(row["Weight_Normalized"])
        port_c = float(base_cagr_port) + shock_for_portfolio(shocks_df, scen, wts)
        bench_c = float(base_cagr_bench) + shock_for_benchmark(shocks_df, scen, bench)
        rows.append({"Scenario": scen, "Weight_Normalized": w, "PortCAGR": port_c, "BenchCAGR": bench_c})

    df = pd.DataFrame(rows)
    swc = float((df["Weight_Normalized"] * df["PortCAGR"]).sum())

    if r3mod_or_none is not None:
        r3af_val, comps = r3mod_or_none.compute_r3af_from_df(
            df,
            scenario_col="Scenario",
            weight_col="Weight_Normalized",
            port_col="PortCAGR",
            bench_col="BenchCAGR",
        )
        return swc, float(r3af_val), comps, "external"

    r3af_val, comps = compute_r3af_fallback(df)
    return swc, float(r3af_val), comps, "fallback"


# ----------------- PORTFOLIO UTIL -----------------
def validate_weights(name: str, wts: dict[str, float]):
    s = float(sum(wts.values()))
    if not np.isfinite(s) or abs(s - 1.0) > 1e-6:
        raise ValueError(f"Portfolio {name} weights must sum to 1.0; got {s:.6f}")


# ----------------- MAIN -----------------
def main():
    scenario_weights = load_scenario_weights(SCENARIOS_XLSX)

    print("Loading OHLCV from zip…")
    ohlcv = load_zip_ohlcv(ZIP_PATH)
    print(f"Loaded tickers: {len(ohlcv)}")

    rets = {t: log_returns(df["px"]) for t, df in ohlcv.items()}

    # ---- Custom portfolios ----
    Latest = {
        "AVGE": 0.20,
        "SPMO": 0.10,
        "GRID": 0.10,
        "XAR": 0.10,
        "XSD": 0.15,
        "XLE": 0.10,
        "SETM": 0.10,
        "ROKT": 0.05,
        "AIRR": 0.05,
        "NUKZ": 0.05,
    }

    LatestC = {
        "AVGE": 0.20,
        "SPMO": 0.05,
        "QUAL": 0.05,
        "GRID": 0.10,
        "XAR": 0.15,
        "XSD": 0.15,
        "XLE": 0.10,
        "SETM": 0.10,
        "ROKT": 0.05,
        "NUKZ": 0.05,
    }

    Current = dict(LatestC)
    Current["XAR"] -= 0.05
    Current["SHLD"] = Current.get("SHLD", 0.0) + 0.05

    custom = {"Latest": Latest, "LatestC": LatestC, "Current": Current}
    for k, v in custom.items():
        validate_weights(k, v)

    all_ports: dict[str, dict[str, float]] = {}
    all_ports.update(try_load_canonical())
    all_ports.update(try_load_frontier())
    all_ports.update(custom)

    # Determine tickers needed
    tickers_used = set()
    for wts in all_ports.values():
        tickers_used.update([t.upper() for t in wts.keys()])
    for comps in PROXY_SPECS.values():
        tickers_used.update([c.upper() for c in comps])
    tickers_used.add(BENCHMARK)

    # Inception overrides
    inception_path = Path(INCEPTION_OVERRIDES_CSV)
    if inception_path.exists():
        inception = load_inception_overrides(INCEPTION_OVERRIDES_CSV)
        print(f"Loaded inception overrides: {len(inception)}")
    else:
        print(f"\nMissing {INCEPTION_OVERRIDES_CSV}. Inferring inceptions from your OHLCV (volume + sanity)…")
        inception = infer_inceptions_from_ohlcv(ohlcv)
        inferred_df = pd.DataFrame(
            [{"Ticker": k, "InceptionDate": v.date().isoformat()} for k, v in sorted(inception.items())]
        )
        out_path = OUTDIR / "inception_overrides_INFERRED.csv"
        inferred_df.to_csv(out_path, index=False)
        print(f"Saved inferred inception overrides to: {out_path}")

    # Scenario shocks
    shocks_path = Path(SCENARIO_SHOCKS_CSV)
    if shocks_path.exists():
        shocks_df = load_scenario_shocks(SCENARIO_SHOCKS_CSV)
        print(f"Loaded scenario shocks: {len(shocks_df)} rows")
    else:
        print(f"\nMissing {SCENARIO_SHOCKS_CSV}. Auto-generating scenario shocks…")
        shocks_df = build_auto_scenario_shocks(scenario_weights, list(tickers_used), bench=BENCHMARK)
        auto_path = OUTDIR / "scenario_shocks_AUTO.csv"
        shocks_df.to_csv(auto_path, index=False)
        print(f"Saved auto shocks to: {auto_path}")

    # Apply inception masks
    for t, d in inception.items():
        if t in rets:
            rets[t] = rets[t].where(rets[t].index >= d)

    # Union index
    union_sets = [set(r.dropna().index) for r in rets.values() if r is not None]
    full_idx = pd.DatetimeIndex(sorted(set().union(*union_sets))) if union_sets else pd.DatetimeIndex([])
    if len(full_idx) == 0:
        raise SystemExit("No valid return dates after inception masking. Check your data/zip.")
    print(f"\nUnion index (post-inception mask): {full_idx.min().date()} → {full_idx.max().date()} ({len(full_idx)} days)")

    ext = {t: rets[t].reindex(full_idx) for t in rets}

    # Proxies
    proxy_w = {}
    print("\nFitting proxies…")
    for target, comps in PROXY_SPECS.items():
        target = target.upper()
        if target not in ext:
            continue

        comp_r = {c.upper(): ext[c.upper()] for c in comps if c.upper() in ext}
        if len(comp_r) < 2:
            continue

        w = fit_proxy_weights(rets[target], comp_r)
        if w is None:
            w = {c: 1 / len(comp_r) for c in comp_r}

        proxy_w[target] = w
        pr = proxy_series(w, comp_r, full_idx)

        if target in inception:
            inc = inception[target]
            pr = pr.where(pr.index < inc)

        ext[target] = stitch(ext[target], pr)

    if proxy_w:
        pw = pd.DataFrame(
            [{"Target": t, "Component": c, "Weight": w} for t, d in proxy_w.items() for c, w in d.items()]
        ).sort_values(["Target", "Weight"], ascending=[True, False])
        pw.to_csv(OUTDIR / "proxy_weights.csv", index=False)
        print("Saved:", OUTDIR / "proxy_weights.csv")

    # R3AF module (optional)
    r3mod = _load_r3af_module_if_present(R3AF_PY)
    print("\nR3AF:", "EXTERNAL" if r3mod else "FALLBACK")

    if BENCHMARK not in ext:
        raise SystemExit(f"Benchmark {BENCHMARK} missing from data/zip.")

    bench_full = ext[BENCHMARK].dropna()

    missing = sorted([t for t in tickers_used if t not in ext])
    if missing:
        print("\nWARNING: Missing tickers in zip (ports using them may be skipped):")
        print(", ".join(missing))

    print(f"\nScoring {len(all_ports)} portfolios across {len(WINDOWS)} windows…")
    rows = []

    for pname, wts in all_ports.items():
        try:
            r_full = port_returns(wts, ext)
        except Exception as e:
            print(f"Skip {pname}: {e}")
            continue

        for wname, wstart in WINDOWS:
            r = slice_window(r_full, wstart)
            if len(r) < 252:
                continue

            b = bench_full
            if wstart is not None:
                b = b[b.index >= pd.to_datetime(wstart)]
            b = b.dropna()
            if len(b) < 252:
                continue

            C = cagr(r)
            S = sortino(r)
            Cm = calmar(r)
            M = max_dd(r)

            base_cagr_port = C
            base_cagr_bench = cagr(b)

            swc, r3af, comps, mode = compute_swc_and_r3af(
                r3mod,
                scenario_weights,
                shocks_df,
                base_cagr_port,
                base_cagr_bench,
                wts,
                BENCHMARK,
            )

            rows.append({
                "Window": wname,
                "Portfolio": pname,
                "Start": r.index.min().date(),
                "End": r.index.max().date(),
                "N_days": len(r),
                "CAGR": C,
                "Sortino": S,
                "Calmar": Cm,
                "MaxDD": M,
                "SWC": swc,
                "R3AF": r3af,
                "R3_mode": mode,
                # If external R3AF: keys are TailAlpha/Convexity/TailBeta/TailOrtho
                "R3_asym": comps.get("asymmetry", np.nan),
                "R3_conv": comps.get("convexity", np.nan),
                "R3_recov": comps.get("recovery", np.nan),
                "R3_TailAlpha": comps.get("TailAlpha", np.nan),
                "R3_Convexity": comps.get("Convexity", np.nan),
                "R3_TailBeta": comps.get("TailBeta", np.nan),
                "R3_TailOrtho": comps.get("TailOrtho", np.nan),
            })

    out = pd.DataFrame(rows)
    out_path = OUTDIR / "scoreboard_r3_swc_multiwindow.csv"
    out.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

    for wname in out["Window"].unique():
        sub = out[out["Window"] == wname].copy().sort_values(["SWC", "R3AF", "Calmar"], ascending=False)
        print(f"\nTop 10 — {wname} (rank: SWC, R3AF, Calmar):")
        print(sub.head(10).to_string(index=False))

    focus = ["Latest", "LatestC", "Current", "P1", "P2", "P4", "P11"]
    keep = out[out["Portfolio"].isin(focus)].copy()
    for wname in keep["Window"].unique():
        sub = keep[keep["Window"] == wname].copy().sort_values(["SWC", "R3AF", "Calmar"], ascending=False)
        print(f"\n=== FOCUS RANK — {wname} (SWC, R3AF, Calmar) ===")
        print(sub[["Portfolio", "CAGR", "SWC", "R3AF", "R3_mode", "Sortino", "Calmar", "MaxDD"]].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    finally:
        input("\nDone. Press Enter to exit…")
