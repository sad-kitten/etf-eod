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
    "AWQA": ["AQWA"],  # alias safety

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
    """Map scenario names to shock-regime buckets.
    Used to build auto scenario_shocks when explicit scenario_shocks.csv is absent.
    """
    t = _norm_s(s)

    # Core macro regimes
    if "base case" in t or "managed re-industrialization" in t or ("managed" in t and "re-industrialization" in t):
        return "base"
    if "reflation" in t or "industrial up-cycle" in t:
        return "reflation"
    if "stagflation" in t or "fiscal dominance" in t:
        return "stagflation"
    if "deflationary bust" in t or "demand collapse" in t or ("deflation" in t and "bust" in t):
        return "deflation_bust"

    # Credit / liquidity / sovereign
    if "debt overhang" in t or "credit stress" in t:
        return "debt_crisis"
    if "credit accident" in t or "u.s. regional" in t:
        return "debt_crisis"
    if "liquidity:" in t or ("jgb" in t and "fiscal dominance" in t):
        return "liquidity_crisis"
    if "us sovereign debt crisis" in t or "treasury market disorder" in t or ("treasury" in t and "disorder" in t):
        return "sovereign_disorder"

    # Tech regimes
    if "ai breakout" in t or "tech concentration" in t:
        return "ai_boom"
    if "ai bust" in t or "regulation backlash" in t:
        return "ai_bust"

    # Geopolitics
    if "geopolitical fragmentation" in t or "multipolar" in t:
        return "geo_shock"
    if "taiwan hostilities" in t or "kinetic shock" in t:
        return "taiwan_shock"
    if "china hard landing" in t:
        return "china_hard"

    # Sectoral / thematic
    if "energy shock" in t or "supply constraint" in t:
        return "energy_shock"
    if "pandemic" in t or "bio disruption" in t:
        return "pandemic"
    if "climate transition" in t:
        return "climate"
    if "systemic cyber" in t or "infrastructure disruption" in t or ("cyber" in t and "disruption" in t):
        return "cyber_shock"
    if "space race" in t or "tech-industrial rivalry" in t:
        return "ai_boom"
    if "cronyism" in t or "policy distortion" in t:
        return "cronyism"
    if "political polarization" in t or "governance shock" in t:
        return "risk_off"

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

    deflation_bust = {"broad": -0.030, "quality_factor": -0.015, "semis": -0.040, "defense": -0.010,
                      "grid": -0.020, "infra": -0.030, "energy": -0.030, "nuclear": 0.000, "materials": -0.020, "theme_growth": -0.040}

    debt_crisis = {"broad": -0.030, "quality_factor": -0.020, "semis": -0.040, "defense": -0.010,
                   "grid": -0.020, "infra": -0.030, "energy": 0.000, "nuclear": 0.015, "materials": 0.010, "theme_growth": -0.040}

    liquidity_crisis = {"broad": -0.025, "quality_factor": -0.015, "semis": -0.035, "defense": -0.005,
                        "grid": -0.015, "infra": -0.025, "energy": 0.000, "nuclear": 0.010, "materials": 0.005, "theme_growth": -0.040}

    sovereign_disorder = {"broad": -0.030, "quality_factor": -0.020, "semis": -0.040, "defense": 0.000,
                          "grid": -0.020, "infra": -0.030, "energy": 0.000, "nuclear": 0.010, "materials": 0.000, "theme_growth": -0.050}

    energy_shock = {"broad": -0.010, "quality_factor": -0.005, "semis": -0.010, "defense": 0.000,
                    "grid": -0.005, "infra": 0.000, "energy": 0.040, "nuclear": 0.010, "materials": 0.020, "theme_growth": -0.005}

    ai_boom = {"broad": 0.010, "quality_factor": 0.010, "semis": 0.040, "defense": 0.000,
               "grid": 0.005, "infra": 0.000, "energy": -0.005, "nuclear": 0.000, "materials": 0.000, "theme_growth": 0.020}

    ai_bust = {"broad": -0.015, "quality_factor": -0.010, "semis": -0.050, "defense": 0.000,
               "grid": -0.010, "infra": -0.015, "energy": -0.005, "nuclear": 0.000, "materials": -0.005, "theme_growth": -0.060}

    geo_shock = {"broad": -0.020, "quality_factor": -0.015, "semis": -0.030, "defense": 0.030,
                 "grid": -0.010, "infra": -0.010, "energy": 0.010, "nuclear": 0.020, "Also Run": 0.0, "materials": 0.010, "theme_growth": -0.010}

    # remove accidental key if present by dict literal edit below
    geo_shock = {k: v for k, v in geo_shock.items() if k != "Also Run"}

    taiwan_shock = {"broad": -0.030, "quality_factor": -0.020, "semis": -0.060, "defense": 0.050,
                    "grid": -0.015, "infra": -0.020, "energy": 0.020, "nuclear": 0.020, "materials": 0.010, "theme_growth": -0.030}

    cronyism = {"broad": -0.010, "quality_factor": -0.008, "semis": -0.010, "defense": 0.000,
                "grid": 0.000, "infra": 0.000, "energy": 0.010, "nuclear": 0.000, "materials": 0.000, "theme_growth": 0.000}

    china_hard = {"broad": -0.020, "quality_factor": -0.015, "semis": -0.020, "defense": 0.010,
                  "grid": -0.010, "infra": -0.010, "energy": -0.005, "nuclear": 0.005, "materials": -0.005, "theme_growth": -0.010}

    pandemic = {"broad": -0.020, "quality_factor": -0.010, "semis": 0.000, "defense": -0.010,
                "grid": -0.010, "infra": -0.020, "energy": -0.020, "nuclear": -0.005, "materials": -0.010, "theme_growth": -0.005}

    cyber_shock = {"broad": -0.020, "quality_factor": -0.010, "semis": -0.030, "defense": 0.010,
                   "grid": -0.010, "infra": -0.020, "energy": 0.000, "nuclear": 0.005, "materials": -0.005, "theme_growth": -0.030}

    climate = {"broad": -0.010, "quality_factor": 0.000, "semis": 0.005, "defense": 0.000,
               "grid": 0.020, "infra": 0.010, "energy": -0.020, "nuclear": 0.010, "materials": 0.010, "theme_growth": 0.010}

    return {"base": base, "risk_off": risk_off, "reflation": reflation, "stagflation": stagflation,
            "deflation_bust": deflation_bust, "debt_crisis": debt_crisis, "liquidity_crisis": liquidity_crisis,
            "sovereign_disorder": sovereign_disorder, "energy_shock": energy_shock, "ai_boom": ai_boom,
            "ai_bust": ai_bust, "geo_shock": geo_shock, "taiwan_shock": taiwan_shock, "cronyism": cronyism,
            "china_hard": china_hard, "pandemic": pandemic, "cyber_shock": cyber_shock, "climate": climate}


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
        frontier_cols = []
        for c in df.columns:
            s = str(c)
            if s.startswith("P") and s[1:].isdigit():
                frontier_cols.append(s)
        for col in frontier_cols:
            d = df[["Ticker", col]].set_index("Ticker")[col].to_dict()
            ports[str(col)] = {k.upper(): float(v) / 100.0 for k, v in d.items() if v and float(v) > 0}
        print(f"Loaded frontier portfolios: {len(ports)} (e.g., {sorted(list(ports.keys()))[:10]} …)")
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

        # If TailBeta/TailOrtho are not identifiable (common with bucketed benchmark shocks),
        # the external module may yield NaN r3af even though TailAlpha/Convexity are informative.
        # Approximate by treating TailOrtho as 0.0 (penalize missing orthogonality signal).
        r3af_f = float(r3af_val) if np.isfinite(r3af_val) else float("nan")
        if not np.isfinite(r3af_f) and isinstance(comps, dict):
            ta = float(comps.get("TailAlpha", np.nan))
            cx = float(comps.get("Convexity", np.nan))
            to = comps.get("TailOrtho", np.nan)
            to = float(to) if np.isfinite(to) else 0.0
            if np.isfinite(ta) and np.isfinite(cx):
                r3af_f = 0.55 * ta + 0.25 * cx + 0.20 * to
                return swc, float(r3af_f), comps, "external_approx"

        return swc, float(r3af_f), comps, "external"

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
    # Optional: load additional tickers from standalone CSVs (same format as yfinance CSVs)
    for _extra in ["AQWA.csv"]:
        p = Path(_extra)
        if p.exists():
            try:
                df_extra = pd.read_csv(p)
                ohlcv[p.stem.upper()] = clean_ohlcv_df(df_extra)
                print(f"Loaded extra OHLCV: {p.stem.upper()} from {p.name}")
            except Exception as e:
                print(f"WARNING: failed to load {p.name}: {e}")
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

    # CurQQEW: start from Current, swap SPMO -> QQEW (same 5% weight)
    CurQQEW = dict(Current)
    _w = CurQQEW.pop("SPMO", 0.0)
    if _w > 0:
        CurQQEW["QQEW"] = CurQQEW.get("QQEW", 0.0) + _w

    # CurrXT: start from Current, swap SPMO -> XT
    CurrXT = dict(Current)
    _w = CurrXT.pop("SPMO", 0.0)
    if _w > 0:
        CurrXT["XT"] = CurrXT.get("XT", 0.0) + _w

    # CurrROBO: start from Current, swap SPMO -> ROBO
    CurrROBO = dict(Current)
    _w = CurrROBO.pop("SPMO", 0.0)
    if _w > 0:
        CurrROBO["ROBO"] = CurrROBO.get("ROBO", 0.0) + _w

    # CurrIHI: start from Current, swap SPMO -> IHI
    CurrIHI = dict(Current)
    _w = CurrIHI.pop("SPMO", 0.0)
    if _w > 0:
        CurrIHI["IHI"] = CurrIHI.get("IHI", 0.0) + _w
    # CurrQUAL: start from Current, swap SPMO -> QUAL (adds to existing QUAL weight)
    CurrQUAL = dict(Current)
    _w = CurrQUAL.pop("SPMO", 0.0)
    if _w > 0:
        CurrQUAL["QUAL"] = CurrQUAL.get("QUAL", 0.0) + _w
    # CurrPAVEQQ: start from Current, swap SETM -> PAVE, and (SPMO + SHLD) -> QQEW
    CurrPAVEQQ = dict(Current)
    _w = CurrPAVEQQ.pop("SETM", 0.0)
    if _w > 0:
        CurrPAVEQQ["PAVE"] = CurrPAVEQQ.get("PAVE", 0.0) + _w
    _w = CurrPAVEQQ.pop("SPMO", 0.0)
    if _w > 0:
        CurrPAVEQQ["QQEW"] = CurrPAVEQQ.get("QQEW", 0.0) + _w
    _w = CurrPAVEQQ.pop("SHLD", 0.0)
    if _w > 0:
        CurrPAVEQQ["QQEW"] = CurrPAVEQQ.get("QQEW", 0.0) + _w

    # CurrHybrid: a stronger transplant toward the repeat winners (P4/P11) while keeping some of Current's DNA
    # (integer weights as %): AVGE10, QQEW15, PAVE10, XAR15, XSD20, GRID10, XLE10, QUAL5, NUKZ5
    CurrHybrid = {
        "AVGE": 0.10,
        "QQEW": 0.15,
        "PAVE": 0.10,
        "XAR": 0.15,
        "XSD": 0.20,
        "GRID": 0.10,
        "XLE": 0.10,
        "QUAL": 0.05,
        "NUKZ": 0.05,
    }
    # CurrConsistentA: "Competitive-Consistent" blend (engine + ballast)
    CurrConsistentA = {
        "XSD": 0.18,
        "XAR": 0.12,
        "QQEW": 0.12,
        "PAVE": 0.12,
        "AVGE": 0.15,
        "GRID": 0.08,
        "XLE": 0.08,
        "QUAL": 0.07,
        "IHI": 0.05,
        "NUKZ": 0.03,
    }

    # CurrConsistentB: steadier variant (more global ballast, slightly less engine)
    CurrConsistentB = {
        "AVGE": 0.20,
        "XSD": 0.15,
        "XAR": 0.12,
        "QQEW": 0.10,
        "PAVE": 0.10,
        "GRID": 0.08,
        "XLE": 0.08,
        "QUAL": 0.07,
        "IHI": 0.05,
        "NUKZ": 0.05,
    }
    # --- Bridge portfolios: designed to compete across eras (blend P4/P11 engine with Current ballast) ---
    CurrBridgeA = {  # P4 engine + ballast (removes SOXX/CIBR in favor of AVGE/XLE)
        "XSD": 0.20,
        "XAR": 0.20,
        "QQEW": 0.20,
        "PAVE": 0.10,
        "GRID": 0.10,
        "AVGE": 0.10,
        "XLE": 0.10,
    }

    CurrBridgeB = {  # More Current ballast, still heavy engine
        "AVGE": 0.15,
        "QQEW": 0.15,
        "XSD": 0.20,
        "XAR": 0.15,
        "PAVE": 0.10,
        "GRID": 0.10,
        "XLE": 0.10,
        "QUAL": 0.05,
    }

    CurrBridgeC = {  # Slightly steadier; keeps engine strong but spreads ballast
        "XSD": 0.18,
        "XAR": 0.18,
        "QQEW": 0.18,
        "PAVE": 0.12,
        "GRID": 0.08,
        "AVGE": 0.12,
        "XLE": 0.10,
        "QUAL": 0.04,
    }
    # --- Top-6 attempt portfolios (designed to be competitive across all eras) ---
    Top6AttemptA = {  # P1-like engine with MAX/2020 ballast
        "XSD": 0.20,
        "XAR": 0.20,
        "QQEW": 0.15,
        "PAVE": 0.15,
        "GRID": 0.10,
        "XLE": 0.10,
        "AVGE": 0.10,
    }

    Top6AttemptB = {  # slightly steadier variant with a touch of quality
        "XSD": 0.18,
        "XAR": 0.18,
        "QQEW": 0.14,
        "PAVE": 0.14,
        "GRID": 0.08,
        "XLE": 0.12,
        "AVGE": 0.12,
        "QUAL": 0.04,
    }
    # --- Top-6 attempt extensions (tune P4-style core to improve MAX/2020 without killing 2010/2012) ---
    Top6AttemptC = {  # Replace P4's (CIBR+SOXX) sleeve with (XLE+QUAL)
        "QQEW": 0.20,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "PAVE": 0.10,
        "XLE": 0.10,
        "QUAL": 0.10,
    }

    Top6AttemptD = {  # Replace only SOXX -> XLE; keep CIBR for cyber/geopolitical regimes
        "QQEW": 0.20,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "PAVE": 0.10,
        "CIBR": 0.10,
        "XLE": 0.10,
    }
    # --- Top-6 attempt variant to improve POST_COVID without SPMO ---
    # Top6AttemptF: based on Top6AttemptC but shifts 3% from PAVE to SETM (adds momentum/quality tilt without adding SPMO)
    Top6AttemptF = {
        "QQEW": 0.20,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "PAVE": 0.07,
        "SETM": 0.03,
        "XLE": 0.10,
        "QUAL": 0.10,
    }
    # Top6AttemptE: based on Top6AttemptC but shifts 3% from GRID to SETM
    Top6AttemptE = {
        "QQEW": 0.20,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.07,
        "SETM": 0.03,
        "PAVE": 0.10,
        "XLE": 0.10,
        "QUAL": 0.10,
    }
    # Top6AttemptG: based on Top6AttemptF but shifts an additional 2% from PAVE to SETM (PAVE 5%, SETM 5%)
    Top6AttemptG = {
        "QQEW": 0.20,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "PAVE": 0.05,
        "SETM": 0.05,
        "XLE": 0.10,
        "QUAL": 0.10,
    }


    # Top6AttemptH: based on Top6AttemptG but adds 2% SPMO funded from GRID (GRID 8%, SPMO 2%)
    Top6AttemptH = {
        "QQEW": 0.20,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.08,
        "SPMO": 0.02,
        "PAVE": 0.05,
        "SETM": 0.05,
        "XLE": 0.10,
        "QUAL": 0.10,
    }
    # Top6AttemptI: based on Top6AttemptG, shift 2% from GRID -> QQEW (keep SETM at 5%)
    Top6AttemptI = {
        "QQEW": 0.22,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.08,
        "PAVE": 0.05,
        "SETM": 0.05,
        "XLE": 0.10,
        "QUAL": 0.10,
    }


    # Top6AttemptJ: based on Top6AttemptG, shift 2% from XLE -> QQEW (keep SETM at 5%)
    Top6AttemptJ = {
        "QQEW": 0.22,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "PAVE": 0.05,
        "SETM": 0.05,
        "XLE": 0.08,
        "QUAL": 0.10,
    }
    # Top6AttemptK: based on Top6AttemptG, add 2% SPMO funded from PAVE (PAVE 3%, SPMO 2%)
    Top6AttemptK = {
        "QQEW": 0.20,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "PAVE": 0.03,
        "SETM": 0.05,
        "SPMO": 0.02,
        "XLE": 0.10,
        "QUAL": 0.10,
    }


    # Top6AttemptL: based on Top6AttemptG, add 3% SPMO funded from PAVE (PAVE 2%, SPMO 3%)
    Top6AttemptL = {
        "QQEW": 0.20,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "PAVE": 0.02,
        "SETM": 0.05,
        "SPMO": 0.03,
        "XLE": 0.10,
        "QUAL": 0.10,
    }
    # Top6AttemptLAQWA: Top6AttemptL with 2% AQWA funded from PAVE (PAVE 2% -> 0%, AQWA 0% -> 2%)
    Top6AttemptLAQWA = {
        "QQEW": 0.20,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "PAVE": 0.00,
        "AQWA": 0.02,
        "SETM": 0.05,
        "SPMO": 0.03,
        "XLE": 0.10,
        "QUAL": 0.10,
    }


    # Top6AttemptLIHI: Top6AttemptL with 3% IHI funded from PAVE (2%) + GRID (1%)
    Top6AttemptLIHI = {
        "QQEW": 0.20,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.09,
        "PAVE": 0.00,
        "IHI": 0.03,
        "SETM": 0.05,
        "SPMO": 0.03,
        "XLE": 0.10,
        "QUAL": 0.10,
    }


    # Top6AttemptLIHI_XLE: Top6AttemptL with 3% IHI funded from XLE (10% -> 7%) to preserve PAVE
    Top6AttemptLIHI_XLE = {
        "QQEW": 0.20,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "PAVE": 0.02,
        "IHI": 0.03,
        "SETM": 0.05,
        "SPMO": 0.03,
        "XLE": 0.07,
        "QUAL": 0.10,
    }
    # ---- Defensive / Quality / Energy tilt variants (less QQEW/XT/ROKT/SPMO; more SHLD/QUAL/XLE) ----

    # CurrDefensiveA: start from Current, move SPMO+ROKT into QUAL/XLE/SHLD
    CurrDefensiveA = {
        "AVGE": 0.20,
        "QUAL": 0.08,
        "XLE": 0.15,
        "SHLD": 0.07,
        "GRID": 0.10,
        "XSD": 0.15,
        "XAR": 0.10,
        "SETM": 0.10,
        "NUKZ": 0.05,
    }

    # Top6DefensiveL: start from Top6AttemptL engine, reduce QQEW+SPMO; add SHLD and more QUAL/XLE
    Top6DefensiveL = {
        "QQEW": 0.14,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "PAVE": 0.02,
        "SETM": 0.05,
        "XLE": 0.12,
        "QUAL": 0.12,
        "SHLD": 0.05,
    }
    # ---- User-specified blends (auto-normalized to sum to 1.0) ----
    # Note: you wrote AWQA; treated as AQWA.
    blend1_raw = {
        "AVGE": 0.10,
        "GRID": 0.10,
        "NUKZ": 0.05,
        "PAVE": 0.05,
        "QUAL": 0.05,
        "QQEW": 0.10,
        "SHLD": 0.05,
        "SETM": 0.05,
        "XLE": 0.10,
        "XSD": 0.15,
        "IHI": 0.05,
    }
    _s = float(sum(blend1_raw.values()))
    blend1 = {k: v / _s for k, v in blend1_raw.items()}

    blend2_raw = {
        "AVGE": 0.10,
        "GRID": 0.10,
        "NUKZ": 0.05,
        "PAVE": 0.05,
        "QUAL": 0.05,
        "QQEW": 0.05,
        "SHLD": 0.05,
        "SETM": 0.05,
        "XLE": 0.10,
        "XSD": 0.15,
        "IHI": 0.05,
        "AQWA": 0.05,  # (AWQA -> AQWA)
    }
    _s = float(sum(blend2_raw.values()))
    blend2 = {k: v / _s for k, v in blend2_raw.items()}
    
    blend3 = {
        "AVGE": 0.10,
        "GRID": 0.10,
        "NUKZ": 0.05,
        "PAVE": 0.05,
        "QUAL": 0.10,
        "QQEW": 0.05,
        "SETM": 0.05,
        "SHLD": 0.05,
        "XAR": 0.15,
        "XLE": 0.15,
        "XSD": 0.15,
    }
    # ---- Top6DefensiveL improvement candidates (surgical tweaks) ----
    Top6DefensiveL_Momo = {
        "QQEW": 0.14,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.02,
        "SETM": 0.05,
        "XLE": 0.12,
        "QUAL": 0.12,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }

    Top6DefensiveL_Qual = {
        "QQEW": 0.14,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SETM": 0.05,
        "XLE": 0.12,
        "QUAL": 0.14,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }

    Top6DefensiveL_Health = {
        "QQEW": 0.14,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SETM": 0.05,
        "XLE": 0.12,
        "QUAL": 0.12,
        "SHLD": 0.05,
        "IHI": 0.02,
        "PAVE": 0.00,
    }
    # ---- Push further from Top6DefensiveL_Momo ----
    Top6DefensiveL_MomoV2 = {
        "QQEW": 0.14,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.03,
        "SETM": 0.05,
        "XLE": 0.12,
        "QUAL": 0.11,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }

    Top6DefensiveL_MomoV3 = {
        "QQEW": 0.14,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.04,
        "SETM": 0.05,
        "XLE": 0.12,
        "QUAL": 0.10,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }

    Top6DefensiveL_MomoV4 = {
        "QQEW": 0.13,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.04,
        "SETM": 0.05,
        "XLE": 0.12,
        "QUAL": 0.11,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }
    # MomoV5: based on MomoV4, shift 1% from QQEW -> SETM (QQEW 12%, SETM 6%)
    Top6DefensiveL_MomoV5 = {
        "QQEW": 0.12,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.04,
        "SETM": 0.06,
        "XLE": 0.12,
        "QUAL": 0.11,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }
    # ---- Next push from MomoV5 (preserve QQEW, fund SETM from elsewhere) ----
    # MomoV6: fund +1% SETM from -1% SHLD (QQEW kept at 13%)
    Top6DefensiveL_MomoV6 = {
        "QQEW": 0.13,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.04,
        "SETM": 0.06,
        "XLE": 0.12,
        "QUAL": 0.11,
        "SHLD": 0.04,
        "PAVE": 0.00,
    }

    # MomoV7: fund +1% SETM from -1% GRID (QQEW kept at 13%)
    Top6DefensiveL_MomoV7 = {
        "QQEW": 0.13,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.09,
        "SPMO": 0.04,
        "SETM": 0.06,
        "XLE": 0.12,
        "QUAL": 0.11,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }
    # MomoV8: based on MomoV5, shift 1% from QQEW -> SETM (QQEW 11%, SETM 7%)
    Top6DefensiveL_MomoV8 = {
        "QQEW": 0.11,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.04,
        "SETM": 0.07,
        "XLE": 0.12,
        "QUAL": 0.11,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }
    # ---- Variants from MomoV8: fund +1% SETM from non-QQEW sources ----
    # MomoV9_GRID: SETM 8% funded by GRID 10% -> 9% (QQEW stays 11%)
    Top6DefensiveL_MomoV9_GRID = {
        "QQEW": 0.11,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.09,
        "SPMO": 0.04,
        "SETM": 0.08,
        "XLE": 0.12,
        "QUAL": 0.11,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }

    # MomoV10_SHLD: SETM 8% funded by SHLD 5% -> 4% (QQEW stays 11%)
    Top6DefensiveL_MomoV10_SHLD = {
        "QQEW": 0.11,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.04,
        "SETM": 0.08,
        "XLE": 0.12,
        "QUAL": 0.11,
        "SHLD": 0.04,
        "PAVE": 0.00,
    }
    # ---- Next lever after SETM limit: micro-tilt between QUAL and SPMO (hold everything else constant) ----
    # MomoV11: +1% SPMO funded by -1% QUAL (SPMO 5%, QUAL 10%)
    Top6DefensiveL_MomoV11 = {
        "QQEW": 0.11,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.05,
        "SETM": 0.07,
        "XLE": 0.12,
        "QUAL": 0.10,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }

    # MomoV12: +2% SPMO funded by -2% QUAL (SPMO 6%, QUAL 9%)
    Top6DefensiveL_MomoV12 = {
        "QQEW": 0.11,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.06,
        "SETM": 0.07,
        "XLE": 0.12,
        "QUAL": 0.09,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }
    # MomoV13: +1% SPMO funded by -1% QUAL (SPMO 7%, QUAL 8%)
    Top6DefensiveL_MomoV13 = {
        "QQEW": 0.11,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.07,
        "SETM": 0.07,
        "XLE": 0.12,
        "QUAL": 0.08,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }
    # MomoV14: +1% SPMO funded by -1% QUAL (SPMO 8%, QUAL 7%)
    Top6DefensiveL_MomoV14 = {
        "QQEW": 0.11,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.08,
        "SETM": 0.07,
        "XLE": 0.12,
        "QUAL": 0.07,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }
    # MomoV15: +1% SPMO funded by -1% QUAL (SPMO 9%, QUAL 6%)
    Top6DefensiveL_MomoV15 = {
        "QQEW": 0.11,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.09,
        "SETM": 0.07,
        "XLE": 0.12,
        "QUAL": 0.06,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }
    # ---- ETF_ERA_2010 improvement probes (off Top6DefensiveL_MomoV15) ----
    # TestA: +1% QQEW funded by -1% SPMO (tilt slightly toward 2010-era engine)
    Top6DefensiveL_MomoV16_QQEW = {
        "QQEW": 0.12,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.08,
        "SETM": 0.07,
        "XLE": 0.12,
        "QUAL": 0.06,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }

    # TestB: +1% PAVE funded by -1% SHLD (reintroduce small 2010-era ballast)
    Top6DefensiveL_MomoV16_PAVE = {
        "QQEW": 0.11,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.09,
        "SETM": 0.07,
        "XLE": 0.12,
        "QUAL": 0.06,
        "SHLD": 0.04,
        "PAVE": 0.01,
    }

    # TestC: fund PAVE using some SPMO.
    # +2% PAVE funded by -2% SPMO (keeps 2020+ helper alive, but adds meaningful PAVE exposure)
    Top6DefensiveL_MomoV16_SPMO2PAVE = {
        "QQEW": 0.11,
        "XSD": 0.20,
        "XAR": 0.20,
        "GRID": 0.10,
        "SPMO": 0.07,
        "SETM": 0.07,
        "XLE": 0.12,
        "QUAL": 0.06,
        "SHLD": 0.05,
        "PAVE": 0.02,
    }
    # ---- Evergreen9 and No-momentum comparators (for cross-era ranking) ----
    Evergreen9 = {
        "XSD": 0.20,
        "XAR": 0.20,
        "XLE": 0.12,
        "QQEW": 0.11,
        "GRID": 0.10,
        "SPMO": 0.09,
        "SETM": 0.07,
        "QUAL": 0.06,
        "SHLD": 0.05,
        "PAVE": 0.00,
    }

    # No-momentum baseline (SPMO removed; SPHQ used; QUAL moved into XSD)
    Evergreen9_NoMomoB_SPHQ_XSD = {
        "XSD": 0.26,
        "XAR": 0.20,
        "XLE": 0.12,
        "QQEW": 0.11,
        "GRID": 0.10,
        "SPHQ": 0.09,
        "SETM": 0.07,
        "SHLD": 0.05,
        "QUAL": 0.00,
    }

    # QQEW sleeve swaps
    Evergreen9_NoMomoB_SPHQ_XSD_QQQE = {
        "XSD": 0.26,
        "XAR": 0.20,
        "XLE": 0.12,
        "QQQE": 0.11,
        "GRID": 0.10,
        "SPHQ": 0.09,
        "SETM": 0.07,
        "SHLD": 0.05,
        "QUAL": 0.00,
    }

    Evergreen9_NoMomoB_SPHQ_XSD_RSP = {
        "XSD": 0.26,
        "XAR": 0.20,
        "XLE": 0.12,
        "RSP": 0.11,
        "GRID": 0.10,
        "SPHQ": 0.09,
        "SETM": 0.07,
        "SHLD": 0.05,
        "QUAL": 0.00,
    }

    # ---- Next surgical step: shift 1% between SETM and SPHQ in the QQQE version ----
    # Variant 1: SPHQ +1, SETM -1  (SPHQ 10, SETM 6)
    Evergreen9_NoMomoB_QQQE_SPHQ10_SETM6 = {
        "XSD": 0.26,
        "XAR": 0.20,
        "XLE": 0.12,
        "QQQE": 0.11,
        "GRID": 0.10,
        "SPHQ": 0.10,
        "SETM": 0.06,
        "SHLD": 0.05,
        "QUAL": 0.00,
    }

    # Variant 2: SPHQ -1, SETM +1  (SPHQ 8, SETM 8)
    Evergreen9_NoMomoB_QQQE_SPHQ8_SETM8 = {
        "XSD": 0.26,
        "XAR": 0.20,
        "XLE": 0.12,
        "QQQE": 0.11,
        "GRID": 0.10,
        "SPHQ": 0.08,
        "SETM": 0.08,
        "SHLD": 0.05,
        "QUAL": 0.00,
    }
    # ---- Further surgical step: shift 1% more between SETM and SPHQ (QQQE version) ----
    # Variant 3: SPHQ 7, SETM 9
    Evergreen9_NoMomoB_QQQE_SPHQ7_SETM9 = {
        "XSD": 0.26,
        "XAR": 0.20,
        "XLE": 0.12,
        "QQQE": 0.11,
        "GRID": 0.10,
        "SPHQ": 0.07,
        "SETM": 0.09,
        "SHLD": 0.05,
        "QUAL": 0.00,
    }

    # Variant 4: SPHQ 9, SETM 7
    Evergreen9_NoMomoB_QQQE_SPHQ9_SETM7 = {
        "XSD": 0.26,
        "XAR": 0.20,
        "XLE": 0.12,
        "QQQE": 0.11,
        "GRID": 0.10,
        "SPHQ": 0.09,
        "SETM": 0.07,
        "SHLD": 0.05,
        "QUAL": 0.00,
    }
    # ---- Additional NoMomoB QQQE micro-shifts (1% knobs) ----
    # +1% XSD funded by -1% GRID
    Evergreen9_NoMomoB_QQQE_XSD27_GRID9 = {
        "XSD": 0.27,
        "XAR": 0.20,
        "XLE": 0.12,
        "QQQE": 0.11,
        "GRID": 0.09,
        "SPHQ": 0.09,
        "SETM": 0.07,
        "SHLD": 0.05,
        "QUAL": 0.00,
    }

    # +1% XSD funded by -1% SHLD
    Evergreen9_NoMomoB_QQQE_XSD27_SHLD4 = {
        "XSD": 0.27,
        "XAR": 0.20,
        "XLE": 0.12,
        "QQQE": 0.11,
        "GRID": 0.10,
        "SPHQ": 0.09,
        "SETM": 0.07,
        "SHLD": 0.04,
        "QUAL": 0.00,
    }

    # ---- User-specified multi-sleeve blend (weights in whole %; sums to 100) ----
    # Note: your provided weights summed to 101%; adjusted by reducing CIBR from 21% -> 20% to total 100%.
    SingleBest_Blend_Beta102 = {
        "CIBR": 0.20,
        "QQEW": 0.13,
        "XSD": 0.12,
        "GRID": 0.11,
        "QMOM": 0.09,
        "QUAL": 0.08,
        "VTI": 0.04,
        "XLE": 0.04,
        "MTUM": 0.03,
        "SPHQ": 0.03,
        "PAVE": 0.03,
        "SCHD": 0.02,
        "SPLV": 0.02,
        "URA": 0.02,
        "SPMO": 0.01,
        "VXUS": 0.01,
        "RSP": 0.01,
        "XAR": 0.01,
        # "USMV": 0.00,  # explicitly omitted (0%)
    }






















































































    
    
    
    
    
    
    custom = {
        # With-momentum
        "Evergreen9": Evergreen9,

        # No-momentum baselines + swaps
        "Evergreen9_NoMomoB_SPHQ_XSD": Evergreen9_NoMomoB_SPHQ_XSD,
        "Evergreen9_NoMomoB_SPHQ_XSD_QQQE": Evergreen9_NoMomoB_SPHQ_XSD_QQQE,
        "Evergreen9_NoMomoB_SPHQ_XSD_RSP": Evergreen9_NoMomoB_SPHQ_XSD_RSP,

        # Next surgical step variants (QQQE version)
        "Evergreen9_NoMomoB_QQQE_SPHQ10_SETM6": Evergreen9_NoMomoB_QQQE_SPHQ10_SETM6,
        "Evergreen9_NoMomoB_QQQE_SPHQ8_SETM8": Evergreen9_NoMomoB_QQQE_SPHQ8_SETM8,
        "Evergreen9_NoMomoB_QQQE_SPHQ7_SETM9": Evergreen9_NoMomoB_QQQE_SPHQ7_SETM9,
        "Evergreen9_NoMomoB_QQQE_SPHQ9_SETM7": Evergreen9_NoMomoB_QQQE_SPHQ9_SETM7,
        "Evergreen9_NoMomoB_QQQE_XSD27_GRID9": Evergreen9_NoMomoB_QQQE_XSD27_GRID9,
        "Evergreen9_NoMomoB_QQQE_XSD27_SHLD4": Evergreen9_NoMomoB_QQQE_XSD27_SHLD4,
        "SingleBest_Blend_Beta102": SingleBest_Blend_Beta102,
    }

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
                "R3_TailAlpha": comps.get("TailAlpha", np.nan) if isinstance(comps, dict) else np.nan,
                "R3_Convexity": comps.get("Convexity", np.nan) if isinstance(comps, dict) else np.nan,
                "R3_TailBeta": comps.get("TailBeta", np.nan) if isinstance(comps, dict) else np.nan,
                "R3_TailOrtho": comps.get("TailOrtho", np.nan) if isinstance(comps, dict) else np.nan,
                "R3_asym": comps.get("asymmetry", np.nan) if isinstance(comps, dict) else np.nan,
                "R3_conv": comps.get("convexity", np.nan) if isinstance(comps, dict) else np.nan,
                "R3_recov": comps.get("recovery", np.nan) if isinstance(comps, dict) else np.nan,
            })

    out = pd.DataFrame(rows)
    # Drop the mode/fallback column and any columns that are entirely NaN across the full output
    for _col in ["R3_mode", "mode", "fallback"]:
        if _col in out.columns:
            out = out.drop(columns=[_col])
    out = out.loc[:, ~out.isna().all()]
    out_path = OUTDIR / "scoreboard_r3_swc_multiwindow.csv"
    out.to_csv(out_path, index=False)
    print("\nSaved:", out_path)

    for wname in out["Window"].unique():
        sub = out[out["Window"] == wname].copy()
        _cols = [c for c in ["SWC","R3AF","Calmar"] if c in sub.columns]
        if _cols:
            sub = sub.sort_values(_cols, ascending=[False]*len(_cols))
        print("\nTop 15 — {} (rank: SWC, R3AF, Calmar):".format(wname))
        disp = sub.head(15).copy()
        for _col in ["R3_mode", "mode", "fallback"]:
            if _col in disp.columns:
                disp = disp.drop(columns=[_col])
        disp = disp.loc[:, ~disp.isna().all()]
        print(disp.to_string(index=False))


if __name__ == "__main__":
    main()
