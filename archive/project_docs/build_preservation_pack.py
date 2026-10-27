from __future__ import annotations

import json, math, hashlib, zipfile, os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls

BASE = Path('/mnt/data')
OUTROOT = BASE / 'KEEP_Preservation_Pack_2026Q1'

# ---------- helpers ----------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_ff5(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=3)
    df = df.rename(columns={df.columns[0]: 'Date'})
    df['Date'] = df['Date'].astype(str)
    df = df[df['Date'].str.match(r'^\d{6}$', na=False)].copy()
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'] + '01', format='%Y%m%d') + pd.offsets.MonthEnd(0)
    return df


def parse_mom(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=13)
    df = df.rename(columns={df.columns[0]: 'Date'})
    df['Date'] = df['Date'].astype(str)
    df = df[df['Date'].str.match(r'^\d{6}$', na=False)].copy()
    df['Mom'] = pd.to_numeric(df['Mom'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'] + '01', format='%Y%m%d') + pd.offsets.MonthEnd(0)
    return df[['Date', 'Mom']]


def load_monthly_returns(zip_path: Path, tickers: List[str]) -> Dict[str, pd.Series]:
    z = zipfile.ZipFile(zip_path)
    out = {}
    for t in tickers:
        fn = f'{t}.csv'
        if fn not in z.namelist():
            continue
        df = pd.read_csv(z.open(fn))
        df['Date'] = pd.to_datetime(df['Date'])
        price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
        s = pd.Series(pd.to_numeric(df[price_col], errors='coerce').values, index=df['Date']).dropna().sort_index()
        m = s.resample('ME').last().pct_change() * 100.0
        out[t] = m
    return out


def fit_factor_model(returns_m: pd.Series, factors: pd.DataFrame) -> dict:
    df = returns_m.rename('ret').to_frame().join(factors.set_index('Date'), how='inner').dropna()
    df['excess'] = df['ret'] - df['RF']
    X = df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']].values
    X = np.column_stack([np.ones(len(X)), X])
    y = df['excess'].values
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    alpha_m = float(coef[0])
    betas = coef[1:].astype(float)
    n = int(len(df))
    shrink_w = n / (n + 60)
    alpha_m_shrunk = alpha_m * shrink_w
    betas_shrunk = betas * shrink_w
    # simple r2
    yhat = X @ coef
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return {
        'n': n,
        'r2': r2,
        'shrink_w': shrink_w,
        'alpha_m': alpha_m,
        'alpha_m_shrunk': alpha_m_shrunk,
        'Mkt-RF': float(betas_shrunk[0]),
        'SMB': float(betas_shrunk[1]),
        'HML': float(betas_shrunk[2]),
        'RMW': float(betas_shrunk[3]),
        'CMA': float(betas_shrunk[4]),
        'Mom': float(betas_shrunk[5]),
    }


def build_factor_fit_row(fit: dict, premia_df: pd.DataFrame) -> pd.Series:
    rows = []
    for _, r in premia_df.iterrows():
        val = (
            float(r['RF'])
            + fit['Mkt-RF'] * float(r['Mkt-RF'])
            + fit['SMB'] * float(r['SMB'])
            + fit['HML'] * float(r['HML'])
            + fit['RMW'] * float(r['RMW'])
            + fit['CMA'] * float(r['CMA'])
            + fit['Mom'] * float(r['Mom'])
            + 12.0 * fit['alpha_m_shrunk']
        )
        rows.append(val)
    return pd.Series(rows, index=premia_df['Scenario'].tolist(), dtype=float)


def compute_proxy_weights(target: pd.Series, comps: Dict[str, pd.Series]) -> Dict[str, float]:
    df = target.rename('target').to_frame()
    for k, s in comps.items():
        df[k] = s
    df = df.dropna()
    X = df[list(comps.keys())].values
    y = df['target'].values
    w, _ = nnls(X, y)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    return {k: float(v) for k, v in zip(comps.keys(), w)}


def compute_port_row(weights: Dict[str, float], etf_rows: pd.DataFrame) -> pd.Series:
    row = pd.Series(0.0, index=etf_rows.columns, dtype=float)
    for t, w in weights.items():
        row += (w / 100.0) * etf_rows.loc[t]
    return row


def cew_30y(row: pd.Series, weights: pd.Series) -> float:
    arr = np.log1p(row.values / 100.0)
    return float(np.exp(30.0 * np.sum(weights.values * arr)))


def weighted_quantile(x, w, q):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    idx = np.argsort(x)
    x, w = x[idx], w[idx]
    cw = np.cumsum(w) / np.sum(w)
    return float(np.interp(q, cw, x))


def compute_r3af(port_row: pd.Series, bench_row: pd.Series, w: pd.Series) -> Tuple[float, dict]:
    # R3AF is computed on decimal returns, not percentage-point inputs.
    r = port_row.values.astype(float) / 100.0
    b = bench_row.values.astype(float) / 100.0
    ww = w.values.astype(float)
    ql = weighted_quantile(b, ww, 0.20)
    qh = weighted_quantile(b, ww, 0.80)
    bad = b <= ql
    good = b >= qh
    mid = ~(bad | good)

    def wmean(x, ws):
        return float(np.sum(x * ws) / np.sum(ws)) if np.sum(ws) > 0 else float('nan')

    mu_bad_r = wmean(r[bad], ww[bad])
    mu_mid_r = wmean(r[mid], ww[mid])
    mu_good_r = wmean(r[good], ww[good])
    mu_bad_b = wmean(b[bad], ww[bad])

    tail_alpha = mu_bad_r - mu_bad_b
    convexity = (mu_good_r - mu_mid_r) - (mu_mid_r - mu_bad_r)

    # weighted beta in bad tail
    rb = r[bad]
    bb = b[bad]
    wb = ww[bad]
    if len(rb) >= 2 and wb.sum() > 0:
        bw = np.sum(wb * bb) / np.sum(wb)
        rw = np.sum(wb * rb) / np.sum(wb)
        vb = np.sum(wb * (bb - bw) ** 2)
        tail_beta = float(np.sum(wb * (bb - bw) * (rb - rw)) / vb) if vb > 1e-18 else np.nan
    else:
        tail_beta = np.nan
    tail_ortho = 1.0 - min(2.0, abs(tail_beta)) if np.isfinite(tail_beta) else np.nan
    r3af = 0.55 * tail_alpha + 0.25 * convexity + 0.20 * tail_ortho
    return float(r3af), {
        'TailAlpha': float(tail_alpha),
        'Convexity': float(convexity),
        'TailBeta': float(tail_beta) if np.isfinite(tail_beta) else np.nan,
        'TailOrtho': float(tail_ortho) if np.isfinite(tail_ortho) else np.nan,
    }


def make_stress_weights(kind: str, weights: pd.Series, severity: pd.Series, bench_row: pd.Series) -> pd.Series:
    swc_bench = float(np.sum(weights.values * bench_row.values))
    badness = np.maximum(0.0, swc_bench - bench_row.values)
    denom = max(1e-9, swc_bench - np.min(bench_row.values))
    badness = badness / denom
    sev = severity.values.astype(float)
    w = weights.values.astype(float)
    sqrtw = np.sqrt(w)
    if kind == 'additive':
        lam = 2.643
        raw = sqrtw * (sev + lam * badness)
    elif kind == 'multiplicative':
        lam = 0.567
        raw = sqrtw * sev * (1 + lam * badness)
    elif kind == 'power':
        lam = 0.323
        raw = sqrtw * np.power(sev, 0.8) * np.power(0.25 + badness, lam)
    else:
        raise ValueError(kind)
    raw = np.maximum(raw, 0)
    raw = raw / raw.sum()
    return pd.Series(raw, index=weights.index)


def robust_metrics(port_row: pd.Series, bench_row: pd.Series, w: pd.Series, severity: pd.Series, stress_kind: str) -> dict:
    equal_w = pd.Series(np.repeat(1 / len(w), len(w)), index=w.index)
    stress_w = make_stress_weights(stress_kind, w, severity, bench_row)
    blend_w = 0.50 * w + 0.25 * equal_w + 0.25 * stress_w
    blend_w = blend_w / blend_w.sum()
    rets = port_row.copy()
    robust_cew = cew_30y(rets, blend_w)
    cew_can = cew_30y(rets, w)
    cew_eq = cew_30y(rets, equal_w)
    cew_stress = cew_30y(rets, stress_w)
    shortfall = float(np.sum(blend_w.values * np.maximum(0.0, 5.0 - rets.values)))
    swc = float(np.sum(w.values * rets.values))
    r3af, comps = compute_r3af(rets, bench_row, w)
    ordered = rets.sort_values()
    worst = float(ordered.iloc[0])
    k = max(1, math.ceil(0.1 * len(ordered)))
    worst_dec = float(ordered.iloc[:k].mean())
    best = float(ordered.iloc[-1])
    return {
        'Robust_CEW_30y': robust_cew,
        'CEW_Canonical_30y': cew_can,
        'CEW_Equal_30y': cew_eq,
        'CEW_Stress_30y': cew_stress,
        'SWCAGR': swc,
        'R3AF': r3af,
        'Shortfall_lt5': shortfall,
        'Worst_Scenario_CAGR': worst,
        'Worst_Decile_CAGR': worst_dec,
        'Best_Scenario_CAGR': best,
        **comps,
    }


def realized_metrics(weights: Dict[str, float], monthly_returns: Dict[str, pd.Series], start='2024-02-29', end='2026-02-28') -> dict:
    series = []
    for t, wt in weights.items():
        if t not in monthly_returns:
            return {'Realized_CAGR': np.nan, 'Realized_Vol': np.nan, 'Sortino': np.nan, 'MaxDD': np.nan, 'Calmar': np.nan}
        series.append(monthly_returns[t].rename(t))
    df = pd.concat(series, axis=1).sort_index()
    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))].copy()
    # drop first row if any missing, then equalize on available rows
    df = df.dropna(how='any')
    if df.empty:
        return {'Realized_CAGR': np.nan, 'Realized_Vol': np.nan, 'Sortino': np.nan, 'MaxDD': np.nan, 'Calmar': np.nan}
    w = np.array([weights[c] / 100.0 for c in df.columns])
    pret = df.values @ w
    s = pd.Series(pret, index=df.index)
    n = len(s)
    terminal = float(np.prod(1 + s / 100.0))
    years = n / 12.0
    cagr = terminal ** (1 / years) - 1 if years > 0 and terminal > 0 else np.nan
    vol = float(s.std(ddof=1) * np.sqrt(12) / 100.0) if n > 1 else np.nan
    downside = s.copy()
    downside[downside > 0] = 0
    downside_dev = float(np.sqrt((downside / 100.0).pow(2).mean()) * np.sqrt(12))
    sortino = (cagr / downside_dev) if downside_dev and downside_dev > 0 else np.nan
    wealth = (1 + s / 100.0).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    maxdd = float(dd.min())
    calmar = (cagr / abs(maxdd)) if maxdd < 0 else np.nan
    return {
        'Realized_CAGR': cagr * 100.0,
        'Realized_Vol': vol * 100.0,
        'Sortino': float(sortino),
        'MaxDD': maxdd * 100.0,
        'Calmar': float(calmar),
    }


def main():
    # clean output dir
    if OUTROOT.exists():
        for p in sorted(OUTROOT.rglob('*'), reverse=True):
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                try:
                    p.rmdir()
                except OSError:
                    pass
    ensure_dir(OUTROOT)
    for sub in ['00_admin','01_sources','02_working','03_derived']:
        ensure_dir(OUTROOT / sub)

    # load source files
    mat = pd.ExcelFile(BASE / 'KEEP_Scenario_CAGR_Matrix_2026Q1_plus_CTA_overlay.xlsx')
    scenario_weights = pd.read_excel(BASE / 'KEEP_Scenarios_Canonical_2026Q1.xlsx', sheet_name='Scenario_Weights')
    scenario_weights = scenario_weights.rename(columns={'Weight_Normalized':'Weight'})[['Scenario','Weight','Severity_1to5','Canonical_Parent','Notes_Merged']]
    scen = pd.read_excel(BASE / 'KEEP_Scenario_CAGR_Matrix_2026Q1_plus_CTA_overlay.xlsx', sheet_name='Scenarios')
    premia = pd.read_excel(BASE / 'KEEP_Scenario_CAGR_Matrix_2026Q1_plus_CTA_overlay.xlsx', sheet_name='Scenario_Factor_Premia_%')
    premia = premia.rename(columns={premia.columns[0]:'Scenario'})
    etf = pd.read_excel(BASE / 'KEEP_Scenario_CAGR_Matrix_2026Q1_plus_CTA_overlay.xlsx', sheet_name='ETF_Scenario_CAGR')
    etf = etf.rename(columns={etf.columns[0]:'Ticker'}).set_index('Ticker')
    deltas = pd.read_excel(BASE / 'KEEP_Scenario_CAGR_Matrix_2026Q1_plus_CTA_overlay.xlsx', sheet_name='Scenario_Deltas_%')
    deltas = deltas.rename(columns={deltas.columns[0]:'Ticker'})
    betas_current = pd.read_excel(BASE / 'KEEP_Scenario_CAGR_Matrix_2026Q1_plus_CTA_overlay.xlsx', sheet_name='Betas_Shrunk')
    port_scen = pd.read_excel(BASE / 'KEEP_Scenario_CAGR_Matrix_2026Q1_plus_CTA_overlay.xlsx', sheet_name='Portfolio_Scenario_CAGR')
    port_scen = port_scen.rename(columns={port_scen.columns[0]:'Portfolio'}).set_index('Portfolio')
    portfolios = pd.read_excel(BASE / 'KEEP_Portfolios_Canonical_2026Q1.xlsx', sheet_name='Portfolios')
    registry = json.load(open(BASE / 'Portfolio_Scenario_Analysis_Metric_Registry_2026-03-16.json'))

    # source extracts
    scenario_weights.to_csv(OUTROOT / '01_sources' / 'scenario_weights.csv', index=False)
    premia.to_csv(OUTROOT / '01_sources' / 'scenario_factor_premia.csv', index=False)
    etf.reset_index().to_csv(OUTROOT / '01_sources' / 'etf_scenario_cagr_current.csv', index=False)
    betas_current.to_csv(OUTROOT / '01_sources' / 'betas_shrunk_current.csv', index=False)
    portfolios.to_csv(OUTROOT / '01_sources' / 'portfolios_canonical_long.csv', index=False)
    pd.DataFrame({'key': list(registry.keys()), 'value': [json.dumps(v) if isinstance(v, (dict,list)) else v for v in registry.values()]}).to_csv(OUTROOT / '01_sources' / 'v8c_metric_registry_flat.csv', index=False)

    # factor fits and rebasing
    ff = parse_ff5(BASE / 'F-F_Research_Data_5_Factors_2x3.csv').merge(parse_mom(BASE / 'F-F_Momentum_Factor.csv'), on='Date', how='inner')
    needed = ['CTA','SHLD','GD','LMT','RTX','CIBR','XAR','AIRR','NUKZ','USMV','SETM','GRID','XLE','XSD','CBOE']
    mrets = load_monthly_returns(BASE / 'yfinance.zip', needed)
    cta_fit = fit_factor_model(mrets['CTA'], ff)
    shld_fit = fit_factor_model(mrets['SHLD'], ff)
    factor_rows = {}
    for t, fit in [('CTA_FACTORFIT', cta_fit), ('SHLD_FACTORFIT', shld_fit), ('GD_FACTORFIT', fit_factor_model(mrets['GD'], ff)), ('LMT_FACTORFIT', fit_factor_model(mrets['LMT'], ff)), ('RTX_FACTORFIT', fit_factor_model(mrets['RTX'], ff)), ('CBOE_FACTORFIT', fit_factor_model(mrets['CBOE'], ff))]:
        factor_rows[t] = build_factor_fit_row(fit, premia)

    proxy_weights = compute_proxy_weights(mrets['SHLD'], {k:mrets[k] for k in ['GD','CIBR','XAR','LMT','RTX']})
    shld_proxy = (
        proxy_weights['GD'] * factor_rows['GD_FACTORFIT'] +
        proxy_weights['CIBR'] * etf.loc['CIBR'] +
        proxy_weights['XAR'] * etf.loc['XAR'] +
        proxy_weights['LMT'] * factor_rows['LMT_FACTORFIT'] +
        proxy_weights['RTX'] * factor_rows['RTX_FACTORFIT']
    )
    shld_rebased = 0.50 * etf.loc['SHLD'] + 0.30 * factor_rows['SHLD_FACTORFIT'] + 0.20 * shld_proxy

    working_rows = pd.DataFrame([
        pd.Series(cta_fit, name='CTA_FACTORFIT_BETAS'),
        pd.Series(shld_fit, name='SHLD_FACTORFIT_BETAS'),
        pd.Series(proxy_weights, name='SHLD_PROXY_WEIGHTS'),
    ]).reset_index().rename(columns={'index':'Item'})
    working_rows.to_csv(OUTROOT / '02_working' / 'rebase_components.csv', index=False)

    alt_rows = pd.concat([
        factor_rows['CTA_FACTORFIT'].rename('CTA_FACTORFIT'),
        etf.loc['CTA_OVERLAY'].rename('CTA_OVERLAY'),
        etf.loc['SHLD'].rename('SHLD_CURRENT'),
        factor_rows['SHLD_FACTORFIT'].rename('SHLD_FACTORFIT'),
        factor_rows['CBOE_FACTORFIT'].rename('CBOE_FACTORFIT'),
        shld_proxy.rename('SHLD_PROXY_BASKET'),
        shld_rebased.rename('SHLD_REBASED'),
    ], axis=1).T
    alt_rows.index.name = 'Ticker'
    alt_rows.reset_index().to_csv(OUTROOT / '02_working' / 'rebased_asset_rows.csv', index=False)

    # extended scenarios as recipes
    ext = pd.DataFrame([
        {'Scenario':'Caribbean regime fracture / Cuba failure','Recipe':'25% Political Polarization / Governance Shock + 20% Energy Shock / Supply Constraint + 20% Geopolitical Fragmentation – Trade/Tech Bifurcation + 15% Geopolitical Fragmentation – Cyber Systemic Event + 10% Credit Accident (U.S. Regional) + 10% China Hard Landing','Weighting_Type':'custom blend','Status':'proposed'},
        {'Scenario':'Consumer debt crunch','Recipe':'30% Credit Stress – Shadow/Private Credit Accident + 25% Credit Accident (U.S. Regional) + 20% Debt Overhang – Financial Repression + 15% Stagflation / Fiscal Dominance + 10% Political Polarization / Governance Shock','Weighting_Type':'custom blend','Status':'proposed'},
        {'Scenario':'Helium / chokepoint industrial shock','Recipe':'35% Energy Shock / Supply Constraint + 25% Geopolitical Fragmentation – Trade/Tech Bifurcation + 15% AI Breakout – Energy/Infrastructure Bottleneck + 10% Geopolitical Fragmentation – Cyber Systemic Event + 10% Liquidity: JGB + US Fiscal Dominance + 5% Political Polarization / Governance Shock','Weighting_Type':'custom blend','Status':'proposed'},
        {'Scenario':'Treasury market air pocket','Recipe':'45% Liquidity: JGB + US Fiscal Dominance + 30% Stagflation / Fiscal Dominance + 15% Debt Overhang – Financial Repression + 10% Credit Accident (U.S. Regional)','Weighting_Type':'custom blend','Status':'proposed'},
        {'Scenario':'Grid sabotage / long blackout','Recipe':'40% Geopolitical Fragmentation – Cyber Systemic Event + 25% Political Polarization / Governance Shock + 20% Energy Shock / Supply Constraint + 15% Credit Accident (U.S. Regional)','Weighting_Type':'custom blend','Status':'proposed'},
    ])
    ext.to_csv(OUTROOT / '02_working' / 'extended_scenarios_recipes.csv', index=False)

    # portfolio candidates
    portfolio_defs = {
        'Evergreen10': {'CTA':15.0,'SHLD':15.0,'XAR':15.0,'NUKZ':10.0,'SETM':10.0,'USMV':10.0,'XSD':10.0,'CIBR':5.0,'GRID':5.0,'XLE':5.0},
        'Evg10+': {'CTA':17.5,'SHLD':15.0,'XAR':15.0,'NUKZ':10.0,'SETM':10.0,'USMV':12.5,'XSD':7.5,'CIBR':5.0,'GRID':2.5,'XLE':5.0},
        'Evg10++': {'CTA':17.5,'SHLD':17.5,'XAR':15.0,'NUKZ':10.0,'SETM':7.5,'USMV':12.5,'XSD':7.5,'CIBR':5.0,'GRID':2.5,'XLE':5.0},
        'Ensemble20': {'CTA':20.0,'XAR':20.0,'SHLD':20.0,'NUKZ':20.0,'AIRR':16.2,'CIBR':3.8},
        'Conservative20': {'AIRR':20.0,'CIBR':20.0,'NUKZ':20.0,'SHLD':20.0,'XAR':20.0},
        'Tactical20': {'CTA':20.0,'CIBR':20.0,'NUKZ':20.0,'SHLD':20.0,'XAR':20.0},
        'v8c-shld': {'CBOE':15.0,'SHLD':15.0,'XAR':15.0,'NUKZ':10.0,'SETM':10.0,'USMV':10.0,'XSD':10.0,'CIBR':5.0,'GRID':5.0,'XLE':5.0},
    }
    rows=[]
    for p, ws in portfolio_defs.items():
        for t,w in ws.items():
            rows.append({'Portfolio':p,'Ticker':t,'Weight_%':w})
    pd.DataFrame(rows).to_csv(OUTROOT / '02_working' / 'portfolio_candidates.csv', index=False)

    # world results
    scen_idx = scen['Scenario']
    w = scen.set_index('Scenario')['Weight_Normalized']
    sev = scen.set_index('Scenario')['Severity_1to5']
    bench_row = port_scen.loc['Evergreen', scen_idx]
    bench_row.index = scen_idx

    asset_worlds = []
    for cta_variant, cta_row in [('CTA_FACTORFIT', factor_rows['CTA_FACTORFIT']), ('CTA_OVERLAY', etf.loc['CTA_OVERLAY'])]:
        for shld_variant, shld_row in [('SHLD_CURRENT', etf.loc['SHLD']), ('SHLD_REBASED', shld_rebased)]:
            label = f'{cta_variant}__{shld_variant}'
            etf_world = etf[scen_idx].copy()
            etf_world.loc['CTA'] = cta_row.values
            etf_world.loc['SHLD'] = shld_row.values
            etf_world.loc['CBOE'] = factor_rows['CBOE_FACTORFIT'].values
            asset_worlds.append((label, etf_world))

    model_rows=[]
    summary_rows=[]
    for pname, pweights in portfolio_defs.items():
        world_metrics=[]
        for aw_label, etf_world in asset_worlds:
            port_row = compute_port_row(pweights, etf_world)
            for stress_kind in ['additive','multiplicative','power']:
                mets = robust_metrics(port_row, bench_row, w, sev, stress_kind)
                mets.update({'Portfolio':pname,'AssetWorld':aw_label,'StressTilt':stress_kind})
                if aw_label == 'CTA_OVERLAY__SHLD_CURRENT' and stress_kind == 'additive':
                    mets.update(realized_metrics(pweights, mrets))
                else:
                    mets.update({'Realized_CAGR':np.nan,'Realized_Vol':np.nan,'Sortino':np.nan,'MaxDD':np.nan,'Calmar':np.nan})
                model_rows.append(mets)
                world_metrics.append(mets)
        dfw = pd.DataFrame(world_metrics)
        summary_rows.append({
            'Portfolio': pname,
            'Robust_CEW_min': dfw['Robust_CEW_30y'].min(),
            'Robust_CEW_p25': dfw['Robust_CEW_30y'].quantile(0.25),
            'Robust_CEW_median': dfw['Robust_CEW_30y'].median(),
            'SWCAGR_min': dfw['SWCAGR'].min(),
            'SWCAGR_median': dfw['SWCAGR'].median(),
            'R3AF_min': dfw['R3AF'].min(),
            'R3AF_median': dfw['R3AF'].median(),
            'Shortfall_max': dfw['Shortfall_lt5'].max(),
            'Worst_Scenario_min': dfw['Worst_Scenario_CAGR'].min(),
            'Best_Scenario_max': dfw['Best_Scenario_CAGR'].max(),
            'Realized_CAGR_prefworld': dfw['Realized_CAGR'].dropna().iloc[0] if dfw['Realized_CAGR'].dropna().size else np.nan,
            'Sortino_prefworld': dfw['Sortino'].dropna().iloc[0] if dfw['Sortino'].dropna().size else np.nan,
            'MaxDD_prefworld': dfw['MaxDD'].dropna().iloc[0] if dfw['MaxDD'].dropna().size else np.nan,
        })
    worlds_df = pd.DataFrame(model_rows)
    worlds_df.to_csv(OUTROOT / '03_derived' / 'portfolio_model_worlds.csv', index=False)
    summary_df = pd.DataFrame(summary_rows).sort_values(['Robust_CEW_p25','R3AF_min'], ascending=[False,False])
    summary_df.to_csv(OUTROOT / '03_derived' / 'portfolio_summary_ensemble.csv', index=False)

    # preferred world scenario rows for key portfolios
    pref_etf = next(v for k,v in asset_worlds if k == 'CTA_OVERLAY__SHLD_CURRENT')
    key_rows=[]
    for pname in ['Evergreen10','Evg10+','Evg10++','Ensemble20','v8c-shld']:
        pr = compute_port_row(portfolio_defs[pname], pref_etf)
        for s, val in pr.items():
            key_rows.append({'Portfolio':pname,'Scenario':s,'CAGR_%':val})
    pd.DataFrame(key_rows).to_csv(OUTROOT / '03_derived' / 'portfolio_scenario_rows_preferred_world.csv', index=False)

    # governance / methodology docs
    readme = f"""# KEEP Preservation Pack 2026Q1

This pack is the new canonical local-sync package.

## Design
- `01_sources/` = direct extracts from uploaded canonical workbooks and registry.
- `02_working/` = rebased assumptions, alternative asset rows, extended scenarios, candidate portfolios.
- `03_derived/` = rebuilt results across uncertainty worlds.

## Fresh approach
The pack separates **immutable source truth** from **mutable working assumptions** and **rebuildable derived outputs**.
That prevents one workbook from silently mixing exact values, provisional rows, and optimizer experiments.

## Recommended local-sync practice
1. Keep this folder locally as the working baseline.
2. When you change a source workbook, rebuild this pack.
3. Compare `manifest.json` hashes before uploading to the ChatGPT project file-store.
4. Upload the whole zip or the whole folder, never piecemeal sheets.

## Status calls
- Production benchmark: `Evg10++`
- Research challenger: `Ensemble20`
- Stress-test artifact only, not production: any unconstrained optimizer result dominated by SHLD concentration

## Uncertainty ensemble used here
- CTA variants: factor-fit, overlay
- SHLD variants: current row, rebased row
- Stress tilts: additive, multiplicative, power

That yields 12 model worlds for robust ranking.
"""
    (OUTROOT / '00_admin' / 'README.md').write_text(readme)

    methodology = """# Methodology snapshot

## Core metrics
- Canonical SWCAGR = weighted average scenario CAGR under canonical weights.
- CEW 30y = exp(30 * sum(weight * log(1 + scenario_cagr/100))).
- Robust CEW = 50% canonical + 25% equal + 25% stress-tilted blend.
- R3AF = 0.55 * TailAlpha + 0.25 * Convexity + 0.20 * TailOrtho.
- Shortfall <5 = weighted average gap below 5% CAGR under the robust blend.

## Stress tilt families
- additive: sqrt(w) * (severity + lambda * badness)
- multiplicative: sqrt(w) * severity * (1 + lambda * badness)
- power: sqrt(w) * severity^0.8 * (0.25 + badness)^lambda

All three are calibrated to the v8 exact stress anchor rather than treated as revealed truth.

## Rebased assets
- CTA is preserved as two rows: factor-fit and overlay.
- SHLD is preserved as current row, factor-fit row, proxy-basket row, and rebased row.
- SHLD rebased row = 50% current + 30% factor-fit + 20% proxy-basket.

## Extended scenarios
These are stored as recipe rows, not silently merged into canonical weights.
They are intended for manual review before promotion.

## Interpretation rule
Use `portfolio_summary_ensemble.csv` for governance.
Use `portfolio_model_worlds.csv` only for forensic inspection.
"""
    (OUTROOT / '00_admin' / 'methodology_snapshot.md').write_text(methodology)

    sync = """# Manual sync checklist

1. Keep the full folder locally.
2. Rebuild with `python build_preservation_pack.py` after any source change.
3. Check `manifest.json` and `portfolio_summary_ensemble.csv` for drift.
4. Replicate the updated zip into the ChatGPT project file-store.
5. Do not upload ad hoc intermediate workbooks without also updating this pack.
"""
    (OUTROOT / '00_admin' / 'manual_sync_checklist.md').write_text(sync)

    recommendations = {
        'production_benchmark': 'Evg10++',
        'research_challenger': 'Ensemble20',
        'cta_working_row': 'CTA_OVERLAY',
        'shld_working_row': 'SHLD_REBASED for research / SHLD_CURRENT for continuity',
        'notes': [
            'This pack intentionally preserves exact sources separately from provisional rows.',
            'Use ensemble quartile metrics before promoting a challenger.',
            'Do not treat unconstrained concentration-heavy outputs as production without separate stress review.'
        ]
    }
    json.dump(recommendations, open(OUTROOT / '00_admin' / 'recommendations.json', 'w'), indent=2)

    # manifest
    files = []
    for p in sorted(OUTROOT.rglob('*')):
        if p.is_file():
            files.append({
                'path': str(p.relative_to(OUTROOT)),
                'bytes': p.stat().st_size,
                'sha256': sha256_file(p),
            })
    manifest = {'package':'KEEP_Preservation_Pack_2026Q1','files':files}
    json.dump(manifest, open(OUTROOT / '00_admin' / 'manifest.json', 'w'), indent=2)

    # zip package
    zip_path = BASE / 'KEEP_Preservation_Pack_2026Q1.zip'
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUTROOT.rglob('*')):
            if p.is_file():
                zf.write(p, arcname=str(Path('KEEP_Preservation_Pack_2026Q1') / p.relative_to(OUTROOT)))

    print('Created', OUTROOT)
    print('Created', zip_path)
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    main()
