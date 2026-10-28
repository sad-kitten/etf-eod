"""Addendum 40 — severe-state dependence of C5_SMH3b / K7.
Part A: gold history vs deck R8k gold rows.
Part B: deck-arithmetic sensitivity of the books' scenario CAGRs to gold / trend haircuts.
Part C: regime-conditional covariance from etf-eod monthly returns.
Inputs: etf-eod monthly_returns.csv; deck R8k + scenarios_25 (saved locally); gold year-end 1968-2004 (London PM, onlygold.com);
        CPI-U December values 1967-2025 (FRED CPIAUCSL).
"""
import numpy as np, pandas as pd, io, json, sys
S = "/tmp/claude-0/-home-claude-etf-eod/c0b7ea70-d990-5608-af82-d3af2d0fd5d4/scratchpad"
m = pd.read_csv("/home/claude/etf-eod/monthly_returns.csv", index_col=0)
m = m.loc[:"2026-09"]                      # drop partial 2026-10
m.index = pd.PeriodIndex(m.index, freq="M")

# ---------- Part A: gold ----------
gold_ye = {1968:43.50,1969:41.00,1970:38.90,1971:44.60,1972:63.84,1973:106.48,1974:183.77,1975:139.29,1976:133.77,1977:161.10,
1978:208.10,1979:459.00,1980:594.90,1981:400.00,1982:447.00,1983:380.00,1984:308.00,1985:327.00,1986:390.90,1987:486.50,1988:410.15,
1989:401.00,1990:386.20,1991:353.15,1992:333.00,1993:391.75,1994:383.25,1995:387.00,1996:369.00,1997:287.05,1998:288.70,1999:290.25,
2000:272.65,2001:276.50,2002:342.75,2003:417.25,2004:435.60}
cpi_dec = {1967:34.0,1968:35.6,1969:37.7,1970:39.8,1971:41.1,1972:42.5,1973:46.3,1974:51.9,1975:55.6,1976:58.4,1977:62.3,1978:67.9,
1979:76.9,1980:86.4,1981:94.1,1982:97.7,1983:101.4,1984:105.5,1985:109.5,1986:110.8,1987:115.6,1988:120.7,1989:126.3,1990:134.2,
1991:138.2,1992:142.3,1993:146.3,1994:150.1,1995:153.9,1996:159.1,1997:161.8,1998:164.4,1999:168.8,2000:174.6,2001:177.4,2002:181.8,
2003:185.5,2004:191.7,2005:198.1,2006:203.1,2007:211.445,2008:211.398,2009:217.347,2010:220.472,2011:227.223,2012:231.221,2013:234.719,
2014:236.252,2015:237.761,2016:242.637,2017:247.805,2018:252.767,2019:258.63,2020:262.045,2021:280.845,2022:298.832,2023:308.741,
2024:317.604,2025:326.031}
cpi_dec[2026] = 334.131   # Aug 2026 (latest), used for the Sep-2026 endpoint
# chain spot gold from GLD (ER 0.40%/yr added back) from Dec-2004 year-end
gld = m["GLD"].dropna().loc["2005-01":]
spot = 435.60 * np.cumprod(1 + gld.values + 0.0040/12)
spot = pd.Series(spot, index=gld.index)
for y in range(2005, 2026):
    gold_ye[y] = float(spot.loc[f"{y}-12"])
gold_ye[2026] = float(spot.loc["2026-09"])   # partial year: Sep-2026
chk = {y: gold_ye[y] for y in (2010, 2015, 2020, 2024, 2025, 2026)}
print("gold chained check (onlygold 2010 1420.25, 2015 1060.00):", {k: round(v, 1) for k, v in chk.items()})

def cagr(a, b, n): return (b / a) ** (1 / n) - 1
rows = []
for L in (10, 20, 30, 40):
    for y0 in range(1968, 2027 - L):
        y1 = y0 + L
        n = L if y1 < 2026 else L - 0.25
        nom = cagr(gold_ye[y0], gold_ye[y1], n)
        real = cagr(gold_ye[y0] / cpi_dec[y0], gold_ye[y1] / cpi_dec[y1], n)
        rows.append(dict(window=L, start=y0, end=y1, nominal_cagr_pct=round(100 * nom, 2), real_cagr_pct=round(100 * real, 2)))
G = pd.DataFrame(rows)
G.to_csv(f"{S}/Gold_Rolling_Windows_20261005.csv", index=False)
summ = G.groupby("window").agg(n=("start", "count"), nom_min=("nominal_cagr_pct", "min"), nom_med=("nominal_cagr_pct", "median"),
                               nom_max=("nominal_cagr_pct", "max"), real_min=("real_cagr_pct", "min"), real_med=("real_cagr_pct", "median"),
                               real_max=("real_cagr_pct", "max"),
                               share_nom_below_7p5=("nominal_cagr_pct", lambda s: round(100 * (s < 7.5).mean())),
                               share_real_neg=("real_cagr_pct", lambda s: round(100 * (s < 0).mean())))
print(summ)
for y0 in (1971, 1975, 1980, 1985, 2000):
    n = 2026 - y0 - 0.25
    print(f"gold {y0}->Sep-2026 nominal {100*cagr(gold_ye[y0], gold_ye[2026], n):.2f}%  real {100*cagr(gold_ye[y0]/cpi_dec[y0], gold_ye[2026]/cpi_dec[2026], n):.2f}%")
print("40y windows:", G[G.window == 40][["start", "end", "nominal_cagr_pct", "real_cagr_pct"]].to_string(index=False))

# ---------- Part B: deck arithmetic ----------
deck = pd.read_csv(f"{S}/deck_r8k.csv", index_col=0)
scen = pd.read_csv(f"{S}/scenarios_25.csv")
w = scen.set_index("Scenario")["Weight_Normalized"]
sev = scen.set_index("Scenario")["Severity_1to5"]
deck = deck[w.index]  # column order
ER = dict(GLDM=0.10, SHLD=0.50, CIBR=0.59, DBMF=0.85, NUKZ=0.85, URNM=0.75, XLE=0.08, XAR=0.35, SMH=0.35, BRKB=0, CME=0, FNV=0, MSCI=0,
          IVV=0.03, VT=0.07, GLD=0.40)
books = {
 "C5_SMH3b": dict(GLDM=15, SHLD=15, CIBR=15, DBMF=15, NUKZ=8, BRKB=5, CME=5, FNV=5, MSCI=5, URNM=5, XLE=3, XAR=2, SMH=2),
 "K7":       dict(GLDM=20, SHLD=20, CIBR=20, DBMF=20, URNM=10, MSCI=5, FNV=5),
 "IVV_100":  dict(IVV=100),
}
groups = {"gold": ["GLDM", "FNV"], "trend": ["DBMF"], "nuclear": ["NUKZ", "URNM"], "defense": ["SHLD", "XAR"], "cyber": ["CIBR"],
          "energy": ["XLE"], "semis": ["SMH"], "singles": ["BRKB", "CME", "MSCI"], "broad": ["IVV"]}

def book_scen(bk, d):
    wt = pd.Series(bk, dtype=float) / 100
    return (d.loc[wt.index].mul(wt, axis=0)).sum(axis=0)   # gross arithmetic weighted CAGR per scenario

def book_er(bk): return sum(v * ER[k] for k, v in bk.items()) / 100

out = []
for name, bk in books.items():
    bs = book_scen(bk, deck)
    er = book_er(bk)
    mean = float((bs * w).sum()); mn = float(bs.min()); mn_s = bs.idxmin()
    sev5 = bs[sev[bs.index] == 5]; sev45 = bs[sev[bs.index] >= 4]
    out.append(dict(book=name, deck="R8k", er_pct=round(er, 3), mean_gross=round(mean, 2), mean_net=round(mean - er, 2),
                    worst_gross=round(mn, 2), worst_net=round(mn - er, 2), worst_scenario=mn_s,
                    sev5_mean_net=round(float((sev5 * w[sev5.index]).sum() / w[sev5.index].sum()) - er, 2),
                    sev45_mean_net=round(float((sev45 * w[sev45.index]).sum() / w[sev45.index].sum()) - er, 2)))
B = pd.DataFrame(out); print(B.to_string(index=False))

# per-scenario decomposition for C5_SMH3b (contribution by group, pp of gross CAGR)
bk = books["C5_SMH3b"]; wt = pd.Series(bk, dtype=float) / 100
dec = pd.DataFrame({g: deck.loc[[t for t in ts if t in bk]].mul(wt.reindex([t for t in ts if t in bk]), axis=0).sum(axis=0)
                    for g, ts in groups.items() if any(t in bk for t in ts)})
dec["book_gross"] = dec.sum(axis=1); dec["IVV"] = deck.loc["IVV"]; dec["severity"] = sev; dec["weight"] = w.round(4)
dec = dec.round(2)
dec.to_csv(f"{S}/SevereState_Decomposition_C5_SMH3b_20261005.csv")
print(dec.sort_values("book_gross").head(8).to_string())

# sensitivities
def apply(d, gold_pp=0, trend_pp=0, trend_scope="sev45", gold_override=None):
    d = d.copy()
    for t in ("GLDM", "GLD", "FNV"):
        d.loc[t] = d.loc[t] - gold_pp
    if trend_pp:
        cols = [c for c in d.columns if sev[c] >= 4] if trend_scope == "sev45" else list(d.columns)
        d.loc["DBMF", cols] = d.loc["DBMF", cols] - trend_pp
    if gold_override:
        for c, v in gold_override.items():
            for t in ("GLDM", "GLD"):
                d.loc[t, c] = min(d.loc[t, c], v)
            d.loc["FNV", c] = min(d.loc["FNV", c], v + 1.0)   # royalty: gold path +1 pp
    return d

S_disinfl = {"AI Abundance – Disinflationary Boom": 2.5, "AI Breakout – Productivity Boom / Concentration": 2.5,
             "Base Case – Managed Re-industrialization": 5.0, "Reflation / Industrial Up-Cycle": 5.0, "China Hard Landing": 5.0}
variants = {
 "deck R8k": apply(deck),
 "gold −1 pp all scenarios": apply(deck, gold_pp=1),
 "gold −2 pp all scenarios": apply(deck, gold_pp=2),
 "gold −3 pp all scenarios": apply(deck, gold_pp=3),
 "1980–2000 analog (gold 2.5% in productivity booms, 5% in base/reflation/China)": apply(deck, gold_override=S_disinfl),
 "trend −3 pp in sev-4/5": apply(deck, trend_pp=3),
 "gold −2 pp + trend −3 pp sev-4/5": apply(deck, gold_pp=2, trend_pp=3),
 "gold −2 pp + trend −3 pp + analog": apply(deck, gold_pp=2, trend_pp=3, gold_override=S_disinfl),
}
sens = []
for vname, d in variants.items():
    for name, bk in books.items():
        bs = book_scen(bk, d); er = book_er(bk)
        mean = float((bs * w).sum()) - er; mn = float(bs.min()) - er
        sev5 = bs[sev[bs.index] == 5]
        sens.append(dict(variant=vname, book=name, mean_net=round(mean, 2), worst_net=round(mn, 2), worst_scenario=bs.idxmin(),
                         sev5_mean_net=round(float((sev5 * w[sev5.index]).sum() / w[sev5.index].sum()) - er, 2),
                         gate_worst_5p25="pass" if mn >= 5.25 else "FAIL"))
SE = pd.DataFrame(sens)
base = SE[SE.variant == "deck R8k"].set_index("book")
SE["d_mean_pp"] = SE.apply(lambda r: round(r.mean_net - base.loc[r.book, "mean_net"], 2), axis=1)
SE["d_worst_pp"] = SE.apply(lambda r: round(r.worst_net - base.loc[r.book, "worst_net"], 2), axis=1)
SE.to_csv(f"{S}/SevereState_Sensitivity_20261005.csv", index=False)
print(SE.to_string(index=False))

# terminal-multiple elasticity: fixed real draw 3% of W0, 40y, deterministic
def term(g, d=0.03, T=40):
    return (1 + g) ** T - d * ((1 + g) ** T - 1) / g
g0 = 0.0592   # real gross g that reproduces C5_SMH3b geo 5.445 under this convention
for dg in (0, -0.005, -0.01, -0.0025):
    print(f"g {100*(g0+dg):.2f}% -> terminal {term(g0+dg):.2f}x  ({100*(term(g0+dg)/term(g0)-1):+.1f}%)")

# ---------- Part C: regime-conditional covariance ----------
ivv = m["IVV"].dropna()
idx = (1 + ivv).cumprod(); peak = idx.cummax(); dd = idx / peak - 1
# state known at month start = drawdown at end of previous month
state = pd.Series("normal", index=ivv.index)
state[dd.shift(1) <= -0.10] = "drawdown"
stress = ivv <= -0.05
print("IVV months", len(ivv), "drawdown-regime months", int((state == "drawdown").sum()), "stress months (<=-5%)", int(stress.sum()))
# Markov persistence
s = (state == "drawdown").astype(int)
p_stay = ((s.shift(1) == 1) & (s == 1)).sum() / (s.shift(1) == 1).sum()
p_enter = ((s.shift(1) == 0) & (s == 1)).sum() / (s.shift(1) == 0).sum()
print(f"P(stay drawdown)={p_stay:.3f}  P(enter)={p_enter:.3f}  mean spell={1/(1-p_stay):.1f} months  stationary share={p_enter/(p_enter+1-p_stay):.3f}")

proxies = {"GLDM": "GLD", "SHLD": "ITA", "NUKZ": "NLR", "CIBR": "HACK", "URNM": "URA", "BRKB": "BRK-B"}
def series(t):
    if t == "BRKB": return m["BRK-B"]
    s = m[t].copy() if t in m else pd.Series(np.nan, index=m.index)
    if t in proxies and proxies[t] != "BRK-B":
        p = m[proxies[t]]
        s = s.where(s.notna(), p)   # fill pre-launch months with the proxy
    return s
tick = ["GLDM", "FNV", "DBMF", "URNM", "NUKZ", "SHLD", "XAR", "CIBR", "XLE", "SMH", "BRKB", "CME", "MSCI", "IVV"]
R = pd.DataFrame({t: series(t) for t in tick})
R["GLDM_pure"] = m["GLDM"]; R["GLD"] = m["GLD"]; R["SHLD_pure"] = m["SHLD"]; R["NUKZ_pure"] = m["NUKZ"]; R["CIBR_pure"] = m["CIBR"]; R["URNM_pure"] = m["URNM"]
R["state"] = state.reindex(R.index); R["stress"] = stress.reindex(R.index)

def cond_stats(col, mask, label):
    if col == "IVV":
        x = R.loc[mask, ["IVV"]].dropna()
        if len(x) < 8: return None
        return dict(row=col, regime=label, n=len(x), mean_m_pct=round(100 * x[col].mean(), 2), vol_ann_pct=round(100 * x[col].std() * 12 ** .5, 1), corr_ivv=1.0, beta_ivv=1.0)
    x = R.loc[mask, [col, "IVV"]].dropna()
    if len(x) < 8: return None
    b = np.polyfit(x["IVV"].values, x[col].values, 1)[0]
    return dict(row=col, regime=label, n=len(x), mean_m_pct=round(100 * x[col].mean(), 2), vol_ann_pct=round(100 * x[col].std() * 12 ** .5, 1),
                corr_ivv=round(x[col].corr(x["IVV"]), 2), beta_ivv=round(b, 2))
cs = []
last48 = R.index >= R.index[-48]
for col in tick + ["GLD", "GLDM_pure", "SHLD_pure", "NUKZ_pure", "CIBR_pure", "URNM_pure"]:
    for mask, label in ((R.index == R.index, "all"), (R.state == "normal", "normal"), (R.state == "drawdown", "drawdown"),
                        (R.stress == True, "stress<=-5%"), (last48, "last48")):
        r = cond_stats(col, mask, label)
        if r: cs.append(r)
CS = pd.DataFrame(cs)
first = {c: str(R[c].first_valid_index()) for c in CS.row.unique()}
CS["history_from"] = CS.row.map(first)
CS.to_csv(f"{S}/Regime_Conditional_Rows_20261005.csv", index=False)
print(CS[CS.regime.isin(["normal", "drawdown", "stress<=-5%"])].pivot(index="row", columns="regime", values=["corr_ivv", "mean_m_pct", "vol_ann_pct"]).round(2).to_string())

# book series (proxy-filled) from 2019-06 (DBMF start); book ex-DBMF from 2011-10 (XAR)
def book_series(bk, start):
    wt = pd.Series(bk, dtype=float) / 100
    X = R.loc[start:, wt.index]
    X = X.dropna()
    return (X * wt).sum(axis=1), X.index[0]
bookC5, s0 = book_series(books["C5_SMH3b"], "2019-06")
bookK7, _ = book_series(books["K7"], "2019-06")
bkx = {k: v for k, v in books["C5_SMH3b"].items() if k != "DBMF"}
bookC5x, s1 = book_series(bkx, "2011-10")
R["C5_SMH3b_proxy"] = bookC5; R["K7_proxy"] = bookK7; R["C5_exDBMF_proxy"] = bookC5x / 0.85
ball = R[["GLDM", "FNV", "DBMF"]].mul(pd.Series({"GLDM": 15, "FNV": 5, "DBMF": 15}) / 35, axis=1).sum(axis=1, min_count=3)
eq = R[["SHLD", "CIBR", "NUKZ", "BRKB", "CME", "MSCI", "URNM", "XLE", "XAR", "SMH"]].mul(
    pd.Series({"SHLD": 15, "CIBR": 15, "NUKZ": 8, "BRKB": 5, "CME": 5, "MSCI": 5, "URNM": 5, "XLE": 3, "XAR": 2, "SMH": 2}) / 65, axis=1).sum(axis=1, min_count=10)
R["ballast_sleeve"] = ball; R["equity_sleeve"] = eq
bs_ = []
for col in ["C5_SMH3b_proxy", "K7_proxy", "C5_exDBMF_proxy", "ballast_sleeve", "equity_sleeve"]:
    for mask, label in ((R.index == R.index, "all"), (R.state == "normal", "normal"), (R.state == "drawdown", "drawdown"),
                        (R.stress == True, "stress<=-5%"), (last48, "last48")):
        r = cond_stats(col, mask, label)
        if r: bs_.append(r)
BS = pd.DataFrame(bs_); BS["history_from"] = BS.row.map({c: str(R[c].first_valid_index()) for c in BS.row.unique()})
# ballast vs equity sleeve correlation by regime
for mask, label in ((R.index == R.index, "all"), (R.state == "normal", "normal"), (R.state == "drawdown", "drawdown"), (R.stress == True, "stress<=-5%")):
    x = R.loc[mask, ["ballast_sleeve", "equity_sleeve"]].dropna()
    BS.loc[len(BS)] = dict(row="corr(ballast, equity sleeve)", regime=label, n=len(x), mean_m_pct=np.nan, vol_ann_pct=np.nan,
                           corr_ivv=round(x["ballast_sleeve"].corr(x["equity_sleeve"]), 2), beta_ivv=np.nan, history_from=str(x.index[0]))
BS.to_csv(f"{S}/Regime_Conditional_Books_20261005.csv", index=False)
print(BS.to_string(index=False))

# conditional covariance matrices (annualized, %^2) for the C5 rows, proxy-filled, from 2019-06
rowsC = ["GLDM", "SHLD", "CIBR", "DBMF", "NUKZ", "BRKB", "CME", "FNV", "MSCI", "URNM", "XLE", "XAR", "SMH"]
X = R.loc["2019-06":, rowsC + ["state", "stress"]].dropna(subset=rowsC)
for label, mask in (("all", X.index == X.index), ("normal", X.state == "normal"), ("drawdown", X.state == "drawdown")):
    cov = X.loc[mask, rowsC].cov() * 12 * 1e4
    cov.round(1).to_csv(f"{S}/Cov_{label}_C5rows_201906_20261005.csv")
    corr = X.loc[mask, rowsC].corr()
    print(label, "n=", int(mask.sum()), "mean pairwise corr", round(corr.values[np.triu_indices(len(rowsC), 1)].mean(), 2),
          "GLDM-IVV-ish via SHLD corr", round(corr.loc["GLDM", "SHLD"], 2))
# worst IVV months table
worst = R.loc[R["IVV"].nsmallest(12).index, ["IVV", "GLDM", "FNV", "DBMF", "URNM", "SHLD", "CIBR", "C5_SMH3b_proxy", "C5_exDBMF_proxy"]]
print((worst * 100).round(1).to_string())
(worst * 100).round(1).to_csv(f"{S}/Worst_IVV_Months_20261005.csv")
# episodes
for a, b, lab in (("2008-09", "2009-02", "GFC Sep08-Feb09"), ("2020-02", "2020-03", "Covid Feb-Mar20"), ("2022-01", "2022-09", "2022 bear"), ("2025-02", "2025-04", "2025 tariff shock")):
    seg = R.loc[a:b, ["IVV", "GLDM", "FNV", "DBMF", "URNM", "SHLD", "CIBR", "C5_SMH3b_proxy", "C5_exDBMF_proxy"]]
    cum = ((1 + seg).prod() - 1) * 100
    print(lab, cum.round(1).to_dict())
