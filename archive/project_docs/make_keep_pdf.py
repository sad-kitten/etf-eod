# (FULL SCRIPT CONTENT)
# NOTE: This script expects these files in the same folder:
# - yfinance.zip
# - KEEP_Scenario_CAGR_Matrix_2026Q1.xlsx
# - KEEP_Scenarios_Canonical_2026Q1.xlsx
# - metrics_r3af.py
# - F-F_Research_Data_5_Factors_2x3.csv
# - F-F_Momentum_Factor.csv

import os, zipfile, pandas as pd, numpy as np
from pandas.tseries.offsets import MonthEnd
import runpy

from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

BASE = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# BASE PORTFOLIOS (COMPLIANT)
# -----------------------------
# Floors include: SETM>=3 and NUKZ>=3 (satisfied below)
ports = {
    "Evg++": {
        "USMV":12,"XAR":13,"CIBR":11,"XSD":10,"GRID":9,"XLE":11,"SPLV":5,"QUAL":5,"AVDE":4,
        "AVUV":4,"NUKZ":4,"SETM":4,"SHLD":5,"MOAT":1,"ROKT":2
    },
    "Evergreen15": {
        "XSD":15,"AVDE":10,"GRID":10,"XAR":10,"AVUV":5,"CIBR":5,"COWZ":5,"MOAT":5,"NUKZ":5,
        "QUAL":5,"ROKT":5,"SETM":5,"SHLD":5,"USMV":5,"XLE":5
    },
    # FIXED to sum 100 after SETM>=3: VALQ 2 -> 1
    "Cont_RobustMaxSWC20": {
        "XAR":11,"XSD":10,"SHLD":10,"XLE":10,"AVDE":6,"SPLV":7,"QUAL":6,"CIBR":5,"GRID":5,
        "MOAT":5,"NUKZ":5,"USMV":5,"ROKT":3,"AVUV":2,"COWZ":2,"PAVE":2,"VALQ":1,"VFQY":2,"SETM":3
    },
    # FIXED to sum 100 after SETM>=3: MOAT 4 -> 3
    "Cont_WiseMaxSWC20": {
        "XAR":13,"SHLD":12,"XSD":11,"XLE":8,"USMV":5,"CIBR":7,"COWZ":6,"NUKZ":6,"GRID":5,"SPLV":5,
        "QUAL":5,"MOAT":3,"AVDE":4,"VFQY":3,"DFAS":2,"ROKT":2,"SETM":3
    },
    "Cont_WiseRunnerUp20": {
        "USMV":20,"XSD":14,"CIBR":13,"XLE":10,"XAR":13,"SHLD":11,"GRID":6,"NUKZ":5,"PAVE":2,"SETM":3,"QQEW":1,"ROKT":2
    },
    # FIXED to sum 100 after SETM>=3: MOAT 7 -> 6
    "Cont_WiseMaxSWC20_XLE10": {
        "XAR":15,"SHLD":12,"XSD":11,"XLE":10,"USMV":4,"CIBR":7,"MOAT":6,"COWZ":6,"NUKZ":6,"GRID":5,"SPLV":5,
        "AVDE":3,"VFQY":3,"DFAS":2,"ROKT":2,"SETM":3
    }
}

def _sum_check(dct):
    bad = {k: sum(v.values()) for k,v in dct.items() if sum(v.values()) != 100}
    if bad:
        raise ValueError(f"Portfolios not summing to 100: {bad}")

_sum_check(ports)

# -----------------------------------------
# A/B VARIANTS (Original vs A vs B choices)
# -----------------------------------------
variants = {
    "Evergreen15": {
        "Original": ports["Evergreen15"],
        # Choice A: XLE +3, GRID +2 funded by MOAT -2, QUAL -2, COWZ -1
        "A": {"XLE":8, "GRID":12, "MOAT":3, "QUAL":3, "COWZ":4},
        # Choice B: XLE +5 funded by MOAT -2, QUAL -1, COWZ -1, USMV -1
        "B": {"XLE":10, "MOAT":3, "QUAL":4, "COWZ":4, "USMV":4},
    },
    "Cont_RobustMaxSWC20": {
        "Original": ports["Cont_RobustMaxSWC20"],
        # Choice A: GRID +2 funded by SPLV -2
        "A": {"GRID":7, "SPLV":5},
        # Choice B: GRID +2 funded by USMV -2
        "B": {"GRID":7, "USMV":3},
    },
    "Cont_WiseMaxSWC20": {
        "Original": ports["Cont_WiseMaxSWC20"],
        # Choice A: XLE +2 funded by USMV -2
        "A": {"XLE":10, "USMV":3},
        # Choice B: XLE +2 funded by MOAT -1 and QUAL -1
        "B": {"XLE":10, "MOAT":2, "QUAL":4},
    },
    "Cont_WiseRunnerUp20": {
        "Original": ports["Cont_WiseRunnerUp20"],
        # Choice A: USMV -5; GRID +1; XLE +2; NUKZ +2
        "A": {"USMV":15, "GRID":7, "XLE":12, "NUKZ":7},
        # Choice B: USMV -5; GRID +2; XLE +2 (no NUKZ change)
        "B": {"USMV":16, "GRID":8, "XLE":12},
    },
}

def apply_changes(base, changes):
    out = dict(base)
    out.update(changes)
    s = sum(out.values())
    if s != 100:
        raise ValueError(f"Weights must sum to 100, got {s} (off by {s-100}) for changes={changes}")
    return out

test_ports = {}
for pname, opts in variants.items():
    base = opts["Original"]
    test_ports[f"{pname}_Original"] = base
    test_ports[f"{pname}_A"] = apply_changes(base, opts["A"])
    test_ports[f"{pname}_B"] = apply_changes(base, opts["B"])

_sum_check(test_ports)

# -----------------------------------------
# PRICE HISTORY + HIST METRICS
# -----------------------------------------
zf = zipfile.ZipFile(os.path.join(BASE, "yfinance.zip"))

def load_close(ticker):
    with zf.open(f"{ticker}.csv") as f:
        d = pd.read_csv(f)
    d = d[d["Date"].notna()].copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d = d.dropna(subset=["Close"]).set_index("Date").sort_index()
    return d["Close"]

def monthly_returns(ticker):
    s = load_close(ticker)
    m = s.resample("ME").last()  # month-end (newer pandas)
    return m.pct_change().dropna()

start_me = pd.Timestamp("2024-02-29")
end_me   = pd.Timestamp("2026-02-28")

etf_list = sorted({t for p in test_ports.values() for t in p})
rets = {}
for t in etf_list + ["VTI"]:
    if f"{t}.csv" in zf.namelist():
        r = monthly_returns(t)
        r = r[(r.index>=start_me) & (r.index<=end_me)]
        rets[t] = r

common_idx = None
for r in rets.values():
    common_idx = r.index if common_idx is None else common_idx.intersection(r.index)
for k in list(rets.keys()):
    rets[k] = rets[k].reindex(common_idx).dropna()
n_months = len(common_idx)

def port_returns(weights_pct):
    w = pd.Series(weights_pct)/100.0
    missing = [t for t in w.index if t not in rets]
    if missing:
        raise ValueError(f"Missing returns for: {missing}")
    mat = np.column_stack([rets[t].values for t in w.index])
    return pd.Series(mat.dot(w.values), index=common_idx)

def cagr(r):
    total = (1+r).prod()
    years = len(r)/12.0
    return total**(1/years)-1

def maxdd(r):
    wealth = (1+r).cumprod()
    peak = wealth.cummax()
    dd = wealth/peak - 1
    return dd.min()

def calmar(r):
    m = abs(maxdd(r))
    return np.nan if m==0 else cagr(r)/m

def sortino(r, rf=0.0):
    ex = r - rf/12.0
    down = np.minimum(ex,0.0)
    dd = np.sqrt((down**2).mean())*np.sqrt(12)
    ann = np.mean(ex)*12
    return np.nan if dd==0 else ann/dd

hist_metrics = {}
for name,w in test_ports.items():
    r = port_returns(w)
    hist_metrics[name] = {
        "Current net CAGR": cagr(r)*100.0,
        "MaxDD": maxdd(r)*100.0,
        "Calmar": calmar(r),
        "Sortino": sortino(r)
    }

# -----------------------------------------
# SCENARIO METRICS (SWC + R3AF)
# -----------------------------------------
matrix_path = os.path.join(BASE, "KEEP_Scenario_CAGR_Matrix_2026Q1.xlsx")
etf_scen = pd.read_excel(matrix_path, sheet_name="ETF_Scenario_CAGR", index_col=0)  # % annual
scen_df  = pd.read_excel(os.path.join(BASE, "KEEP_Scenarios_Canonical_2026Q1.xlsx"))
w_scen   = pd.Series(scen_df["Weight_Normalized"].values, index=scen_df["Scenario"].values)

sf = pd.read_excel(matrix_path, sheet_name="Scenario_Factor_Premia_%", index_col=0) # % annual

ff5 = pd.read_csv(os.path.join(BASE, "F-F_Research_Data_5_Factors_2x3.csv"), skiprows=3)
mom = pd.read_csv(os.path.join(BASE, "F-F_Momentum_Factor.csv"), skiprows=12)
ff5 = ff5[ff5["Unnamed: 0"].astype(str).str.fullmatch(r"\d{6}")].copy()
mom = mom[mom["Unnamed: 0"].astype(str).str.fullmatch(r"\d{6}")].copy()
ff5["Date"] = pd.to_datetime(ff5["Unnamed: 0"].astype(str)+"01", format="%Y%m%d")
mom["Date"] = pd.to_datetime(mom["Unnamed: 0"].astype(str)+"01", format="%Y%m%d")
f = ff5[["Date","Mkt-RF","SMB","HML","RMW","CMA","RF"]].merge(mom[["Date","Mom"]], on="Date", how="inner")
for c in ["Mkt-RF","SMB","HML","RMW","CMA","RF","Mom"]:
    f[c] = pd.to_numeric(f[c], errors="coerce")/100.0
f = f.dropna()
f["MonthEnd"] = f["Date"] + MonthEnd(0)

vti_r = monthly_returns("VTI").dropna()
reg = pd.DataFrame({"MonthEnd": vti_r.index, "VTI_ret": vti_r.values}).merge(f, on="MonthEnd", how="inner")
Y = (reg["VTI_ret"] - reg["RF"]).values
X = reg[["Mkt-RF","SMB","HML","RMW","CMA","Mom"]].values
X = np.column_stack([np.ones(len(X)), X])
b = np.linalg.lstsq(X, Y, rcond=None)[0]
alpha = b[0]
beta = pd.Series(b[1:], index=["Mkt-RF","SMB","HML","RMW","CMA","Mom"])

vti_scen = sf["RF"] + (sf[beta.index]*beta.values).sum(axis=1) + alpha*100.0
vti_scen = vti_scen.reindex(etf_scen.columns)

r3 = runpy.run_path(os.path.join(BASE, "metrics_r3af.py"))
compute_r3af = r3["compute_r3af_from_df"]

def port_scenario(weights_pct):
    w = pd.Series(weights_pct)/100.0
    missing = [t for t in w.index if t not in etf_scen.index]
    if missing:
        raise ValueError(f"Scenario matrix missing ETFs: {missing}")
    return (etf_scen.loc[w.index].T * w.values).sum(axis=1).reindex(w_scen.index)

scenario_metrics = {}
for name,w in test_ports.items():
    ps = port_scenario(w)
    swc = float((ps*w_scen).sum())
    df = pd.DataFrame({
        "Scenario": w_scen.index,
        "Weight": w_scen.values,
        "Portfolio": ps.values,
        "Benchmark": vti_scen.reindex(w_scen.index).values
    })
    r3af, _ = compute_r3af(df, scenario_col="Scenario", weight_col="Weight",
                           port_col="Portfolio", bench_col="Benchmark")
    scenario_metrics[name] = {"SWC": swc, "R3AF": r3af}

# Console comparison
rows = []
for name in test_ports:
    rows.append({
        "Portfolio": name,
        "SWC": scenario_metrics[name]["SWC"],
        "R3AF": scenario_metrics[name]["R3AF"],
        "Current net CAGR": hist_metrics[name]["Current net CAGR"],
        "MaxDD": hist_metrics[name]["MaxDD"],
        "Calmar": hist_metrics[name]["Calmar"],
        "Sortino": hist_metrics[name]["Sortino"],
    })
df_compare = pd.DataFrame(rows).sort_values(["Portfolio"])
print("\n=== Original vs A vs B (metrics) ===")
print(df_compare.to_string(index=False))

# -----------------------------------------
# PDF (ONE PAGE, LANDSCAPE)
# -----------------------------------------
def sum_row(w, tickers): return sum(w.get(t,0) for t in tickers)

composites = [
    ("SHLD+CIBR+XAR+ROKT", ["SHLD","CIBR","XAR","ROKT"]),
    ("NUKZ+XLE+GRID+SETM", ["NUKZ","XLE","GRID","SETM"]),
]

tickers = sorted(set().union(*[set(w) for w in test_ports.values()]))

rows = []
for t in tickers:
    rows.append([t] + [test_ports[p].get(t,0) for p in test_ports])

for cname, ts in composites:
    rows.append([cname] + [sum_row(test_ports[p], ts) for p in test_ports])

rows += [
    ["MaxDD"] + [hist_metrics[p]["MaxDD"] for p in test_ports],
    ["SWC"] + [scenario_metrics[p]["SWC"] for p in test_ports],
    ["Current net CAGR"] + [hist_metrics[p]["Current net CAGR"] for p in test_ports],
    ["Calmar"] + [hist_metrics[p]["Calmar"] for p in test_ports],
    ["Sortino"] + [hist_metrics[p]["Sortino"] for p in test_ports],
    ["R3AF"] + [scenario_metrics[p]["R3AF"] for p in test_ports],
]

cols = ["Ticker/Metric"] + list(test_ports.keys())
df = pd.DataFrame(rows, columns=cols)

pdf_path = os.path.join(BASE, "KEEP_Portfolios_AB_Compare_2026Q1.pdf")
doc = SimpleDocTemplate(pdf_path, pagesize=landscape(letter),
                        leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=18)
styles = getSampleStyleSheet()
story = []
story.append(Paragraph("KEEP 2026Q1 — Original vs A vs B (weights, composites, metrics)", styles["Title"]))
story.append(Spacer(1,4))
story.append(Paragraph(f"Hist window: {common_idx.min().date()} → {common_idx.max().date()} (monthly, n={n_months}).",
                       styles["Normal"]))
story.append(Spacer(1,6))

data = [cols]
metric_names = {"MaxDD","SWC","Current net CAGR","Calmar","Sortino","R3AF"}
comp_names = {c for c,_ in composites}

for _, r in df.iterrows():
    key = r["Ticker/Metric"]
    line = [key]
    for p in test_ports:
        v = r[p]
        if key in metric_names:
            if key in {"MaxDD","SWC","Current net CAGR"}:
                line.append(f"{float(v):.2f}")
            else:
                line.append(f"{float(v):.3f}")
        else:
            line.append(str(int(v)))
    data.append(line)

page_w = landscape(letter)[0] - 36
col0 = 180
other = (page_w - col0) / (len(cols)-1)
tbl = Table(data, colWidths=[col0] + [other]*(len(cols)-1), repeatRows=1)

style = [
    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
    ("FONTSIZE",(0,0),(-1,-1),6),
    ("TOPPADDING",(0,0),(-1,-1),1),
    ("BOTTOMPADDING",(0,0),(-1,-1),1),
    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e6e6e6")),
    ("GRID",(0,0),(-1,-1),0.25,colors.grey),
    ("ALIGN",(1,1),(-1,-1),"RIGHT"),
    ("ALIGN",(0,0),(0,-1),"LEFT"),
]
for i in range(1,len(data)):
    k = data[i][0]
    if k in comp_names:
        style += [("BACKGROUND",(0,i),(-1,i),colors.HexColor("#f3f7ff")),
                  ("FONTNAME",(0,i),(-1,i),"Helvetica-Bold")]
    if k in metric_names:
        style += [("BACKGROUND",(0,i),(-1,i),colors.HexColor("#fff6e6")),
                  ("FONTNAME",(0,i),(-1,i),"Helvetica-Bold")]
tbl.setStyle(TableStyle(style))
story.append(tbl)
doc.build(story)

print("Wrote PDF:", pdf_path)
