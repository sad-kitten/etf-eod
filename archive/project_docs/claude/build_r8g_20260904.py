"""R8g — RPRX on real history replaces the Addendum 20 proxy row.

The measured record (74 months): beta 0.46 to IVV, alpha +0.61%/yr with t = 0.07 (indistinguishable from
zero), correlation 0.31, down-capture 0.28. Correlation and down-capture that low mean RPRX is NOT an
equity-beta sleeve, so — by the same rule applied to DBMF/CTA/KMLM in Addendum 19 — the naive
geometric rule (7.50 + realised excess = -3.82, measured against a 17.5%/yr market) does not apply.

Row = beta-consistent base (alpha set to zero, since it is not significant) + a halved judgment premium
for the contractual structure the beta regression cannot see (95% adjusted-EBITDA margin, 35+ approved
products, receipts guided +7-10%), + the Addendum 19 overlays scaled to the smaller row.
    base    = 0.46 x 7.50 = 3.46
    premium = +1.5  (the Addendum 19 premium, applied once not twice, on a much lower base)
    shape   = 0.80 x deviation of the 0.65*XLV+0.35*IBB anchor about its mean
"""
import numpy as np, pandas as pd
import lib_pathce as L

BETA, IVV_ROW, PREM, SHAPE = 0.46, 7.50, 1.5, 0.80
xlv = np.array([float(L.DECK.loc['XLV', s]) for s in L.SC])
ibb = np.array([float(L.DECK.loc['IBB', s]) for s in L.SC])
anchor = 0.65 * xlv + 0.35 * ibb
OV = {"Credit Stress – Shadow/Private Credit Accident": -1.0,
      "Credit Accident (U.S. Regional)": -1.0,
      "Deflationary Bust – Balance-Sheet Recession": 0.5,
      "Pandemic / Bio Disruption": 1.5,
      "Taiwan Hostilities / Kinetic Shock": 0.5,
      "Broad Kinetic Conflict – Multi-Theater": 0.5,
      "Geopolitical Fragmentation – Cyber Systemic Event": 0.3,
      "AI Breakout – Productivity Boom / Concentration": 0.5,
      "Cronyism / Policy Distortion": -0.8,
      "Political Polarization / Governance Shock": -0.8}
row = BETA * IVV_ROW + PREM + SHAPE * (anchor - anchor.mean()) + np.array([OV.get(s, 0.0) for s in L.SC])

d = pd.read_csv("Deck_Recal_R8f_20260904.csv", index_col=0)
if "RPRX_proxy" in d.index:
    d = d.drop(index="RPRX_proxy")
d.loc["RPRX", list(d.columns)] = np.round(row, 4)
d.to_csv("Deck_Recal_R8g_20260904.csv")

g = pd.read_csv("out/Row_Grades_R8f_20260904.csv")
g = g[g.Ticker != "RPRX_proxy"]
g = pd.concat([g, pd.DataFrame([{"Ticker": "RPRX", "Grade": "B", "Meaning":
    "Royalty Pharma, real history from etf-eod (74 mo from 2020-07); row = beta-consistent base "
    "(0.46 x IVV_row, alpha t=0.07 so set to zero) + 1.5 premium for contractual royalty structure "
    "+ 0.80 x anchor shape + policy/leverage overlays"}])], ignore_index=True)
g.to_csv("out/Row_Grades_R8g_20260904.csv", index=False)

print(f"RPRX row (real-data basis): mean {row.mean():.2f}  min {row.min():.2f}  max {row.max():.2f}  std {row.std():.2f}")
print(f"  superseded: 8.04 (Add.19 derived), 7.29 (Add.20 proxy)")
for c in ['IVV', 'XLV', 'IBB', 'XBI', 'CME', 'DBMF', 'GLDM']:
    r = np.array([float(L.DECK.loc[c, s]) for s in L.SC])
    print(f"  vs {c:5s} mean {r.mean():5.2f} min {r.min():6.2f}")
