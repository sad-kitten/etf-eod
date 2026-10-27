"""R8i — adds the AI Buildout Disappointment scenario the deck has lacked (Addendum 16 §2-3).

Construction exactly as specified there:
    AIBUST = 0.50*Credit Stress + 0.35*Base + 0.15*Stagflation
             + power complex -3.0pp, semis -2.5pp, speculative tech -1.5pp, energy -1.0pp, ballast +0.5pp
applied to EVERY ticker in the deck (overlay 0 for tickers in no set). Severity 3 (weighted parent severity 3.3).

Weight: Addendum 16 said 4-6%, funded from the three AI-positive scenarios. Within that band the weight is set so
that AI-boom and AI-bust carry EQUAL probability mass: the three AI-positive scenarios total raw 11.0 of 116 (9.48%);
5.5 raw is moved to AIBUST, leaving 5.5 raw (4.74%) positive and 4.74% negative. Total deck mass is unchanged and
the other 21 scenarios are untouched. The robust weight vector (PI_ROBUST) recomputes from the file automatically.
"""
import numpy as np, pandas as pd, os
DECK_IN, SCEN_IN = "Deck_Recal_R8h_20260905.csv", "scenarios_24_R6_20260831.csv"
DECK_OUT, SCEN_OUT = "Deck_Recal_R8i_20260909.csv", "scenarios_25_R8i_20260909.csv"
NAME = "AI Buildout Disappointment – Power/Capex De-rating"

d = pd.read_csv(DECK_IN); d["Ticker"] = d["Ticker"].str.strip(); d = d.set_index("Ticker")
C = lambda k: d[[c for c in d.columns if c.startswith(k)][0]]
POWER = {'NUKZ','URNM','CEG','NLR','GRID','SETM','URA','AQWA'}; SEMI = {'SMH','SOXX','XSD','XLK','FTEC'}
SPEC = {'QTUM','BOTZ','ROBO','KOMP','XT','PRNT','ROKT','JEDI'}; ENER = {'XLE','GUNR','PDBC'}
BALL = {'GLDM','DBMF','CTA','KMLM','PSLV','TLT','IEF'}
ov = pd.Series(0.0, index=d.index)
for S, v in ((POWER, -3.0), (SEMI, -2.5), (SPEC, -1.5), (ENER, -1.0), (BALL, 0.5)):
    ov[[t for t in d.index if t in S]] = v
col = 0.50 * C("Credit Stress") + 0.35 * C("Base Case") + 0.15 * C("Stagflation") + ov
d[NAME] = np.round(col, 4)
d.to_csv(DECK_OUT)

s = pd.read_csv(SCEN_IN)
ai = s["Scenario"].str.startswith("AI ")
move = 5.5; tot = s.loc[ai, "Weight_raw"].sum()
s.loc[ai, "Weight_raw"] = s.loc[ai, "Weight_raw"] * (tot - move) / tot
new = pd.DataFrame([dict(Scenario_ID=25, Scenario=NAME, Weight_raw=move, Severity_1to5=3, Canonical_Parent=NAME,
    Definition="Datacentre power-demand growth halves; PPA premiums compress; power complex de-rates; semi capex cycle rolls; broad market soft, not systemic.",
    Weight_Normalized=np.nan)])
s = pd.concat([s, new], ignore_index=True)
s["Weight_Normalized"] = s["Weight_raw"] / s["Weight_raw"].sum()
s.to_csv(SCEN_OUT, index=False)

print(f"deck {DECK_OUT}: {d.shape[0]} tickers x {d.shape[1]} scenarios; scenarios {SCEN_OUT}: {len(s)} rows, weights sum {s.Weight_Normalized.sum():.6f}")
print(f"AI-positive mass {s.loc[s.Scenario.str.startswith('AI '), 'Weight_Normalized'].sum()*100:.2f}%  AI-bust {s.loc[s.Scenario==NAME,'Weight_Normalized'].iloc[0]*100:.2f}%")
print("\nAIBUST column for the recommended books' holdings vs their Base and worst rows:")
for t in ['GLDM','SHLD','CIBR','DBMF','NUKZ','URNM','XLE','BRKB','CME','FNV','MSCI','XAR','SETM','HACK','IVV','SMH','CEG','GRID','CTA','KMLM']:
    if t in d.index:
        r = d.loc[t]; print(f"  {t:5s} AIBUST {r[NAME]:6.2f}   Base {C('Base Case')[t]:6.2f}   Credit {C('Credit Stress')[t]:6.2f}   row min {r.drop(NAME).min():6.2f}")
