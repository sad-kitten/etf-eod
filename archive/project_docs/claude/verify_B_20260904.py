"""Engine-exact verification (60k paths, seed 42, taxes, basis B) of search_B finalists next to the reference books.
Columns: base geo/P5/ruin, -200bp, stress vol (1.18 x vol48 for new books; analog ratios for the reference books),
surprise-adjusted CE (S4 = 0.10 for new books), lifetime tax, worst net CAGR, DRB40@6%, vol48, singles.
Run: R7=1 R8=1 DECK_FILE=Deck_Recal_R8b_20260903.csv SCEN_FILE=scenarios_24_R6_20260831.csv python3 verify_B.py"""
import sys, json, numpy as np, pandas as pd
import lib_pathce as L
import policy_engine2 as E
import search_SACE as SS
from core import DECKM, ERV, sigma_of, wvec, IDX, SINGLES

THEM = ['SHLD','NUKZ','CIBR','GRID','XAR','URNM','CEG','JEDI']; SING = ['CME','MSCI','CEG','CBOE','BRKB','FNV']
SV = pd.read_csv("out/Policy_Basis_20260903.csv").set_index("Port")["stress_vol_pct"] / 100.0
S4 = {"P2":0.06,"R3b":0.10,"R3b_B5":0.10,"R3dK":0.12,"R1b_final":0.10,"R2_final":0.08}
books = json.load(open("pkg/enginev2/books.json"))
books["R3b_B5"] = {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.11,'CIBR':.08,'GRID':.05,'CME':.05,'BRKB':.05,'CTA':.05,'URNM':.03,'XAR':.03,'CEG':.03,'FNV':.04,'MSCI':.03}
cands = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}
import os
todo = ([(k, books[k]) for k in ["R3b", "R3b_B5", "R1b_final", "R2_final"]] if os.environ.get("REFS", "1") == "1" else []) + list(cands.items())
rows = []
for nm, d in todo:
    w = wvec(d); r = DECKM.T @ w - float(ERV @ w); sig = sigma_of(w)
    mc = E.suite(r, sig, E.POL); mcL = E.suite(r - 0.02, sig, E.POL)
    sv = float(SV[nm]) if nm in SV.index else 1.18 * sig
    mcS = E.suite(r, sv, E.POL)
    wt = float(SS.TW @ w); ws = float(SS.SW @ w)          # widened theme set, all single names, S4 by rule (search_SACE)
    slope = (np.log(E.geo(mc)) - np.log(E.geo(mcL))) / 2
    pen = 0.9 * slope + 0.8 * wt * slope + SS.s4(w) + 0.25 * ws
    rg = mc[("robust", "G")]
    rows.append(dict(Port=nm, singles=sum(1 for t in SINGLES if d.get(t, 0) > 0), geo=round(E.geo(mc), 2), P5=round(rg["P5"], 2), ruin=rg["ruin"],
                     LS_geo=round(E.geo(mcL), 2), LS_P5=round(mcL[("robust","G")]["P5"], 2), LS_ruin=mcL[("robust","G")]["ruin"],
                     SV_geo=round(E.geo(mcS), 2), SV_P5=round(mcS[("robust","G")]["P5"], 2), SV_ruin=mcS[("robust","G")]["ruin"],
                     SACE=round(E.geo(mc) * np.exp(-pen), 2), tax=round(rg["tax_paid"], 3), worst_pct=round(float(r.min()) * 100, 2),
                     DRB6=round(L.drb40(r, h=0.06), 4), vol48=round(sig * 100, 2), theme_wt=round(wt, 2), S4=round(SS.s4(w), 3), pen=round(pen, 3),
                     weights=" ".join(f"{t}{int(round(x*100))}" for t, x in sorted(d.items(), key=lambda kv: -kv[1]))))
    print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv(os.environ.get("OUT", "out/SearchB_Verify_20260904.csv"), index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 120); print(df.drop(columns="weights").to_string(index=False))
