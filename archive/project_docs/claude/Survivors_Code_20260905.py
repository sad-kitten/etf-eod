"""Apply the owner's filter set and return the surviving distinct books.
Filters: (1) gate FAILs out, (2) grade-C weight cap, (3) risk-share caps (concentration),
(4) essential duplicates collapsed -- two books are the same book if less than DUP of the
portfolio has to move to turn one into the other."""
import numpy as np, pandas as pd, json, os, warnings; warnings.filterwarnings('ignore')
import lib_pathce as L, search_B as S
from core import UNIV, IDX, DECKM, ERV, sigma_of, wvec, feasible, CONVS
CW_CAP, RS_CAP, NRS_CAP, DUP = 0.20, 0.38, 0.24, 0.03
CW = np.array([1.0 if S.grades.get(t, "B") == "C" else 0.0 for t in UNIV])
mr = L.monthly_returns(); hist = {t: int(mr[t].notna().sum()) for t in mr.columns}

BK = {}
for f in ["out/rankbooks.json", "out/finalbooks2.json", "out/searchFILT.json", "out/searchFILT2.json"]:
    if os.path.exists(f):
        for k, v in json.load(open(f)).items(): BK.setdefault(k, v)
def geo_fast(w, M):
    r = M.T @ w - float(ERV @ w); sig = sigma_of(w)
    return float(np.exp(np.mean([np.log(S.run_B(r, c[0], c[1], sig)["PathCE"]) for c in CONVS])))

cand = []
for nm, d in BK.items():
    d = {t: v for t, v in d.items() if v > 0}
    w = wvec(d)
    if abs(w.sum()-1) > 1e-9 or not feasible(w): continue
    ns, mx = S.risk_shares(w); cw = float(CW @ w); fails = S.gate_fails(w)
    rec = dict(Port=nm, w=w, d=d, nuc=ns, name=mx, C=cw, fails=fails,
               gfast=geo_fast(w, DECKM), hcc=geo_fast(w, S.HCC),
               lt5=sum(v for t, v in d.items() if hist.get(t, 999) < 60))
    cand.append(rec)
print(f"pool {len(cand)}")
drop = {}
keep0 = []
for c in cand:
    if c["fails"]: drop.setdefault("FAIL gates", []).append(c["Port"]); continue
    if c["C"] > CW_CAP + 1e-9: drop.setdefault("grade-C weight", []).append(c["Port"]); continue
    if c["nuc"] > RS_CAP + 1e-9: drop.setdefault("nuclear risk share", []).append(c["Port"]); continue
    if c["name"] > NRS_CAP + 1e-9: drop.setdefault("single-name risk share", []).append(c["Port"]); continue
    keep0.append(c)
keep0.sort(key=lambda c: -c["gfast"])
uniq, dup_of = [], {}
for c in keep0:
    hit = None
    for u in uniq:
        if 0.5*np.abs(c["w"]-u["w"]).sum() <= DUP + 1e-9: hit = u; break
    if hit is None: uniq.append(c)
    else: dup_of.setdefault(hit["Port"], []).append(c["Port"])
for k, v in drop.items(): print(f"  dropped, {k:24s} ({len(v):2d}): {', '.join(sorted(v))}")
print(f"  collapsed as essential duplicates: " + ("; ".join(f"{k} = {', '.join(v)}" for k, v in dup_of.items()) or "none"))
print(f"\nSURVIVORS: {len(uniq)}")
json.dump({c["Port"]: {t: round(float(v), 2) for t, v in c["d"].items()} for c in uniq}, open("out/survivors.json", "w"), indent=1)
json.dump({c["Port"]: dict(nuc=round(c["nuc"],3), name=round(c["name"],3), C=round(c["C"],3),
           lt5=round(c["lt5"],3), hcc_pct=round((c["hcc"]/c["gfast"]-1)*100,2), dups=dup_of.get(c["Port"], []))
           for c in uniq}, open("out/survivor_filters.json","w"), indent=1)
for c in uniq: print(f"  {c['Port']:12s} geo_fast {c['gfast']:.3f}  nuc {c['nuc']*100:3.0f}%  name {c['name']*100:3.0f}%  C {c['C']*100:3.0f}%  <5y {c['lt5']*100:3.0f}%  hcc {(c['hcc']/c['gfast']-1)*100:+.2f}%")
import pandas as pd, json, numpy as np
d = pd.read_csv("out/Survivors_R8h_20260905.csv").set_index("Port")
F = json.load(open("out/survivor_filters.json")); BK = json.load(open("out/survivors.json"))
import warnings; warnings.filterwarnings('ignore')
import search_B as S
from core import ERV, wvec
d = d.sort_values("geo_x", ascending=False)
rows = []
print(f"{'#':>2}  {'BOOK':12s} {'geo x':>6s} {'real%':>6s} {'nom%':>5s} {'nom x':>6s} {'SACE x':>6s} {'g3%':>5s} {'P5%':>5s} {'P25%':>5s} {'vol':>5s} {'nuc':>4s} {'name':>5s} {'C':>4s} {'<5y':>4s} {'hcc%':>6s} {'ER':>6s} {'sing':>4s} {'n':>3s} {'sum':>4s}")
for i, p in enumerate(d.index, 1):
    r = d.loc[p]; f = F[p]; w = {t: v for t, v in BK[p].items() if v > 0}
    ks = sorted(w, key=lambda t: (-w[t], t)); tot = sum(round(w[t]*100) for t in ks)
    SING = {'CME','MSCI','CEG','CBOE','FNV'}; ns = sum(1 for t in ks if t in SING)
    er = float(ERV @ wvec(w))*100
    nom = ((1+r.geo_cagr/100)*1.025-1)*100
    g3 = r.ce3_cagr
    print(f"{i:2d}  {p:12s} {r.geo_x:6.3f} {r.geo_cagr:6.2f} {nom:5.2f} {r.geo_x*1.025**40:6.2f} {r.SACE_x:6.3f} {g3:5.2f} {r.cagr_P5:5.2f} {r.cagr_P25:5.2f} {r.vol48:5.2f} "
          f"{f['nuc']*100:4.0f} {f['name']*100:5.0f} {f['C']*100:4.0f} {f['lt5']*100:4.0f} {f['hcc_pct']:6.2f} {er:6.3f} {ns:4d} {len(ks):3d} {tot:4d}")
    rows.append(dict(rank=i, Port=p, geo_x=r.geo_x, real_cagr=r.geo_cagr, nom_cagr=round(nom,2),
        nom_x=round(r.geo_x*1.025**40,2), SACE_x=r.SACE_x, g3_cagr=g3, P5_cagr=r.cagr_P5, P25_cagr=r.cagr_P25,
        vol48=r.vol48, nuc_rs=round(f['nuc']*100), name_rs=round(f['name']*100), C_wt=round(f['C']*100),
        lt5y_wt=round(f['lt5']*100), hcc_pct=f['hcc_pct'], ER=round(er,3), singles=ns, holdings=len(ks),
        collapsed_duplicates=";".join(f['dups']),
        weights=" ".join(f"{t}{int(round(w[t]*100))}" for t in ks)))
print()
for i, p in enumerate(d.index, 1):
    w = {t: v for t, v in BK[p].items() if v > 0}; ks = sorted(w, key=lambda t: (-w[t], t))
    dup = F[p]["dups"]
    print(f"{i:2d}. {p:12s} " + " · ".join(f"{t} {int(round(w[t]*100))}" for t in ks) + (f"   [also: {', '.join(dup)}]" if dup else ""))
pd.DataFrame(rows).to_csv("out/Survivors_Filtered_20260905.csv", index=False)
"""Diverse seeds that already satisfy the filter set, so the climb starts admissible."""
import json, numpy as np, warnings; warnings.filterwarnings('ignore')
import search_B as S
from core import wvec, feasible, UNIV, IDX
CW = np.array([1.0 if S.grades.get(t,"B")=="C" else 0.0 for t in UNIV])
SEEDS = {
 "F1_bal":   {'CIBR':.15,'DBMF':.15,'GLDM':.15,'SHLD':.15,'NUKZ':.09,'BRKB':.05,'CME':.05,'FNV':.05,'MSCI':.05,'XLE':.05,'URNM':.04,'HACK':.02},
 "F2_bal2":  {'CIBR':.15,'DBMF':.15,'GLDM':.15,'SHLD':.15,'NUKZ':.10,'BRKB':.05,'CME':.05,'MSCI':.05,'XLE':.05,'URNM':.04,'KMLM':.02,'SETM':.02,'FTEC':.02},
 "F3_xar":   {'CIBR':.15,'DBMF':.15,'GLDM':.15,'SHLD':.14,'NUKZ':.09,'BRKB':.05,'CME':.05,'FNV':.05,'MSCI':.05,'XLE':.05,'XAR':.04,'URNM':.03},
 "F4_grid":  {'CIBR':.15,'DBMF':.15,'GLDM':.15,'SHLD':.14,'NUKZ':.09,'BRKB':.05,'CME':.05,'GRID':.05,'MSCI':.05,'XLE':.05,'URNM':.03,'PSLV':.04},
 "F5_noxle": {'CIBR':.15,'DBMF':.15,'GLDM':.15,'SHLD':.15,'NUKZ':.10,'BRKB':.05,'CME':.05,'FNV':.05,'MSCI':.05,'XLI':.05,'URNM':.03,'CTA':.02},
 "F6_cta":   {'CIBR':.14,'DBMF':.15,'GLDM':.15,'SHLD':.14,'NUKZ':.09,'CTA':.06,'BRKB':.05,'CME':.05,'FNV':.05,'MSCI':.05,'XLE':.04,'URNM':.03},
 "F7_pm":    {'CIBR':.15,'DBMF':.15,'GLDM':.15,'SHLD':.14,'NUKZ':.09,'PSLV':.05,'BRKB':.05,'CME':.05,'MSCI':.05,'XLE':.05,'URNM':.03,'XAR':.04},
 "F8_qual":  {'CIBR':.15,'DBMF':.15,'GLDM':.15,'SHLD':.14,'NUKZ':.09,'BRKB':.05,'CME':.05,'JQUA':.05,'MSCI':.05,'XLE':.05,'URNM':.03,'FNV':.04},
 "F9_semi":  {'CIBR':.15,'DBMF':.15,'GLDM':.15,'SHLD':.14,'NUKZ':.09,'BRKB':.05,'CME':.05,'FNV':.05,'MSCI':.05,'XLE':.04,'SMH':.04,'URNM':.04},
 "F10_setm": {'CIBR':.15,'DBMF':.15,'GLDM':.15,'SHLD':.14,'NUKZ':.08,'BRKB':.05,'CME':.05,'FNV':.05,'MSCI':.05,'XLE':.05,'SETM':.04,'URNM':.04},
 "F11_ivv":  {'CIBR':.15,'DBMF':.15,'GLDM':.15,'SHLD':.14,'NUKZ':.09,'BRKB':.05,'CME':.05,'IVV':.05,'MSCI':.05,'XLE':.05,'FNV':.04,'URNM':.03},
 "F12_gunr": {'CIBR':.15,'DBMF':.15,'GLDM':.15,'SHLD':.14,'NUKZ':.09,'BRKB':.05,'CME':.05,'FNV':.05,'GUNR':.05,'MSCI':.05,'XLE':.04,'URNM':.03},
}
ok = {}
for nm, d in SEEDS.items():
    w = wvec(d); tot = sum(d.values())
    ns, mx = S.risk_shares(w); cw = float(CW @ w); f = S.gate_fails(w)
    good = abs(tot-1) < 1e-9 and feasible(w) and ns <= 0.38 and mx <= 0.24 and cw <= 0.20
    print(f"{nm:10s} sum {tot:.2f} feas {feasible(w)!s:5s} nuc {ns*100:4.0f}% name {mx*100:4.0f}% C {cw*100:3.0f}%  gates {'PASS' if not f else ','.join(sorted(set(x.split(':')[1] for x in f)))}  {'-> keep' if good else '-> drop'}")
    if good: ok[nm] = d
json.dump(ok, open("out/filter_seeds.json","w"), indent=1)
print(f"\n{len(ok)} admissible seeds")
