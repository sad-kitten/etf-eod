# ===== uncertainty.py =====
"""E1 — parameter-uncertainty (Bayesian predictive) evaluation for EngineV2 (Addendum 55b, 2026-10-24).

Why: deck rows are estimates. Over 50 years an error in a row's expected return acts as a permanent drift, so its share of
terminal dispersion rivals the year-to-year noise the engine already simulates. The central engine treats every row as known.
Here each path draws, once for its whole horizon:
  (a) row errors, pp/yr, mean 0 (the deck stays the posterior mean):
        equity factor 1.25 (every equity row, incl. thematic, singles, miners, RSST's equity sleeve)
        thematic factor 1.5 (standing THEMATIC rows) + theme-block factor 1.5 each (nuclear, defence, cyber, semis, energy)
        gold factor 1.5 (gold family) · trend factor 2.0 in normal states, x1.5 in severity-4/5 states (DBMF, KMLM, CTA, RSST)
        idiosyncratic by row grade: A 0.75, B 1.25, C 2.0; single companies 2.5
  (b) the sigma level: standing 48-month window or the 120-month anchored window (M1), probability 1/2 each (D7)
Specs: U1 = sds above; U½ / U2 = all sds x0.5 / x2; the sigma mix is on in all three; U0 = everything off.
Draws come from a separate seeded stream (2718 fast, 31415 engine-exact): scenario paths and return shocks are the central
engine's, so books are compared on common random numbers and U0 reproduces the central engine exactly.
"""
import inspect, numpy as np
import lib_pathce as L, standing as ST, search_B as S, policy_engine2 as E
from core import UNIV, IDX, N, DECKM, ERV, SINGLES, sigma_of

SPEC1 = dict(eq=1.25, them=1.5, block=1.5, gold=1.5, trend=2.0, trend_stress=1.5, A=0.75, B=1.25, C=2.0, single=2.5)
NON_EQUITY = set("IEF TLT GOVZ LTPZ SCHP VTIP BIL SHY AGG TIP STIP HYG LQD EMB TAIL BTAL GLDM GLD PSLV DBMF KMLM CTA "
                 "CTA_OVERLAY PDBC IBIT".split())
GOLD_U = set(ST.GOLD_FAM) | set(ST.BLOCKS["gold"])
TREND_U = {"DBMF": 1.0, "KMLM": 1.0, "CTA": 1.0, "RSST": 1.0, "DBMFh": 1.0, "RSSTh": 1.0}
# stacked funds: factor loadings = their notional sleeves (RSST 1 equity + 1 trend; GDE 0.9 equity + 0.9 gold; NTSX 0.9 equity)
SLEEVE_SCALE = {"NTSX": dict(eq=0.9), "GDE": dict(eq=0.9, gold=0.9), "DBMFh": dict(eq=0.0)}
BLOCKS_U = dict(nuclear=ST.BLOCKS["nuclear"], defence=ST.BLOCKS["defence"], cyber=ST.BLOCKS["cyber"],
                semis=ST.BLOCKS["semis"], energy=["XLE", "AMLP"])
FACTORS = ["equity", "thematic", "gold", "trend"] + [f"block:{b}" for b in BLOCKS_U]

def loadings(spec=SPEC1):
    """(normal-state loadings, stress-state loadings, idiosyncratic sd), decimal per year; rows = UNIV order."""
    Ln = np.zeros((N, len(FACTORS))); idio = np.zeros(N)
    for t, i in IDX.items():
        if t not in NON_EQUITY: Ln[i, 0] = spec["eq"]
        if t in ST.THEMATIC: Ln[i, 1] = spec["them"]
        if t in GOLD_U: Ln[i, 2] = spec["gold"]
        Ln[i, 3] = TREND_U.get(t, 0.0) * spec["trend"]
        for k, names in enumerate(BLOCKS_U.values()):
            if t in names: Ln[i, 4 + k] = spec["block"]
        idio[i] = spec["single"] if t in SINGLES else spec.get(str(S.grades.get(t, "B")), spec["B"])
        sc = SLEEVE_SCALE.get(t, {})
        if "eq" in sc: Ln[i, 0] = spec["eq"] * sc["eq"]
        if "gold" in sc: Ln[i, 2] = spec["gold"] * sc["gold"]
    Ls = Ln.copy(); Ls[:, 3] *= spec["trend_stress"]
    return Ln / 100.0, Ls / 100.0, idio / 100.0

class Draws:
    """Per-path permanent row errors (npaths x N) for normal and stress states, plus the sigma-window coin."""
    def __init__(self, npaths, seed, spec=SPEC1):
        rng = np.random.default_rng(seed)
        F = rng.standard_normal((npaths, len(FACTORS))); U = rng.standard_normal((npaths, N)); self.coin = rng.random(npaths)
        Ln, Ls, idio = loadings(spec)
        self.En = F @ Ln.T + U * idio; self.Es = F @ Ls.T + U * idio
    def book(self, w, k=1.0, sigmix=True):
        dn = k * (self.En @ w); ds = k * (self.Es @ w)
        if sigmix:
            ratio = float(np.sqrt(max(float(w @ ST.COV120 @ w), 1e-12))) / sigma_of(w)
            m = np.where(self.coin < 0.5, 1.0, ratio)
        else:
            m = np.ones(len(self.coin))
        return dn, ds, m
    def book_sd(self, w, k=1.0):
        return float(np.std(k * (self.En @ w))), float(np.std(k * (self.Es @ w)))

# ------------------------------------------------------------------ fast evaluator (mirror of standing.run_fast)
def run_fast_u(r, sigv, pin, conv, dn, ds, m):
    FE = S.FE; s = FE.S[pin]; sig = sigv[s] * m[:, None]
    mu = np.log1p(np.maximum(r[s] + dn[:, None] + (ds - dn)[:, None] * ST.STRESS[s], -0.95))
    if conv == "G": mu = mu - sig * sig / 2
    X = np.exp(mu + sig * FE.Z) / (1.0 + S.INFL)
    W = np.ones(FE.npaths); red = np.zeros(FE.npaths, bool); ru = np.zeros(FE.npaths, bool)
    for t in range(1, S.T + 1):
        W = W * X[:, t - 1]
        red = np.where(W < 0.80, True, np.where(W >= 1.0, False, red))
        draw = np.maximum(0.03 * np.where(red, 0.75, 1.0) - (S.SS if t >= 11 else 0.0), 0.0)
        W = W - draw; ru |= W <= 0; W = np.maximum(W, 1e-9)
    lg = np.log(W)
    return dict(PathCE=float(np.exp(lg.mean())), CE3=float(np.mean(W ** -2.0) ** -0.5), CE5=float(np.mean(W ** -4.0) ** -0.25),
                CE2=float(np.mean(W ** -1.0) ** -1.0), P5=float(np.quantile(W, .05)), P10=float(np.quantile(W, .10)), ruin=int(ru.sum()), pW1=float((W < 1.0).mean()))

CONVS = [("robust", "G"), ("robust", "L"), ("canonical", "G"), ("canonical", "L")]
def evaluate_u(w, D, k=1.0, sigmix=True, deck=None):
    M = DECKM if deck is None else deck
    r = M.T @ w - float(ERV @ w)
    dn, ds, m = (np.zeros(S.FE.npaths), np.zeros(S.FE.npaths), np.ones(S.FE.npaths)) if D is None else D.book(w, k, sigmix)
    out = {}
    for pin in ("robust", "canonical"):
        sigv, _ = ST.sigma_vec(w, pin)
        for conv in ("G", "L"): out[(pin, conv)] = run_fast_u(r, sigv, pin, conv, dn, ds, m)
    mlog = lambda key: float(np.mean([np.log(out[c][key]) for c in CONVS]))
    return dict(geo=np.exp(mlog("PathCE")), ce3=np.exp(mlog("CE3")), ce5=np.exp(mlog("CE5")), P5=out[("robust", "G")]["P5"],
                ruin=out[("robust", "G")]["ruin"], pW1=float(np.mean([out[c]["pW1"] for c in CONVS])))

# ------------------------------------------------------------------ engine-exact (policy_engine2.run, two lines changed)
_src = inspect.getsource(E.run)
_a, _b = "        sg = sigma[s] if vec else sigma\n", "        mu = np.log1p(r_net[s]) - "
assert _src.count(_a) == 1 and _src.count(_b) == 1 and _src.startswith("def run(rng, r_net, pi, conv, sigma, pol=POL):")
_src = (_src.replace("def run(rng, r_net, pi, conv, sigma, pol=POL):",
                     "def run_unc(rng, r_net, pi, conv, sigma, pol=POL, dn=0.0, ds=0.0, m=1.0):", 1)
            .replace(_a, "        sg = (sigma[s] if vec else sigma) * m\n")
            .replace(_b, "        mu = np.log1p(r_net[s] + dn + (ds - dn) * STRESS_U[s]) - "))
_ns = dict(vars(E)); _ns["STRESS_U"] = ST.STRESS
exec(compile(_src, "uncertainty.run_unc", "exec"), _ns); run_unc = _ns["run_unc"]

def suite_u(r_net, sigma, dn=0.0, ds=0.0, m=1.0, pol=E.POL):
    """policy_engine2.suite with per-path row errors and sigma multiplier (same rng(42) per pin => same streams)."""
    out = {}
    for pin, pi in (("robust", L.PI_ROBUST), ("canonical", L.PI_CANON)):
        rng = np.random.default_rng(42)
        sg = sigma[pin] if isinstance(sigma, dict) else sigma
        for conv in ("G", "L"): out[(pin, conv)] = run_unc(rng, r_net, pi, conv, sg, pol, dn, ds, m)
    return out

def tail_stats(r, pi=None):
    """D4 floor statistics on a per-scenario net CAGR vector: worst, CVaR10/CVaR15 (pi-weighted mean of the lowest 10% / 15%
    of probability mass; robust pi by default), severity-5 weighted mean (canonical weights), mean of the worst two."""
    pi = L.PI_ROBUST if pi is None else pi
    o = np.argsort(r); rs, ps = r[o], pi[o] / pi.sum()
    def cvar(q):
        c = np.cumsum(ps); take = np.clip(q - (c - ps), 0.0, ps); return float((take * rs).sum() / q)
    sev5 = L.SEV >= 5; w = L.W_CAN
    return dict(worst=float(rs[0]), worst_scen=L.SC[int(o[0])], second=float(rs[1]), cvar10=cvar(0.10), cvar15=cvar(0.15),
                sev5=float((w[sev5] * r[sev5]).sum() / w[sev5].sum()), worst2=float(rs[:2].mean()))

# ===== eval_u.py =====
"""Fast screen for Addendum 55b: central v3.4 gates + E1 parameter-uncertainty metrics + D4 floor statistics, one row per book.
Run (env_a54.sh sourced: deck R8n, or DECK=R8o; env_stk.sh: DECK=R8m): python3 eval_u.py out.csv books1.json [books2.json ...]
Fast evaluator = pre-tax simplified wealth model (6000 paths x 4 conventions, FAST_P5 1.06 ~ engine-exact P5 1.00); use the
engine-exact report for decisions. Columns: central geo/ce3/P5/fails; U1 (sds x1 + sigma mix), U1 without the sigma mix,
U½, U2; book error sd; D4 tail statistics of the per-scenario net CAGR vector."""
import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import standing as ST, stk, search_B as S, uncertainty as U, lib_pathce as L
from core import wvec, DECKM, ERV, IDX
D = U.Draws(S.FE.npaths, 2718)
books = {}
for f in sys.argv[2:]:
    for nm, d in json.load(open(f)).items():
        d = {t: v for t, v in d.items() if v > 0}
        if all(t in IDX for t in d) and nm not in books: books[nm] = d
rows = []
cg = lambda x: (x ** (1 / float(L.T)) - 1) * 100
for nm, d in books.items():
    w = wvec(d); r = DECKM.T @ w - float(ERV @ w); e = ST.evaluate_fast(w); v = ST.violations(w)
    _sl = ST.SIGMA_LEVEL; ST.SIGMA_LEVEL = "cov120"
    try: e1 = ST.evaluate_fast(w)
    finally: ST.SIGMA_LEVEL = _sl
    u1, u1n, uh, u2 = (U.evaluate_u(w, D, k, mix) for k, mix in ((1, True), (1, False), (.5, True), (2, True)))
    ts = U.tail_stats(r); sw, ssc = ST.stressed_worst(w); sdn, sds = D.book_sd(w)
    rows.append({"Port": nm, "geo": round(e["geo"], 3), "ce3 CAGR": round(cg(e["ce3"]), 2), "P5": round(e["P5"], 3),
                 "gates": "PASS" if not e["fails"] else "; ".join(e["fails"]), "M1 gates": "PASS" if not e1["fails"] else "; ".join(e1["fails"]), "M1 ce3 CAGR": round(cg(e1["ce3"]), 2), "constraints": "ok" if not v else "; ".join(v),
                 "U1 geo": round(u1["geo"], 3), "U1 ce3 CAGR": round(cg(u1["ce3"]), 2), "U1 ce5 CAGR": round(cg(u1["ce5"]), 2),
                 "U1 P5": round(u1["P5"], 3), "U1 P(W<1)": round(u1["pW1"], 3), "U1nomix ce3 CAGR": round(cg(u1n["ce3"]), 2),
                 "Uhalf ce3 CAGR": round(cg(uh["ce3"]), 2), "Uhalf P5": round(uh["P5"], 3), "U2 ce3 CAGR": round(cg(u2["ce3"]), 2),
                 "U2 P5": round(u2["P5"], 3), "err sd pp": round(sdn * 100, 2), "err sd stress pp": round(sds * 100, 2),
                 "worst": round(ts["worst"] * 100, 2), "worst scen": ts["worst_scen"][:24], "2nd": round(ts["second"] * 100, 2),
                 "CVaR10": round(ts["cvar10"] * 100, 2), "CVaR15": round(ts["cvar15"] * 100, 2), "sev5 mean": round(ts["sev5"] * 100, 2),
                 "stressed worst": round(sw * 100, 2), "ER": round(float(ERV @ w) * 100, 3),
                 "weights": " ".join(f"{t}{int(round(x * 100))}" for t, x in sorted(d.items(), key=lambda kv: -kv[1]))})
    print(nm, rows[-1]["geo"], rows[-1]["ce3 CAGR"], rows[-1]["U1 ce3 CAGR"], rows[-1]["U1 P5"], rows[-1]["gates"], flush=True)
df = pd.DataFrame(rows).set_index("Port").sort_values("U1 ce3 CAGR", ascending=False); df.to_csv(sys.argv[1])
print(df[["geo", "ce3 CAGR", "U1 ce3 CAGR", "U1 P5", "Uhalf ce3 CAGR", "U2 ce3 CAGR", "err sd pp", "worst", "CVaR15", "gates", "constraints"]].to_string())

# ===== search_u.py =====
"""Robust climb for Addendum 55b: search_v3.climb (repair phase + hill climb, v3.4 central gates, standing constraints) with the
objective replaced by the E1 predictive objective: mean over the 4 conventions of log CE(gamma=3) under U1 (row-error sds x UNC_K,
default 1, sigma-window mix on unless UNC_MIX=0), fast evaluator on common random numbers (Draws seed 2718).
Run (env_stk.sh sourced): SEEDS_FILE=books_ref.json OUTJSON=out/climb_x.json NO_STACK=1 python3 search_u.py C5_SMH3b"""
import os, sys, json, numpy as np, warnings; warnings.filterwarnings("ignore")
import standing as ST, stk, search_B as S, uncertainty as U, search_v3 as SV
from core import DECKM, ERV, IDX
K = float(os.environ.get("UNC_K", "1.0")); MIX = os.environ.get("UNC_MIX", "1") == "1"
KEY = {"1": "PathCE", "2": "CE2", "3": "CE3", "5": "CE5"}[os.environ.get("OBJ_GAMMA", "3")]   # objective CE at gamma
D = U.Draws(S.FE.npaths, 2718)
def objective(w):
    r = DECKM.T @ w - float(ERV @ w); dn, ds, m = D.book(w, K, MIX); vals = []
    for pin in ("robust", "canonical"):
        sigv, _ = ST.sigma_vec(w, pin)
        for conv in ("G", "L"): vals.append(np.log(U.run_fast_u(r, sigv, pin, conv, dn, ds, m)[KEY]))
    return float(np.mean(vals))
SV.objective = objective
# ---- optional second deck (DECK2_FILE, same universe): objective = mean of the two decks' objectives; gates must pass on both
import pandas as pd, lib_pathce as L
from core import UNIV
if os.environ.get("DECK2_FILE"):
    _D2 = pd.read_csv(os.path.join(L.DECKS, os.environ["DECK2_FILE"])); _D2["Ticker"] = _D2["Ticker"].str.strip(); _D2 = _D2.set_index("Ticker")
    _D2.loc["IAI"] = _D2.loc[["CME", "CBOE", "NDAQ"]].mean()
    M2 = np.array([[_D2.loc[t, sc] for sc in L.SC] for t in UNIV]) / 100.0; MS2 = ST.stressed_deck(M2); M1 = DECKM
    def _obj_on(w, M):
        r = M.T @ w - float(ERV @ w); dn, ds, m = D.book(w, K, MIX); vals = []
        for pin in ("robust", "canonical"):
            sigv, _ = ST.sigma_vec(w, pin)
            for conv in ("G", "L"): vals.append(np.log(U.run_fast_u(r, sigv, pin, conv, dn, ds, m)[KEY]))
        return float(np.mean(vals))
    def _eval_on(w, M, MS):
        saved = ST.DECK_STRESSED; ST.DECK_STRESSED = MS
        try: return ST.evaluate_fast(w, deck=M)
        finally: ST.DECK_STRESSED = saved
    def _deficit_on(v, M, MS):
        e = _eval_on(v, M, MS); r = M.T @ v - float(ERV @ v); saved = ST.DECK_STRESSED; ST.DECK_STRESSED = MS
        try: sw = ST.stressed_worst(v)[0]
        finally: ST.DECK_STRESSED = saved
        d = max(0.0, ST.GATES["worst"] - float(r.min())) * 100 + (max(0.0, ST.STRESS_FLOOR - sw) * 100 if ST.STRESS_GATE else 0.0)
        d += max(0.0, ST.FAST_P5 - e["P5"]) + max(0.0, ST.GATES["R3AF"] - ST.r3af_std(r)) * 10
        d += 1.0 * (e["ruin"] > 0) + 1.0 * (ST.run_fast(r - 0.02, ST.sigma_vec(v, "robust")[0], "robust", "G")["ruin"] > 0)
        return d
    SV.objective = lambda w: 0.5 * (_obj_on(w, M1) + _obj_on(w, M2))
    SV.gates_ok = lambda v: not _eval_on(v, M1, ST.DECK_STRESSED)["fails"] and not _eval_on(v, M2, MS2)["fails"]
    SV.deficit = lambda v: _deficit_on(v, M1, ST.DECK_STRESSED) + _deficit_on(v, M2, MS2)
    objective = SV.objective
# ---- DUAL_SIGMA=1: Addendum 54 (other session) D7 co-equal gate -- every accepted move must pass the gates under the 48-month
# sigma and under SIGMA_LEVEL=cov120 (M1); the objective's sigma mix is unaffected (it uses its own per-path coin).
if os.environ.get("DUAL_SIGMA", "0") == "1":
    _g, _d = SV.gates_ok, SV.deficit
    def _in_m1(f, v):
        saved = ST.SIGMA_LEVEL; ST.SIGMA_LEVEL = "cov120"
        try: return f(v)
        finally: ST.SIGMA_LEVEL = saved
    SV.gates_ok = lambda v: _g(v) and _in_m1(_g, v)
    SV.deficit = lambda v: _d(v) + _in_m1(_d, v)
if __name__ == "__main__":
    BK = json.load(open(os.environ.get("SEEDS_FILE", "books_ref.json"))); out = {}
    for tag in sys.argv[1:]:
        w, cur = SV.climb({t: v for t, v in BK[tag].items() if v > 0}, tag)
        out[tag + "_U"] = {t: round(float(w[IDX[t]]), 2) for t in S.held(w)}
        e = ST.evaluate_fast(w); u = U.evaluate_u(w, D, K, MIX)
        print(f"[{tag}] FINAL U-{KEY} {np.exp(cur):.3f} | central geo {e['geo']:.3f} ce3 {e['ce3']:.3f} P5 {e['P5']:.2f} worst {e['worst']*100:.2f}"
              f" fails {e['fails']} | U1 geo {u['geo']:.3f} ce3 {u['ce3']:.3f} P5 {u['P5']:.3f} | {S.show(w)}", flush=True)
        json.dump(out, open(os.environ.get("OUTJSON", "out/climb_U.json"), "w"), indent=1)

# ===== report_u.py =====
"""Engine-exact report for Addendum 55b: report_v3.profile (central, v3.4 gates, all standard columns) + E1 predictive columns
(policy_engine2.run with per-path permanent row errors and the sigma-window mix; Draws seed 31415 on 60k paths, rng(42) streams
otherwise identical to the central run) for U1, U½, U2, + D4 tail statistics. Optional POL_YIELDS="y_ord,y_qual" adds a central
run with the taxable-account distribution yields recalibrated (realized-performance lever: asset location).
Run (env_a54.sh sourced: deck R8n, or DECK=R8o; env_stk.sh: DECK=R8m): python3 report_u.py books.json out.csv [name1,name2,...]"""
import sys, os, json, time, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import standing as ST, stk, report_v3 as R, uncertainty as U, policy_engine2 as E, lib_pathce as L
from core import wvec, ERV, IDX
DX = U.Draws(L.NP_, 31415)
POLY = os.environ.get("POL_YIELDS")
def cg(x): return (x ** (1 / float(L.T)) - 1) * 100
BK = json.load(open(sys.argv[1])); keep = sys.argv[3].split(",") if len(sys.argv) > 3 else list(BK)
rows = []
for nm in keep:
    d = {t: v for t, v in BK[nm].items() if v > 0}
    if not all(t in IDX for t in d): print("skip", nm, [t for t in d if t not in IDX]); continue
    t0 = time.time(); row = R.profile(nm, d)
    w = wvec(d); r = R.M0.T @ w - float(ERV @ w); sig = {p: ST.sigma_vec(w, p)[0] for p in ("robust", "canonical")}
    for tag, k in (("U1", 1.0), ("Uhalf", 0.5), ("U2", 2.0)):
        dn, ds, m = DX.book(w, k, True); mc = U.suite_u(r, sig, dn, ds, m); W = R.pooled(mc); rg = mc[("robust", "G")]
        pc = W ** (1 / float(L.T)) - 1
        row.update({f"{tag} geo x": round(R.ce_cagr(W, 1)[0], 3), f"{tag} real CAGR %": round(R.ce_cagr(W, 1)[1], 2),
                    f"{tag} g3 CE-CAGR %": round(R.ce_cagr(W, 3)[1], 2), f"{tag} g5 CE-CAGR %": round(R.ce_cagr(W, 5)[1], 2),
                    f"{tag} P5 x": round(rg["P5"], 3), f"{tag} P5 path CAGR %": round(float(np.quantile(pc, .05)) * 100, 2),
                    f"{tag} P(W<1) %": round(float((W < 1).mean()) * 100, 1), f"{tag} ruin": int(rg["ruin"])})
    sdn, sds = DX.book_sd(w); ts = U.tail_stats(r)
    row.update({"err sd pp": round(sdn * 100, 2), "CVaR10 %": round(ts["cvar10"] * 100, 2), "CVaR15 %": round(ts["cvar15"] * 100, 2),
                "sev5 mean %": round(ts["sev5"] * 100, 2), "2nd worst %": round(ts["second"] * 100, 2)})
    if POLY:
        yo, yq = (float(x) for x in POLY.split(",")); pol = dict(E.POL, y_ord=yo, y_qual=yq)
        mc = E.suite(r, sig, pol); W = R.pooled(mc)
        row.update({"loc geo x": round(R.ce_cagr(W, 1)[0], 3), "loc g3 CE-CAGR %": round(R.ce_cagr(W, 3)[1], 2),
                    "loc P5 x": round(mc[("robust", "G")]["P5"], 3), "loc lifetime tax x": round(mc[("robust", "G")]["tax_paid"], 3)})
    row["secs total"] = round(time.time() - t0); rows.append(row)
    print(nm, row["geo x"], row["g3 CE-CAGR %"], row["P5 terminal x"], row["gates"], "|", row["U1 geo x"], row["U1 g3 CE-CAGR %"],
          row["U1 P5 x"], row["U2 g3 CE-CAGR %"], row["secs total"], "s", flush=True)
pd.DataFrame(rows).set_index("Port").to_csv(sys.argv[2])

# ===== stk.py =====
"""Stacked-fund wiring (Addendum 53 conventions, re-implemented 2026-10-24 for Addendum 55b). Import after standing and before
report_v3 / search_v3. Active only when RSST is in the universe (ENGINEV2_EXTRA=RSST with the *_STK deck, panel and grades):
RSST counts as ballast (core.BAL, standing.BALLAST) and takes the trend haircut (-3.0 pp in severity-4/5) on the stressed deck.
NO_STACK=1 removes RSST from the search pool (books that do not use fund-embedded leverage)."""
import os
import core, standing as ST, search_B as S
ACTIVE = "RSST" in core.IDX
if ACTIVE:
    for lst in (core.BAL, ST.BALLAST, ST.TREND):
        if "RSST" not in lst: lst.append("RSST")
    ST.DECK_STRESSED = ST.stressed_deck()
    if os.environ.get("NO_STACK", "0") == "1":   # books without fund-embedded leverage: no stacked fund in the search pool
        for _t in ("RSST", "RSSB", "NTSX", "GDE", "RSSTh"):
            if _t in S.POOL: S.POOL.remove(_t)

# ---- STK_A54=1 (env_a54.sh): the other session's Addendum 54 / deck R8n conventions, from R8n_Code_20261024.py (standing,
# search_v3) and StackedETF_Code_20261024.py (core.BAL). Off by default; nothing above changes. RSST's ballast/BAL/trend-stress
# membership comes from the ACTIVE block above (their STACK_STRESS RSST trend_sev45 x1.0 == ST.TREND membership).
A54 = os.environ.get("STK_A54", "0") == "1"
if A54:
    _I = core.IDX
    if "RSSB" in _I and "RSSB" not in core.BAL: core.BAL.append("RSSB")                     # their core.BAL lists RSST and RSSB
    if "GDE" in _I and "GDE" not in ST.BLOCKS["gold"]: ST.BLOCKS["gold"].append("GDE")     # D5 look-through: GDE in the gold block
    _stressed_base = ST.stressed_deck
    def _stressed_a54(M=None):
        """Their STACK_STRESS for GDE: 0.9 x the gold-family haircut (-1.5 pp) in every scenario."""
        M = _stressed_base(M)
        if "GDE" in _I: M[_I["GDE"]] += 0.9 * ST.STRESS_SPEC["gold_all"] / 100.0
        return M
    ST.stressed_deck = _stressed_a54; ST.DECK_STRESSED = ST.stressed_deck()
    if os.environ.get("PREF", "0") == "1":   # Addendum 43 owner preferences with the A54 look-through (their violations(), PREF=1)
        GOLDLINK = {"GLDM": 1, "GLD": 1, "FNV": 1, "WPM": 1, "GDX": 1, "SIL": 1, "PSLV": 1, "GDE": 0.9}
        TRENDW = ["DBMF", "KMLM", "CTA", "RSST"]
        _violations_base = ST.violations
        def _violations_pref(w):
            v = _violations_base(w)
            g = sum(w[_I[t]] * k for t, k in GOLDLINK.items() if t in _I); tr = sum(w[_I[t]] for t in TRENDW if t in _I)
            if g > 0.15 + 1e-9: v.append(f"gold-linked {g*100:.1f}%>15")
            if tr < g - 1e-9: v.append(f"trend {tr*100:.0f}%<gold {g*100:.0f}%")
            return v
        ST.violations = _violations_pref
    for _t in [x for x in os.environ.get("EXCL", "DBMFh,RSSTh,QTUM").split(",") if x]:   # their search_v3 EXCL (pool and singles)
        for _lst in (S.POOL, S.S1):
            if _t in _lst: _lst.remove(_t)

# ===== build_r8o.py =====
"""Deck R8o candidate (Addendum 55b, 2026-10-24) = deck R8n of Addendum 54 (Deck_R8n_A54_20261024.csv, byte-identical to the
Project's Deck_R8n_20261024.csv) with the KMLM and CTA rows set to trend parity on the reshaped DBMF row:
KMLM = DBMF + 0.05 and CTA = DBMF - 0.09 (gross), i.e. the same net-of-ER return as DBMF in every scenario (ERs 0.90 / 0.76 /
0.85). Every other row, including RSST (built from DBMF), is unchanged. The robust pi is frozen at R8n's
(data/decks/pi_robust_R8n_A54.csv; set PI_ROBUST_FILE) so that the parity change does not reweight scenarios through the EVG11
benchmark (CTA 10% of EVG11).  Run with env_a54.sh sourced: python3 build_r8o.py"""
import os, numpy as np, pandas as pd
import lib_pathce as L                      # under env_a54.sh this loads R8n and its recomputed robust pi
H = os.path.dirname(os.path.abspath(__file__)); DK = os.path.join(H, "data", "decks")
np.savetxt(os.path.join(DK, "pi_robust_R8n_A54.csv"), L.PI_ROBUST, delimiter=",", fmt="%.17g")
D = pd.read_csv(os.path.join(DK, "Deck_R8n_A54_20261024.csv")); D["Ticker"] = D["Ticker"].str.strip(); D = D.set_index("Ticker")
O = D.copy(); O.loc["KMLM"] = D.loc["DBMF"] + 0.05; O.loc["CTA"] = D.loc["DBMF"] - 0.09
O.reset_index().to_csv(os.path.join(DK, "Deck_R8o_20261024.csv"), index=False)
w = L.W_CAN; d = O.loc[["KMLM", "CTA"]] - D.loc[["KMLM", "CTA"]]
print("R8o - R8n, pp (canonical mean, min, max):",
      {t: (round(float((d.loc[t].values * w).sum() / w.sum()), 2), round(d.loc[t].min(), 2), round(d.loc[t].max(), 2)) for t in ("KMLM", "CTA")})

# ===== build_decks.py =====
"""Early deck candidates for Addendum 55b (2026-10-24); the parity deck here is on R8m and is superseded by build_r8o.py. Writes, under data/decks/ and data/:
  Deck_R8mP_20261024.csv        R8m with the D3 trend-parity correction: KMLM = DBMF + 0.05, CTA = DBMF - 0.09 (gross rows), so the
                                three trend rows carry the same net-of-ER return in every scenario (DBMF row unchanged).
  Deck_R8m_STK_20261024.csv     R8m + RSST synthetic row (Addendum 53 formula: IVV + (DBMF - BIL) - 0.30 gross; ER 0.99)
  Deck_R8mP_STK_20261024.csv    R8mP + the same RSST row
  panel_stk_20261024.csv        etf-eod monthly panel + synthetic RSST month = IVV + DBMF - BIL - 0.30/1200 (Addendum 53)
  Row_Grades_R8j_STK.csv        R8j grades + RSST grade B;  ER_R8.json gets RSST 0.99 (new key only; no existing value moves)
"""
import json, os, pandas as pd
H = os.path.dirname(os.path.abspath(__file__)); DK = os.path.join(H, "data", "decks")
D = pd.read_csv(os.path.join(DK, "Deck_Recal_R8m_20261006.csv")); D["Ticker"] = D["Ticker"].str.strip(); D = D.set_index("Ticker")
N = D.copy()
N.loc["KMLM"] = D.loc["DBMF"] + 0.05
N.loc["CTA"] = D.loc["DBMF"] - 0.09
N.reset_index().to_csv(os.path.join(DK, "Deck_R8mP_20261024.csv"), index=False)
for tag, X in (("R8m", D), ("R8mP", N)):
    Y = X.copy(); Y.loc["RSST"] = X.loc["IVV"] + (X.loc["DBMF"] - X.loc["BIL"]) - 0.30
    Y.reset_index().to_csv(os.path.join(DK, f"Deck_{tag}_STK_20261024.csv"), index=False)
M = pd.read_csv("/home/claude/etf-eod/monthly_returns.csv", index_col=0)
M["RSST"] = M["IVV"] + M["DBMF"] - M["BIL"] - 0.30 / 1200.0
M.to_csv(os.path.join(H, "data", "panel_stk_20261024.csv"))
G = pd.read_csv(os.path.join(DK, "Row_Grades_R8j_20260909.csv"))
if "RSST" not in set(G.Ticker):
    G = pd.concat([G, pd.DataFrame({"Ticker": ["RSST"], "Grade": ["B"], "Meaning": ["synthetic stacked row (Addendum 53)"]})])
G.to_csv(os.path.join(DK, "Row_Grades_R8j_STK.csv"), index=False)
er = json.load(open(os.path.join(DK, "ER_R8.json"))); er.setdefault("RSST", 0.99)
json.dump(er, open(os.path.join(DK, "ER_R8.json"), "w"), indent=1)
d = (N.loc[["KMLM", "CTA"]] - D.loc[["KMLM", "CTA"]])
print("R8n minus R8m (pp), pi-canonical-weighted mean / min / max:")
w = pd.read_csv(os.path.join(DK, "scenarios_25_R8i_20260909.csv"))["Weight_Normalized"].values
for t in ("KMLM", "CTA"): print(t, round(float((d.loc[t].values * w).sum() / w.sum()), 2), round(d.loc[t].min(), 2), round(d.loc[t].max(), 2))
print("RSST R8m row (Base, Taiwan, min):", round(D.loc["IVV"].iloc[0] + D.loc["DBMF"].iloc[0] - D.loc["BIL"].iloc[0] - 0.3, 2),
      round(D.loc["IVV"].iloc[11] + D.loc["DBMF"].iloc[11] - D.loc["BIL"].iloc[11] - 0.3, 2))

# ===== assemble_r55.py =====
"""Assemble the Addendum 55b engine-exact report (one row per deck x book) and the fast-screen file for the Project."""
import pandas as pd, numpy as np
O = "/home/claude/engine/out/"
KEEP = ["geo x", "real CAGR %", "g3 CE-CAGR %", "g5 CE-CAGR %", "P5 terminal x", "P5 path CAGR %", "P(W50<1) %", "worst scen %",
        "stressed worst %", "gates", "constraints", "U1 geo x", "U1 real CAGR %", "U1 g3 CE-CAGR %", "U1 g5 CE-CAGR %", "U1 P5 x",
        "U1 P5 path CAGR %", "U1 P(W<1) %", "Uhalf g3 CE-CAGR %", "Uhalf P5 x", "U2 g3 CE-CAGR %", "U2 P5 x", "U2 P(W<1) %",
        "err sd pp", "CVaR10 %", "CVaR15 %", "sev5 mean %", "2nd worst %", "lifetime tax x", "loc geo x", "loc g3 CE-CAGR %",
        "loc P5 x", "loc lifetime tax x", "ER %", "holdings", "largest RS %", "gold RS %", "defence RS %", "ballast wt %", "weights"]
parts = []
for f, deck in (("exact_R8n_final.csv", "R8n"), ("exact_R8n_g2m.csv", "R8n"), ("exact_R8n_g2ma.csv", "R8n"), ("exact_R8o.csv", "R8o"),
                ("exact_R8o_g2ma.csv", "R8o"), ("exact_R8m_chk.csv", "R8m"), ("exact_R8m_g2ma.csv", "R8m")):
    d = pd.read_csv(O + f, index_col=0); d.insert(0, "deck", deck); parts.append(d)
X = pd.concat(parts)
M1 = pd.concat([pd.read_csv(O + "exact_R8n_final_M1.csv", index_col=0).assign(deck="R8n"),
                pd.read_csv(O + "exact_R8n_g2ma_M1.csv", index_col=0).assign(deck="R8n"),
                pd.read_csv(O + "exact_R8o_M1.csv", index_col=0).assign(deck="R8o")])
X = X.reset_index().rename(columns={"index": "Port"})
m = M1.reset_index().rename(columns={"index": "Port"})[["Port", "deck", "geo x", "g3 CE-CAGR %", "P5 terminal x", "gates"]]
m.columns = ["Port", "deck", "M1 geo x", "M1 g3 CE-CAGR %", "M1 P5 x", "M1 gates"]
X = X.merge(m, on=["Port", "deck"], how="left")
cols = ["Port", "deck"] + [c for c in KEEP if c in X.columns] + ["M1 geo x", "M1 g3 CE-CAGR %", "M1 P5 x", "M1 gates"]
X = X[cols]
# R8o rows for books without KMLM/CTA equal their R8n rows (robust pi frozen at R8n's): copy for completeness
for nm in ("C5_SMH3b", "G2", "HS_B"):
    if not ((X.Port == nm) & (X.deck == "R8o")).any():
        r = X[(X.Port == nm) & (X.deck == "R8n")].copy(); r["deck"] = "R8o (= R8n: no KMLM/CTA)"; X = pd.concat([X, r])
X.to_csv(O + "Report_T50_R55_20261024.csv", index=False)
S = []
for f, deck in (("scr_A54.csv", "R8n"), ("scr_A54_r55.csv", "R8n"), ("scr_A54_g2mv.csv", "R8n"), ("scr_R8o_r55.csv", "R8o"),
                ("scr_R8o_g2mv.csv", "R8o"), ("scr_R8m_g2mv.csv", "R8m"), ("scr_R8m.csv", "R8m")):
    d = pd.read_csv(O + f, index_col=0); d = d[~d.index.str.endswith("_h")]; d.insert(0, "deck", deck); S.append(d)
S = pd.concat(S).reset_index().drop_duplicates(subset=["Port", "deck"]).set_index("Port")
S.to_csv(O + "Screen_R55_fast_20261024.csv")
print(X[["Port", "deck", "geo x", "g3 CE-CAGR %", "P5 terminal x", "U1 geo x", "U1 g3 CE-CAGR %", "U1 P5 x", "U2 g3 CE-CAGR %",
         "M1 g3 CE-CAGR %", "gates", "M1 gates"]].to_string(index=False))
print(len(X), "exact rows;", len(S), "fast rows")

# ===== env_stk.sh =====
# Addendum 55b environment (R8m checks): v3.4 gates, T=50, cap 20%, stacked extras present (RSST appended after the base universe).
# DECK selects R8m (Deck_R8m_STK_20261024.csv) or R8mP = R8m + trend parity (Deck_R8mP_STK_20261024.csv + PI_ROBUST_FILE frozen at R8m).
cd /home/claude/engine; export PYTHONPATH=/home/claude/engine
export ENGINEV2_DATA=/home/claude/engine/data SCEN_FILE=scenarios_25_R8i_20260909.csv \
       GRADES_FILE=/home/claude/engine/data/decks/Row_Grades_R8j_STK.csv \
       ENGINEV2_PANEL=/home/claude/engine/data/panel_stk_20261024.csv ENGINEV2_END=2026-08 NPATHS=6000 \
       ENGINEV2_T=50 STRESS_GATE=1 ETF_CAP=0.20 ENGINEV2_EXTRA=RSST OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "${DECK:-R8m}" = "R8mP" ]; then export DECK_FILE=Deck_R8mP_STK_20261024.csv PI_ROBUST_FILE=/home/claude/engine/data/decks/pi_robust_R8m_20261006.csv
else export DECK_FILE=Deck_R8m_STK_20261024.csv; unset PI_ROBUST_FILE; fi

# ===== env_a54.sh =====
# Addendum 54 environment: the other session's deck R8n (2026-10-24) and conventions on this engine. v3.4 gates, T=50, cap 20%,
# stressed-floor gate on; stacked extras RSST,RSSB,NTSX,GDE and trend-half rows DBMFh,RSSTh appended after the base universe.
#   deck   Deck_R8n_A54_20261024.csv = their Deck_R8n_20261024.csv byte for byte (R8m + D3 trend-shape correction on
#          DBMF/KMLM/CTA/CTA_OVERLAY + D6 stacked rows with BIL+0.30 financing and the -0.50/-0.25 pp structure haircut)
#   grades Row_Grades_R8j_A54.csv = R8j + RSST/RSSB/GDE/RSSTh grade C, NTSX/DBMFh grade B
#   panel  panel_a54_20261024.csv = etf-eod 2026-10-20 pull + their synthetic monthly series (financing spread only)
#   ER     ER_R8.json (new keys only): RSST 0.99, RSSB 0.39, NTSX 0.20, GDE 0.20, DBMFh 0.85, RSSTh 0.99
#   STK_A54=1 with a54_hook first on PYTHONPATH: stk.py is imported right after standing, also under plain report_v3.py and
#          search_v3.py: RSST in core.BAL/BALLAST/trend stress, RSSB in core.BAL, GDE in the gold block with 0.9 x the gold stress,
#          EXCL removed from the search pool. Prefix PREF=1 for the owner-preference constraints (gold-linked <= 15%, GDE 0.9).
# M1 co-equal gate (D7): a book is admissible only if it PASSes both runs, as is (48-month sigma) and with SIGMA_LEVEL=cov120:
#   python3 report_v3.py books_a54.json out/x_vol48.csv;  SIGMA_LEVEL=cov120 python3 report_v3.py books_a54.json out/x_M1.csv
# PI_ROBUST is recomputed from the R8n deck (not frozen). NPATHS=2000 is their search setting; report_v3 always runs 60k paths.
cd /home/claude/engine; export PYTHONPATH=/home/claude/engine/a54_hook:/home/claude/engine
export ENGINEV2_DATA=/home/claude/engine/data DECK_FILE=Deck_R8n_A54_20261024.csv SCEN_FILE=scenarios_25_R8i_20260909.csv \
       GRADES_FILE=/home/claude/engine/data/decks/Row_Grades_R8j_A54.csv \
       ENGINEV2_PANEL=/home/claude/engine/data/panel_a54_20261024.csv ENGINEV2_END=2026-08 NPATHS=2000 \
       ENGINEV2_T=50 STRESS_GATE=1 ETF_CAP=0.20 ENGINEV2_EXTRA=RSST,RSSB,NTSX,GDE,DBMFh,RSSTh STK_A54=1 \
       EXCL=DBMFh,RSSTh,QTUM,SPGI,WM,CEG,ICE,CBOE,NDAQ,V,COIN,IBKR,WPM \
       OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PI_ROBUST_FILE DECK_SHIFT STRESS_DECK ENGINEV2_RPRX SIGMA_LEVEL PREF NO_STACK
# Addendum 55b additions (this session): fast-evaluator paths 6000 for screens/climbs; DECK=R8o selects R8n + KMLM/CTA parity
# (KMLM = DBMF + 0.05, CTA = DBMF - 0.09 gross on the reshaped rows) with the robust pi frozen at R8n's.
export NPATHS=6000
if [ "${DECK:-R8n}" = "R8o" ]; then export DECK_FILE=Deck_R8o_20261024.csv PI_ROBUST_FILE=/home/claude/engine/data/decks/pi_robust_R8n_A54.csv; fi

# ===== a54_hook/sitecustomize.py =====
"""Addendum 54 hook. env_a54.sh puts this directory first on PYTHONPATH. With STK_A54=1, the first `import standing` also imports
`stk` right after standing has run, so plain `python3 report_v3.py ...` / `python3 search_v3.py ...` get the same stacked-fund
wiring that report_u.py / search_u.py get by importing stk themselves. Without STK_A54=1 this file only chains the system
sitecustomize (if any) and does nothing else."""
import os, sys, importlib.abc, importlib.machinery

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.machinery.PathFinder.find_spec("sitecustomize", [p for p in sys.path if os.path.abspath(p or os.curdir) != _here])
if _spec is not None and _spec.origin and os.path.abspath(_spec.origin) != os.path.abspath(__file__):
    with open(_spec.origin, encoding="utf-8") as _f:
        exec(compile(_f.read(), _spec.origin, "exec"), {"__name__": "sitecustomize", "__file__": _spec.origin})


class _StkAfterStanding(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "standing":
            return None
        sys.meta_path.remove(self)                      # one-shot
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        exec_base = spec.loader.exec_module

        def exec_module(module):
            exec_base(module)
            import stk  # noqa: F401  (applies the stacked / STK_A54 wiring once standing is fully executed)

        spec.loader.exec_module = exec_module
        return spec


if os.environ.get("STK_A54", "0") == "1":
    sys.meta_path.insert(0, _StkAfterStanding())

# ===== lib_pathce.py (2026-10-24 addition, env-gated; default behaviour unchanged) =====
74:if os.environ.get("PI_ROBUST_FILE"):   # 2026-10-24 (Addendum 55b): freeze the robust pi from a file, e.g. R8m's pi under the R8n
75:    PI_ROBUST = np.loadtxt(os.environ["PI_ROBUST_FILE"], delimiter=","); PI_ROBUST = PI_ROBUST / PI_ROBUST.sum()   # parity deck
76-PI_CANON = W_CAN / W_CAN.sum()
77-ST_W = stress_weights()
# ===== uncertainty.py zero-error check (t_u0.py) =====
import json, time, numpy as np, warnings; warnings.filterwarnings("ignore")
import standing as ST, stk, search_B as S, uncertainty as U, policy_engine2 as E, lib_pathce as L
from core import wvec, DECKM, ERV, UNIV
bk = json.load(open("books_ref.json")); w = wvec(bk["C5_SMH3b"]); r = DECKM.T @ w - float(ERV @ w)
mx = 0.0
for pin in ("robust", "canonical"):
    sigv, _ = ST.sigma_vec(w, pin)
    for conv in ("G", "L"):
        a = ST.run_fast(r, sigv, pin, conv); b = U.run_fast_u(r, sigv, pin, conv, np.zeros(S.FE.npaths), np.zeros(S.FE.npaths), np.ones(S.FE.npaths))
        mx = max(mx, max(abs(a[k] - b[k]) for k in ("PathCE", "CE3", "CE5", "P5", "P10")))
print("fast U0 vs standing.run_fast max abs diff:", mx)
e = ST.evaluate_fast(w); print("central fast C5_SMH3b: geo %.4f ce3 %.4f P5 %.4f worst %.4f fails %s" % (e["geo"], e["ce3"], e["P5"], e["worst"]*100, e["fails"]))
D = U.Draws(S.FE.npaths, 2718)
for nm in ("C5_SMH3b", "K7", "HS_B", "XAR8_XLU4_QTUM3"):
    w = wvec(bk[nm]); sd = D.book_sd(w)
    out = {tag: U.evaluate_u(w, D, k, mix) for tag, k, mix in (("U1", 1, True), ("U1nomix", 1, False), ("Uhalf", .5, True), ("U2", 2, True))}
    print(nm, "book error sd n/s pp: %.2f/%.2f" % (sd[0]*100, sd[1]*100), {t: "geo %.3f ce3 %.3f P5 %.3f pW1 %.3f" % (o["geo"], o["ce3"], o["P5"], o["pW1"]) for t, o in out.items()})
t0 = time.time(); w = wvec(bk["C5_SMH3b"]); sig = {p: ST.sigma_vec(w, p)[0] for p in ("robust", "canonical")}
a = E.suite(r, sig, E.POL); b = U.suite_u(r, sig)
print("exact U0 vs policy_engine2.suite max abs diff:", max(abs(a[c][k] - b[c][k]) for c in a for k in ("PathCE", "P5", "P10")), "%.0fs" % (time.time() - t0))
