"""Account-aware policy engine (2026-09-04): the owner's actual account sizes (note-to-self, 2026-08-28),
2026 federal brackets held fixed in real terms (they are CPI-indexed), Missouri 4.7% with the TY2025+ 100%
capital-gains subtraction (HB 594), inherited-IRA 10-year rule with bracket-fill distributions, Roth + HSA
as a tax-free reserve drawn last. Simulation is in REAL dollars: g_real = g_nom / (1+INFL); the brokerage
cost basis is nominal, so its real value decays by 1/(1+INFL) each year (the inflation tax on gains).
Same scenario Markov process, seeds and conventions as lib_pathce.run_engine / policy_engine.py.

Run: R7=1 R8=1 DECK_FILE=Deck_Recal_R8b_20260903.csv SCEN_FILE=scenarios_24_R6_20260831.csv python3 policy_engine2.py
"""
import json, sys
import numpy as np, pandas as pd
import lib_pathce as L
from core import DECKM, ERV, sigma_of, wvec, CONVS

INFL = 0.025
ACCT = dict(ira=384120.14 + 67216.37,                      # IRA_1 + IRA_3 — both treated as inherited (see note)
            brk=199677.93 + 596544.64 + 129629.35,         # Brokerage_v + _s + _f (taxable)
            roth=500318.19, hsa=122292.16)                 # tax-free reserve: Roth (qualified), HSA (medical)
W0 = float(sum(ACCT.values()))                             # 1,999,798.78
DRAW0 = 0.03 * W0                                          # real 3% of initial corpus (spending need, pre-tax)
FLEX = dict(lo=0.80, cut=0.75, hi=1.00)                    # guardrail on total REAL wealth vs initial
FED = {  # tax year 2026 (Rev. Proc. 2025-32); taxable-income thresholds; held fixed in real terms
    "single": dict(std=16100, std65=2050, cg0=49450, cg15=545500,
                   br=[(12400, .10), (50400, .12), (105700, .22), (201775, .24), (256225, .32), (640600, .35), (np.inf, .37)]),
    "mfj":    dict(std=32200, std65=3300, cg0=98900, cg15=613700,
                   br=[(24800, .10), (100800, .12), (211400, .22), (403550, .24), (512450, .32), (768700, .35), (np.inf, .37)])}
MO_RATE = 0.047   # Missouri individual rate (TY2025); capital gains 100% subtracted for individuals from TY2025; dividends taxed
POL = dict(status="mfj", fill=0.10, ira_years=10, gain0=0.50, y_ord=0.010, y_qual=0.005, age65_year=7, amort_r=0.05,
           rmd_f0=27.1, ss_start=11, ss=45000.0, ss_taxable=0.85, ss_offsets=True)
# Owner's answers (2026-09-04): both IRAs inherited and on RMDs; married filing jointly; Social Security from age 70;
# spending stays 3% real and Social Security reduces the corpus draw from year 11 (ss_offsets=True = reporting basis).
# fill: IRA distributions each year fill ordinary taxable income to the top of this bracket (and at least the level
#       payment that empties the IRA by ira_years, at least the RMD floor, at least the spending shortfall) — all out in
#       year ira_years.  rmd_f0: single-life factor at age 60 (2022 table), reduced by 1 each year (annual RMD floor).
# gain0: unrealized-gain share of the brokerage at start (unknown; 0.50 assumed; irrelevant at this draw)
# y_ord/y_qual: brokerage distribution yield (ordinary / qualified) with the book held pro rata in every account
# age65_year: extra standard deduction from this year.  ss: owner's Social Security at 70 in today's dollars (PLACEHOLDER —
#       replace with the SSA age-70 estimate), starting in year ss_start (age 70 = 2037 = year 11); ss_taxable share counts as
#       federal ordinary income (Missouri exempts SS).  ss_offsets: True = spending fixed at 3% real and SS reduces the corpus
#       draw; False = SS is additional spending, the corpus draw stays 3% real and pays only the tax SS adds.

def bracket_tax(ti, br):
    tax = np.zeros_like(ti); lo = 0.0
    for top, rate in br:
        tax += np.clip(ti - lo, 0.0, top - lo) * rate; lo = top
    return tax

def fed_tax(ordinc, pref, P, std):
    """Federal tax: ordinary income through the brackets, preferential income (qualified dividends + LTCG)
    stacked on top at 0/15/20%; the standard deduction offsets ordinary income first, then preferential."""
    ti_ord = np.maximum(ordinc - std, 0.0)
    ti_pref = np.maximum(pref - np.maximum(std - ordinc, 0.0), 0.0)
    lo = ti_ord; hi = ti_ord + ti_pref
    a15 = np.maximum(0.0, np.minimum(hi, P["cg15"]) - np.maximum(lo, P["cg0"]))
    a20 = np.maximum(0.0, hi - np.maximum(lo, P["cg15"]))
    return bracket_tax(ti_ord, P["br"]) + 0.15 * a15 + 0.20 * a20

def mo_tax(ordinc, qual, std):
    return MO_RATE * np.maximum(ordinc + qual - std, 0.0)      # capital gains exempt; dividends and IRA income taxed

def run(rng, r_net, pi, conv, sigma, pol=POL):
    NP_, T, LAM = L.NP_, L.T, L.LAM
    P = FED[pol["status"]]; fill_top = [top for top, rate in P["br"] if rate == pol["fill"]][0]
    N = pol["ira_years"]; r = pol["amort_r"]
    s = rng.choice(len(pi), size=NP_, p=pi)
    ira = np.full(NP_, ACCT["ira"]); brk = np.full(NP_, ACCT["brk"]); roth = np.full(NP_, ACCT["roth"] + ACCT["hsa"])
    basis = brk * (1.0 - pol["gain0"])
    ru = np.zeros(NP_, bool); red = np.zeros(NP_, bool); brk_out = np.zeros(NP_, bool); roth_tap = np.zeros(NP_, bool)
    yrs_red = np.zeros(NP_); tax_paid = np.zeros(NP_); tax_yr = np.zeros(T); ira_yr = np.zeros(T)
    for t in range(1, T + 1):
        j = rng.random(NP_) < LAM
        if j.any(): s[j] = rng.choice(len(pi), size=int(j.sum()), p=pi)
        mu = np.log1p(r_net[s]) - (sigma * sigma / 2 if conv == "G" else 0.0)
        g = np.exp(mu + sigma * rng.standard_normal(NP_)) / (1.0 + INFL)       # real gross return, all accounts
        d_ord = pol["y_ord"] * brk; d_qual = pol["y_qual"] * brk                # brokerage distributions (cash out)
        rmd = ira / max(pol["rmd_f0"] - (t - 1), 1.0)                            # annual RMD floor on prior year-end balance
        ira = ira * g; roth = roth * g; brk = np.maximum(brk * g - d_ord - d_qual, 0.0)
        basis = basis / (1.0 + INFL)
        W = ira + brk + roth
        red = np.where(W < FLEX["lo"] * W0, True, np.where(W >= FLEX["hi"] * W0, False, red))
        need = DRAW0 * np.where(red, FLEX["cut"], 1.0); yrs_red += red
        std = P["std"] + (P["std65"] if t >= pol["age65_year"] else 0.0)
        ss = pol["ss"] if t >= pol["ss_start"] else 0.0                          # Social Security (real), from age 70
        ss_ord = pol["ss_taxable"] * ss                                          # federally taxable part; Missouri exempt
        # inherited IRA: bracket-fill / level payment / RMD floor / cover-the-need, all out by year N
        if t < N:
            n_left = N - t + 1
            room = np.maximum(0.0, std + fill_top - d_ord - ss_ord)
            amort = ira * r / (1.0 - (1.0 + r) ** (-n_left))
            need_gross = np.maximum(0.0, need - d_ord - d_qual - (ss if pol["ss_offsets"] else 0.0)) / (1.0 - pol["fill"] - MO_RATE)
            take = np.minimum(ira, np.maximum(np.maximum(np.maximum(room, amort), need_gross), rmd))
        elif t == N:
            take = ira.copy()
        else:
            take = np.zeros(NP_)
        ira = ira - take; ira_yr[t - 1] = take.mean()
        ordinc = d_ord + take                                                    # corpus-sourced ordinary income
        gf = np.where(brk > 1e-9, np.clip(1.0 - basis / np.maximum(brk, 1e-9), 0.0, 1.0), 0.0)
        base_cash = d_ord + d_qual + take + (ss if pol["ss_offsets"] else 0.0)
        def total_tax(S):
            t_all = fed_tax(ordinc + ss_ord, d_qual + S * gf, P, std) + mo_tax(ordinc, d_qual, std)
            if pol["ss_offsets"] or ss == 0.0: return t_all
            return fed_tax(ordinc, d_qual + S * gf, P, std) + mo_tax(ordinc, d_qual, std)   # SS pays its own incremental tax
        S = np.zeros(NP_)
        for _ in range(20):                                                    # fixed point: sale S covers need + tax (contraction <= 0.2)
            S = np.maximum(0.0, need + total_tax(S) - base_cash)
        S = np.minimum(S, brk)
        tax = total_tax(S)
        cash = base_cash + S - tax
        surplus = np.maximum(cash - need, 0.0); short = np.maximum(need - cash, 0.0)
        basis = basis * np.where(brk > 1e-9, 1.0 - S / np.maximum(brk, 1e-9), 1.0); brk = brk - S
        brk = brk + surplus; basis = basis + surplus                         # surplus reinvested at full basis
        take_r = np.minimum(short, roth); roth = roth - take_r; roth_tap |= take_r > 1.0
        ru |= (short - take_r) > 1.0
        brk_out |= brk <= 1.0   # taxable account exhausted at some point
        tax_paid += tax; tax_yr[t - 1] = tax.mean()
        ira = np.maximum(ira, 0.0); brk = np.maximum(brk, 0.0); roth = np.maximum(roth, 0.0); basis = np.minimum(basis, brk)
    W = (ira + brk + roth) / W0
    lg = np.log(np.maximum(W, 1e-9))
    return dict(PathCE=float(np.exp(lg.mean())), P10=float(np.quantile(W, .10)), P5=float(np.quantile(W, .05)),
                ruin=int(ru.sum()), yrs_red=float(yrs_red.mean() / T), tax_paid=float(tax_paid.mean() / W0),
                tax_yr=tax_yr, ira_yr=ira_yr, brk_out=float(brk_out.mean()), roth_tap=float(roth_tap.mean()),
                roth_share=float((roth / W0 / np.maximum(W, 1e-9)).mean()))

def suite(r_net, sigma, pol=POL):
    out = {}
    for pin, pi in (("robust", L.PI_ROBUST), ("canonical", L.PI_CANON)):
        rng = np.random.default_rng(42)
        for conv in ("G", "L"):
            out[(pin, conv)] = run(rng, r_net, pi, conv, sigma, pol)
    return out

def geo(mc): return float(np.exp(np.mean([np.log(mc[c]["PathCE"]) for c in CONVS])))

def row(nm, mc, pol):
    rg = mc[("robust", "G")]
    return dict(Port=nm, variant=pol.get("name", "base"), geo=round(geo(mc), 2), P5=round(rg["P5"], 2), ruin=rg["ruin"],
                yrs_reduced_pct=round(rg["yrs_red"] * 100, 1), tax_total_share_W0=round(rg["tax_paid"], 3),
                tax_yr1_10_mean=round(float(rg["tax_yr"][:10].mean())), tax_yr11_40_mean=round(float(rg["tax_yr"][10:].mean())),
                ira_dist_yr1_10_mean=round(float(rg["ira_yr"][:10].mean())), brk_exhausted_pct=round(rg["brk_out"] * 100, 2),
                roth_tapped_pct=round(rg["roth_tap"] * 100, 2), roth_share_of_terminal=round(rg["roth_share"], 2))

if __name__ == "__main__":
    books = json.load(open("pkg/enginev2/books.json"))
    books["R3b_B5"] = {'GLDM':.15,'SHLD':.15,'DBMF':.15,'NUKZ':.11,'CIBR':.08,'GRID':.05,'CME':.05,'BRKB':.05,'CTA':.05,'URNM':.03,'XAR':.03,'CEG':.03,'FNV':.04,'MSCI':.03}
    NAMES = ["P2", "R3b", "R3b_B5", "R3dK", "R1b_final", "R2_final"]
    rows = []
    print(f"W0 = {W0:,.2f}  draw0 = {DRAW0:,.2f}  shares: ira {ACCT['ira']/W0:.3f} brk {ACCT['brk']/W0:.3f} roth+hsa {(ACCT['roth']+ACCT['hsa'])/W0:.3f}", flush=True)
    # Reporting basis (B): SS offsets the draw. Columns as in Addendum 9: base, -200bp level stress, stress volatility
    # (analog-ratio vol from Policy_Basis_20260903.csv), surprise-adjusted CE (same recipe as Addenda 4-9); (A) kept for reference.
    SV = pd.read_csv("out/Policy_Basis_20260903.csv").set_index("Port")["stress_vol_pct"] / 100.0
    THEM = ['SHLD','NUKZ','CIBR','GRID','XAR','URNM','CEG','JEDI']; SING = ['CME','MSCI','CEG','CBOE','BRKB','FNV']
    S4 = {"P2":0.06,"R3b":0.10,"R3b_B5":0.10,"R3dK":0.12,"R1b_final":0.10,"R2_final":0.08}
    ADDL = dict(name="(A) SS additional spending", ss_offsets=False)
    for nm in NAMES:
        d = books[nm]; w = wvec(d); r = DECKM.T @ w - float(ERV @ w); sig = sigma_of(w)
        mc = suite(r, sig, POL); rows.append(row(nm, mc, POL)); print(rows[-1], flush=True)
        if nm == "R3b":
            rg = mc[("robust", "G")]
            print("R3b mean tax by year 1-14:", np.round(rg["tax_yr"][:14]), "\n    mean IRA dist by year 1-10:", np.round(rg["ira_yr"][:10]), flush=True)
        pol = dict(POL); pol["name"] = "(B) -200bp"
        mcL = suite(r - 0.02, sig, pol); rows.append(row(nm, mcL, pol)); print(rows[-1], flush=True)
        pol = dict(POL); pol["name"] = f"(B) stress vol {SV[nm]*100:.2f}%"
        mcS = suite(r, float(SV[nm]), pol); rows.append(row(nm, mcS, pol)); print(rows[-1], flush=True)
        wt = sum(d.get(t, 0) for t in THEM); ws = sum(d.get(t, 0) for t in SING)
        slope = (np.log(geo(mc)) - np.log(geo(mcL))) / 2
        pen = 0.6 * slope + 0.8 * wt * slope + S4[nm] + 0.30 * slope + 0.25 * ws
        rows.append(dict(Port=nm, variant="(B) surprise-adjusted CE", geo=round(geo(mc) * np.exp(-pen), 2), P5=np.nan, ruin=np.nan)); print(rows[-1], flush=True)
        pol = dict(POL); pol.update(ADDL)
        mc = suite(r, sig, pol); rows.append(row(nm, mc, pol)); print(rows[-1], flush=True)
    VARS = [dict(name="(B) SS=30k", ss=30000.0), dict(name="(B) SS=60k", ss=60000.0),
            dict(name="(B) location-aware yield", y_ord=0.001, y_qual=0.007),
            dict(name="(B) fill=12%", fill=0.12), dict(name="(A) location-aware yield", ss_offsets=False, y_ord=0.001, y_qual=0.007)]
    for nm in ["R3b", "R3dK"]:
        d = books[nm]; w = wvec(d); r = DECKM.T @ w - float(ERV @ w); sig = sigma_of(w)
        for v in VARS:
            pol = dict(POL); pol.update(v)
            mc = suite(r, sig, pol); rows.append(row(nm, mc, pol)); print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("out/Policy_Accounts_20260904.csv", index=False)
    pd.set_option("display.width", 250); print(df.to_string())
