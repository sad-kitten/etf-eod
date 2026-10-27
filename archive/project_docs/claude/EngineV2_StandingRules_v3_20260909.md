# EngineV2 Standing Rules v3 (2026-09-09) — Addendum 31 items 1–12, as implemented

Code: `standing.py` (constraints, absolute gates, regime sigma, reporting set), `search_v3.py` (climb on the reported criterion under the rules), `report_v3.py` (the reporting set, engine-exact), `build_r8j.py` (row-admission rule), `Reaudit_Schedule_20260909.md`. Every rule below is enforced in code, not by convention.

## 1. Objective (items 1, 2)
Primary criterion: **CE-CAGR at γ = 3** on basis (B), engine-exact, 4 conventions × 60k paths. Reported beside γ = 1 (expected wealth, the former geo-PathCE) and γ = 5. **Decision rule: a book must be non-dominated on the (γ = 1, γ = 3) pair among admissible, gate-passing books; the owner picks the point.** The hand-built surprise penalty (SACE) is retired as a criterion; its constraint-like ingredients live in §3. Every search climbs on the criterion it reports (`OBJ=CE3` or `OBJ=GEO`), from multiple seeds, under §3 and §2 simultaneously.

## 2. Absolute gates (items 3, 4, 5) — `standing.GATES`
On basis (B), robust-G convention, current deck:
| gate | threshold | basis |
|---|---|---|
| ruin paths | 0 | base deck |
| P5 terminal | ≥ 1.00 × corpus, real | base deck — the bad decile keeps its purchasing power after 40 years of drawing |
| worst scenario net CAGR | ≥ 5.5% | base deck |
| R3AF vs EVG11 | ≥ −0.002 | base deck |
| −200 bp deck | ruin 0, P5 ≥ 0.45 | stress |
| C-haircut deck (HCC) | ruin 0, P5 ≥ 0.85 | stress |
Not gated, reported: DRB40@6% (knife-edged — a 1% trade doubled it), RCEW40 (a return measure, not a safety one; gating it excluded legitimate reliability books), and the smooth replacement **shortfall6** = π-weighted expected shortfall of scenario net CAGR below 6%. **Headline level = the −100 bp column**; −200 bp remains the stress gate. Gates are absolute — nothing is measured against P1 any more.

## 3. Standing constraints (items 6, 7, 8) — `standing.violations`
| constraint | value |
|---|---|
| feasibility (unchanged) | long-only, ETF ≤ 15%, singles ≤ 5% / 20% aggregate, semis ≤ 12%, defence ≤ 35%, nuclear ≤ 35% weight, ballast set ≥ 20%, PM-equity ≤ 10%, ≤ 14 names, 1% grid, 2% minimum |
| nuclear complex risk share | ≤ 38% of book variance |
| any single holding risk share | ≤ 24% |
| any theme block (defence, gold, cyber, semis) | ≤ 35% of variance |
| grade-C weight | ≤ 20% |
| **weight in funds with < 60 months of history** | **≤ 30%** |
| single names (BRKB exempt) | ≤ 3 |
| **ballast floor: DBMF + CTA + KMLM** | **≥ 9%** (until §5 makes it unnecessary) |
| duplicates | two books are one book if < 3% of the portfolio must move |

## 4. Deck and rows (items 9, 10) — `build_r8j.py`, `Reaudit_Schedule_20260909.md`
Admission: beta- and geometric-consistency audit before any row enters; alpha zeroed unless t ≥ 2; over-claims corrected at half the gap; under-claims left, but **reported as the symmetric-deck sensitivity** (CME +3.74, MSCI +2.32, BRKB +0.92, FNV +1.70 — deck R8j_sym, never used for selection). **Rows on < 36 months are grade C unless anchored** (a grade-A/B sector fund with ≥ 120 months and correlation ≥ 0.5), and in either case the premium over the highest-correlation anchor is capped at **+1.0 pp** (parallel shift, shape preserved). Applied on R8j: **NUKZ 10.24 → 8.65** (anchor NLR, corr 0.88, premium +2.59 capped); SHLD unchanged (anchor PPA, premium +0.51, stays B); JEDI / EUAD / IBIT grade C, no change. Re-audit schedule in the file named above; every re-audit is a deck revision and every deck revision re-runs gates on all standing books.

## 5. Method (item 11) — `standing.sigma_vec`
Regime-conditional volatility: covariance estimated separately in bottom-tercile equity months and the rest (full pairwise panel; short-history names represented by their sector anchor in the conditional panel; ratio shrunk halfway toward 1). Scenarios of severity ≥ 4 use the stress σ, others the normal σ, rescaled so the π-weighted variance equals the standing 48-month σ — the *level* is unchanged, only its allocation across states. Result: book stress/normal ratios of 1.09–1.15 after shrinkage, nearly uniform across books. **The channel is real but small**: managed futures' crisis value already enters through their scenario rows (DBMF's Credit row 6.73 vs Base 6.09), not through covariance. The ballast floor in §3 therefore stays.

## 6. Reporting set (items 4, 12) — `report_v3.py`
With every recommendation: the dial (γ = 1/3/5 CE-CAGR), P5 terminal and P5/P25 path CAGR, −100 bp headline, −200 bp and HCC stress, symmetric-deck sensitivity, worst scenario, shortfall6, DRB40 and RCEW40 (reported), gates, constraints, risk shares by block, grade-C and < 60-month weight, ballast weight, singles, holdings, ER, lifetime tax, vol48, stress-σ ratio. **Tie bands: 0.03 × on geo, 0.02 pp/yr on γ = 3 CE-CAGR** — inside them, choose on cost and structure, never on the number. Draw sensitivity (retained CAGR at 2.5% vs 3.0%) is reported on request via `policy_engine2.POL`.
