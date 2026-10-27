# EngineV2 Standing Rules v3.1 (2026-09-10) — v3 with the Addendum 33 gate set ratified

Supersedes `EngineV2_StandingRules_v3_20260909.md`. Only §2 changed; §1 and §3–§6 are restated so this is the single current statement. Code: `standing.py` (constraints, gates, regime sigma, reporting set), `search_v3.py`, `report_v3.py`, `build_r8j.py`, `Reaudit_Schedule_20260909.md`. Every rule is enforced in code.

## 1. Objective (unchanged)
Primary criterion: **CE-CAGR at γ = 3** on basis (B), engine-exact, 4 conventions × 60k paths, reported beside γ = 1 (expected wealth) and γ = 5. **Decision rule: a book must be non-dominated on the (γ = 1, γ = 3) pair among admissible, gate-passing books; the owner picks the point.** Every search climbs on the criterion it reports (`OBJ=CE3` or `OBJ=GEO`) from **all eight seeds** (see Addendum 34 §3 for why this is now stated), under §2 and §3 simultaneously.

## 2. Absolute gates — v3.1, ratified 2026-09-10 (`standing.GATES`, `standing.FAST_P5`)
On basis (B), robust-G convention, current deck:
| gate | threshold | basis |
|---|---|---|
| ruin paths | 0 | base deck — tripwire |
| **P5 terminal** | **≥ 1.00 × corpus, real, engine-exact**; **≥ 1.06 on the fast evaluator at search time** | tail — the bad decile keeps its purchasing power after 40 years of drawing |
| **worst-scenario net CAGR** | **≥ 5.25%** (5.5 reported as "comfortable") | regime floor, at the deck's resolution |
| R3AF vs EVG11 | ≥ −0.002 | benchmark-relative anti-fragility — the one orthogonal gate |
| −200 bp deck: ruin | 0 | tripwire |
**Reported, not gated:** −200 bp P5, HCC P5 and ruin, DRB40@6%, RCEW40, shortfall6, P10. Gates are absolute; nothing is measured against P1. Gates are for safety only — a return floor is never a gate.

The two P5 thresholds are one gate. The fast evaluator (search basis, no taxes) overstates P5 by about +0.045 (14 books: +0.048, sd 0.012; 5-seed means: +0.043 to +0.050), so 1.06 on the fast evaluator is 1.00 exact plus one standard deviation of the offset. **The fast threshold is a calibrated constant, not a rule**: it was calibrated at the fast evaluator's 6,000-path, seed-777 configuration and must be re-derived whenever the deck, the basis (taxes, draw, SS offset) or the fast evaluator changes. The exact 1.00 is the rule; `report_v3.py` is the arbiter; every search endpoint is confirmed engine-exact before it is called admissible.

## 3. Standing constraints (unchanged) — `standing.violations`
| constraint | value |
|---|---|
| feasibility | long-only, ETF ≤ 15%, singles ≤ 5% / 20% aggregate, semis ≤ 12%, defence ≤ 35%, nuclear ≤ 35% weight, ballast set ≥ 20%, PM-equity ≤ 10%, ≤ 14 names, 1% grid, 2% minimum |
| nuclear complex risk share | ≤ 38% of book variance |
| any single holding risk share | ≤ 24% |
| any theme block (defence, gold, cyber, semis) | ≤ 35% of variance |
| grade-C weight | ≤ 20% |
| weight in funds with < 60 months of history | ≤ 30% |
| single names (BRKB exempt) | ≤ 3 |
| ballast floor: DBMF + CTA + KMLM | ≥ 9% |
| duplicates | two books are one book if < 3% of the portfolio must move |
These values were adopted by the owner's instruction to apply Addendum 31 in full; they have not been individually examined the way the gates now have (Addendum 34 §5).

## 4. Deck and rows (unchanged) — `build_r8j.py`, `Reaudit_Schedule_20260909.md`
Beta- and geometric-consistency audit before any row enters; alpha zeroed unless t ≥ 2; over-claims corrected at half the gap; under-claims left but reported as the symmetric-deck sensitivity (R8j_sym, never used for selection). Rows on < 36 months are grade C unless anchored (grade-A/B sector fund, ≥ 120 months, correlation ≥ 0.5); premium over the highest-correlation anchor capped at +1.0 pp. Every re-audit is a deck revision; **every deck revision re-runs gates on all standing books and re-derives `FAST_P5`**.

## 5. Method (unchanged) — `standing.sigma_vec`
Regime-conditional volatility: covariance from bottom-tercile equity months versus the rest (short-history names via their sector anchor in the conditional panel; ratio shrunk halfway toward 1); severity ≥ 4 scenarios use the stress σ; rescaled so the π-weighted variance equals the standing 48-month σ. Book ratios 1.08–1.15.

## 6. Reporting set (unchanged) — `report_v3.py`
The dial (γ = 1/3/5), P5 terminal and P5/P25 path CAGR, −100 bp headline, −200 bp and HCC stress, symmetric-deck sensitivity, worst scenario, shortfall6, DRB40 and RCEW40 (reported), gates, constraints, risk shares by block, grade-C and < 60-month weight, ballast weight, singles, holdings, ER, lifetime tax, vol48, stress-σ ratio. Tie bands: 0.03 × on geo, 0.02 pp/yr on γ = 3 — inside them, choose on cost and structure. Noise floors (engine-exact, 6 seeds): geo 0.0095, P5 0.009; fast evaluator P5: sd ≈ 0.017 across seeds at 4k–14k paths.
