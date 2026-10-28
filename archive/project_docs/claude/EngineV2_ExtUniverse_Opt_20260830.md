# EngineV2 Extended-Universe PathCE Optimization — 2026-08-30

Optimization of the P1 baseline {GLDM 15 · SHLD 15 · DBMF 15 · NUKZ 15 · CIBR 10 · GRID 10 · CTA 5 · CME 5 · JEDI 5 · MSCI 5} over a 36-name US-listed universe (deck coverage ∩ price history), per user request "optimize for same 40-year PathCE resilient performance considering SMH ROKT XAR and whatever else US-listed you advise."

## Validation anchors (store reproduced to the digit before any optimization)

- V1/V2 (robust π, G/L) reproduced exactly on one shared rng(42) stream: PathCE 7.61 / 11.25, P10 2.3 / 3.4, ruin 365 / 186.
- V3/V4 (canonical π) reproduced exactly on a **fresh** rng(42) stream: 10.39 / 14.78, ruin 198 / 94. Finding: the "post-hoc documentary variants" in EngineV2_Rebuild_Variants_20260829.csv were produced with a reset generator, not the continued stream (continued stream gives 10.28 / 14.87). Resolves variants-file provenance; does not resolve the 12.35 target.
- AR2 vol pipeline reproduced exactly: P1 vol48 = 11.618% (monthly Adj_Close, 48m window ending 2026-08 partial, live-weight renormalization, ddof=1; JEDI live 11/48).
- WER(P1) = 0.532% from AR1 prospectus table.

## Method

- Engine mechanics per AR3/AR7 unchanged: 20-scenario canonical deck (Deck_Recal_R1R4_20260822), yearly Markov stay-or-redraw λ=1/8, lognormal σ = portfolio vol48, 3% fixed year-end draw, 60k paths, T=40, seed 42.
- Spec ambiguity (AR5 drift G/L; AR6 π robust/canonical) handled by evaluating all four conventions; search objective = mean ln PathCE across the four ("spec-robust geo-PathCE"); nothing is accepted unless it beats P1 under **every** convention.
- Estimation-error discipline: a "haircut deck" shrinks every non-incumbent row 25% toward the per-scenario cross-universe median; final objective = min(score, score_haircut) (maximin), and all floors must also hold under the haircut deck. Rationale: newcomer rows carry more model risk than the recalibrated incumbents.
- Hard floors vs P1 under all 4 conventions × both decks: P10 ≥, P5 ≥, ruin ≤; plus worst-scenario net CAGR ≥ 5.58%, RCEW40_robust ≥ 12.93, DRB40 ≤ P1, R3AF ≥ P1 − 0.002 (R3AF vs EVG20260811 benchmark per metrics_r3af.py).
- Constraints: long-only; ETF cap 15%; single stocks 5% each / 20% aggregate; semis (SMH+SOXX+XSD) ≤ 12%; defense/space sleeve ≤ 35%; nuclear/power sleeve ≤ 35%; ballast (GLDM+DBMF+CTA+KMLM+IEF) ≥ 20%; ≤ 14 names; 1% grid, min position 2%.
- Search: curated sleeve tests + greedy stochastic transfer search (CRN Monte Carlo, 8–14k paths), finalists re-verified at engine-exact 60k.
- "Jagged RCEW40" has no definition in the store; standard robust RCEW40 (rcew40.py 50/25/25) used in its place and flagged.

## Results (60k engine-exact; PathCE by convention; haircut in parens)

| Metric | P1 baseline | R2 Evolution (recommended) | R1 Maximin (aggressive) |
|---|---|---|---|
| geo-PathCE (4-conv) | 10.71 | **13.47** (12.79) | 16.09 (14.86) |
| PathCE robust-G / robust-L | 7.61 / 11.25 | 9.53 / 14.22 | 11.27 / 16.84 |
| PathCE canon-G / canon-L | 10.39 / 14.78 | 12.99 / 18.73 | 15.59 / 22.64 |
| P10 (robG…canL) | 2.33/3.37/3.20/4.57 | 2.81/4.10/3.87/5.61 | 3.22/4.75/4.53/6.62 |
| P5 robust-G | 1.37 | 1.67 | 1.92 |
| Ruin/60k (robG) | 366 | 297 | 259 |
| Worst-scenario net CAGR | 5.58% (Taiwan) | 6.04% (Taiwan) | 6.22% (Taiwan) |
| RCEW40 robust | 12.93 | 16.05 | 18.84 |
| R3AF | −0.0539 | −0.0452 | −0.0552 |
| WER / vol48 | 0.532% / 11.62% | 0.494% / 12.07% | 0.387% / 12.33% |
| Sortino48 / maxDD48 | 1.81 / −7.9% | 1.80 / −8.1% | 1.90 / −8.2% |
| One-way turnover | — | 13% | 34% |

All floors pass for both candidates under all 4 conventions × both decks at 60k. DRB40 = 0 for all three (every scenario clears the 5% hurdle) — gate currently uninformative; consider raising hurdle to 6% next recal.

### Weights

| Ticker | P1 | R2 | R1 |
|---|---|---|---|
| GLDM | 15 | 15 | 15 |
| SHLD | 15 | 15 | 15 |
| DBMF | 15 | 15 | 11 |
| NUKZ | 15 | 13 | 10 |
| CIBR | 10 | 9 | 5 |
| GRID | 10 | 6 | 0 |
| CTA | 5 | 4 | 0 |
| CME | 5 | 5 | 5 |
| JEDI | 5 | 0 | 0 |
| MSCI | 5 | 5 | 5 |
| URNM | — | 4 | 3 |
| XAR | — | 3 | 0 |
| CEG | — | 3 | 5 |
| XLE | — | 3 | 9 |
| QTUM | — | 0 | 10 |
| CBOE | — | 0 | 5 |
| KMLM | — | 0 | 2 |

R2 trades from P1: sell JEDI −5, GRID −4, NUKZ −2, CIBR −1, CTA −1; buy URNM +4, XAR +3, CEG +3, XLE +3.

## Verdicts on the three named tickers

- **XAR — in (3%).** XAR's deck row strictly dominates JEDI's in all 20 scenarios (e.g., Taiwan 5.62 vs 1.39; Space Race 14.47 vs 8.96) at lower ER (0.35 vs 0.50). JEDI→XAR is the cleanest single upgrade in the universe; standalone swap passes every floor and adds ~+1.0 geo-PathCE.
- **SMH — out.** Every funded add (3–5% from CIBR/GRID) breaches multiple floors: ruin up under all four conventions, worst-scenario falls 5.58→5.24–5.36% (Taiwan row −4.91%), R3AF −0.054→−0.12/−0.16. Its AI-boom upside does not compensate under log-wealth with a 3% draw. Semis exposure is already carried indirectly via QTUM (R1) and the AI-bottleneck rows of GRID/NUKZ/CEG. Revisit only if the deck's Taiwan/cyber severity for semis is softened.
- **ROKT — out, narrowly.** ROKT+5 (JEDI out) fails only worst-scenario (5.47 vs 5.58) and R3AF; the XAR route captures the same space/defense theme with materially better tail rows (ROKT Taiwan −0.66 vs XAR +5.62). If ROKT conviction is strong, ≤3% funded from CIBR is the least-damaging placement, but it cannot pass the current floors.

## Other advised names (evaluated, 36-name universe)

Selected: URNM (uranium miners; strongest resilience row after CEG — worst 7.43%, stagflation 15.6, energy shock 14.1), CEG (nuclear operator; best row in deck, capped 5% single-stock), XLE (energy-shock/stagflation hedge; 14.7/14.2 rows), CBOE + KMLM (R1: exchange-vol toll road, third trend style). Evaluated and not selected: IEF, PPA, ITA, EUAD, PAVE, SETM, NLR, SOXX, XSD, HACK, IHAK-class, QUAL, COWZ, AVUV, MOAT, ICE, SPGI, NDAQ (each either dilutes PathCE or fails to earn weight under the floors). No deck rows exist for TLT/ZROZ/GOVZ/BRKB/WPM/FNV/V — deriving them via the R1–R4 pipeline is the natural next deck recal if long-duration or royalty ballast is wanted.

## Open flags

1. **QTUM tension.** The deck rates QTUM top-decile resilient growth (base 12.88, worst 4.92 — better than CIBR's Taiwan row); the 2026-08-21 Evergreen decision cut QTUM to 0. R1 holds 10% QTUM on the deck's numbers. Either the deck row is stale or the conviction deserves revisiting — recommend resolving before adopting R1; R2 deliberately holds QTUM at 0.
2. R1 drops GRID/CTA entirely and runs 34% turnover — adopt only if the deck rows for QTUM/XLE/CEG/URNM are trusted at face value; R2 is the deck-uncertainty-aware recommendation.
3. ER table: incumbents per AR1; additions at prospectus values (QTUM 0.40 and ROKT 0.45 verified 2026-08-30; URNM assumed 0.75, XAR 0.35, XLE 0.09 — immaterial at these weights: ±0.10pp WER ≈ ∓0.5 PathCE).
4. Sensitivities (robust legs): R2 keeps PathCE ≥ 8.99/13.50 at WER+0.10pp and 9.07/13.99 at σ=12.36%, both far above P1's corresponding 7.15/10.67 and 6.64/10.72.

Files: EngineV2_ExtUniverse_Weights_20260830.csv, EngineV2_ExtUniverse_Metrics_20260830.csv, EngineV2_ExtUniverse_Code_20260830.py.
