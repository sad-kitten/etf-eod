# GitHub handoff, step 7 (2026-10-04): etf-eod patch 0003, multibagger shortlist tickers — DONE

Continuation of `claude/GitHub_Handoff_20260901.md` (steps 1-6). Unlike the 2026-09 sessions, this session attached `sad-kitten/etf-eod` with push access mid-session (add_repo), so the patch was applied end to end here; no repo-bound task was needed.

## What changed
- Patch: `claude/etf-eod_patch_0003_shortlist_tickers_20261004.patch` (sha256 `212982b59d947b2fb79dccd9aeefec4875a1012b1152a48e0bffe2f52b87d9c1`, 117 lines, 3,445 bytes). One commit `55e72a8` on main `a982101`: `tickers.txt` +12 symbols (+1 comment line), one README line. No script or workflow change.
- Symbols: `3697.T` SHIFT, `3923.T` Rakus, `4194.T` Visional, `APPF`, `CSU.TO`, `KNSL`, `KSPI`, `MEDP`, `NU`, `ONON`, `PAY`, `TOI.V`. First non-US listings; files are `prices/CSU.TO.csv`, `prices/4194.T.csv` and so on (only dashes are stripped), and `validate.py` maps the stem back to the symbol.

## What was run (all 2026-10-04, UTC)
1. Branch `claude/add-shortlist-2026-10-04` pushed; PR #4 opened via the REST API (GraphQL is blocked for `gh pr create`; use `gh api repos/.../pulls`).
2. `refresh-prices` dispatched on the branch with the 12-symbol `tickers` input: run 37230991519, success (20:09-20:10), commit `0471251` "refresh 2026-10-04". 12/12 pulled, 181/181 ok, no stale, no non-positive Adj_Close.
3. PR #4 merged (merge commit `563c5e4`). The push to main triggered the full refresh: run 37231128088, success (20:11-20:16), commit `a6cb76f`. Main: 181/181 ok, 12/12 new symbols, 0 stale, last_date 2026-10-02 (2026-10-01 for Tokyo).
4. Branch deletion is refused by the proxy (both the refs API and `git push --delete`); delete `claude/add-shortlist-2026-10-04` in the GitHub UI if wanted.

## New files on main
| symbol | first_date | rows | note |
|---|---|---|---|
| 3697.T | 2014-11-13 | 2922 | JPY; max gap 11 days (Golden Week) |
| 3923.T | 2015-12-09 | 2661 | JPY; max gap 11 days |
| 4194.T | 2021-04-21 | 1330 | JPY |
| APPF | 2015-06-26 | 2834 | |
| CSU.TO | 2007-10-25 | 4753 | CAD; 1,853 rows flagged Repaired? (all 2007-2015, dividend-adjustment hints; prices consistent, C$24.60 -> C$2,981) |
| KNSL | 2016-07-28 | 2560 | |
| KSPI | 2024-01-19 | 679 | US ADS listing only: 32 months, thin history |
| MEDP | 2016-08-11 | 2550 | |
| NU | 2021-12-09 | 1208 | |
| ONON | 2021-09-15 | 1268 | |
| PAY | 2021-05-26 | 1345 | |
| TOI.V | 2021-02-02 | 1423 | CAD |

## Price metrics for the shortlist (from the new files; `claude/Shortlist_PriceMetrics_20261004.csv`)
vol48 = annualized standard deviation of the last 48 month-end Adj_Close returns (partial 2026-10 excluded; KSPI has 32 months). Beta vs IVV on the same monthly window; local currency for CAD and JPY names, so not FX-adjusted. CAGR from first file date to 2026-10-02 (Tokyo 10-01). Drawdowns on daily Adj_Close. Range = position between the 12-month low and high.

| ticker | vol48 | beta | CAGR since first | 12-mo return | max drawdown | drawdown now | range |
|---|---|---|---|---|---|---|---|
| NU | 44.0% | 0.69 | 5.6% | -11.8% | -72% | -28% | 26% |
| ONON | 53.8% | 2.30 | -2.5% | -27.2% | -69% | -52% | 17% |
| KNSL | 40.6% | 1.14 | 33.2% | -25.3% | -47% | -40% | 21% |
| PAY | 62.3% | 0.83 | 1.4% | -0.7% | -81% | -31% | 44% |
| TOI.V | 31.3% | 0.62 | 6.8% | -39.9% | -57% | -54% | 9% |
| 4194.T | 34.9% | 1.02 | 4.3% | -18.6% | -50% | -28% | 50% |
| APPF | 37.9% | 0.96 | 26.7% | -19.7% | -55% | -37% | 50% |
| MEDP | 41.9% | 1.08 | 35.4% | +11.8% | -43% | -5% | 87% |
| KSPI | 35.7% | 1.07 | 4.1% | +27.0% | -49% | -25% | 71% |
| CSU.TO | 26.7% | 0.73 | 30.3% | -23.4% | -56% | -42% | 41% |
| 3923.T | 42.2% | 1.97 | 25.6% | -19.3% | -73% | -56% | 51% |
| 3697.T | 55.1% | 1.79 | 20.9% | -20.8% | -76% | -63% | 43% |

Reading: the four names with less than five years of history (NU, ONON, PAY, TOI.V, plus KSPI) show IPO-era drawdowns of 57-81% and near-zero CAGRs since listing, which is the thin-history pattern the engine's fair-weather adjustment exists for; the long-history compounders (KNSL, MEDP, CSU, APPF, Rakus, SHIFT) have 21-35% CAGRs with 43-76% maximum drawdowns along the way. The weekly Monday 09:20 UTC run now maintains all 181 symbols.
