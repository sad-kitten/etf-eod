# EngineV2 full engine (50-year, v3.4 gates, E1-F)

Rebuilt 2026-10-26 (Addendum 57) from investment-lab@4bad722 (`lib_pathce.py`, `core.py`) plus the Project code bundles
(Bench_2FFL_Code_20260919: `search_B.py`, `policy_engine2.py`; R8n_Code_20261024: `standing.py`, `report_v3.py`;
Factor_Code_20261025: `uncertainty.py`, `report_u.py`).

Run from a clone of this repository:

```
source engine/env57.sh
SINGLES_MAX=99 python3 report_u.py books57.json out.csv [Book1,Book2]
```

- Price panel: `../monthly_returns.csv` (this repo). 48-month σ window ends ENGINEV2_END=2026-08.
- Deck: `Deck_R8q_20261026.csv` = R8p + ZTS row. Scenarios: 25-scenario R8i. 60,000 paths x 4 conventions, 50 years.
- Patches vs the bundles: `ENGINEV2_T` (horizon), `ENGINEV2_EXTRA` (extra tickers), SINGLES += UTHR/VRTX/ISRG/ZTS,
  `SINGLES_MAX` / `SINGLES_TOT` / `CW_CAP` env overrides (owner dropped the 3-singles count cap 2026-10-26).
- Grades file `Row_Grades_R8j_A57.csv` covers the book universe; tickers absent default to grade B.

Validation (2026-10-26): C5_SMH3b reproduces Report_T50_R56 exactly (geo 7.596, γ3 2.43, P5 1.29; U1 7.44 / 1.39 / 0.85);
R56_A reproduces the Addendum 56 real-UTHR postscript (7.59 / 2.81 / 1.63; U1 7.50 / 2.01 / 1.10).
Search: `SEEDS_FILE=... OUTJSON=... PREF=1 LOCK=UTHR:0.05 EXCL=<bonds,...> python3 search_f.py SEED` (seed must be admissible).
