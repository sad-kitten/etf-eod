# etf-eod

Daily end-of-day prices for a fixed universe of ETFs and stocks, pulled from Yahoo Finance
by a GitHub Actions workflow and committed back to this repository as plain CSV files.

## Files

| Path | What it is |
|---|---|
| `tickers.txt` | The universe: one Yahoo Finance symbol per line, `#` starts a comment. |
| `prices/<SYMBOL>.csv` | One file per symbol with daily history from 2000-01-01. Dashes are removed from the file name (`BRK-B` -> `prices/BRKB.csv`). |
| `pull_status.csv` | Result of the latest pull, one row per symbol. |
| `validation.csv` | Data-quality summary for every file in `prices/`. |
| `monthly_returns.csv` | Wide panel of month-end `Adj_Close` percent changes, one column per symbol. |
| `scripts/pull.py` | Reads `tickers.txt`, downloads each symbol with `yfinance`, writes `prices/` and `pull_status.csv`. |
| `scripts/validate.py` | Reads `prices/*.csv`, writes `validation.csv` and `monthly_returns.csv`. |
| `.github/workflows/refresh.yml` | The `refresh-prices` workflow that runs pull, validate and commit. |

## Schema

### `prices/<SYMBOL>.csv`

| Column | Notes |
|---|---|
| `Date` | Trading date, `YYYY-MM-DD`. |
| `Open`, `High`, `Low`, `Close` | Unadjusted prices as reported by Yahoo (`auto_adjust=False`). |
| `Adj_Close` | Close adjusted for dividends and splits. Use this for returns. |
| `Volume` | Shares traded, integer (missing values written as 0). |
| `Capital_Gains` | Capital-gain distributions per share. Present only for funds that report them; the column is absent for stocks. |
| `Repaired?` | `True` when `yfinance` (`repair=True`) fixed a bad print such as a 100x price error or a missing split adjustment. |

### `pull_status.csv`

| Column | Notes |
|---|---|
| `ticker` | Yahoo symbol as written in `tickers.txt` (`BRK-B`, not `BRKB`). |
| `ok` | `True` if the download succeeded within 3 attempts. |
| `rows` | Number of rows written (0 on failure). |
| `err` | Reason for the last failed attempt (first 160 characters, one line), empty on success. Yahoo's own message is recorded, e.g. `$XYZ: possibly delisted; no timezone found`. |

A full run rewrites the file for the whole universe. A subset run (see below) updates only the
symbols it pulled and keeps every other row from the previous file. If every symbol in a run
fails (Yahoo outage, rate limit) or a Python dependency is missing, `pull.py` exits non-zero so
the workflow stops before the commit step and the previous good files stay in place.

### `validation.csv`

| Column | Notes |
|---|---|
| `ticker` | Yahoo symbol (mapped back from the file name via `tickers.txt`; falls back to the file stem). |
| `first_date`, `last_date` | First and last `Date` in the file. |
| `rows` | Number of data rows in the file. |
| `max_gap_days` | Largest number of calendar days between two consecutive rows. Normal weekends and holidays give 3 to 4; anything much larger is a hole in the history. |
| `stale` | `True` when `last_date` is more than 10 days before the run date (UTC). |
| `adj_close_nonpositive` | Count of rows where `Adj_Close <= 0`. |
| `repaired_rows` | Count of rows where `Repaired?` is `True`. |

### `monthly_returns.csv`

Indexed by `month` (`YYYY-MM`). Each column is a symbol; each cell is the percent change of the
last `Adj_Close` observed in that month versus the last `Adj_Close` of the previous month
(0.05 means +5%). Values are rounded to 6 decimals. A cell is empty for the symbol's first
month and for months before it started trading. The final row is month-to-date if the latest
pull does not extend to the last trading day of that month.

## Cadence

The `refresh-prices` workflow runs:

- every Monday at 09:20 UTC (on the default branch);
- on any push that changes `tickers.txt`, `.github/workflows/refresh.yml` or `scripts/**`;
- on demand from the Actions tab (`workflow_dispatch`). The optional `tickers` input takes a
  comma-separated list of symbols to pull instead of the whole universe.

Runs share the `refresh-prices` concurrency group, so two refreshes never write at the same
time. Each run does `pull -> validate -> commit`, committing `prices/`, `pull_status.csv`,
`validation.csv` and `monthly_returns.csv` with the message `refresh <date>`.

## Adding a ticker

1. Add the Yahoo symbol on its own line in `tickers.txt` (case does not matter, dashes stay:
   `BRK-B`).
2. Commit and push. The push triggers a full refresh, which creates `prices/<SYMBOL>.csv` and adds
   the symbol to `pull_status.csv`, `validation.csv` and `monthly_returns.csv`.
3. To fetch just the new symbol right away, run the workflow manually with the `tickers` input
   set to that symbol. Symbols passed to `tickers` must already be in `tickers.txt`.

To drop a symbol, delete its line in `tickers.txt` and delete `prices/<SYMBOL>.csv`;
`validate.py` reports every file it finds in `prices/`, whether or not it is still in the universe.

## Running locally

```bash
pip install "yfinance[repair]" pandas   # [repair] adds scipy + scikit-learn, required for repair=True
python scripts/pull.py                  # or: SUBSET=VTI,BRK-B python scripts/pull.py
python scripts/validate.py
```

## Consuming the data with pandas

```python
import pandas as pd
prices = pd.read_csv("prices/VOO.csv", index_col="Date", parse_dates=True)
monthly = pd.read_csv("monthly_returns.csv", index_col="month")
```

Replace the local paths with `https://raw.githubusercontent.com/sad-kitten/etf-eod/main/...`
to read straight from GitHub.
