"""
Download historical OHLCV for the “missing” ETFs into CSV files.

Default source: Yahoo Finance via yfinance (most reliable for ETFs).
Fallback: Stooq via pandas_datareader (often missing newer/smaller ETFs).

Output: ./prices/<TICKER>.csv
"""

from __future__ import annotations

import os
import time
from datetime import date
from typing import Iterable, Optional

TICKERS = ["GLD","STIP"]

START = "2000-01-01"          # adjust as you like
END = date.today().isoformat()
OUTDIR = "prices"

# ---- helpers ----

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_csv(df, ticker: str, outdir: str) -> str:
    # Normalize columns a bit (works for yfinance + stooq)
    df = df.copy()
    if "Date" not in df.columns:
        df.reset_index(inplace=True)
        if df.columns[0] != "Date":
            df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    outpath = os.path.join(outdir, f"{ticker}.csv")
    df.to_csv(outpath, index=False)
    return outpath

def try_yfinance(tickers: Iterable[str], start: str, end: str):
    try:
        import yfinance as yf
    except Exception as e:
        print("yfinance not available:", e)
        return None

    print("Using yfinance…")
    results = {}
    for t in tickers:
        try:
            # auto_adjust=False keeps raw OHLC; set True if you prefer adj OHLC
            df = yf.download(
                t,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                print(f"  - {t}: no data returned")
                continue

            # yfinance uses 'Adj Close' — keep it
            results[t] = df
            print(f"  - {t}: {len(df):,} rows")
            time.sleep(0.25)  # gentle rate limiting
        except Exception as e:
            print(f"  - {t}: error {e}")
    return results

def try_stooq(tickers: Iterable[str], start: str, end: str):
    try:
        import pandas_datareader.data as web
    except Exception as e:
        print("pandas_datareader not available:", e)
        return None

    print("Using Stooq fallback…")
    results = {}
    for t in tickers:
        try:
            # Stooq expects lowercase and sometimes suffixes;
            # For US ETFs it’s often just the lowercase ticker.
            df = web.DataReader(t.lower(), "stooq", start=start, end=end)
            if df is None or df.empty:
                print(f"  - {t}: no data on Stooq")
                continue

            # Stooq returns newest-first; reverse to chronological
            df = df.sort_index()
            results[t] = df
            print(f"  - {t}: {len(df):,} rows")
            time.sleep(0.2)
        except Exception as e:
            print(f"  - {t}: error {e}")
    return results

def main(tickers: list[str], start: str, end: str, outdir: str) -> None:
    ensure_outdir(outdir)

    remaining = set(tickers)

    yf_res = try_yfinance(remaining, start, end) or {}
    for t, df in yf_res.items():
        outpath = save_csv(df, t, outdir)
        print("Saved:", outpath)
        remaining.discard(t)

    if remaining:
        stooq_res = try_stooq(remaining, start, end) or {}
        for t, df in stooq_res.items():
            outpath = save_csv(df, t, outdir)
            print("Saved:", outpath)
            remaining.discard(t)

    if remaining:
        print("\nCould not download these tickers from either source:")
        for t in sorted(remaining):
            print(" -", t)
        print("\nIf this happens, Yahoo sometimes blocks temporarily. Re-run later, or use a VPN, or add a longer sleep.")

if __name__ == "__main__":
    # Install deps if needed:
    #   pip install yfinance pandas pandas_datareader
    main(TICKERS, START, END, OUTDIR)
