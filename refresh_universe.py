"""Refresh NSE main-board equity symbols from the official NSE equity master CSV."""
from io import BytesIO
from pathlib import Path
import sys
import time
import pandas as pd
import requests

CSV_URL = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
HOME_URL = "https://www.nseindia.com/"
OUTPUT = Path(__file__).resolve().parent / "universe.csv"
ALLOWED_SERIES = {"EQ", "BE", "BZ"}
MIN_EXPECTED_SYMBOLS = 500
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
    "Accept": "text/csv,application/csv,*/*",
    "Referer": HOME_URL,
    "Connection": "keep-alive",
}

def main():
    session = requests.Session()
    session.headers.update(HEADERS)
   
    response = session.get(CSV_URL, timeout=60)
    response.raise_for_status()
    if not response.content or b"SYMBOL" not in response.content[:5000].upper():
        raise RuntimeError("NSE did not return the expected equity master CSV.")

    df = pd.read_csv(BytesIO(response.content), encoding="latin1")
    df.columns = [str(c).replace("\ufeff", "").strip().upper() for c in df.columns]
    if not {"SYMBOL", "SERIES"}.issubset(df.columns):
        raise RuntimeError("NSE CSV is missing SYMBOL or SERIES columns.")

    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    df["SERIES"] = df["SERIES"].astype(str).str.strip().str.upper()
    if "NAME OF COMPANY" in df.columns:
        names = df["NAME OF COMPANY"].fillna("").astype(str).str.upper()
        df = df[~names.str.contains(r"\bETF\b|\bBEES\b|RIGHTS ENTITLEMENT", regex=True)]

    df = df[
        df["SERIES"].isin(ALLOWED_SERIES)
        & df["SYMBOL"].str.fullmatch(r"[A-Z0-9&_-]+", na=False)
    ].drop_duplicates("SYMBOL").sort_values("SYMBOL")

    if len(df) < MIN_EXPECTED_SYMBOLS:
        raise RuntimeError(
            f"Only {len(df)} eligible symbols found. Existing universe.csv was not changed."
        )

    cols = ["SYMBOL", "SERIES"]
    if "NAME OF COMPANY" in df.columns:
        cols.append("NAME OF COMPANY")
    result = df[cols].rename(
        columns={"SYMBOL": "Symbol", "SERIES": "Series", "NAME OF COMPANY": "Company"}
    )
    result.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"Saved {len(result)} NSE main-board equity symbols to {OUTPUT.name}.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: NSE universe refresh failed: {exc}", file=sys.stderr)
        raise
