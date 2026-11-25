import io
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

OUT = Path("nse_daily_history.csv.gz")
LOOKBACK_CALENDAR_DAYS = 380
MIN_TRADING_DAYS = 200
BASE_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{}.csv"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.nseindia.com/all-reports",
})
try:
    session.get("https://www.nseindia.com/", timeout=20)
except requests.RequestException as exc:
    print(f"Session note: {exc}")

required = {
    "SYMBOL", "SERIES", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
    "CLOSE_PRICE", "TTL_TRD_QNTY", "TURNOVER_LACS",
}
parts = []
trading_dates = []
missing_dates = 0
errors = []
today = date.today()
first_day = today - timedelta(days=LOOKBACK_CALENDAR_DAYS)

for day_offset in range(LOOKBACK_CALENDAR_DAYS, 0, -1):
    day = today - timedelta(days=day_offset)
    if day.weekday() >= 5:
        continue

    tag = day.strftime("%d%m%Y")
    url = BASE_URL.format(tag)
    response = None

    for attempt in range(3):
        try:
            response = session.get(url, timeout=35)
            if response.status_code == 404:
                response = None
                missing_dates += 1
                break
            if response.status_code in (429, 500, 502, 503, 504):
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            errors.append(f"{day.isoformat()}: {exc}")
            response = None
            break

    if response is None:
        continue

    try:
        frame = pd.read_csv(io.BytesIO(response.content))
        frame.columns = [str(c).strip().upper() for c in frame.columns]
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"missing columns {missing}")

        frame["SERIES"] = frame["SERIES"].astype(str).str.strip().str.upper()
        equity = frame[frame["SERIES"] == "EQ"].copy()
        if equity.empty:
            raise ValueError("no EQ-series rows")

        for col in ["OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY", "TURNOVER_LACS"]:
            equity[col] = pd.to_numeric(
                equity[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )

        equity["DATE"] = day.isoformat()
        equity["SYMBOL"] = equity["SYMBOL"].astype(str).str.strip().str.upper()
        equity = equity.rename(columns={
            "OPEN_PRICE": "OPEN",
            "HIGH_PRICE": "HIGH",
            "LOW_PRICE": "LOW",
            "CLOSE_PRICE": "CLOSE",
            "TTL_TRD_QNTY": "VOLUME",
            "TURNOVER_LACS": "TURNOVER_LAKHS",
        })
        equity["TURNOVER_CRORES"] = equity["TURNOVER_LAKHS"] / 100.0
        equity = equity[[
            "DATE", "SYMBOL", "OPEN", "HIGH", "LOW", "CLOSE",
            "VOLUME", "TURNOVER_CRORES",
        ]].dropna(subset=["CLOSE", "VOLUME"])
        equity = equity[equity["SYMBOL"].ne("") & equity["CLOSE"].gt(0)]

        if len(equity) < 100:
            raise ValueError(f"only {len(equity)} usable EQ rows")

        parts.append(equity)
        trading_dates.append(day)
        if len(trading_dates) % 25 == 0:
            print(f"Downloaded {len(trading_dates)} trading days; latest: {day}; rows: {sum(len(p) for p in parts):,}")
    except Exception as exc:
        errors.append(f"{day.isoformat()}: {exc}")

    time.sleep(0.20)

if not parts:
    raise SystemExit("FAIL: no historical NSE trading-day files were collected")

history = pd.concat(parts, ignore_index=True)
history = history.drop_duplicates(subset=["DATE", "SYMBOL"], keep="last")
history["DATE"] = pd.to_datetime(history["DATE"])
history = history.sort_values(["SYMBOL", "DATE"]).reset_index(drop=True)
history.to_csv(OUT, index=False, compression="gzip", date_format="%Y-%m-%d")

print("\nNSE HISTORICAL DATABASE SUMMARY")
print(f"Requested range: {first_day.isoformat()} to {(today - timedelta(days=1)).isoformat()}")
print(f"Trading days downloaded: {len(trading_dates)}")
print(f"Weekday dates with no file (holiday/non-trading): {missing_dates}")
print(f"Unique symbols: {history['SYMBOL'].nunique():,}")
print(f"Total daily rows: {len(history):,}")
print(f"First stored date: {history['DATE'].min().date()}")
print(f"Last stored date: {history['DATE'].max().date()}")
print(f"Compressed database size: {OUT.stat().st_size:,} bytes")
print("Sample 3BBLACKBIO rows:")
sample = history[history["SYMBOL"] == "3BBLACKBIO"].tail(5)
print(sample.to_string(index=False) if not sample.empty else "No sample rows found")
if errors:
    print(f"Non-404 errors: {len(errors)}")
    for item in errors[:20]:
        print(item)

if len(trading_dates) < MIN_TRADING_DAYS:
    raise SystemExit(
        f"FAIL: only {len(trading_dates)} trading days found; "
        f"need at least {MIN_TRADING_DAYS} for a reliable 1-year database"
    )
if history["SYMBOL"].nunique() < 1000:
    raise SystemExit("FAIL: fewer than 1,000 unique symbols in historical database")

print("\nPASS: NSE historical database created successfully.")
