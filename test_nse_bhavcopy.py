import io
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

OUT = Path("nse_bhavcopy_test")
OUT.mkdir(exist_ok=True)

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "Accept": "text/csv,text/plain,*/*",
    "Referer": "https://www.nseindia.com/all-reports",
})
try:
    session.get("https://www.nseindia.com/", timeout=20)
except requests.RequestException as exc:
    print(f"Note: NSE homepage session setup did not complete: {exc}")

required = {
    "SYMBOL", "SERIES", "DATE1", "PREV_CLOSE", "OPEN_PRICE",
    "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY",
    "TURNOVER_LACS",
}

today = date.today()
downloaded = None
used_date = None
last_error = None

# NSE does not publish a new equity bhavcopy on weekends/holidays.
for offset in range(1, 11):
    day = today - timedelta(days=offset)
    date_tag = day.strftime("%d%m%Y")
    url = f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{date_tag}.csv"
    print(f"Trying {day.isoformat()}: {url}")
    try:
        response = session.get(url, timeout=35)
        response.raise_for_status()
        if not response.content or b"SYMBOL" not in response.content[:5000].upper():
            raise ValueError("Response is empty or does not look like the NSE Full Bhavcopy CSV")
        frame = pd.read_csv(io.BytesIO(response.content))
        frame.columns = [str(c).strip().upper() for c in frame.columns]
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing expected NSE columns: {missing}")
        frame["SERIES"] = frame["SERIES"].astype(str).str.strip().str.upper()
        equity = frame[frame["SERIES"] == "EQ"].copy()
        if equity.empty:
            raise ValueError("CSV parsed, but it contains no EQ-series equity rows")
        for col in ["PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY", "TURNOVER_LACS"]:
            equity[col] = pd.to_numeric(equity[col].astype(str).str.replace(",", ""), errors="coerce")
        equity = equity.dropna(subset=["CLOSE_PRICE", "TTL_TRD_QNTY"])
        if len(equity) < 100:
            raise ValueError(f"Only {len(equity)} usable EQ rows; expected at least 100")
        downloaded = equity
        used_date = day
        raw_path = OUT / f"sec_bhavdata_full_{date_tag}.csv"
        raw_path.write_bytes(response.content)
        break
    except Exception as exc:
        last_error = exc
        print(f"  Not usable: {exc}")

if downloaded is None:
    print("FAIL: Could not download and parse an NSE Full Bhavcopy from the last 10 calendar days.")
    print(f"Last error: {last_error}")
    sys.exit(1)

sample_cols = ["SYMBOL", "SERIES", "CLOSE_PRICE", "TTL_TRD_QNTY", "TURNOVER_LACS"]
sample = downloaded[sample_cols].head(10)
print("\nPASS: Downloaded and parsed an actual NSE Full Bhavcopy.")
print(f"Trade date: {used_date.isoformat()}")
print(f"Usable EQ-series rows: {len(downloaded):,}")
print("Sample rows:")
print(sample.to_string(index=False))
sample.to_csv(OUT / "parsed_eq_sample.csv", index=False)
(OUT / "TEST_RESULT.txt").write_text(
    f"PASS\nTrade date: {used_date.isoformat()}\nUsable EQ rows: {len(downloaded)}\n",
    encoding="utf-8",
)
