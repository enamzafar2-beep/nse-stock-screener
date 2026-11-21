# NSE Stock Screener — Daily Results Update

## Files
- `app.py`: Streamlit dashboard, manual scanner, persistent daily results.
- `.github/workflows/daily_scan.yml`: weekday scheduled scan at 4:30 PM IST (11:00 UTC), plus manual workflow dispatch.
- `universe.csv`: symbols used by the scheduled scan. Add NSE symbols in the `Symbol` column.
- `daily_scan_results.csv`: created/updated by the workflow and displayed in the dashboard.

## Setup
1. Upload/replace `app.py`, `requirements.txt`, and `README.md` in the repository root.
2. Create `.github/workflows/` and upload `daily_scan.yml` there.
3. Upload `universe.csv` in the repository root.
4. In GitHub, open Settings → Actions → General and ensure workflow permissions allow read and write access to repository contents.
5. Open Actions → NSE daily scan → Run workflow to test immediately. The scheduled job then runs weekdays at 4:30 PM IST.
6. Streamlit Community Cloud will redeploy after the commit; refresh the dashboard to see saved results.

The scheduled scan uses the symbols in `universe.csv` and the default thresholds in `app.py`. Manual dashboard scans use the symbols and thresholds selected in the app.

## Data limitations
Yahoo Finance through yfinance is unofficial; data may be delayed, incomplete, throttled or inaccurate. This is a research screener, not investment advice.


## Full NSE universe
The daily GitHub Actions scan refreshes `universe.csv` from NSE official main-board equity securities (`EQUITY_L.csv`) before scanning. SME securities are published separately by NSE and are not included. Market-cap, 3-month momentum, EMA proximity, reversal, volume, entry and target filters remain unchanged. Data retrieval for this broad universe uses Yahoo Finance through yfinance and may be incomplete or rate-limited; check the scan error count.
