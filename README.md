# NSE Stock Screener — Daily Results Update

## Files
- `app.py`: Streamlit dashboard, manual scanner, persistent daily results.
- `.github/workflows/daily_nse_pipeline.yml`: refreshes the NSE EQ universe and verified Full Bhavcopy history, then runs both Momentum and Pullback scans. It is scheduled for weekdays at 6:00 PM IST and can also be started manually.
- `universe.csv`: symbols used by the scheduled scan. Add NSE symbols in the `Symbol` column.
- `nse_daily_history.csv.gz`: approximately two years of verified NSE EQ daily OHLCV and turnover history.
- `industry_classification.csv`: reviewed company-to-sector/industry classification map.
- `daily_scan_results.csv`: separate date-stamped Momentum and Pullback results, displayed in the dashboard.

## Setup
1. Upload/replace `app.py`, `requirements.txt`, and `README.md` in the repository root.
2. Create `.github/workflows/` and upload `daily_scan.yml` there.
3. Upload `universe.csv` in the repository root.
4. In GitHub, open Settings → Actions → General and ensure workflow permissions allow read and write access to repository contents.
5. Open Actions → NSE daily scan → Run workflow to test immediately. The scheduled job then runs weekdays at 4:30 PM IST.
6. Streamlit Community Cloud will redeploy after the commit; refresh the dashboard to see saved results.

The automated pipeline refreshes the NSE universe, rebuilds and validates the historical database, and runs both strategies using the default thresholds in `app.py`. Manual dashboard scans use the symbols and thresholds selected in the app. Each day's Momentum and Pullback results are retained independently. GitHub scheduled workflows run only from the repository's default branch; the pipeline must be deployed to that branch before its schedule becomes active.

## Data limitations
NSE Full Bhavcopy supplies stock OHLCV and turnover. Yahoo Finance through yfinance remains the source for the Nifty 500 proxy and company market-cap / PEG metadata; those fields may be delayed, incomplete, throttled or inaccurate. Raw NSE prices are not corporate-action adjusted, so splits and bonuses can distort historical returns and moving averages. This is a research screener, not investment advice.


## Full NSE universe
The daily GitHub Actions scan refreshes `universe.csv` from NSE official main-board equity securities (`EQUITY_L.csv`) before scanning. SME securities are published separately by NSE and are not included. Market-cap, 3-month momentum, EMA proximity, reversal, volume, entry and target filters remain unchanged. Data retrieval for this broad universe uses Yahoo Finance through yfinance and may be incomplete or rate-limited; check the scan error count.
