# NSE Stock Screener — Daily Results Update

## Files
- `app.py`: Streamlit dashboard, manual scanner, persistent daily results.
- `.github/workflows/daily_nse_pipeline.yml`: refreshes the NSE EQ universe and verified Full Bhavcopy history, runs Momentum and Pullback scans, and updates the forward-performance ledger. Scheduled weekdays at 6:00 PM IST; also supports manual runs.
- `universe.csv`: symbols used by the scheduled scan.
- `nse_daily_history.csv.gz`: verified NSE EQ daily OHLCV and turnover history.
- `industry_classification.csv`: reviewed company-to-sector/industry classification map.
- `daily_scan_results.csv`: date-stamped Momentum and Pullback candidates.
- `update_performance_tracker.py`: calculates forward outcomes from sessions after each candidate's signal date.
- `performance_tracker.csv`: one row per unique signal-date / strategy / symbol, with 5-, 10- and 20-session outcomes.

## Forward performance tracker
The daily workflow rebuilds the tracker from the full scan history and verified NSE prices, so pending outcomes automatically mature as new sessions arrive. It records close-to-close returns, maximum favourable excursion (highest high), maximum adverse excursion (lowest low), and the session date for each completed horizon. Nifty 500 proxy returns and excess returns are included when historical Yahoo Finance data is available; a benchmark download failure leaves those optional fields blank and does not fail the NSE pipeline. The tracker does not change scanner filters or issue buy/sell instructions.

A horizon is measured using the next 5, 10 or 20 verified NSE trading sessions strictly after the signal date. A result remains Pending until the full horizon exists. Historical prices are raw NSE bhavcopy prices, not corporate-action adjusted; splits and bonuses can distort measured returns and excursions.

## Pipeline
GitHub Actions refreshes the official NSE EQ universe, rebuilds and validates the verified Bhavcopy history, runs both scan modes, updates the performance tracker, and commits generated CSVs. The schedule runs weekdays at 6:00 PM IST from the default branch. Use Actions → Daily NSE Full-Data Scanner → Run workflow to start a manual run.

## Data limitations
NSE Full Bhavcopy supplies stock OHLCV and turnover. Yahoo Finance through yfinance remains the source for the Nifty 500 proxy and company market-cap / PEG metadata; those fields may be delayed, incomplete, throttled or inaccurate. This is a research screener, not investment advice.
