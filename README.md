# NSE Momentum Pullback Scanner

## Run locally
1. Install Python 3.10 or newer.
2. Extract this ZIP and open a terminal in the folder.
3. Run:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\\Scripts\\activate
   # macOS/Linux:
   source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Open the local URL printed by Streamlit (usually http://localhost:8501).

## Universe
Upload a CSV with a `Symbol` column containing NSE symbols such as `RELIANCE`, `TCS`, `SBIN` (without `.NS`), or paste symbols in the app.

## Base filters
- Market cap strictly greater than ₹1,000 crore by default.
- 3-month gain strictly greater than 60%, using about 63 trading sessions.
- Latest close within ±2% of EMA 10, EMA 20, or EMA 50.

## Bullish reversal layer
A stock is marked `Bullish reversal` when an EMA support zone was tested during the last three sessions and the latest candle shows at least one of:
- Hammer-like bullish candle
- Bullish engulfing candle
- Reclaim of an EMA after the prior close was below it

Volume confirmation is displayed when latest volume is at least 1.5x the previous 20-session average. The app also estimates entry, stop, and 2R/3R targets for reversal candidates.

## Data limitations
Yahoo Finance via yfinance is unofficial for NSE data and may be delayed, missing, throttled, or inaccurate. Market cap can be stale. Verify all candidates against a licensed market-data provider or broker before trading. This is a research screener, not investment advice.
