import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="NSE Momentum Pullback Scanner", page_icon="📈", layout="wide")

st.title("📈 NSE Momentum Pullback Scanner")
st.caption("Finds NSE-listed stocks with market cap > ₹1,000 crore, 3-month price gain > 60%, and price within 2% of EMA 10, 20 or 50.")

with st.sidebar:
    st.header("Scanner settings")
    min_mcap = st.number_input("Minimum market cap (₹ crore)", min_value=1, value=1000, step=100)
    min_return = st.number_input("Minimum 3-month return (%)", min_value=0.0, value=60.0, step=5.0)
    proximity = st.number_input("EMA proximity (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.5)
    lookback = st.selectbox("EMA calculation timeframe", ["Daily"], index=0)
    workers = st.slider("Concurrent requests", 1, 12, 5)
    st.caption("Data source: Yahoo Finance via yfinance. Quotes may be delayed, incomplete or rate-limited. Reversal signals are rule-based candle patterns, not guaranteed reversals.")

st.subheader("1. Choose the NSE universe")
st.write("Upload a CSV containing a `Symbol` column (NSE symbols without `.NS`), or paste symbols below. For a broad scan, upload the NSE equity list or your preferred universe.")
uploaded = st.file_uploader("Upload NSE symbols CSV", type=["csv"])
default_symbols = "RELIANCE,HDFCBANK,ICICIBANK,SBIN,INFY,TCS,LT,BEL,HAL,MAZDOCK,CDSL,MCX"
symbols_text = st.text_area("Or paste comma/newline-separated NSE symbols", value=default_symbols, height=90)

symbols = []
if uploaded is not None:
    try:
        universe = pd.read_csv(uploaded)
        cols = {c.lower().strip(): c for c in universe.columns}
        if "symbol" not in cols:
            st.error("CSV must contain a column named `Symbol`.")
        else:
            symbols = universe[cols["symbol"]].dropna().astype(str).str.strip().tolist()
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
if not symbols:
    symbols = [s.strip().upper().replace(".NS", "") for s in symbols_text.replace("\n", ",").split(",") if s.strip()]
symbols = list(dict.fromkeys(s for s in symbols if s and s not in {"SYMBOL", "NAN"}))

st.write(f"**{len(symbols)} symbols loaded**")
scan = st.button("🔎 Run scanner", type="primary", use_container_width=True)

@st.cache_data(ttl=900, show_spinner=False)
def get_history(symbol):
    return yf.download(f"{symbol}.NS", period="6mo", interval="1d", auto_adjust=True,
                       progress=False, threads=False, timeout=20)

@st.cache_data(ttl=3600, show_spinner=False)
def get_market_cap(symbol):
    info = yf.Ticker(f"{symbol}.NS").get_info()
    return info.get("marketCap")

def get_column(frame, name):
    col = frame[name]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col.astype(float)

def inspect_symbol(symbol):
    try:
        hist = get_history(symbol)
        if hist is None or hist.empty or len(hist) < 55:
            return None, f"{symbol}: insufficient price history"

        close = get_column(hist, "Close").dropna()
        high = get_column(hist, "High").reindex(close.index)
        low = get_column(hist, "Low").reindex(close.index)
        open_ = get_column(hist, "Open").reindex(close.index)
        volume = get_column(hist, "Volume").reindex(close.index).fillna(0)
        if len(close) < 55:
            return None, f"{symbol}: insufficient price history"

        last = float(close.iloc[-1])
        ref = float(close.iloc[-64]) if len(close) >= 64 else float(close.iloc[0])
        ret = (last / ref - 1) * 100 if ref else None
        ema_series = {n: close.ewm(span=n, adjust=False).mean() for n in (10, 20, 50)}
        emas = {n: float(series.iloc[-1]) for n, series in ema_series.items()}
        near = [n for n, ema in emas.items() if abs(last / ema - 1) * 100 <= proximity]
        if not near or ret is None or ret <= min_return:
            return None, None
        cap = get_market_cap(symbol)
        if cap is None or cap / 1e7 <= min_mcap:
            return None, None

        # A support interaction means the candle's low came within the EMA band.
        # Check the latest candle and the two preceding candles for an EMA test.
        ema_touches = []
        for n, series in ema_series.items():
            recent_ema = series.iloc[-3:]
            recent_low = low.iloc[-3:]
            touched = ((recent_low <= recent_ema * (1 + proximity / 100)) &
                       (recent_low >= recent_ema * (1 - proximity / 100))).any()
            if touched:
                ema_touches.append(n)

        o, h, l, c = map(float, (open_.iloc[-1], high.iloc[-1], low.iloc[-1], close.iloc[-1]))
        prev_o, prev_c = float(open_.iloc[-2]), float(close.iloc[-2])
        body = abs(c - o)
        candle_range = max(h - l, 1e-9)
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        bullish = c > o
        hammer = bullish and lower_wick >= max(body * 1.8, candle_range * 0.35) and upper_wick <= candle_range * 0.30
        engulfing = bullish and prev_c < prev_o and o <= prev_c and c >= prev_o
        ema_reclaim = any(float(close.iloc[-2]) < float(ema_series[n].iloc[-2]) and
                          c > float(ema_series[n].iloc[-1]) for n in (10, 20, 50))
        bullish_reversal = bool(ema_touches and (hammer or engulfing or ema_reclaim))

        avg_vol = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else 0
        vol_ratio = float(volume.iloc[-1] / avg_vol) if avg_vol > 0 else None
        volume_confirm = vol_ratio is not None and vol_ratio >= 1.5
        signal_parts = []
        if hammer:
            signal_parts.append("Hammer")
        if engulfing:
            signal_parts.append("Bullish engulfing")
        if ema_reclaim:
            signal_parts.append("EMA reclaim")
        if bullish_reversal:
            signal = "Bullish reversal" + (": " + ", ".join(signal_parts))
        elif ema_touches:
            signal = "EMA tested — wait for reversal"
        else:
            signal = "Near EMA — no reversal candle"

        nearest = min(near, key=lambda n: abs(last / emas[n] - 1))
        atr = float((high - low).rolling(14).mean().iloc[-1])
        support_ema = min((emas[n] for n in (ema_touches or near)), key=lambda x: abs(last - x))
        stop = min(l, support_ema - 0.5 * atr) if atr > 0 else l
        risk = max(last - stop, 0)
        return {
            "Symbol": symbol, "Price (₹)": last, "Market cap (₹ Cr)": cap / 1e7,
            "3M return (%)": ret, "EMA match": ", ".join(map(str, near)),
            "Nearest EMA": f"EMA {nearest}", "Distance to EMA (%)": (last / emas[nearest] - 1) * 100,
            "Bullish signal": signal, "Reversal confirmed": bullish_reversal,
            "Volume (x avg)": vol_ratio, "Volume confirmation": volume_confirm,
            "Indicative entry (₹)": last if bullish_reversal else None,
            "Stop-loss (₹)": stop if bullish_reversal else None,
            "Target 1 (2R) (₹)": last + 2 * risk if bullish_reversal else None,
            "Target 2 (3R) (₹)": last + 3 * risk if bullish_reversal else None,
            "EMA 10 (₹)": emas[10], "EMA 20 (₹)": emas[20], "EMA 50 (₹)": emas[50],
            "As of": str(close.index[-1].date())
        }, None
    except Exception as e:
        return None, f"{symbol}: {str(e)[:120]}"

if scan:
    if not symbols:
        st.warning("Add at least one NSE symbol or upload a CSV.")
    else:
        results, errors = [], []
        progress = st.progress(0, text="Starting scan…")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(inspect_symbol, s): s for s in symbols}
            done = 0
            for future in as_completed(futures):
                row, err = future.result()
                if row:
                    results.append(row)
                if err:
                    errors.append(err)
                done += 1
                progress.progress(done / len(futures), text=f"Checked {done} of {len(futures)} symbols")
        progress.empty()
        st.session_state["scan_results"] = pd.DataFrame(results)
        st.session_state["scan_errors"] = errors

if "scan_results" in st.session_state:
    df = st.session_state["scan_results"]
    a, b, c = st.columns(3)
    a.metric("Qualified stocks", len(df))
    b.metric("Symbols checked", len(symbols))
    c.metric("Scan date", pd.Timestamp.now(tz="Asia/Kolkata").strftime("%d %b %Y, %I:%M %p"))
    if df.empty:
        st.info("No stocks matched all filters in this scan. Try a larger universe or review the thresholds.")
    else:
        df = df.sort_values(["Reversal confirmed", "3M return (%)", "Market cap (₹ Cr)"],
                            ascending=[False, False, False])
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                         "Price (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                         "Market cap (₹ Cr)": st.column_config.NumberColumn(format="%.0f"),
                         "3M return (%)": st.column_config.NumberColumn(format="%.2f%%"),
                         "Distance to EMA (%)": st.column_config.NumberColumn(format="%.2f%%"),
                         "Volume (x avg)": st.column_config.NumberColumn(format="%.2fx"),
                         "Indicative entry (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                         "Stop-loss (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                         "Target 1 (2R) (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                         "Target 2 (3R) (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                         "EMA 10 (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                         "EMA 20 (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                         "EMA 50 (₹)": st.column_config.NumberColumn(format="₹%.2f"),
                     })
        st.download_button("⬇️ Download results CSV", df.to_csv(index=False).encode("utf-8"),
                           "nse_momentum_pullback_results.csv", "text/csv")
    errors = st.session_state.get("scan_errors", [])
    if errors:
        with st.expander(f"Skipped / data errors ({len(errors)})"):
            st.write("\n".join(errors[:200]))

st.divider()
st.subheader("How the scanner qualifies a stock")
st.markdown("""
- **Market cap:** latest Yahoo Finance market-cap field must be strictly greater than the threshold.
- **3-month move:** latest adjusted daily close versus approximately 63 trading sessions earlier must be strictly greater than the threshold.
- **Pullback:** latest close must be within the selected percentage above or below at least one of EMA 10, EMA 20 or EMA 50.
- **EMA calculation:** daily closing prices, exponentially weighted with `adjust=False`.
- **Bullish reversal signals:** hammer candle, bullish engulfing candle, or a close reclaiming an EMA after the prior close was below it. The candle must also have tested an EMA support zone within the last three sessions.
- **Volume confirmation:** latest volume at least 1.5× the average of the prior 20 sessions; shown as an additional confirmation, not a mandatory filter.
- **Trade levels:** indicative entry at the latest close, stop below the latest candle low / EMA support with an ATR buffer, and 2R/3R targets. These are screening estimates, not order instructions.

**Important:** This is a screening tool, not an execution system or investment recommendation. Yahoo Finance is an unofficial data source for this purpose; market-cap fields and prices can be stale or missing, and the app does not guarantee real-time NSE data. Verify candidates against your broker/NSE data before trading. The 3-month return uses the available trading sessions and can differ slightly from a calendar-quarter return.
""")
