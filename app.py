import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import math

import pandas as pd
import streamlit as st
import yfinance as yf

ROOT = Path(__file__).parent
HISTORY_FILE = ROOT / "daily_scan_results.csv"
UNIVERSE_FILE = ROOT / "universe.csv"
IST = "Asia/Kolkata"
BENCHMARK = "^CRSLDX"  # Nifty 500 on Yahoo Finance


def fetch_nse_symbols():
    """Refresh from the already-maintained universe file/workflow where possible."""
    # NSE blocks some automated requests. The GitHub Actions refresh_universe.py
    # remains the authoritative universe refresher; this button reads that file.
    if UNIVERSE_FILE.exists():
        frame = pd.read_csv(UNIVERSE_FILE)
        col = next((c for c in frame.columns if str(c).strip().lower() == "symbol"), None)
        if col:
            symbols = sorted(set(frame[col].dropna().astype(str).str.strip().str.upper()))
            if len(symbols) >= 500:
                return symbols
    raise RuntimeError("universe.csv is unavailable or contains fewer than 500 symbols. Run the GitHub Actions universe refresh first.")


def load_symbols(refresh=False):
    if UNIVERSE_FILE.exists():
        try:
            frame = pd.read_csv(UNIVERSE_FILE)
            col = next((c for c in frame.columns if str(c).strip().lower() == "symbol"), None)
            if col:
                symbols = sorted(set(frame[col].dropna().astype(str).str.upper().str.replace(".NS", "", regex=False).str.strip()))
                if symbols:
                    return symbols
        except Exception:
            pass
    return ["RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "INFY", "TCS", "LT", "BEL", "HAL", "MAZDOCK", "CDSL", "MCX"]


def _download_history(symbol, period="1y"):
    hist = yf.download(f"{symbol}.NS", period=period, interval="1d", auto_adjust=True,
                       progress=False, threads=False, timeout=25)
    if hist is None or hist.empty:
        return None
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    return hist


def _column(hist, name):
    value = hist[name]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return value.astype(float)


def benchmark_return():
    try:
        hist = yf.download(BENCHMARK, period="6mo", interval="1d", auto_adjust=True,
                           progress=False, threads=False, timeout=25)
        if hist is None or hist.empty:
            return None
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        close = _column(hist, "Close").dropna()
        if len(close) < 127:
            return None
        return float((close.iloc[-1] / close.iloc[-127] - 1) * 100)
    except Exception:
        return None


def inspect_symbol(symbol, min_mcap=1000, min_return=50, min_price=90,
                   min_above_low=1.5, min_turnover=10, use_ema=False,
                   proximity=2, use_peg=False, max_peg=2.0):
    try:
        hist = _download_history(symbol, "1y")
        if hist is None or len(hist) < 127:
            return None, f"{symbol}: insufficient daily history"
        close = _column(hist, "Close").dropna()
        high = _column(hist, "High").reindex(close.index)
        low = _column(hist, "Low").reindex(close.index)
        volume = _column(hist, "Volume").reindex(close.index).fillna(0)
        if len(close) < 127:
            return None, f"{symbol}: insufficient daily history"

        last = float(close.iloc[-1])
        low_52w = float(low.tail(252).min())
        ret_6m = float((last / float(close.iloc[-127]) - 1) * 100)
        avg_turnover_cr = float((close.tail(30) * volume.tail(30)).mean() / 1e7)
        if last <= min_price or low_52w <= 0 or last / low_52w < min_above_low:
            return None, None
        if ret_6m <= min_return or avg_turnover_cr < min_turnover:
            return None, None

        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.get_info() or {}
        market_cap = info.get("marketCap")
        market_cap_cr = float(market_cap) / 1e7 if market_cap else None
        if market_cap_cr is None or market_cap_cr <= min_mcap:
            return None, None
        sector = info.get("sector") or "Unknown"
        peg = info.get("pegRatio")
        try:
            peg = float(peg) if peg is not None else None
            if peg is not None and (not math.isfinite(peg) or peg <= 0):
                peg = None
        except (TypeError, ValueError):
            peg = None
        if use_peg and (peg is None or peg > max_peg):
            return None, None

        ema_values = {n: float(close.ewm(span=n, adjust=False).mean().iloc[-1]) for n in (10, 20, 50)}
        ema_near = [n for n, value in ema_values.items() if abs(last / value - 1) * 100 <= proximity]
        if use_ema and not ema_near:
            return None, None

        return {
            "Symbol": symbol,
            "Price (₹)": last,
            "Market cap (₹ Cr)": market_cap_cr,
            "6M return (%)": ret_6m,
            "52W low (₹)": low_52w,
            "Above 52W low (x)": last / low_52w,
            "30D avg turnover (₹ Cr)": avg_turnover_cr,
            "Sector": sector,
            "PEG": peg,
            "EMA match": ", ".join(map(str, ema_near)) if ema_near else "—",
            "EMA 10 (₹)": ema_values[10],
            "EMA 20 (₹)": ema_values[20],
            "EMA 50 (₹)": ema_values[50],
            "As of": str(close.index[-1].date()),
        }, None
    except Exception as exc:
        return None, f"{symbol}: {str(exc)[:160]}"


def run_scan(symbols, min_mcap=1000, min_return=50, min_price=90,
             min_above_low=1.5, min_turnover=10, min_rs=80,
             strong_sector_only=True, use_ema=False, proximity=2,
             use_peg=False, max_peg=2.0, workers=8):
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(inspect_symbol, symbol, min_mcap, min_return, min_price,
                        min_above_low, min_turnover, use_ema, proximity,
                        use_peg, max_peg): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            row, err = future.result()
            if row:
                rows.append(row)
            if err:
                errors.append(err)

    df = pd.DataFrame(rows)
    if df.empty:
        return df, errors

    # RS score is the percentile rank of each stock's 6-month excess return
    # over the Nifty 500 benchmark, among stocks with usable data in this scan.
    nifty500_return = benchmark_return()
    df["Nifty 500 6M return (%)"] = nifty500_return
    if nifty500_return is None:
        errors.append("Nifty 500 benchmark data unavailable; RS >= 80 cannot be verified, so no results are being passed through.")
        return pd.DataFrame(), errors
    else:
        df["Excess return vs Nifty 500 (%)"] = df["6M return (%)"] - nifty500_return
        df["RS score"] = (df["Excess return vs Nifty 500 (%)"].rank(pct=True) * 100).round(1)

    # Sector strength: median 6-month return for each sector; retain sectors
    # that both beat the Nifty 500 and fall in the top 30% of sector medians.
    sector_perf = df[df["Sector"] != "Unknown"].groupby("Sector")["6M return (%)"].median()
    if not sector_perf.empty:
        strong_names = set(sector_perf[sector_perf >= sector_perf.quantile(0.70)].index)
        strong_names = set(name for name in strong_names if sector_perf[name] > nifty500_return)
        df["Sector median 6M return (%)"] = df["Sector"].map(sector_perf)
        df["Strong sector"] = df["Sector"].isin(strong_names)
    else:
        df["Sector median 6M return (%)"] = None
        df["Strong sector"] = False

    df = df[df["RS score"] >= min_rs]
    if strong_sector_only:
        df = df[df["Strong sector"]]
    if not df.empty:
        df = df.sort_values(["RS score", "6M return (%)", "30D avg turnover (₹ Cr)"],
                            ascending=[False, False, False], na_position="last")
    return df, errors


def save_daily_results(df, symbols, errors):
    now = pd.Timestamp.now(tz=IST)
    scan_date = now.strftime("%Y-%m-%d")
    df = df.copy()
    if df.empty:
        df = pd.DataFrame(columns=["Symbol", "Price (₹)", "Market cap (₹ Cr)", "6M return (%)",
                                   "52W low (₹)", "Above 52W low (x)", "30D avg turnover (₹ Cr)",
                                   "Sector", "PEG", "RS score", "As of"])
    df.insert(0, "Scan date", scan_date)
    df.insert(1, "Scan time IST", now.strftime("%H:%M:%S"))
    df["Symbols checked"] = len(symbols)
    df["Scan errors"] = len(errors)
    old = pd.read_csv(HISTORY_FILE) if HISTORY_FILE.exists() else pd.DataFrame()
    if not old.empty and "Scan date" in old:
        old = old[old["Scan date"].astype(str) != scan_date]
    combined = pd.concat([old, df], ignore_index=True)
    combined.to_csv(HISTORY_FILE, index=False)
    return combined


# GitHub Actions scheduled mode uses the same baseline defaults as the dashboard.
if os.getenv("NSE_DAILY_SCAN") == "1":
    symbols = load_symbols()
    results, errors = run_scan(symbols)
    save_daily_results(results, symbols, errors)
    print(f"Daily scan completed: {len(symbols)} symbols, {len(results)} matches, {len(errors)} errors")
    raise SystemExit(0)


st.set_page_config(page_title="NSE Strength & Quality Scanner", page_icon="📈", layout="wide")
st.title("📈 NSE Strength & Quality Scanner")
st.caption("End-of-day stock screening for swing and positional research. Not a live intraday scanner.")

with st.sidebar:
    st.header("Core filters")
    min_mcap = st.number_input("Minimum market cap (₹ crore)", min_value=1, value=1000, step=100)
    min_return = st.number_input("Minimum 6-month return (%)", min_value=0.0, value=50.0, step=5.0)
    min_price = st.number_input("Minimum share price (₹)", min_value=1.0, value=90.0, step=5.0)
    min_above_low = st.number_input("Minimum price / 52-week low", min_value=1.0, value=1.5, step=0.1)
    min_turnover = st.number_input("Minimum 30-day average turnover (₹ crore)", min_value=0.0, value=10.0, step=1.0)
    min_rs = st.slider("Minimum Relative Strength score", 0, 100, 80)
    strong_sector_only = st.checkbox("Only strong sectors", value=True)
    st.header("Optional filters")
    use_ema = st.checkbox("Enable EMA 10/20/50 proximity filter", value=False)
    proximity = st.number_input("EMA proximity (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.5)
    use_peg = st.checkbox("Enable PEG filter (missing PEG values excluded)", value=False)
    max_peg = st.number_input("Maximum PEG ratio", min_value=0.1, value=2.0, step=0.1)
    workers = st.slider("Concurrent requests", 1, 12, 6)

st.subheader("1. NSE universe")
symbols = load_symbols()
st.write(f"**{len(symbols):,} symbols loaded** from `universe.csv`.")
if st.button("↻ Reload saved NSE universe", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

if st.button("🔎 Run end-of-day scanner", type="primary", use_container_width=True):
    if not symbols:
        st.warning("No symbols found. Check universe.csv in your GitHub repository.")
    else:
        with st.spinner(f"Scanning {len(symbols):,} symbols. This may take some time..."):
            results, errors = run_scan(
                symbols, min_mcap, min_return, min_price, min_above_low,
                min_turnover, min_rs, strong_sector_only, use_ema,
                proximity, use_peg, max_peg, workers
            )
            save_daily_results(results, symbols, errors)
            st.session_state["latest_results"] = results
            st.session_state["latest_errors"] = errors
        st.success(f"Scan completed: {len(results)} matches; {len(errors)} data errors.")

if "latest_results" in st.session_state:
    st.subheader("2. Latest scan results")
    latest = st.session_state["latest_results"]
    if latest.empty:
        st.info("No stocks matched all selected filters. Try relaxing one or more filters.")
    else:
        st.dataframe(latest, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download latest results CSV", latest.to_csv(index=False).encode("utf-8"),
                           "nse_strength_scanner_results.csv", "text/csv")
    errs = st.session_state.get("latest_errors", [])
    if errs:
        with st.expander(f"Data errors ({len(errs)})"):
            st.write(errs[:100])

st.divider()
st.subheader("3. Saved daily scan history")
if HISTORY_FILE.exists():
    try:
        history = pd.read_csv(HISTORY_FILE)
        if not history.empty and "Scan date" in history:
            dates = sorted(history["Scan date"].dropna().astype(str).unique(), reverse=True)
            selected = st.selectbox("Choose scan date", dates)
            day = history[history["Scan date"].astype(str) == selected].copy()
            a, b, c = st.columns(3)
            a.metric("Matching stocks", len(day))
            b.metric("Symbols checked", int(day["Symbols checked"].max()) if "Symbols checked" in day and len(day) else 0)
            c.metric("Scan date", selected)
            if not day.empty:
                st.dataframe(day.drop(columns=["Scan date", "Scan time IST"], errors="ignore"),
                             use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download selected day's results", day.to_csv(index=False).encode("utf-8"),
                                   f"nse_scan_{selected}.csv", "text/csv")
        else:
            st.info("No saved daily scans yet.")
    except Exception as exc:
        st.warning(f"Could not read saved history: {exc}")
else:
    st.info("No saved daily scans yet. The next scheduled scan will appear here.")

st.divider()
st.subheader("4. Filter definitions and data notes")
st.markdown("""
- **Relative Strength score:** percentile rank of each stock's 6-month excess return over the Nifty 500 benchmark, across the scanned symbols. The benchmark is Yahoo Finance's `^CRSLDX` series; if benchmark data is unavailable, the RS filter is not applied and scores are blank.
- **Strong sectors:** sectors whose median 6-month stock return is in the top 30% of available sectors and, when benchmark data is available, exceeds the Nifty 500 return.
- **Price and liquidity:** latest EOD close above ₹90 by default; at least 1.5× the 52-week low; 30-day average daily turnover is the mean of close × volume, in ₹ crore.
- **Market cap and momentum:** market cap above ₹1,000 crore and 6-month return above 50% by default.
- **EMA filter:** optional; when enabled, close must be within the selected percentage of EMA 10, 20 or 50.
- **PEG:** optional; when enabled, only positive PEG values at or below the selected maximum are retained. Missing PEG values are excluded in that mode.
- All calculations use daily end-of-day data. Intraday screening and the agreed swing/positional backtests are separate future stages.

**Data caution:** Yahoo Finance via yfinance is unofficial and may be delayed, incomplete, rate-limited or inaccurate. Market cap, sector and PEG fields can be missing or stale. Verify candidates with NSE and your broker. This is a research screener, not investment advice.
""")
