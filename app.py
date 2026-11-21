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
BENCHMARK = "^CRSLDX"  # Yahoo Finance symbol used here for Nifty 500

def load_symbols():
    if UNIVERSE_FILE.exists():
        try:
            frame = pd.read_csv(UNIVERSE_FILE)
            col = next((c for c in frame.columns if str(c).strip().lower() == "symbol"), None)
            if col:
                symbols = sorted(set(frame[col].dropna().astype(str).str.upper()
                                     .str.replace(".NS", "", regex=False).str.strip()))
                if symbols:
                    return symbols
        except Exception:
            pass
    return ["RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN", "INFY", "TCS",
            "LT", "BEL", "HAL", "MAZDOCK", "CDSL", "MCX"]

def download_history(symbol):
    hist = yf.download(f"{symbol}.NS", period="1y", interval="1d",
                       auto_adjust=True, progress=False, threads=False, timeout=25)
    if hist is None or hist.empty:
        return None
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    return hist

def series(hist, name):
    value = hist[name]
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    return value.astype(float)

def benchmark_return():
    try:
        hist = yf.download(BENCHMARK, period="6mo", interval="1d",
                           auto_adjust=True, progress=False, threads=False, timeout=25)
        if hist is None or hist.empty:
            return None
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        close = series(hist, "Close").dropna()
        if len(close) < 100:
            return None
        return float((close.iloc[-1] / close.iloc[0] - 1) * 100)
    except Exception:
        return None

def inspect_symbol(symbol, scan_type, min_mcap, min_return, min_price,
                   min_above_low, min_turnover, proximity, use_peg, max_peg):
    try:
        hist = download_history(symbol)
        if hist is None or len(hist) < 127:
            return None, f"{symbol}: insufficient daily history"
        close = series(hist, "Close").dropna()
        high = series(hist, "High").reindex(close.index)
        low = series(hist, "Low").reindex(close.index)
        open_price = series(hist, "Open").reindex(close.index)
        volume = series(hist, "Volume").reindex(close.index).fillna(0)

        last = float(close.iloc[-1])
        low_52w = float(low.tail(252).min())
        ret_6m = float((last / float(close.iloc[-127]) - 1) * 100)
        avg_turnover_cr = float((close.tail(30) * volume.tail(30)).mean() / 1e7)
        if last <= min_price or low_52w <= 0 or last / low_52w < min_above_low:
            return None, None
        if avg_turnover_cr < min_turnover:
            return None, None

        # EMA calculations use end-of-day adjusted closing prices.
        ema = {n: close.ewm(span=n, adjust=False).mean() for n in (10, 20, 50)}
        ema_values = {n: float(v.iloc[-1]) for n, v in ema.items()}
        ema_near = [n for n, value in ema_values.items()
                    if abs(last / value - 1) * 100 <= proximity]

        # Pullback qualification: a substantial prior six-month advance,
        # rising EMA50, price holding close to EMA20/EMA50, and not deeply
        # below EMA50. This deliberately does NOT require today's 6M return
        # to remain above the momentum scanner's 50% threshold.
        six_month_high = float(high.tail(127).max())
        six_month_low = float(low.tail(127).min())
        prior_advance_pct = ((six_month_high / six_month_low) - 1) * 100 if six_month_low > 0 else 0
        ema50_rising = len(ema[50]) >= 11 and float(ema[50].iloc[-1]) > float(ema[50].iloc[-11])
        near_ema20 = abs(last / ema_values[20] - 1) * 100 <= proximity
        near_ema50 = abs(last / ema_values[50] - 1) * 100 <= proximity
        holding_ema50 = last >= ema_values[50] * 0.98
        pullback_ok = (prior_advance_pct >= 50 and ema50_rising and
                       (near_ema20 or near_ema50) and holding_ema50)

        if scan_type == "Momentum":
            if ret_6m < min_return:
                return None, None
        elif not pullback_ok:
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

        if scan_type == "Pullback":
            # A first-pass recovery label, not a guaranteed reversal signal.
            turning_up = (last > float(close.iloc[-2]) and
                          last > float(open_price.iloc[-1]))
            setup = "Pullback turning up" if turning_up else "Near EMA support"
        else:
            setup = "Momentum"

        return {
            "Symbol": symbol,
            "Setup": setup,
            "Price (₹)": last,
            "Market cap (₹ Cr)": market_cap_cr,
            "6M return (%)": ret_6m,
            "Prior 6M swing advance (%)": prior_advance_pct,
            "52W low (₹)": low_52w,
            "Above 52W low (x)": last / low_52w,
            "30D avg turnover (₹ Cr)": avg_turnover_cr,
            "Sector": sector,
            "PEG": peg,
            "EMA match": ", ".join(str(n) for n, matched in ((20, near_ema20), (50, near_ema50)) if matched)
                         if (near_ema20 or near_ema50) else "—",
            "EMA 10 (₹)": ema_values[10],
            "EMA 20 (₹)": ema_values[20],
            "EMA 50 (₹)": ema_values[50],
            "As of": str(close.index[-1].date()),
        }, None
    except Exception as exc:
        return None, f"{symbol}: {str(exc)[:160]}"

def run_scan(symbols, scan_type, min_mcap=1000, min_return=50, min_price=90,
             min_above_low=1.5, min_turnover=10, min_rs=80,
             strong_sector_only=True, proximity=2, use_peg=False,
             max_peg=2.0, workers=6):
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(inspect_symbol, symbol, scan_type, min_mcap, min_return,
                        min_price, min_above_low, min_turnover, proximity,
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

    nifty500_return = benchmark_return()
    df["Nifty 500 6M return (%)"] = nifty500_return
    if nifty500_return is None:
        errors.append("Nifty 500 benchmark data unavailable; RS score cannot be verified.")
        return pd.DataFrame(), errors

    df["Excess return vs Nifty 500 (%)"] = df["6M return (%)"] - nifty500_return
    df["RS score"] = (df["Excess return vs Nifty 500 (%)"].rank(pct=True) * 100).round(1)

    sector_perf = df[df["Sector"] != "Unknown"].groupby("Sector")["6M return (%)"].median()
    if not sector_perf.empty:
        strong_names = set(sector_perf[sector_perf >= sector_perf.quantile(0.70)].index)
        strong_names = {name for name in strong_names if sector_perf[name] > nifty500_return}
        df["Sector median 6M return (%)"] = df["Sector"].map(sector_perf)
        df["Strong sector"] = df["Sector"].isin(strong_names)
    else:
        df["Sector median 6M return (%)"] = None
        df["Strong sector"] = False

    df = df[df["RS score"] >= min_rs]
    if strong_sector_only:
        df = df[df["Strong sector"]]
    if not df.empty:
        sort_cols = ["RS score", "Prior 6M swing advance (%)", "30D avg turnover (₹ Cr)"]
        df = df.sort_values(sort_cols, ascending=[False, False, False], na_position="last")
    return df, errors

def save_daily_results(df, symbols, errors):
    now = pd.Timestamp.now(tz=IST)
    scan_date = now.strftime("%Y-%m-%d")
    saved = df.copy()
    if saved.empty:
        saved = pd.DataFrame(columns=["Symbol", "Setup", "Price (₹)", "Market cap (₹ Cr)",
                                      "6M return (%)", "RS score", "As of"])
    saved.insert(0, "Scan date", scan_date)
    saved.insert(1, "Scan time IST", now.strftime("%H:%M:%S"))
    saved["Symbols checked"] = len(symbols)
    saved["Scan errors"] = len(errors)
    old = pd.read_csv(HISTORY_FILE) if HISTORY_FILE.exists() else pd.DataFrame()
    if not old.empty and "Scan date" in old:
        old = old[old["Scan date"].astype(str) != scan_date]
    pd.concat([old, saved], ignore_index=True).to_csv(HISTORY_FILE, index=False)

# Scheduled GitHub Actions scan retains the original momentum-screen behaviour.
if os.getenv("NSE_DAILY_SCAN") == "1":
    symbols = load_symbols()
    results, errors = run_scan(symbols, "Momentum")
    save_daily_results(results, symbols, errors)
    print(f"Daily momentum scan: {len(symbols)} symbols, {len(results)} matches, {len(errors)} errors")
    raise SystemExit(0)

st.set_page_config(page_title="NSE Strength & EMA Pullback Scanner",
                   page_icon="📈", layout="wide")
st.title("📈 NSE Strength & EMA Pullback Scanner")
st.caption("End-of-day research scanner for swing and positional setups. Not a live intraday scanner.")

with st.sidebar:
    st.header("Scanner type")
    scan_type = st.radio("Choose what to find", ["Momentum", "Pullback"],
                         help="Pullback finds stocks with a prior strong advance that are now near a rising EMA20 or EMA50.")
    st.header("Core filters")
    min_mcap = st.number_input("Minimum market cap (₹ crore)", min_value=1, value=1000, step=100)
    min_return = st.number_input("Minimum 6-month return (%)", min_value=0.0, value=50.0, step=5.0,
                                 disabled=(scan_type == "Pullback"))
    min_price = st.number_input("Minimum share price (₹)", min_value=1.0, value=90.0, step=5.0)
    min_above_low = st.number_input("Minimum price / 52-week low", min_value=1.0, value=1.5, step=0.1)
    min_turnover = st.number_input("Minimum 30-day average turnover (₹ crore)", min_value=0.0, value=10.0, step=1.0)
    min_rs = st.slider("Minimum Relative Strength score", 0, 100, 80)
    strong_sector_only = st.checkbox("Only strong sectors", value=True)
    st.header("EMA pullback settings")
    proximity = st.number_input("Distance from EMA20/EMA50 (%)", min_value=0.5, max_value=8.0,
                                value=2.0, step=0.5)
    use_peg = st.checkbox("Enable PEG filter (missing PEG values excluded)", value=False)
    max_peg = st.number_input("Maximum PEG ratio", min_value=0.1, value=2.0, step=0.1)
    workers = st.slider("Concurrent requests", 1, 12, 6)

st.subheader("1. NSE universe")
symbols = load_symbols()
st.write(f"**{len(symbols):,} symbols loaded** from `universe.csv`.")
if st.button("↻ Reload saved NSE universe", use_container_width=True):
    st.rerun()

if st.button("🔎 Run end-of-day scanner", type="primary", use_container_width=True):
    if not symbols:
        st.warning("No symbols found. Check universe.csv in your GitHub repository.")
    else:
        with st.spinner(f"Scanning {len(symbols):,} symbols. This may take some time..."):
            results, errors = run_scan(
                symbols, scan_type, min_mcap, min_return, min_price, min_above_low,
                min_turnover, min_rs, strong_sector_only, proximity, use_peg,
                max_peg, workers
            )
            save_daily_results(results, symbols, errors)
            st.session_state["latest_results"] = results
            st.session_state["latest_errors"] = errors
            st.session_state["latest_scan_type"] = scan_type
        st.success(f"{scan_type} scan completed: {len(results)} matches; {len(errors)} data errors.")

if "latest_results" in st.session_state:
    st.subheader(f"2. Latest {st.session_state.get('latest_scan_type', '')} scan results")
    latest = st.session_state["latest_results"]
    if latest.empty:
        st.info("No stocks matched all selected filters. You can review the filters, or try a later scan.")
    else:
        st.dataframe(latest, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download latest results CSV",
                           latest.to_csv(index=False).encode("utf-8"),
                           "nse_scanner_results.csv", "text/csv")
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
                st.download_button("⬇️ Download selected day's results",
                                   day.to_csv(index=False).encode("utf-8"),
                                   f"nse_scan_{selected}.csv", "text/csv")
        else:
            st.info("No saved daily scans yet.")
    except Exception as exc:
        st.warning(f"Could not read saved history: {exc}")
else:
    st.info("No saved daily scans yet.")

st.divider()
st.subheader("4. How the scanner works")
st.markdown("""
- **Momentum mode:** keeps the existing minimum six-month return of 50% by default.
- **Pullback mode:** does not require the current six-month return to remain above 50%. Instead, the stock must have made at least a 50% rise from its six-month low to its six-month high, have a rising EMA50, and be within the chosen distance (2% by default) of EMA20 or EMA50 while remaining no more than 2% below EMA50.
- **Pullback labels:** “Near EMA support” means the stock is in the pullback zone. “Pullback turning up” means the latest close is above the previous close and above the day's open. These are screening labels, not confirmed buy signals.
- **Relative Strength:** percentile rank of six-month excess return versus the Nifty 500 proxy series `^CRSLDX`, among stocks that pass the selected scan's initial filters.
- **Strong sectors:** sectors whose median six-month return is in the top 30% of available sectors and exceeds the benchmark return.
- **Liquidity and quality:** close above ₹90, at least 1.5× the 52-week low, 30-day average daily turnover at least ₹10 crore, and market cap above ₹1,000 crore by default.
- **PEG:** optional; missing PEG values are excluded when the filter is enabled.
- All calculations use daily end-of-day data. Intraday screening and backtesting remain future stages.
**Data caution:** Yahoo Finance via yfinance is unofficial and may be delayed, incomplete, rate-limited or inaccurate. Market cap, sector and PEG can be missing or stale. Verify candidates with NSE and your broker. This is a research screener, not investment advice.
""")
