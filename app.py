import os
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

ROOT = Path(__file__).parent
HISTORY_FILE = ROOT / "daily_scan_results.csv"
UNIVERSE_FILE = ROOT / "universe.csv"
IST = "Asia/Kolkata"

NSE_EQUITY_CSV = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"


def fetch_nse_symbols():
    """Fetch the current main-board equity symbol list from NSE's official CSV."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    response = requests.get(NSE_EQUITY_CSV, headers=headers, timeout=30)
    response.raise_for_status()
    frame = pd.read_csv(BytesIO(response.content), encoding="latin1")
    frame.columns = [str(c).replace("\ufeff", "").strip().upper() for c in frame.columns]
    symbol_col = next((c for c in frame.columns if c in {"SYMBOL", "SYMBOL NAME"}), None)
    if not symbol_col:
        raise ValueError("NSE equity CSV did not contain a SYMBOL column")
    symbols = (frame[symbol_col].dropna().astype(str).str.strip().str.upper()
               .str.replace(r"[^A-Z0-9&_-]", "", regex=True))
    symbols = sorted(set(x for x in symbols if x and x not in {"NAN", "NULL"}))
    if len(symbols) < 500:
        raise ValueError(f"NSE returned only {len(symbols)} symbols; refusing to replace the existing universe")
    UNIVERSE_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Symbol": symbols}).to_csv(UNIVERSE_FILE, index=False)
    return symbols


def load_symbols(refresh=False):
    if refresh:
        try:
            return fetch_nse_symbols()
        except Exception as exc:
            print(f"WARNING: Could not refresh NSE symbol list: {exc}")
    if UNIVERSE_FILE.exists():
        try:
            frame = pd.read_csv(UNIVERSE_FILE)
            col = next((c for c in frame.columns if c.lower().strip() == "symbol"), None)
            if col:
                symbols = sorted(set(frame[col].dropna().astype(str).str.upper().str.replace(".NS", "", regex=False).str.strip()))
                if symbols:
                    return symbols
        except Exception:
            pass
    return ["RELIANCE","HDFCBANK","ICICIBANK","SBIN","INFY","TCS","LT","BEL","HAL","MAZDOCK","CDSL","MCX"]

def inspect_symbol(symbol, min_mcap=1000, min_return=60, proximity=2):
    try:
        hist = yf.download(f"{symbol}.NS", period="6mo", interval="1d", auto_adjust=True,
                           progress=False, threads=False, timeout=25)
        if hist is None or hist.empty or len(hist) < 55:
            return None, f"{symbol}: insufficient price history"
        def series(name):
            x = hist[name]
            if isinstance(x, pd.DataFrame):
                x = x.iloc[:, 0]
            return x.astype(float)
        close = series("Close").dropna()
        high, low = series("High").reindex(close.index), series("Low").reindex(close.index)
        op, volume = series("Open").reindex(close.index), series("Volume").reindex(close.index).fillna(0)
        if len(close) < 55:
            return None, f"{symbol}: insufficient price history"
        last = float(close.iloc[-1])
        ref = float(close.iloc[-64]) if len(close) >= 64 else float(close.iloc[0])
        ret = (last / ref - 1) * 100 if ref else None
        emas_s = {n: close.ewm(span=n, adjust=False).mean() for n in (10,20,50)}
        emas = {n: float(v.iloc[-1]) for n,v in emas_s.items()}
        near = [n for n,e in emas.items() if abs(last/e-1)*100 <= proximity]
        if not near or ret is None or ret <= min_return:
            return None, None
        cap = yf.Ticker(f"{symbol}.NS").get_info().get("marketCap")
        if cap is None or cap/1e7 <= min_mcap:
            return None, None
        touches = []
        for n, ema in emas_s.items():
            if ((low.iloc[-3:] <= ema.iloc[-3:]*(1+proximity/100)) &
                (low.iloc[-3:] >= ema.iloc[-3:]*(1-proximity/100))).any():
                touches.append(n)
        o,h,l,c = map(float,(op.iloc[-1],high.iloc[-1],low.iloc[-1],close.iloc[-1]))
        po,pc = float(op.iloc[-2]),float(close.iloc[-2])
        body=abs(c-o); rng=max(h-l,1e-9)
        lower=min(o,c)-l; upper=h-max(o,c)
        hammer=(c>o and lower>=max(body*1.8,rng*.35) and upper<=rng*.30)
        engulf=(c>o and pc<po and o<=pc and c>=po)
        reclaim=any(float(close.iloc[-2])<float(emas_s[n].iloc[-2]) and c>float(emas_s[n].iloc[-1]) for n in (10,20,50))
        reversal=bool(touches and (hammer or engulf or reclaim))
        avg=float(volume.iloc[-21:-1].mean()) if len(volume)>=21 else 0
        vol_ratio=float(volume.iloc[-1]/avg) if avg>0 else None
        signal_parts=[name for flag,name in [(hammer,"Hammer"),(engulf,"Bullish engulfing"),(reclaim,"EMA reclaim")] if flag]
        signal=("Bullish reversal: "+", ".join(signal_parts)) if reversal else ("EMA tested — wait for reversal" if touches else "Near EMA — no reversal candle")
        nearest=min(near,key=lambda n:abs(last/emas[n]-1))
        atr=float((high-low).rolling(14).mean().iloc[-1])
        support=min((emas[n] for n in (touches or near)),key=lambda x:abs(last-x))
        stop=min(l,support-.5*atr) if atr>0 else l
        risk=max(last-stop,0)
        return {
            "Symbol":symbol,"Price (₹)":last,"Market cap (₹ Cr)":cap/1e7,
            "3M return (%)":ret,"EMA match":", ".join(map(str,near)),
            "Nearest EMA":f"EMA {nearest}","Distance to EMA (%)":(last/emas[nearest]-1)*100,
            "Bullish signal":signal,"Reversal confirmed":reversal,
            "Volume (x avg)":vol_ratio,"Volume confirmation":bool(vol_ratio is not None and vol_ratio>=1.5),
            "Indicative entry (₹)":last if reversal else None,
            "Stop-loss (₹)":stop if reversal else None,
            "Target 1 (2R) (₹)":last+2*risk if reversal else None,
            "Target 2 (3R) (₹)":last+3*risk if reversal else None,
            "EMA 10 (₹)":emas[10],"EMA 20 (₹)":emas[20],"EMA 50 (₹)":emas[50],
            "As of":str(close.index[-1].date())
        }, None
    except Exception as exc:
        return None, f"{symbol}: {str(exc)[:160]}"

def run_scan(symbols, min_mcap=1000, min_return=60, proximity=2, workers=8):
    rows, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(inspect_symbol,s,min_mcap,min_return,proximity):s for s in symbols}
        for future in as_completed(futures):
            row, err = future.result()
            if row: rows.append(row)
            if err: errors.append(err)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Reversal confirmed","3M return (%)","Market cap (₹ Cr)"],
                            ascending=[False,False,False])
    return df, errors

def save_daily_results(df, symbols, errors):
    now = pd.Timestamp.now(tz=IST)
    scan_date = now.strftime("%Y-%m-%d")
    if df.empty:
        df = pd.DataFrame(columns=["Symbol","Price (₹)","Market cap (₹ Cr)","3M return (%)",
                                   "EMA match","Nearest EMA","Distance to EMA (%)","Bullish signal",
                                   "Reversal confirmed","Volume (x avg)","Volume confirmation",
                                   "Indicative entry (₹)","Stop-loss (₹)","Target 1 (2R) (₹)",
                                   "Target 2 (3R) (₹)","EMA 10 (₹)","EMA 20 (₹)","EMA 50 (₹)","As of"])
    df = df.copy()
    df.insert(0,"Scan date",scan_date)
    df.insert(1,"Scan time IST",now.strftime("%H:%M:%S"))
    df["Symbols checked"] = len(symbols)
    df["Scan errors"] = len(errors)
    old = pd.read_csv(HISTORY_FILE) if HISTORY_FILE.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old["Scan date"].astype(str) != scan_date]
    combined = pd.concat([old,df],ignore_index=True)
    combined.to_csv(HISTORY_FILE,index=False)
    return combined

# GitHub Actions scheduled mode: scan the committed universe and persist results.
if os.getenv("NSE_DAILY_SCAN") == "1":
    symbols = load_symbols(refresh=True)
    results, errors = run_scan(symbols)
    save_daily_results(results,symbols,errors)
    print(f"Daily scan completed: {len(symbols)} symbols, {len(results)} matches, {len(errors)} errors")
    raise SystemExit(0)

st.set_page_config(page_title="NSE Momentum Pullback Scanner",page_icon="📈",layout="wide")
st.title("📈 NSE Momentum Pullback Scanner")
st.caption("Daily momentum + EMA pullback scanner with bullish reversal signals and saved scan history.")

with st.sidebar:
    st.header("Scanner settings")
    min_mcap=st.number_input("Minimum market cap (₹ crore)",min_value=1,value=1000,step=100)
    min_return=st.number_input("Minimum 3-month return (%)",min_value=0.0,value=60.0,step=5.0)
    proximity=st.number_input("EMA proximity (%)",min_value=.1,max_value=10.0,value=2.0,step=.5)
    workers=st.slider("Concurrent requests",1,12,8)

st.subheader("1. Choose NSE universe")
st.caption("The scheduled scanner refreshes the official NSE main-board equity list automatically. You can also refresh it below. SME securities are listed separately by NSE and are not included in this main-board universe.")
uploaded=st.file_uploader("Upload NSE symbols CSV",type=["csv"])
default_symbols=",".join(load_symbols())
symbols_text=st.text_area("Or paste comma/newline-separated NSE symbols",value=st.session_state.get("nse_symbols_text", default_symbols),height=90)
symbols=[]
if uploaded is not None:
    try:
        frame=pd.read_csv(uploaded)
        col=next((c for c in frame.columns if c.lower().strip()=="symbol"),None)
        if col: symbols=frame[col].dropna().astype(str).tolist()
        else: st.error("CSV must contain a Symbol column.")
    except Exception as exc: st.error(f"Could not read CSV: {exc}")
if not symbols:
    symbols=[s.strip().upper().replace(".NS","") for s in symbols_text.replace("\n",",").split(",") if s.strip()]
symbols=list(dict.fromkeys(symbols))
if st.button("↻ Refresh official NSE stock list", use_container_width=True):
    with st.spinner("Downloading the latest NSE equity list..."):
        try:
            refreshed = fetch_nse_symbols()
            st.session_state["nse_symbols_text"] = ",".join(refreshed)
            st.success(f"Loaded {len(refreshed)} NSE equity symbols.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not refresh the NSE list: {exc}. Your existing list has been kept.")
if "nse_symbols_text" in st.session_state and uploaded is None:
    symbols = [x.strip().upper().replace(".NS", "") for x in st.session_state["nse_symbols_text"].replace("\n", ",").split(",") if x.strip()]
symbols=list(dict.fromkeys(symbols))
st.write(f"**{len(symbols)} symbols loaded**")

if st.button("🔎 Run scanner now",type="primary",use_container_width=True):
    if not symbols:
        st.warning("Add at least one NSE symbol.")
    else:
        with st.spinner(f"Scanning {len(symbols)} symbols…"):
            results,errors=run_scan(symbols,min_mcap,min_return,proximity,workers)
            save_daily_results(results,symbols,errors)
            st.session_state["latest_results"]=results
            st.session_state["latest_errors"]=errors
        st.success("Scan completed and saved to daily history.")

st.divider()
st.subheader("2. Daily scan results")
if HISTORY_FILE.exists():
    history=pd.read_csv(HISTORY_FILE)
    if not history.empty:
        dates=sorted(history["Scan date"].dropna().astype(str).unique(),reverse=True)
        selected=st.selectbox("Choose scan date",dates)
        day=history[history["Scan date"].astype(str)==selected].copy()
        # Daily summary values are repeated on rows; for zero-match days the file retains headers only.
        a,b,c=st.columns(3)
        a.metric("Bullish reversals",int(day["Reversal confirmed"].astype(str).str.lower().eq("true").sum()) if "Reversal confirmed" in day else 0)
        a2=int(day["Symbols checked"].max()) if "Symbols checked" in day and len(day) else 0
        b.metric("Matching stocks",len(day))
        c.metric("Scan date",selected)
        if not day.empty:
            st.dataframe(day.drop(columns=["Scan date","Scan time IST"],errors="ignore"),
                         use_container_width=True,hide_index=True)
            st.download_button("⬇️ Download selected day's results",
                               day.to_csv(index=False).encode("utf-8"),
                               f"nse_scan_{selected}.csv","text/csv")
        else:
            st.info("No stocks matched the filters on this scan date.")
        st.caption("The history file is stored in the GitHub repository and updated by the scheduled workflow.")
    else:
        st.info("No saved daily scans yet. The first scheduled or manual scan will appear here.")
else:
    st.info("No saved daily scans yet. The first scheduled or manual scan will appear here.")

st.divider()
st.subheader("3. How the scanner works")
st.markdown("""
- Universe: current NSE main-board equity symbols, refreshed from NSE official securities CSV; SME securities are separate.
- Market cap strictly above the selected threshold (default ₹1,000 crore).
- 3-month return above 60%, approximately 63 trading sessions.
- Latest close within the selected percentage of EMA 10, EMA 20 or EMA 50.
- Bullish reversal candidates require a recent EMA support test plus a hammer, bullish engulfing or EMA reclaim.
- Volume confirmation is shown when latest volume is at least 1.5× the previous 20-session average.
- Entry, stop-loss and 2R/3R targets are screening estimates, not trading instructions.

**Data caution:** Yahoo Finance via yfinance is unofficial and may be delayed, incomplete, rate-limited or inaccurate. Market-cap data can be stale. Verify candidates with NSE/broker data. This is a research screener, not investment advice.
""")
