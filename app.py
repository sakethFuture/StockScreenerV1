"""
Market Pulse Screener v4
- NSE + BSE universe, deduplicated by ISIN (keep faster exchange)
- Strict MCap filtering per tier (None MCap = skip)
- MCap ranges: Small 0-30K, Mid 30K-50K, Large 50K+
- Parallel MCap fetch + parallel Screen 2
- Login: Future / Future
"""

from flask import Flask, jsonify, request, render_template, redirect, url_for, session
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import logging
import warnings
import time
import os
import json
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import io

warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("peewee").setLevel(logging.CRITICAL)

app = Flask(__name__)
app.secret_key = "fcc_pulse_2024_xk9"
CORS(app)

CACHE_DIR = os.path.join(os.getcwd(), "screener_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

USERS = {"Future": "Future"}

# ── Tier configs ──────────────────────────────────────────────────────────────
TIER_CONFIG = {
    "smallcap": {
        "label": "Small Cap",
        "mcap_min": 0,
        "mcap_max": 30000,          # 0 – 30,000 Cr
        "spread_max": 5.0,
        "full_thresh": 3.0,
        "mid_thresh": 4.0,
        "dist_down_pct": 0.8,
        "dist_window": 25,
        "dist_pressure": 4,
        "dist_downtrend": 6,
        "ftd_up_pct": 2.5,
        "ftd_day_min": 4,
        "ftd_day_max": 7,
        "index_attempts": ["^CNXSC", "^NSEI", "NIFTYSMLCAP100.NS"],
        "index_label": "Nifty SmallCap 100",
    },
    "midcap": {
        "label": "Mid Cap",
        "mcap_min": 30000,          # 30,000 – 50,000 Cr
        "mcap_max": 50000,
        "spread_max": 4.5,
        "full_thresh": 2.5,
        "mid_thresh": 3.5,
        "dist_down_pct": 1.0,
        "dist_window": 25,
        "dist_pressure": 4,
        "dist_downtrend": 6,
        "ftd_up_pct": 2.0,
        "ftd_day_min": 4,
        "ftd_day_max": 7,
        "index_attempts": ["^CNXMC", "NIFTYMIDCAP100.NS", "^NSEI"],
        "index_label": "Nifty MidCap 100",
    },
    "largecap": {
        "label": "Large Cap",
        "mcap_min": 50000,          # 50,000 Cr+
        "mcap_max": 10_000_000,
        "spread_max": 4.0,
        "full_thresh": 2.0,
        "mid_thresh": 3.0,
        "dist_down_pct": 1.2,
        "dist_window": 25,
        "dist_pressure": 5,
        "dist_downtrend": 7,
        "ftd_up_pct": 1.5,
        "ftd_day_min": 4,
        "ftd_day_max": 7,
        "index_attempts": ["^NSEI", "^CNX100"],
        "index_label": "Nifty 50",
    },
}

TIER_MCAP_LABELS = {
    "smallcap": [
        (0,      1000,  "MICRO (0-1K)"),
        (1000,   5000,  "SMALL (1-5K)"),
        (5000,   30000, "MID-SMALL (5-30K)"),
    ],
    "midcap": [
        (30000,  40000, "LOWER MID (30-40K)"),
        (40000,  50000, "UPPER MID (40-50K)"),
    ],
    "largecap": [
        (50000,  200000,"LARGE (50-200K)"),
        (200000, 10_000_000, "MEGA (200K+)"),
    ],
}

SIGNALS = {
    "CONFIRMED UPTREND": {"color": "green",  "action": "All setups actionable."},
    "UNDER PRESSURE":    {"color": "yellow", "action": "Only FULL stage setups. Spread < 2% only."},
    "RALLY ATTEMPT":     {"color": "blue",   "action": "Build watchlist. Do NOT buy until FTD confirmed."},
    "DOWNTREND":         {"color": "red",    "action": "Watchlist only. No new buys."},
}

# ── Auth ──────────────────────────────────────────────────────────────────────

def safe_download(ticker, period="6mo", interval="1d", timeout=8, **kwargs):
    """
    yf.download with a hard wall-clock timeout.
    Returns None if the call hangs or errors — never blocks the batch.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    def _dl():
        return yf.download(ticker, period=period, interval=interval,
                           auto_adjust=True, progress=False, **kwargs)
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_dl)
        try:
            return fut.result(timeout=timeout)
        except (FuturesTimeout, Exception):
            fut.cancel()
            return None

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()
        if USERS.get(u) == p:
            session["logged_in"] = True
            session["username"] = u
            return redirect(url_for("index"))
        error = "Invalid credentials"
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ── Universe: NSE + BSE, deduped by ISIN ──────────────────────────────────────

def fetch_nse_tickers():
    """Returns list of (ticker, isin, symbol, name) from NSE equity list."""
    try:
        r = requests.get(
            "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"},
            timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        sym_col  = next((c for c in df.columns if "SYMBOL" in c.upper()), None)
        isin_col = next((c for c in df.columns if "ISIN" in c.upper()), None)
        name_col = next((c for c in df.columns if "NAME" in c.upper()), None)
        if not sym_col:
            return []
        rows = []
        for _, row in df.iterrows():
            sym  = str(row[sym_col]).strip()
            if sum(1 for c in sym if c.isalpha()) < 3:
                continue
            isin = str(row[isin_col]).strip() if isin_col else ""
            name = str(row[name_col]).strip() if name_col else sym
            rows.append((sym + ".NS", isin, sym, name))
        return rows
    except Exception as e:
        print(f"NSE fetch failed: {e}")
        return []

def fetch_bse_tickers():
    """BSE HTML fallback — returns (ticker, isin, symbol, name)."""
    try:
        r = requests.get(
            "https://www.bseindia.com/corporates/List_Scrips.aspx",
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Referer": "https://www.bseindia.com/",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            timeout=20)
        tables = pd.read_html(io.BytesIO(r.content))
        for t in tables:
            cols = [str(c).upper() for c in t.columns]
            code_col = next((i for i,c in enumerate(cols) if "CODE" in c), None)
            isin_col = next((i for i,c in enumerate(cols) if "ISIN" in c), None)
            name_col = next((i for i,c in enumerate(cols) if "NAME" in c), None)
            if code_col is not None:
                rows = []
                for _, row in t.iterrows():
                    code = str(row.iloc[code_col]).strip().split(".")[0]
                    if not code.isdigit(): continue
                    isin = str(row.iloc[isin_col]).strip() if isin_col is not None else ""
                    name = str(row.iloc[name_col]).strip() if name_col is not None else code
                    rows.append((code + ".BO", isin, name[:25], name))
                if rows:
                    return rows
        return []
    except Exception as e:
        print(f"BSE HTML parse failed: {e}")
        return []

def fetch_bse_tickers_api():
    """BSE API endpoint — returns (ticker, isin, symbol, name)."""
    try:
        r = requests.get(
            "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
            "?Group=&Scripcode=&industry=&segment=Equity&status=Active",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://www.bseindia.com/",
                "Origin": "https://www.bseindia.com",
            },
            timeout=20)
        r.raise_for_status()
        data = r.json()
        items = data.get("Table", data) if isinstance(data, dict) else data
        rows = []
        for item in items:
            code   = str(item.get("SCRIP_CD",   item.get("scripcode",  ""))).strip()
            isin   = str(item.get("ISIN_NO",    item.get("isin",       ""))).strip()
            name   = str(item.get("SCRIP_NAME", item.get("scrip_name", code))).strip()
            symbol = str(item.get("NSESYMBOL",  item.get("nsesymbol",  ""))).strip()
            display = symbol if (symbol and symbol not in ("nan","")) else name[:25]
            if code.isdigit() and len(code) >= 5:
                rows.append((code + ".BO", isin, display, name))
        return rows
    except Exception as e:
        print(f"BSE API failed: {e}")
        return []

def get_universe():
    """NSE only — BSE removed (numeric codes unusable, MCap/name data unreliable)."""
    cache      = os.path.join(CACHE_DIR, "universe_v2.csv")
    name_cache = os.path.join(CACHE_DIR, "name_map.json")

    if os.path.exists(cache) and os.path.exists(name_cache):
        age = (time.time() - os.path.getmtime(cache)) / 3600
        if age < 12:
            df = pd.read_csv(cache)
            with open(name_cache) as f:
                name_map = json.load(f)
            return df["ticker"].tolist(), name_map, f"cached ({int(age)}h old, {len(df)} stocks)"

    nse_rows    = fetch_nse_tickers()
    tickers     = [t for t, _, _, _ in nse_rows]
    name_map    = {t: {"symbol": s, "name": n} for t, _, s, n in nse_rows}

    pd.DataFrame({"ticker": tickers}).to_csv(cache, index=False)
    with open(name_cache, "w") as f:
        json.dump(name_map, f)

    return tickers, name_map, f"fresh — NSE only: {len(tickers)} stocks"

# ── EMA + 200 DMA ─────────────────────────────────────────────────────────────

def ema(arr, p):
    return pd.Series(arr.astype(float)).ewm(span=p, adjust=False).mean().values

def extract_close(data, ticker, is_batch):
    try:
        if not is_batch or not isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns:
                return data["Close"].dropna().values.flatten().astype(float)
            return None
        l0 = list(data.columns.get_level_values(0))
        l1 = list(data.columns.get_level_values(1))
        if ticker in l0 and "Close" in l1:
            return data[ticker]["Close"].dropna().values.flatten().astype(float)
        if "Close" in l0 and ticker in l1:
            return data["Close"][ticker].dropna().values.flatten().astype(float)
        return None
    except:
        return None

def check_ema(close, cfg):
    if len(close) < 210:
        return None
    e200 = ema(close, 200)
    if float(e200[-1]) <= float(e200[-21]):
        return None
    ev  = {p: float(ema(close, p)[-1]) for p in [8, 13, 21, 34, 55]}
    ev5 = {p: float(ema(close, p)[-6]) for p in [8, 13, 21, 34, 55]}
    def spread(d):
        mn = min(d.values())
        return (max(d.values()) - mn) / mn * 100 if mn > 0 else 999
    s_now = spread(ev)
    s_5d  = spread(ev5)
    if s_now > cfg["spread_max"] or s_now >= s_5d:
        return None
    price  = float(close[-1])
    stage  = "FULL" if s_now <= cfg["full_thresh"] else ("MID" if s_now <= cfg["mid_thresh"] else "FAST")
    high52 = float(np.max(close[-252:])) if len(close) >= 252 else float(np.max(close))
    return {
        "stage":        stage,
        "price":        round(price, 2),
        "spread_8_55":  round(s_now, 2),
        "spread_5d":    round(s_5d, 2),
        "ema200_slope": round(float(e200[-1]) - float(e200[-21]), 2),
        "pct_off_52h":  round((high52 - price) / high52 * 100, 1),
    }

# ── MCap ──────────────────────────────────────────────────────────────────────

def get_mcap(ticker):
    """Returns MCap in INR Crores via info dict (fast_info unreliable on cloud)."""
    USD_TO_INR = 83.5
    try:
        info = yf.Ticker(ticker).info
        mcap = info.get("marketCap") or info.get("regularMarketCap")
        curr = info.get("currency", "INR") or "INR"
        if mcap:
            if curr.upper() == "USD":
                mcap = mcap * USD_TO_INR
            return round(mcap / 1e7, 0)
        shares = info.get("sharesOutstanding")
        price  = info.get("regularMarketPrice") or info.get("currentPrice")
        if shares and price:
            mcap = shares * price
            if curr.upper() == "USD":
                mcap = mcap * USD_TO_INR
            return round(mcap / 1e7, 0)
        return None
    except:
        return None

def mcap_tier_label(mcap_cr, tier_key):
    for lo, hi, label in TIER_MCAP_LABELS.get(tier_key, []):
        if mcap_cr is not None and lo <= mcap_cr < hi:
            return label
    return "UNKNOWN"

# ── Market Pulse ──────────────────────────────────────────────────────────────

def compute_pulse(cfg):
    d = None
    used_ticker = None
    notes = []
    for attempt in cfg["index_attempts"]:
        try:
            tmp = safe_download(attempt, period="6mo", interval="1d", timeout=10)
            if tmp is not None and not tmp.empty and len(tmp) > 5:
                d = tmp
                used_ticker = attempt
                if attempt != cfg["index_attempts"][0]:
                    notes.append(f"Using {attempt}")
                break
        except Exception as e:
            notes.append(f"{attempt}: {e}")

    if d is None or d.empty:
        return {"pulse": "CONFIRMED UPTREND", "error": "Index unavailable", "notes": notes}

    if isinstance(d.columns, pd.MultiIndex):
        close  = d["Close"].iloc[:, 0].dropna().astype(float)
        volume = d["Volume"].iloc[:, 0].dropna().astype(float)
    else:
        close  = d["Close"].dropna().astype(float)
        volume = d["Volume"].dropna().astype(float)

    chg     = close.pct_change() * 100
    avg_vol = volume.rolling(20).mean()

    dist_mask  = (chg <= -cfg["dist_down_pct"]) & (volume > avg_vol)
    dist_count = int(dist_mask.iloc[-cfg["dist_window"]:].sum())
    dist_dates = [str(d2)[:10] for d2 in dist_mask[dist_mask].iloc[-5:].index]

    c          = close.values
    low_idx    = len(c) - 15 + int(np.argmin(c[-15:]))
    low_price  = float(c[low_idx])
    days_since = len(c) - 1 - low_idx
    bounce     = (float(c[-1]) - low_price) / low_price * 100
    in_rally   = 1.0 <= bounce and 1 <= days_since <= 12
    rally_day  = days_since if in_rally else None

    ftd = False; ftd_day = None
    if in_rally:
        for day in range(cfg["ftd_day_min"], min(cfg["ftd_day_max"]+1, days_since+1)):
            idx = low_idx + day
            if idx >= len(c): break
            if (float(chg.iloc[idx]) >= cfg["ftd_up_pct"] and
                    float(volume.iloc[idx]) > float(avg_vol.iloc[idx])):
                ftd = True; ftd_day = day; break

    if dist_count >= cfg["dist_downtrend"]:
        pulse = "RALLY ATTEMPT" if in_rally else "DOWNTREND"
    elif dist_count >= cfg["dist_pressure"]:
        pulse = "UNDER PRESSURE"
    else:
        pulse = "CONFIRMED UPTREND"
    if ftd:
        pulse = "CONFIRMED UPTREND"

    return {
        "pulse":          pulse,
        "signal":         SIGNALS.get(pulse, {}),
        "index":          used_ticker,
        "index_label":    cfg["index_label"],
        "idx_1d":         round(float(chg.iloc[-1]), 2),
        "idx_5d":         round(float((close.iloc[-1]/close.iloc[-6]-1)*100) if len(close)>=6 else 0, 2),
        "dist_count":     dist_count,
        "dist_window":    cfg["dist_window"],
        "dist_pressure":  cfg["dist_pressure"],
        "dist_downtrend": cfg["dist_downtrend"],
        "dist_dates":     dist_dates[-3:],
        "in_rally":       in_rally,
        "rally_day":      rally_day,
        "ftd":            ftd,
        "ftd_day":        ftd_day,
        "ftd_day_min":    cfg["ftd_day_min"],
        "ftd_day_max":    cfg["ftd_day_max"],
        "date":           str(date.today()),
        "notes":          notes,
    }

# ── Screen 2 stock pulse ──────────────────────────────────────────────────────

def analyse_stock_pulse(r, cfg, pulse):
    ticker = r["ticker"]
    try:
        d = safe_download(ticker, period="6mo", interval="1d", timeout=8)
        if d is None or d.empty:
            r.update({"dist_days": None, "ft_days": None, "stock_pulse": "NO DATA", "action": "WATCHLIST"})
            return r
        if isinstance(d.columns, pd.MultiIndex):
            close  = d["Close"].iloc[:, 0].dropna().astype(float)
            volume = d["Volume"].iloc[:, 0].dropna().astype(float)
        else:
            close  = d["Close"].dropna().astype(float)
            volume = d["Volume"].dropna().astype(float)

        chg     = close.pct_change() * 100
        avg_vol = volume.rolling(20).mean()
        dist_mask  = (chg <= -cfg["dist_down_pct"]) & (volume > avg_vol)
        dist_count = int(dist_mask.iloc[-cfg["dist_window"]:].sum())

        c          = close.values
        low_idx    = len(c) - 15 + int(np.argmin(c[-15:]))
        low_price  = float(c[low_idx])
        days_since = len(c) - 1 - low_idx
        bounce     = (float(c[-1]) - low_price) / low_price * 100
        in_rally   = 1.0 <= bounce and 1 <= days_since <= 12

        ftd = False; ftd_day = None
        if in_rally:
            for day in range(cfg["ftd_day_min"], min(cfg["ftd_day_max"]+1, days_since+1)):
                idx = low_idx + day
                if idx >= len(c): break
                if (float(chg.iloc[idx]) >= cfg["ftd_up_pct"] and
                        float(volume.iloc[idx]) > float(avg_vol.iloc[idx])):
                    ftd = True; ftd_day = day; break

        if dist_count >= cfg["dist_downtrend"]:
            sp = "RALLY ATTEMPT" if in_rally else "DOWNTREND"
        elif dist_count >= cfg["dist_pressure"]:
            sp = "UNDER PRESSURE"
        else:
            sp = "CONFIRMED UPTREND"
        if ftd:
            sp = "CONFIRMED UPTREND"

        if pulse == "CONFIRMED UPTREND" and sp in ["CONFIRMED UPTREND", "UNDER PRESSURE"]:
            action = "ACTIONABLE"
        elif pulse == "UNDER PRESSURE" and sp == "CONFIRMED UPTREND" and r.get("spread_8_55", 99) < 2.0:
            action = "CAUTION"
        else:
            action = "WATCHLIST"

        r.update({
            "dist_days":   dist_count,
            "ft_days":     ftd_day,
            "in_rally":    in_rally,
            "rally_day":   days_since if in_rally else None,
            "stock_pulse": sp,
            "ftd_fired":   ftd,
            "action":      action,
        })
    except:
        r.update({"dist_days": None, "ft_days": None, "stock_pulse": "ERROR", "action": "WATCHLIST"})
    return r

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
@login_required
def index():
    return render_template("index.html")

@app.route("/api/pulse/<tier>")
@login_required
def api_pulse(tier):
    if tier not in TIER_CONFIG:
        return jsonify({"error": "Unknown tier"}), 400
    return jsonify(compute_pulse(TIER_CONFIG[tier]))

@app.route("/api/screen/base/<tier>")
@login_required
def api_screen_base(tier):
    """
    Serves instantly from nightly cache.
    If cache is missing, tells user to run nightly_scan.py.
    Manual rescan available via /api/scan/run (background thread).
    """
    if tier not in TIER_CONFIG:
        return jsonify({"error": "Unknown tier"}), 400

    cache_path = os.path.join(CACHE_DIR, f"base_{tier}_{date.today()}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            results = json.load(f)
        return jsonify({
            "type":    "done",
            "results": results,
            "tier":    tier,
            "date":    str(date.today()),
            "source":  "cache",
        })

    # No cache for today
    lock = os.path.join(CACHE_DIR, "scan.lock")
    if os.path.exists(lock):
        # Auto-clear stale lock older than 30 min
        age_min = (time.time() - os.path.getmtime(lock)) / 60
        if age_min > 30:
            os.remove(lock)
        else:
            return jsonify({"type": "scanning", "message": f"Scan in progress ({int(age_min)}m elapsed) — check back shortly."}), 202

    return jsonify({
        "type":    "no_cache",
        "message": "No scan for today yet. Click RESCAN to build cache (~10-15 min).",
    }), 404

@app.route("/api/screen/pulse/<tier>")
@login_required
def api_screen_pulse(tier):
    """Serve Screen 2 from nightly pre-built pulse cache — instant, no live API calls."""
    if tier not in TIER_CONFIG:
        return jsonify({"error": "Unknown tier"}), 400

    # Serve from pre-built cache (built by nightly_scan Step 4)
    pulse_cache = os.path.join(CACHE_DIR, f"pulse_{tier}_{date.today()}.json")
    if os.path.exists(pulse_cache):
        with open(pulse_cache) as f:
            data = json.load(f)
        data["source"] = "cache"
        return jsonify(data)

    # Fallback: compute live if pulse cache missing
    base_path = os.path.join(CACHE_DIR, f"base_{tier}_{date.today()}.json")
    if not os.path.exists(base_path):
        return jsonify({"error": "Run base screen first (click RESCAN)."}), 400

    with open(base_path) as f:
        base_results = json.load(f)

    cfg        = TIER_CONFIG[tier]
    pulse_data = compute_pulse(cfg)
    pulse      = pulse_data["pulse"]

    enriched = [None] * len(base_results)
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(analyse_stock_pulse, dict(r), cfg, pulse): i
                   for i, r in enumerate(base_results)}
        for fut in as_completed(futures):
            i = futures[fut]
            enriched[i] = fut.result()

    action_ord = {"ACTIONABLE": 0, "CAUTION": 1, "WATCHLIST": 2}
    stage_ord  = {"FULL": 0, "MID": 1, "FAST": 2}
    enriched = [e for e in enriched if e is not None]
    enriched.sort(key=lambda x: (
        action_ord.get(x.get("action","WATCHLIST"), 9),
        stage_ord.get(x.get("stage","FAST"), 9),
        x.get("spread_8_55", 99)
    ))

    result = {"results": enriched, "pulse": pulse_data,
              "tier": tier, "date": str(date.today()), "total": len(enriched), "source": "live"}

    with open(pulse_cache, "w") as f:
        json.dump(result, f)

    return jsonify(result)

@app.route("/api/scan/run", methods=["POST"])
@login_required
def api_scan_run():
    """Trigger a background rescan. Non-blocking — returns immediately."""
    import subprocess, sys
    lock = os.path.join(CACHE_DIR, "scan.lock")
    if os.path.exists(lock):
        return jsonify({"status": "already_running"})
    # Write lock
    with open(lock, "w") as f:
        f.write(str(datetime.now()))
    def run_bg():
        try:
            scanner = os.path.join(os.path.dirname(__file__), "nightly_scan.py")
            subprocess.run([sys.executable, scanner], timeout=1800)
        finally:
            if os.path.exists(lock):
                os.remove(lock)
    import threading
    t = threading.Thread(target=run_bg, daemon=True)
    t.start()
    return jsonify({"status": "started", "message": "Scan started in background. Results ready in ~10-15 min."})

@app.route("/api/scan/clear", methods=["POST"])
@login_required
def api_scan_clear():
    """Manually clear a stuck scan lock."""
    lock = os.path.join(CACHE_DIR, "scan.lock")
    if os.path.exists(lock):
        os.remove(lock)
        return jsonify({"status": "cleared"})
    return jsonify({"status": "no_lock"})

@app.route("/api/scan/status")
@login_required
def api_scan_status():
    lock = os.path.join(CACHE_DIR, "scan.lock")
    running = os.path.exists(lock)
    caches = {}
    for tier in TIER_CONFIG:
        cp = os.path.join(CACHE_DIR, f"base_{tier}_{date.today()}.json")
        if os.path.exists(cp):
            with open(cp) as f:
                data = json.load(f)
            caches[tier] = {"count": len(data), "date": str(date.today())}
        else:
            caches[tier] = None
    return jsonify({"scanning": running, "caches": caches, "date": str(date.today())})

@app.route("/api/cache/base/<tier>")
@login_required
def api_cache_base(tier):
    cache_path = os.path.join(CACHE_DIR, f"base_{tier}_{date.today()}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            results = json.load(f)
        return jsonify({"cached": True, "results": results, "date": str(date.today())})
    return jsonify({"cached": False})

if __name__ == "__main__":
    app.run(debug=False, port=5050, threaded=True)
