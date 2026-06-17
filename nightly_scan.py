"""
Nightly Scanner — run this once daily (e.g. 8 PM after market close)
Scans full NSE+BSE universe, caches results for all 3 tiers.
Users get instant load from cache — no waiting in browser.

Schedule:
  Windows Task Scheduler → python nightly_scan.py
  Linux/Mac cron        → 0 20 * * 1-5 cd /path/to/pulse_screener && python nightly_scan.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
import time
import os
import json
import io
from datetime import date, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screener_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

TIER_CONFIG = {
    "smallcap": {
        "label": "Small Cap", "mcap_min": 0, "mcap_max": 30000,
        "spread_max": 5.0, "full_thresh": 3.0, "mid_thresh": 4.0,
    },
    "midcap": {
        "label": "Mid Cap", "mcap_min": 30000, "mcap_max": 50000,
        "spread_max": 4.5, "full_thresh": 2.5, "mid_thresh": 3.5,
    },
    "largecap": {
        "label": "Large Cap", "mcap_min": 50000, "mcap_max": 10_000_000,
        "spread_max": 4.0, "full_thresh": 2.0, "mid_thresh": 3.0,
    },
}

TIER_MCAP_LABELS = {
    "smallcap": [(0,1000,"MICRO (0-1K)"),(1000,5000,"SMALL (1-5K)"),(5000,30000,"MID-SMALL (5-30K)")],
    "midcap":   [(30000,40000,"LOWER MID (30-40K)"),(40000,50000,"UPPER MID (40-50K)")],
    "largecap": [(50000,200000,"LARGE (50-200K)"),(200000,10_000_000,"MEGA (200K+)")],
}

def safe_download(ticker, period="1y", interval="1d", timeout=8, **kwargs):
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

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ── Universe ──────────────────────────────────────────────────────────────────

def fetch_nse():
    try:
        r = requests.get(
            "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"},
            timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.BytesIO(r.content))
        sym_col  = next((c for c in df.columns if "SYMBOL" in c.upper()), None)
        isin_col = next((c for c in df.columns if "ISIN"   in c.upper()), None)
        name_col = next((c for c in df.columns if "NAME"   in c.upper()), None)
        rows = []
        for _, row in df.iterrows():
            sym = str(row[sym_col]).strip()
            if sum(1 for c in sym if c.isalpha()) < 3: continue
            isin = str(row[isin_col]).strip() if isin_col else ""
            name = str(row[name_col]).strip() if name_col else sym
            rows.append((sym + ".NS", isin, sym, name))
        log(f"NSE: {len(rows)} tickers")
        return rows
    except Exception as e:
        log(f"NSE fetch failed: {e}")
        return []

def fetch_bse():
    try:
        r = requests.get(
            "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
            "?Group=&Scripcode=&industry=&segment=Equity&status=Active",
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.bseindia.com/",
                     "Origin": "https://www.bseindia.com"},
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
        log(f"BSE: {len(rows)} tickers")
        return rows
    except Exception as e:
        log(f"BSE fetch failed: {e}")
        return []

def build_universe():
    cache      = os.path.join(CACHE_DIR, "universe_v2.csv")
    name_cache = os.path.join(CACHE_DIR, "name_map.json")
    nse_rows   = fetch_nse()   # NSE only — BSE removed (numeric codes, no MCap data)
    tickers    = [t for t, _, _, _ in nse_rows]
    name_map   = {t: {"symbol": s, "name": n} for t, _, s, n in nse_rows}
    pd.DataFrame({"ticker": tickers}).to_csv(cache, index=False)
    with open(name_cache, "w") as f:
        json.dump(name_map, f)
    log(f"Universe: NSE only — {len(tickers)} tickers")
    return tickers, name_map

# ── EMA screen ────────────────────────────────────────────────────────────────

def ema(arr, p):
    return pd.Series(arr.astype(float)).ewm(span=p, adjust=False).mean().values

def extract_close(data, ticker, is_batch):
    try:
        if not is_batch or not isinstance(data.columns, pd.MultiIndex):
            return data["Close"].dropna().values.flatten().astype(float) if "Close" in data.columns else None
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
    if len(close) < 210: return None
    e200 = ema(close, 200)
    if float(e200[-1]) <= float(e200[-21]): return None   # 200 DMA must be rising
    ev  = {p: float(ema(close, p)[-1]) for p in [8,13,21,34,55]}
    ev5 = {p: float(ema(close, p)[-6]) for p in [8,13,21,34,55]}
    def spread(d):
        mn = min(d.values())
        return (max(d.values()) - mn) / mn * 100 if mn > 0 else 999
    s_now = spread(ev)
    s_5d  = spread(ev5)
    if s_now > cfg["spread_max"] or s_now >= s_5d: return None  # must be compressing
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

def get_mcap(ticker):
    """
    Returns MCap in INR Crores.
    fast_info.market_cap is unreliable on Railway/cloud (returns None).
    Strategy: shares_outstanding × last_price → INR Cr.
    Falls back to regularMarketCap from info if shares not available.
    """
    USD_TO_INR = 83.5
    try:
        tk   = yf.Ticker(ticker)
        info = tk.info
        # Try market cap directly first
        mcap = info.get("marketCap") or info.get("regularMarketCap")
        curr = info.get("currency", "INR") or "INR"
        if mcap:
            if curr.upper() == "USD":
                mcap = mcap * USD_TO_INR
            return round(mcap / 1e7, 0)
        # Fallback: shares × price
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

# ── Main scan ─────────────────────────────────────────────────────────────────

def run_scan():
    start = datetime.now()
    log("=" * 55)
    log("NIGHTLY SCAN STARTING")
    log("=" * 55)

    tickers, name_map = build_universe()
    total   = len(tickers)
    BATCH   = 100

    # ── Step 1: EMA screen (all tickers, no tier filter yet) ──────────────────
    log(f"Step 1: EMA + 200 DMA scan across {total} tickers …")
    ema_hits = []   # list of {ticker, stage, spread, ...}
    batches  = [tickers[i:i+BATCH] for i in range(0, total, BATCH)]

    for i, batch in enumerate(batches):
        try:
            is_b = len(batch) > 1
            data = safe_download(
                batch if is_b else batch[0],
                period="1y", interval="1d", timeout=30,
                group_by="ticker" if is_b else None, threads=True)
            if data is not None and not data.empty:
                for ticker in batch:
                    c = extract_close(data, ticker, is_b)
                    if c is None: continue
                    # Use loosest cfg for initial screen — tier filter happens after MCap
                    res = check_ema(c, {"spread_max": 5.0, "full_thresh": 3.0, "mid_thresh": 4.0})
                    if res:
                        res["ticker"] = ticker
                        nm = name_map.get(ticker, {})
                        res["symbol"] = nm.get("symbol", ticker.replace(".NS","").replace(".BO",""))
                        res["name"]   = nm.get("name", "")
                        ema_hits.append(res)
        except Exception as e:
            pass

        if (i + 1) % 10 == 0:
            pct = round((i+1)*BATCH/total*100, 1)
            log(f"  {(i+1)*BATCH}/{total} scanned ({pct}%) — {len(ema_hits)} EMA hits")

    log(f"Step 1 done: {len(ema_hits)} stocks passed EMA + 200 DMA")

    # ── Step 2: Price-based tier bucketing (no MCap API call — Yahoo blocks on cloud) ──
    # Price proxy for tier: Small <500, Mid 500-2000, Large 2000+
    # This avoids Yahoo .info calls which get rate-limited/blocked on cloud servers
    PRICE_TIERS = {
        "smallcap": (0,     500),
        "midcap":   (500,   2000),
        "largecap": (2000,  9999999),
    }
    log(f"Step 2: Bucketing {len(ema_hits)} stocks by price proxy (no MCap API needed) …")

    # ── Step 3: Split into tiers and cache ────────────────────────────────────
    log("Step 3: Writing cache …")
    today = str(date.today())
    stage_ord = {"FULL": 0, "MID": 1, "FAST": 2}

    for tier_key, cfg in TIER_CONFIG.items():
        p_min, p_max = PRICE_TIERS[tier_key]
        bucket = []
        for r in ema_hits:
            price = r.get("price", 0)
            if price < p_min or price >= p_max: continue
            if r["spread_8_55"] > cfg["spread_max"]: continue
            entry = dict(r)
            entry["mcap_cr"]   = None   # not available without API
            entry["mcap_tier"] = f"~{tier_key.upper()}"
            entry["exchange"]  = "BSE" if r["ticker"].endswith(".BO") else "NSE"
            bucket.append(entry)

        bucket.sort(key=lambda x: (stage_ord.get(x["stage"], 9), x["spread_8_55"]))
        cache_path = os.path.join(CACHE_DIR, f"base_{tier_key}_{today}.json")
        with open(cache_path, "w") as f:
            json.dump(bucket, f)
        log(f"  {cfg['label']}: {len(bucket)} stocks → {cache_path}")

    # ── Step 4: Pulse analysis per tier (cached so users get instant Screen 2) ──
    log("Step 4: Computing market pulse + stock-level pulse for each tier …")

    SIGNALS = {
        "CONFIRMED UPTREND": {"color": "green",  "action": "All setups actionable."},
        "UNDER PRESSURE":    {"color": "yellow", "action": "Only FULL stage setups. Spread < 2% only."},
        "RALLY ATTEMPT":     {"color": "blue",   "action": "Build watchlist. Do NOT buy until FTD confirmed."},
        "DOWNTREND":         {"color": "red",    "action": "Watchlist only. No new buys."},
    }

    def compute_index_pulse(cfg):
        for attempt in cfg["index_attempts"]:
            try:
                d = safe_download(attempt, period="6mo", interval="1d", timeout=15)
                if d is None or d.empty: continue
                close  = (d["Close"].iloc[:,0] if isinstance(d.columns, pd.MultiIndex) else d["Close"]).dropna().astype(float)
                volume = (d["Volume"].iloc[:,0] if isinstance(d.columns, pd.MultiIndex) else d["Volume"]).dropna().astype(float)
                chg     = close.pct_change() * 100
                avg_vol = volume.rolling(20).mean()
                dist_mask  = (chg <= -cfg["dist_down_pct"]) & (volume > avg_vol)
                dist_count = int(dist_mask.iloc[-cfg["dist_window"]:].sum())
                c          = close.values
                low_idx    = len(c) - 15 + int(np.argmin(c[-15:]))
                days_since = len(c) - 1 - low_idx
                bounce     = (float(c[-1]) - float(c[low_idx])) / float(c[low_idx]) * 100
                in_rally   = 1.0 <= bounce and 1 <= days_since <= 12
                ftd = False; ftd_day = None
                if in_rally:
                    for day in range(cfg["ftd_day_min"], min(cfg["ftd_day_max"]+1, days_since+1)):
                        idx2 = low_idx + day
                        if idx2 >= len(c): break
                        if float(chg.iloc[idx2]) >= cfg["ftd_up_pct"] and float(volume.iloc[idx2]) > float(avg_vol.iloc[idx2]):
                            ftd = True; ftd_day = day; break
                if dist_count >= cfg["dist_downtrend"]:
                    pulse = "RALLY ATTEMPT" if in_rally else "DOWNTREND"
                elif dist_count >= cfg["dist_pressure"]:
                    pulse = "UNDER PRESSURE"
                else:
                    pulse = "CONFIRMED UPTREND"
                if ftd: pulse = "CONFIRMED UPTREND"
                return {
                    "pulse": pulse, "signal": SIGNALS.get(pulse, {}),
                    "index": attempt, "index_label": cfg["index_label"],
                    "idx_1d": round(float(chg.iloc[-1]), 2),
                    "idx_5d": round(float((close.iloc[-1]/close.iloc[-6]-1)*100) if len(close)>=6 else 0, 2),
                    "dist_count": dist_count, "dist_window": cfg["dist_window"],
                    "dist_pressure": cfg["dist_pressure"], "dist_downtrend": cfg["dist_downtrend"],
                    "in_rally": in_rally, "rally_day": days_since if in_rally else None,
                    "ftd": ftd, "ftd_day": ftd_day,
                    "ftd_day_min": cfg["ftd_day_min"], "ftd_day_max": cfg["ftd_day_max"],
                    "date": today,
                }
            except Exception as e:
                log(f"  Index {attempt} failed: {e}")
        return {"pulse": "CONFIRMED UPTREND", "error": "Index unavailable", "date": today}

    def analyse_one(r, cfg, pulse):
        ticker = r["ticker"]
        try:
            d = safe_download(ticker, period="6mo", interval="1d", timeout=10)
            if d is None or d.empty:
                r.update({"dist_days": None, "ft_days": None, "stock_pulse": "NO DATA", "action": "WATCHLIST"})
                return r
            close  = (d["Close"].iloc[:,0] if isinstance(d.columns, pd.MultiIndex) else d["Close"]).dropna().astype(float)
            volume = (d["Volume"].iloc[:,0] if isinstance(d.columns, pd.MultiIndex) else d["Volume"]).dropna().astype(float)
            chg     = close.pct_change() * 100
            avg_vol = volume.rolling(20).mean()
            dist_mask  = (chg <= -cfg["dist_down_pct"]) & (volume > avg_vol)
            dist_count = int(dist_mask.iloc[-cfg["dist_window"]:].sum())
            c          = close.values
            low_idx    = len(c) - 15 + int(np.argmin(c[-15:]))
            days_since = len(c) - 1 - low_idx
            bounce     = (float(c[-1]) - float(c[low_idx])) / float(c[low_idx]) * 100
            in_rally   = 1.0 <= bounce and 1 <= days_since <= 12
            ftd = False; ftd_day = None
            if in_rally:
                for day in range(cfg["ftd_day_min"], min(cfg["ftd_day_max"]+1, days_since+1)):
                    idx2 = low_idx + day
                    if idx2 >= len(c): break
                    if float(chg.iloc[idx2]) >= cfg["ftd_up_pct"] and float(volume.iloc[idx2]) > float(avg_vol.iloc[idx2]):
                        ftd = True; ftd_day = day; break
            if dist_count >= cfg["dist_downtrend"]:
                sp = "RALLY ATTEMPT" if in_rally else "DOWNTREND"
            elif dist_count >= cfg["dist_pressure"]:
                sp = "UNDER PRESSURE"
            else:
                sp = "CONFIRMED UPTREND"
            if ftd: sp = "CONFIRMED UPTREND"
            if pulse == "CONFIRMED UPTREND" and sp in ["CONFIRMED UPTREND", "UNDER PRESSURE"]:
                action = "ACTIONABLE"
            elif pulse == "UNDER PRESSURE" and sp == "CONFIRMED UPTREND" and r.get("spread_8_55", 99) < 2.0:
                action = "CAUTION"
            else:
                action = "WATCHLIST"
            r.update({"dist_days": dist_count, "ft_days": ftd_day,
                      "in_rally": in_rally, "rally_day": days_since if in_rally else None,
                      "stock_pulse": sp, "ftd_fired": ftd, "action": action})
        except:
            r.update({"dist_days": None, "ft_days": None, "stock_pulse": "ERROR", "action": "WATCHLIST"})
        return r

    for tier_key, cfg in TIER_CONFIG.items():
        base_path  = os.path.join(CACHE_DIR, f"base_{tier_key}_{today}.json")
        pulse_path = os.path.join(CACHE_DIR, f"pulse_{tier_key}_{today}.json")
        if not os.path.exists(base_path):
            continue
        with open(base_path) as f:
            bucket = json.load(f)
        if not bucket:
            with open(pulse_path, "w") as f:
                json.dump({"results": [], "pulse": {}, "tier": tier_key, "date": today}, f)
            continue

        log(f"  {cfg['label']}: computing index pulse …")
        pulse_data = compute_index_pulse(cfg)
        pulse      = pulse_data["pulse"]
        log(f"  {cfg['label']}: pulse = {pulse}. Analysing {len(bucket)} stocks (10 workers) …")

        enriched = [None] * len(bucket)
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(analyse_one, dict(r), cfg, pulse): i for i, r in enumerate(bucket)}
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                enriched[i] = fut.result()
                done += 1
                if done % 20 == 0:
                    log(f"    {done}/{len(bucket)} analysed")

        action_ord = {"ACTIONABLE": 0, "CAUTION": 1, "WATCHLIST": 2}
        stage_ord2 = {"FULL": 0, "MID": 1, "FAST": 2}
        enriched = [e for e in enriched if e is not None]
        enriched.sort(key=lambda x: (
            action_ord.get(x.get("action","WATCHLIST"), 9),
            stage_ord2.get(x.get("stage","FAST"), 9),
            x.get("spread_8_55", 99)
        ))
        with open(pulse_path, "w") as f:
            json.dump({"results": enriched, "pulse": pulse_data, "tier": tier_key, "date": today, "total": len(enriched)}, f)
        log(f"  {cfg['label']}: {len(enriched)} stocks cached → {pulse_path}")

    elapsed = (datetime.now() - start).seconds // 60
    log(f"SCAN COMPLETE in {elapsed} min. Cache ready for {today}.")
    log("Users will now get instant results from the webapp.")

if __name__ == "__main__":
    run_scan()
