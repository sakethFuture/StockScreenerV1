# CELL 3 - MARKET PULSE ANALYSIS
# Runs on RESULT_DF from Cell 2
# For each stock: distribution days, follow-through day, trend state
# Also computes overall index pulse

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Pulse thresholds (calibrated per stage) ───────────────────────────────────
DIST_DOWN_PCT  = 0.8    # % down = distribution day
DIST_WINDOW    = 25     # sessions to look back
DIST_PRESSURE  = 4      # dist days → under pressure
DIST_DOWNTREND = 6      # dist days → downtrend
FTD_UP_PCT     = 2.5    # % up needed for follow-through day
FTD_DAY_MIN    = 4      # earliest FTD day after low
FTD_DAY_MAX    = 7      # latest FTD day

INDEX_TICKER   = "^NSEI"   # Nifty 50 as market proxy

# ── Helper ────────────────────────────────────────────────────────────────────

def ema(arr, p):
    return pd.Series(arr.astype(float)).ewm(span=p, adjust=False).mean().values

def compute_pulse_from_close(close, volume):
    """
    Given close + volume series, returns pulse dict.
    Works for both index and individual stocks.
    """
    chg     = close.pct_change() * 100
    avg_vol = volume.rolling(20).mean()

    # Distribution days
    dist_mask  = (chg <= -DIST_DOWN_PCT) & (volume > avg_vol)
    dist_count = int(dist_mask.iloc[-DIST_WINDOW:].sum())
    dist_dates = [str(d)[:10] for d in dist_mask[dist_mask].iloc[-5:].index]

    # Rally attempt detection
    c          = close.values
    low_idx    = len(c) - 15 + int(np.argmin(c[-15:]))
    low_price  = float(c[low_idx])
    days_since = len(c) - 1 - low_idx
    bounce     = (float(c[-1]) - low_price) / low_price * 100
    in_rally   = 1.0 <= bounce and 1 <= days_since <= 12

    # Follow-through day
    ftd = False
    ftd_day = None
    if in_rally:
        for day in range(FTD_DAY_MIN, min(FTD_DAY_MAX + 1, days_since + 1)):
            idx = low_idx + day
            if idx >= len(c): break
            if (float(chg.iloc[idx]) >= FTD_UP_PCT and
                    float(volume.iloc[idx]) > float(avg_vol.iloc[idx])):
                ftd = True
                ftd_day = day
                break

    # State
    if dist_count >= DIST_DOWNTREND:
        state = "RALLY ATTEMPT" if in_rally else "DOWNTREND"
    elif dist_count >= DIST_PRESSURE:
        state = "UNDER PRESSURE"
    else:
        state = "CONFIRMED UPTREND"
    if ftd:
        state = "CONFIRMED UPTREND"

    return {
        "state":       state,
        "dist_days":   dist_count,
        "in_rally":    in_rally,
        "rally_day":   days_since if in_rally else None,
        "ftd":         ftd,
        "ftd_day":     ftd_day,
        "dist_dates":  dist_dates[-3:],
    }

# ── Step 1: Index pulse ───────────────────────────────────────────────────────

print("Computing index pulse (Nifty 50) …")
try:
    idx_data = yf.download(INDEX_TICKER, period="6mo", interval="1d",
                           auto_adjust=True, progress=False)
    if isinstance(idx_data.columns, pd.MultiIndex):
        idx_close  = idx_data["Close"].iloc[:, 0].dropna().astype(float)
        idx_volume = idx_data["Volume"].iloc[:, 0].dropna().astype(float)
    else:
        idx_close  = idx_data["Close"].dropna().astype(float)
        idx_volume = idx_data["Volume"].dropna().astype(float)

    mkt = compute_pulse_from_close(idx_close, idx_volume)
    mkt_state = mkt["state"]

    STATE_COLORS = {
        "CONFIRMED UPTREND": "🟢",
        "UNDER PRESSURE":    "🟡",
        "RALLY ATTEMPT":     "🔵",
        "DOWNTREND":         "🔴",
    }
    ACTION_MAP = {
        "CONFIRMED UPTREND": "All FULL + MID setups actionable.",
        "UNDER PRESSURE":    "FULL stage only. Tight spread < 2% only.",
        "RALLY ATTEMPT":     "Build watchlist. NO buys until FTD confirmed.",
        "DOWNTREND":         "Watchlist only. No new positions.",
    }

    print(f"\n{'='*55}")
    print(f"  MARKET PULSE: {STATE_COLORS.get(mkt_state,'')} {mkt_state}")
    print(f"  Distribution days (last {DIST_WINDOW}d): {mkt['dist_days']}  "
          f"[pressure≥{DIST_PRESSURE} / downtrend≥{DIST_DOWNTREND}]")
    print(f"  Rally attempt: {'Yes — Day ' + str(mkt['rally_day']) if mkt['in_rally'] else 'No'}")
    print(f"  Follow-through: {'✓ Day ' + str(mkt['ftd_day']) if mkt['ftd'] else '—'}")
    if mkt['dist_dates']:
        print(f"  Recent dist days: {', '.join(mkt['dist_dates'])}")
    print(f"  Action: {ACTION_MAP.get(mkt_state,'')}")
    print(f"{'='*55}\n")

except Exception as e:
    mkt_state = "CONFIRMED UPTREND"
    print(f"Index fetch failed ({e}) — defaulting to CONFIRMED UPTREND")

# ── Step 2: Per-stock pulse ───────────────────────────────────────────────────

if 'RESULT_DF' not in dir() or RESULT_DF.empty:
    print("RESULT_DF not found — run Cell 2 first.")
else:
    tickers = RESULT_DF["ticker"].tolist()
    print(f"Running stock-level pulse on {len(tickers)} stocks …\n")

    pulse_rows = []
    for i, ticker in enumerate(tickers):
        try:
            d = yf.download(ticker, period="6mo", interval="1d",
                            auto_adjust=True, progress=False)
            if d is None or d.empty:
                raise ValueError("no data")
            if isinstance(d.columns, pd.MultiIndex):
                close  = d["Close"].iloc[:, 0].dropna().astype(float)
                volume = d["Volume"].iloc[:, 0].dropna().astype(float)
            else:
                close  = d["Close"].dropna().astype(float)
                volume = d["Volume"].dropna().astype(float)

            p = compute_pulse_from_close(close, volume)

            # Actionability: cross market pulse × stock pulse
            sp = p["state"]
            if mkt_state == "CONFIRMED UPTREND" and sp in ["CONFIRMED UPTREND", "UNDER PRESSURE"]:
                action = "ACTIONABLE"
            elif mkt_state == "UNDER PRESSURE" and sp == "CONFIRMED UPTREND":
                action = "CAUTION"
            else:
                action = "WATCHLIST"

            row = RESULT_DF[RESULT_DF["ticker"] == ticker].iloc[0].to_dict()
            row.update({
                "stock_pulse": sp,
                "d_days":      p["dist_days"],
                "ft_day":      p["ftd_day"] if p["ftd"] else "—",
                "in_rally":    "Yes" if p["in_rally"] else "—",
                "rally_day":   p["rally_day"] if p["in_rally"] else "—",
                "action":      action,
            })
            pulse_rows.append(row)

        except Exception as e:
            row = RESULT_DF[RESULT_DF["ticker"] == ticker].iloc[0].to_dict()
            row.update({"stock_pulse": "ERROR", "d_days": "—",
                        "ft_day": "—", "in_rally": "—",
                        "rally_day": "—", "action": "WATCHLIST"})
            pulse_rows.append(row)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(tickers)} done …")

    # ── Build final table ─────────────────────────────────────────────────────
    PULSE_DF = pd.DataFrame(pulse_rows)

    action_ord = {"ACTIONABLE": 0, "CAUTION": 1, "WATCHLIST": 2}
    stage_ord  = {"FULL": 0, "MID": 1, "FAST": 2}
    PULSE_DF["_a"] = PULSE_DF["action"].map(action_ord)
    PULSE_DF["_s"] = PULSE_DF["stage"].map(stage_ord)
    PULSE_DF = (PULSE_DF.sort_values(["_a", "_s", "pct_off_52h"])
                        .drop(columns=["_a", "_s"])
                        .reset_index(drop=True))

    DISPLAY_COLS = ["ticker", "name", "stage", "action",
                    "stock_pulse", "d_days", "ft_day",
                    "in_rally", "rally_day", "pct_off_52h"]

    # ── Print by action group ─────────────────────────────────────────────────
    for grp_label in ["ACTIONABLE", "CAUTION", "WATCHLIST"]:
        grp = PULSE_DF[PULSE_DF["action"] == grp_label]
        if grp.empty: continue
        print(f"\n{'─'*70}")
        print(f"  {grp_label} ({len(grp)} stocks)")
        print(f"{'─'*70}")
        print(grp[DISPLAY_COLS].to_string(index=False))

    print(f"\n{'='*70}")
    print(f"  TOTAL: {len(PULSE_DF)} stocks")
    print(f"  ACTIONABLE: {len(PULSE_DF[PULSE_DF['action']=='ACTIONABLE'])}  "
          f"CAUTION: {len(PULSE_DF[PULSE_DF['action']=='CAUTION'])}  "
          f"WATCHLIST: {len(PULSE_DF[PULSE_DF['action']=='WATCHLIST'])}")
    print(f"  Market pulse: {mkt_state}")
    print(f"{'='*70}")

    import builtins
    builtins.PULSE_DF = PULSE_DF
    print("\nPULSE_DF saved. Run Cell 4 to export.")
