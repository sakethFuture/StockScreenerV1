# Market Pulse Screener

Bloomberg-terminal-style webapp version of your `smallcap_pulse_screen.ipynb`.  
Runs locally. No paid data. yfinance + NSE universe.

---

## Setup

```bash
pip install flask flask-cors yfinance pandas numpy requests
python run.py
# Open http://localhost:5050
```

---

## How to use

### 1. Pick your tier
| Tier       | MCap range         | Index        | Philosophy                          |
|------------|--------------------|--------------|--------------------------------------|
| Small Cap  | 0 – 30,000 Cr      | Nifty SC 100 | Volatile, micro-cap needs dividend  |
| Mid Cap    | 10,000 – 50,000 Cr | Nifty MC 100 | Stricter spread, tighter thresholds |
| Large Cap  | 50,000 Cr+         | Nifty 50     | Institution-driven, need more dist  |

### 2. Load Pulse
Reads the relevant index (6-month data) and tells you:
- **CONFIRMED UPTREND** 🟢 — All FULL+MID setups actionable
- **UNDER PRESSURE** 🟡 — FULL only, spread < 2%
- **RALLY ATTEMPT** 🔵 — Build watchlist, NO buys until FTD
- **DOWNTREND** 🔴 — Watchlist only

Distribution day thresholds are calibrated per tier (small caps tolerate more noise).

### 3. Screen 1 — Base screen (EMA + 200 DMA)
Scans full NSE universe (~2400 stocks):
- 200 DMA rising (today > 21 sessions ago)
- EMA 8/13/21/34/55 compressing (spread tightening vs 5 sessions ago)
- Spread ≤ SPREAD_MAX (5% small, 4.5% mid, 4% large)
- Price within ±5% of EMA cluster
- MCap within tier range
- Micro-cap rule (small cap only): dividend yield > 0 if MCap ≤ 1,000 Cr

Results cached for the day. Re-running Screen 2 reuses today's cache.

**Columns**: TICKER · STAGE · MCAP TIER · MCAP (Cr) · DIV % · PRICE · SPREAD 8-55 · SPREAD 5D · % vs EMA · 200 DMA SLOPE · % OFF 52H

### 4. Screen 2 — Pulse analysis
Runs stock-level pulse on Screen 1 results:
- Distribution days per stock (25-session window)
- Follow-through day detection (day 4-7 of rally attempt)
- Rally attempt state

**Columns add**: ACTION · STOCK PULSE · D-DAYS · FT DAY · RALLY · RALLY DAY

**ACTION logic**:
- `ACTIONABLE` — Market + stock both healthy
- `CAUTION` — Market under pressure but stock clean (spread < 2% only)
- `WATCHLIST` — Market in rally attempt or downtrend

### Filters + sort
- Filter by STAGE (FULL / MID / FAST)
- Filter by ACTION (pulse screen only)
- Click any column header to sort
- Search by ticker

---

## EMA stage definitions

| Stage | Spread 8–55 | Meaning                         |
|-------|-------------|----------------------------------|
| FULL  | ≤ 3%        | Tightest compression, best risk  |
| MID   | 3–4%        | Moderate compression             |
| FAST  | 4–5%        | Loose but still valid            |

(Thresholds differ slightly per tier — see `TIER_CONFIG` in `app.py`)

---

## Cache
- Universe: cached 12 hours in `screener_cache/universe.csv`
- Base screen: cached per tier per day in `screener_cache/base_{tier}_{date}.json`
- Reload page — cached data loads automatically

---

## Customise

Edit `TIER_CONFIG` in `app.py`:
- `spread_max`, `full_thresh`, `mid_thresh` — EMA compression thresholds
- `dist_down_pct` — what % down = distribution day
- `dist_pressure`, `dist_downtrend` — distribution day counts for state change
- `ftd_up_pct` — follow-through day minimum % up
