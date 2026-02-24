# polymarket-data

BTC 15-minute Polymarket data recorder for Phase A capture windows.

## What This Does
- Discovers BTC 15m up/down markets from Gamma.
- Polls CLOB top-of-book and midpoint for each active market.
- Captures BTC spot price for context (`coinbase` or `binance`).
- Writes append-only JSONL ticks and resolution outcomes.
- Persists metadata (`tick_seq`, market mapping, status) for restart safety.

## Requirements
- Python 3.9+
- No external Python packages are required.

If your workflow expects `requirements.txt`:
```bash
python3 -m pip install -r requirements.txt
```

## Quick Start
1. Dry-run smoke test:
```bash
python3 polymarket_data_capture.py --once --dry-run --log-level INFO
```
2. One-shot write test:
```bash
python3 polymarket_data_capture.py --once --log-level INFO
```
3. Continuous capture:
```bash
python3 polymarket_data_capture.py --log-level INFO
```

## Recommended Environment
```bash
export PM_DATA_DIR=data
export PM_INTERVAL_SEC=2
export PM_DISCOVER_INTERVAL_SEC=60
export PM_RESOLUTION_INTERVAL_SEC=30
export PM_HEARTBEAT_INTERVAL_SEC=300
export PM_SLUG_PREFIX=btc-updown-15m-
export PM_GAMMA_BASE=https://gamma-api.polymarket.com
export PM_CLOB_BASE=https://clob.polymarket.com
export PM_BTC_PRICE_SOURCE=coinbase
export PM_MAX_RETRIES=3
export PM_BASE_BACKOFF_SEC=0.35
export PM_LOG_LEVEL=INFO
```

Optional auth env vars:
- `PM_GAMMA_AUTH_TOKEN`
- `PM_CLOB_AUTH_TOKEN`

## Data Layout
- Ticks: `data/raw/YYYY-MM-DD/<slug>.jsonl`
- Outcomes: `data/outcomes/YYYY-MM-DD.jsonl`
- Metadata: `data/metadata/markets.json`

`data/` is gitignored by default.

## Healthy Log Signals
- `event=start ...`
- `event=discover markets_seen=<n> tracked=<n> added=<n>`
- `event=market_resolved slug=<slug> resolved=UP|DOWN`
- `event=heartbeat active_markets=<n> unresolved_post_end=<n> ticks_written_total=<n> errors_total=<n> worst_lag_sec=<x>`

## Restart Safety
- JSONL writes are append-only.
- `tick_seq` is persisted and restored.
- Around abrupt restarts, duplicates can still occur; downstream should de-dupe by `(slug, tick_seq)` or `(slug, ts)`.

## Operations Notes
- Discovery uses Gamma active paging plus timestamp-slug probing.
- Midpoint comes from CLOB `midpoint` endpoint (book midpoint is fallback only).
- Resolution detection uses `umaResolutionStatus == "resolved"` and `outcomePrices`.

## Runbook
Detailed operational checklist lives in:
- `tasks/polymarket-data-capture-runbook.md`

## Offline Backtest Simulation (Analysis Only)
This repo also includes an **offline** backtest simulator that reads recorded ticks plus recorded outcomes and simulates a conservative strategy. It never places real trades.

Run with defaults:
```bash
python analyze_simulation.py
```

Example with custom parameters:
```bash
python analyze_simulation.py --stake 10 --fee-rate 0.02 --high-conf 0.65 --low-conf 0.35 --early-time-sec 480 --spread-max 0.10 --dedupe-by mtime --verbose
```

Outputs:
- `results/trades.csv`
- `results/summary.json`

Note: `analyze_simulation.py` requires `pandas` and `numpy` (unlike the recorder script).
