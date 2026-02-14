# Polymarket Data Capture Runbook

## Purpose
Run the Phase A BTC 15m recorder on a VPS and keep it healthy for 48-72 hour collection windows.

## Script
- Recorder: `polymarket_data_capture.py`
- Runtime: Python 3.9+ (no external Python dependencies required)

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

Optional auth env vars if your deployment needs them:
- `PM_GAMMA_AUTH_TOKEN`
- `PM_CLOB_AUTH_TOKEN`

## Data Output
- Ticks: `data/raw/YYYY-MM-DD/<slug>.jsonl`
- Outcomes: `data/outcomes/YYYY-MM-DD.jsonl`
- Metadata: `data/metadata/markets.json`

## What Healthy Logs Look Like
- Startup:
`event=start ...`
- Discovery:
`event=discover markets_seen=<n> tracked=<n> added=<n>`
- Resolution:
`event=market_resolved slug=<slug> resolved=UP|DOWN`
- Heartbeat every 5 min:
`event=heartbeat active_markets=<n> unresolved_post_end=<n> ticks_written_total=<n> errors_total=<n> worst_lag_sec=<x>`

## Restart Safety
- Recorder is append-only for JSONL writes.
- `tick_seq` is persisted in metadata and resumes across restarts.
- Duplicate rows can still occur around abrupt restarts; downstream analysis should de-dupe by `(slug, tick_seq)` or `(slug, ts)`.

## Operational Notes
- Discovery primarily uses Gamma active markets paging and falls back to probing timestamp-based slugs around current time.
- Midpoint is fetched from CLOB `midpoint` endpoint (book-derived midpoint is used only as fallback).
- Resolution is based on market status fields (`umaResolutionStatus == "resolved"`) and final outcome prices.

## Suggested VPS Process Management
1. Use `systemd` or `supervisord` for auto-restart.
2. Redirect stdout/stderr to log files or journald.
3. Rotate logs daily and keep at least 7 days.

## Minimal Validation Checklist (before long run)
1. Run `--once --dry-run` and confirm discover logs appear.
2. Run `--once` and confirm new files appear under `data/raw` and `data/metadata`.
3. Confirm each tick line has `schema_version: 1`, UTC timestamps, and `market_end_ts`.
4. Confirm heartbeat appears by running continuously for >5 minutes.
