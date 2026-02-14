# Polymarket Data Capture Plan (BTC 15m Up/Down)

## Goal
Build a small, data-first foundation for a Polymarket crypto **15‑minute Up/Down** strategy.

We are **not trading yet**. We first capture our own dataset to test whether short-horizon Polymarket odds are systematically miscalibrated, and whether any edge survives spread/fees and realistic fills.

## Non-goals (for now)
- No order placement / no live trading logic.
- No full orderbook snapshots (top-of-book only).
- No complex modeling (start with calibration + simple edge sims).

## Core hypothesis
We are trading **mispriced probability**, not “predicting BTC direction.”

Later we will compute a simple fair probability for Up/Down given:
- time remaining
- recent realized volatility
- distance from interval open (optional)

We only consider trading when:
`market_probability − fair_probability > costs (spread + fees + slippage)`

## Scope
- Asset: **BTC**
- Market type: **15‑minute Up/Down**
- Sampling: **every 2 seconds** (adjust if rate limits require it)

## Rolling markets (important)
BTC 15m Up/Down markets are a **rolling series**. Recorder must discover markets dynamically (no hardcoding).
- Slug prefix: `btc-updown-15m-`
- Slug format: `btc-updown-15m-<unix_timestamp>`

## Phases
### Phase A — Data capture only (no orders)
Run a recorder on a 24/7 VPS.
- Capture **2–3 days** initially (~250–300 resolved samples).
- Keep recording while iterating.

### Phase B — Analysis (go/no-go)
Use captured data to test:
- **Calibration:** when market is X%, does it occur ~X%?
- **Where bias lives:** extremes, time-of-day, volatility regimes.
- **Tradability:** edge after costs and conservative fill assumptions.

### Phase C — Tiny live trading (later)
Only if Phase B shows a robust, tradable bias.
- Limit orders only
- Strict exposure caps
- Kill switch / stop rules

## Recorder requirements
### Market discovery + metadata
For each active market discovered via Gamma:
- Filter by slug prefix `btc-updown-15m-`
- Persist metadata needed to join ticks ↔ outcomes:
  - `slug`, `market_id`, `condition_id`
  - interval window (`start_ts`, `end_ts`) if available
  - `yes_token_id` (and `no_token_id` if available)
  - any resolution rule/oracle identifiers available from the API

### Tick capture (per active market, per tick)
Record top-of-book for YES (MVP) plus enough bookkeeping for integrity.

**Interpretation note:** YES price is treated as “market-implied probability” (subject to spread/fees). We use **midpoint** as the simplest proxy.

### External reference (per tick)
External BTC price is used for feature generation only (volatility, “up” threshold checks, etc.).
- Source: exchange feed initially
- Later: align to the market’s actual resolution oracle (often Chainlink BTC/USD) to avoid false calibration errors

### Lifecycle + resolution (recorder behavior)
- **When to stop polling a market:** stop tick polling when the market is resolved, or when the market is no longer active/available via the discovery API.
- **After `end_ts`:** once `end_ts` passes, switch from high-frequency orderbook polling to a lower-frequency **resolution-check** loop (e.g., every 15–60s) until the market is marked resolved and an outcome is available.
- **How to detect resolution:** use the market metadata/status fields from the Polymarket/Gamma APIs (e.g., resolved/closed + outcome). Do **not** infer resolution from prices collapsing to 0/1.

### Reliability + error handling
- The recorder loop must **never crash** on a single failed request.
- All API calls retry up to **3×** with exponential backoff (+ jitter). If still failing, **skip that tick**, log the error, and continue.

## On-disk schema
Use newline-delimited JSON (JSONL). Append-only.

### Tick schema (JSONL) — `schema_version: 1`
One line per `(market, tick)`:

```json
{
  "schema_version": 1,
  "ts": "ISO8601 UTC",
  "slug": "btc-updown-15m-<unix_ts>",
  "market_id": "string",
  "condition_id": "string",
  "yes_token_id": "string",
  "market_end_ts": "ISO8601 UTC",
  "bid": 0.0,
  "bid_size": 0.0,
  "ask": 0.0,
  "ask_size": 0.0,
  "mid": 0.0,
  "spread": 0.0,
  "btc_price": 0.0,
  "btc_price_source": "string",
  "btc_price_ts": "ISO8601 UTC",
  "interval_open_price": 0.0,
  "vol_1m": null,
  "latency_ms": 0,
  "tick_seq": 0
}
```

Conventions:
- `mid = (bid + ask) / 2`
- `spread = ask − bid`
- `btc_price_ts` is the timestamp of the external price sample used (can differ from `ts` if your feed batches/streams).
- `interval_open_price` = first external BTC price observed after market discovery (good enough for MVP; can be refined later)
- `vol_1m` = standard deviation of log-returns of `btc_price` over the trailing 60 seconds (≈30 samples at 2s cadence). Set to `null` until the rolling buffer is full.
- If `bid` or `ask` is missing, set the missing fields to `null` and set `mid`/`spread` to `null` (avoid using 0 as a sentinel).
- All timestamps are UTC.

### Outcome schema (JSONL)
One line per market once resolved:

```json
{
  "slug": "btc-updown-15m-<unix_ts>",
  "market_id": "string",
  "condition_id": "string",
  "resolved": "UP|DOWN",
  "resolved_ts": "ISO8601 UTC"
}
```

## Storage layout
```
data/raw/YYYY-MM-DD/<slug>.jsonl
data/outcomes/YYYY-MM-DD.jsonl
data/metadata/markets.json
```

## Rough sizing (order-of-magnitude)
- Rows/day ≈ `43,200 × avg_active_markets` (2s cadence).
- Disk/day depends on JSON verbosity; expect **single-digit to tens of MB/day** uncompressed for 1–a few markets. Keep rotation by date; optionally gzip older files.

## Data quality checklist (MVP)
- UTC everywhere; include `latency_ms`.
- Monotonic `tick_seq` per market (helps detect gaps/duplicates).
- Always record `btc_price_source` + `btc_price_ts` so you can audit external-feed staleness/drift.
- Recorder restart is safe (append-only, tolerate duplicates; analysis can de-dupe by `(slug, tick_seq)` or `(slug, ts)`).
- Log discovery/removal of markets and any API errors/rate-limit events.
- Heartbeat every ~5 minutes: active markets count, ticks written, error count, and worst lag (so a VPS run is debuggable).

## Initial success criteria
After ~2–3 days (~250–300 resolved markets):
- Detect consistent miscalibration (especially at extremes and/or specific regimes).
- Show conservative edge remains positive after spread/fees assumptions.
- Confirm liquidity is sufficient for small sizing.

## Next steps (do one at a time)
1) Confirm **exact market resolution rule** (UP vs DOWN definition) and the **truth oracle** used for resolution.
2) Confirm Gamma discovery filters + which fields are reliably present (`market_id`, `condition_id`, tokens, start/end).
3) Lock schema (`schema_version: 1`) and implement recorder + file rotation.
4) Run 48–72 hours.
5) Analyze: calibration curves + simple edge simulation with conservative costs/fills.
