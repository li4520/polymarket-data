# Polymarket Data Capture — Task List (Phase A)

## Header
- Status: ✅ ACTIVE
- Last updated: 2026-02-14
- Tech Plan: `plans/polymarket-data.md`

---

## Next Action
- [x] Confirm Polymarket APIs/auth + choose BTC price feed
  - Chosen endpoints:
  - Gamma discovery/details: `GET https://gamma-api.polymarket.com/markets`
  - CLOB top-of-book: `GET https://clob.polymarket.com/book?token_id=<id>`
  - CLOB midpoint: `GET https://clob.polymarket.com/midpoint?token_id=<id>`
  - Chosen BTC source (MVP default): Coinbase ticker `GET https://api.exchange.coinbase.com/products/BTC-USD/ticker`
  - Auth defaults: unauthenticated public reads; optional headers supported by env vars.
  - Required/optional env vars implemented:
  - `PM_DATA_DIR`, `PM_INTERVAL_SEC`, `PM_DISCOVER_INTERVAL_SEC`, `PM_RESOLUTION_INTERVAL_SEC`, `PM_HEARTBEAT_INTERVAL_SEC`
  - `PM_SLUG_PREFIX`, `PM_GAMMA_BASE`, `PM_CLOB_BASE`, `PM_BTC_PRICE_SOURCE`
  - `PM_DISCOVER_LIMIT`, `PM_DISCOVER_PAGES`, `PM_PROBE_INTERVALS_BACK`, `PM_PROBE_INTERVALS_FORWARD`
  - `PM_REQUEST_TIMEOUT_SEC`, `PM_MAX_RETRIES`, `PM_BASE_BACKOFF_SEC`, `PM_GAMMA_AUTH_TOKEN`, `PM_CLOB_AUTH_TOKEN`, `PM_LOG_LEVEL`

---

## Phases

## Phase 0 — Setup / Validation
- [x] Lock the MVP resolution definition (UP vs DOWN)
  - Acceptance: written in this doc as a precise rule (what two prices are compared, how timestamps are chosen, and what happens on equality/ties if applicable).
  - Verify: recorder uses Gamma market status fields (`umaResolutionStatus == "resolved"`) plus `outcomePrices`/`outcomes` winner mapping (`Up` -> `UP`, `Down` -> `DOWN`) and writes `resolved_ts` from `closedTime` when available.

- [x] Define recorder configuration surface (env vars + CLI flags)
  - Acceptance: list of env vars (names + meaning) and CLI flags (`--interval`, `--data-dir`, `--dry-run`, `--once`, etc.) sufficient to run on a VPS unattended.
  - Verify: implemented in `polymarket_data_capture.py` CLI help and env-backed defaults.

- [x] Confirm required market metadata fields are available from Gamma
  - Acceptance: for discovered markets we can reliably obtain at least `slug`, `market_id`, `condition_id`, `yes_token_id`, and `end_ts` (or a substitute that can be converted into `market_end_ts`).
  - Verify: recorder persists `slug`, `market_id` (`id`), `condition_id` (`conditionId`), `yes_token_id` (from parsed `clobTokenIds`), and `market_end_ts` (`endDate`) in metadata.

## Phase 1 — Core Implementation (Recorder)
- [x] Implement market discovery loop (rolling markets)
  - Acceptance: recorder discovers active markets by slug prefix `btc-updown-15m-` and maintains an in-memory set of “active markets”.
  - Verify: run discovery in `--dry-run` mode and see added/removed market events in logs without crashing.

- [x] Persist `data/metadata/markets.json` (append/update safely)
  - Acceptance: metadata is written so ticks/outcomes can be joined later (at minimum: `slug`, `market_id`, `condition_id`, `yes_token_id`, `market_end_ts`).
  - Verify: start/stop the recorder twice and confirm metadata remains valid JSON and includes newly discovered markets.

- [x] Implement CLOB top-of-book polling for YES
  - Acceptance: for each active market, fetch best bid/ask price+size for the YES token id; missing sides are recorded as `null` (not 0).
  - Verify: in `--once` mode, write a single tick JSON object where `bid/ask/mid/spread` match expectations.

- [x] Implement external BTC spot price fetcher
  - Acceptance: on each tick, record `btc_price`, `btc_price_source`, and `btc_price_ts` (can differ from tick `ts`).
  - Verify: temporarily disconnect the price source and confirm retries/backoff happen and the loop continues (tick records may have `btc_price: null` if source is down).

- [x] Implement `interval_open_price` tracking per market
  - Acceptance: `interval_open_price` is set to the first external BTC price observed after market discovery (per plan) and remains constant for that market.
  - Verify: restart recorder mid-market; confirm the field remains coherent (either persisted per market or re-derived consistently and noted in logs).

- [x] Implement `vol_1m` calculation buffer
  - Acceptance: `vol_1m` is the stdev of log-returns of `btc_price` over trailing 60 seconds; set to `null` until enough samples exist.
  - Verify: unit/integration check that `vol_1m` is `null` for the first ~60s and becomes non-null thereafter given stable price inputs.

- [x] Implement tick serialization (schema_version 1) + per-market `tick_seq`
  - Acceptance: each tick JSONL line matches the plan schema, includes `schema_version: 1`, and has monotonic `tick_seq` per market.
  - Verify: run for 2 minutes and confirm each market’s `tick_seq` increments by 1 per tick with no resets unless explicitly documented.

- [x] Implement JSONL writer + daily rotation
  - Acceptance: ticks are appended to `data/raw/YYYY-MM-DD/<slug>.jsonl` and outcomes to `data/outcomes/YYYY-MM-DD.jsonl`; directory tree is created automatically.
  - Verify: run across a UTC date boundary (or simulate) and confirm a new folder/file is created without losing writes.

- [x] Implement lifecycle handling + resolution checks
  - Acceptance: stop high-frequency tick polling when market is resolved or no longer active; after `market_end_ts`, switch to a lower-frequency resolution-check loop until outcome is available, then write an outcome record.
  - Verify: observe one full market lifecycle end-to-end and confirm exactly one outcome line is written per market.

## Phase 2 — Integration (VPS Run Readiness)
- [x] Add structured logging + 5-minute heartbeat
  - Acceptance: heartbeat logs include active markets count, ticks written, error count, and worst lag; logs are readable for unattended operation.
  - Verify: run for 10+ minutes and confirm heartbeat appears at the intended cadence.

- [x] Add backoff/retry + “never crash” guarantees
  - Acceptance: all external calls retry up to 3× with exponential backoff + jitter; persistent failures skip the tick and continue.
  - Verify: induce failures (bad DNS / 500s / timeouts) and confirm the process keeps running and logs gaps clearly.

## Phase 3 — Testing + Hardening
- [x] Add a minimal smoke test procedure (manual or automated)
  - Acceptance: there’s a documented procedure to verify discovery + one tick write + file paths without running 48–72h.
  - Verify: smoke run validated with `--once --dry-run` and `--once`; see runbook in `tasks/polymarket-data-capture-runbook.md`.

- [ ] Validate schema invariants
  - Acceptance: missing bid/ask never uses 0 as sentinel; timestamps are UTC; `market_end_ts` is present on every tick line.
  - Verify: scan a sample file and confirm invariants hold; record any exceptions as TODOs before long runs.

## Phase 4 — Documentation + Wrap-up
- [x] Write a short runbook for VPS operation
  - Acceptance: includes how to run, required env vars, expected disk growth, log locations, and how to restart safely.
  - Verify: documented in `tasks/polymarket-data-capture-runbook.md`.

---

## Notes / Assumptions
- This task list is for **Phase A (data capture only)**. No trading/order placement tasks belong here.
- Resolution should be detected from **official status/outcome fields**, not inferred from prices.
- If required API fields are missing/inconsistent, update the tech plan and re-align before implementation continues.

## Open Questions (to resolve in Phase 0)
- Which exact Gamma endpoints/fields are used for discovery, and which CLOB endpoints are used for top-of-book?
- What auth is required for those endpoints (if any), and what credentials are available on the VPS?
- Which BTC spot price feed is the MVP choice (and do we have keys / rate limits)?
