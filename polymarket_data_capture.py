#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import logging
import math
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_utc(ts: Optional[dt.datetime] = None) -> str:
    value = ts or utc_now()
    return value.isoformat().replace("+00:00", "Z")


def parse_iso_utc(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(raw).astimezone(dt.timezone.utc)
    except ValueError:
        return None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_json_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, list) else []
        except json.JSONDecodeError:
            return []
    return []


def safe_slug_filename(slug: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in slug)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_last_jsonl_line(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        pos = handle.tell() - 1
        while pos > 0:
            handle.seek(pos)
            if handle.read(1) == b"\n":
                break
            pos -= 1
        if pos <= 0:
            handle.seek(0)
        line = handle.readline().decode("utf-8").strip()
    if not line:
        return None
    try:
        decoded = json.loads(line)
        return decoded if isinstance(decoded, dict) else None
    except json.JSONDecodeError:
        return None


class HttpClient:
    def __init__(
        self,
        default_headers: Dict[str, str],
        max_retries: int = 3,
        base_backoff: float = 0.35,
        timeout_sec: float = 10.0,
    ) -> None:
        self.default_headers = default_headers
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.timeout_sec = timeout_sec

    def get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        query = urllib.parse.urlencode(params or {}, doseq=True)
        full_url = f"{url}?{query}" if query else url
        attempts = self.max_retries + 1
        last_error: Optional[Exception] = None
        headers = dict(self.default_headers)
        if extra_headers:
            headers.update(extra_headers)
        for attempt in range(1, attempts + 1):
            try:
                req = urllib.request.Request(full_url, headers=headers, method="GET")
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                    raw = response.read().decode("utf-8")
                return json.loads(raw)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= attempts:
                    break
                sleep_for = self.base_backoff * (2 ** (attempt - 1)) + random.uniform(0.0, 0.2)
                time.sleep(sleep_for)
        if last_error:
            raise last_error
        raise RuntimeError("unexpected request failure")


@dataclass
class MarketState:
    slug: str
    market_id: str
    condition_id: str
    yes_token_id: str
    no_token_id: Optional[str]
    market_end_ts: str
    market_end_dt: Optional[dt.datetime]
    resolution_source: Optional[str]
    discovered_at: str
    active: bool = True
    last_seen_discovery: Optional[dt.datetime] = None
    interval_open_price: Optional[float] = None
    tick_seq: int = 0
    price_window: deque = field(default_factory=deque)
    resolved: bool = False
    last_resolution_check: Optional[dt.datetime] = None


class Recorder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.data_dir = Path(args.data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.outcomes_dir = self.data_dir / "outcomes"
        self.metadata_path = self.data_dir / "metadata" / "markets.json"
        self.slug_prefix = args.slug_prefix
        self.gamma_base = args.gamma_base.rstrip("/")
        self.clob_base = args.clob_base.rstrip("/")
        self.price_source = args.price_source
        self.discover_limit = args.discover_limit
        self.discover_pages = args.discover_pages
        self.probe_intervals_back = args.probe_intervals_back
        self.probe_intervals_forward = args.probe_intervals_forward
        self.poll_interval = args.interval_sec
        self.discover_interval = args.discover_interval_sec
        self.resolution_interval = args.resolution_interval_sec
        self.heartbeat_interval = args.heartbeat_interval_sec
        self.dry_run = args.dry_run
        self.once = args.once
        self.logger = logging.getLogger("pm_recorder")
        headers = {"User-Agent": "polymarket-data-capture/1.0"}
        self.gamma_headers: Dict[str, str] = {}
        self.clob_headers: Dict[str, str] = {}
        if args.gamma_auth_token:
            self.gamma_headers["Authorization"] = f"Bearer {args.gamma_auth_token}"
        if args.clob_auth_token:
            self.clob_headers["X-API-Key"] = args.clob_auth_token
        self.http = HttpClient(
            headers,
            max_retries=args.max_retries,
            base_backoff=args.base_backoff_sec,
            timeout_sec=args.request_timeout_sec,
        )
        self.markets: Dict[str, MarketState] = {}
        self.meta_store: Dict[str, Any] = {"markets": {}}
        self.resolved_slugs: set[str] = set()
        self.total_ticks_written = 0
        self.total_errors = 0
        self.last_discovery_at: Optional[dt.datetime] = None
        self.last_heartbeat_at = utc_now()
        self.worst_loop_lag_sec = 0.0

    def load_state(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.outcomes_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if self.metadata_path.exists():
            try:
                with self.metadata_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                    if isinstance(loaded, dict):
                        self.meta_store = loaded
            except Exception as exc:  # noqa: BLE001
                self.logger.error("event=metadata_load_error path=%s error=%s", str(self.metadata_path), exc)
                self.meta_store = {"markets": {}}
        self.meta_store.setdefault("markets", {})
        self.load_existing_outcomes()

        for slug, details in self.meta_store["markets"].items():
            state = self.market_state_from_meta(slug, details)
            if state:
                self.markets[slug] = state

        self.restore_tick_seq_counters()

    def market_state_from_meta(self, slug: str, details: Dict[str, Any]) -> Optional[MarketState]:
        yes_token_id = details.get("yes_token_id")
        if not yes_token_id:
            return None
        market_end_ts = details.get("market_end_ts") or ""
        resolved = bool(details.get("resolved", False)) or (slug in self.resolved_slugs)
        state = MarketState(
            slug=slug,
            market_id=str(details.get("market_id", "")),
            condition_id=str(details.get("condition_id", "")),
            yes_token_id=str(yes_token_id),
            no_token_id=details.get("no_token_id"),
            market_end_ts=market_end_ts,
            market_end_dt=parse_iso_utc(market_end_ts),
            resolution_source=details.get("resolution_source"),
            discovered_at=details.get("discovered_at", iso_utc()),
            active=bool(details.get("active", True)),
            interval_open_price=None,
            tick_seq=int(details.get("tick_seq", 0) or 0),
            resolved=resolved,
        )
        if resolved:
            state.active = False
        return state

    def load_existing_outcomes(self) -> None:
        if not self.outcomes_dir.exists():
            return
        for path in sorted(self.outcomes_dir.glob("*.jsonl")):
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    slug = obj.get("slug")
                    if slug:
                        self.resolved_slugs.add(str(slug))

    def restore_tick_seq_counters(self) -> None:
        current_date = utc_now().strftime("%Y-%m-%d")
        for slug, state in self.markets.items():
            path = self.raw_dir / current_date / f"{safe_slug_filename(slug)}.jsonl"
            last_obj = read_last_jsonl_line(path)
            if not last_obj:
                continue
            tick_seq = last_obj.get("tick_seq")
            if isinstance(tick_seq, int) and tick_seq >= 0:
                state.tick_seq = max(state.tick_seq, tick_seq + 1)

    def save_metadata(self) -> None:
        if self.dry_run:
            return
        atomic_write_json(self.metadata_path, self.meta_store)

    def resolve_yes_index(self, outcomes: List[Any]) -> int:
        lowered = [str(item).strip().lower() for item in outcomes]
        if "up" in lowered:
            return lowered.index("up")
        if "yes" in lowered:
            return lowered.index("yes")
        return 0

    def fetch_market_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        data = self.http.get_json(
            f"{self.gamma_base}/markets",
            params={"slug": slug},
            extra_headers=self.gamma_headers,
        )
        if isinstance(data, list) and data:
            first = data[0]
            return first if isinstance(first, dict) else None
        if isinstance(data, dict):
            return data
        return None

    def discover_markets(self) -> None:
        rows: List[Dict[str, Any]] = []
        for page in range(self.discover_pages):
            offset = page * self.discover_limit
            params = {"active": "true", "limit": str(self.discover_limit), "offset": str(offset)}
            page_rows = self.http.get_json(
                f"{self.gamma_base}/markets",
                params=params,
                extra_headers=self.gamma_headers,
            )
            if not isinstance(page_rows, list) or not page_rows:
                break
            rows.extend(item for item in page_rows if isinstance(item, dict))
        now = utc_now()
        seen_slugs: set[str] = set()
        added = 0
        for row in rows:
            slug = str(row.get("slug") or "")
            if not slug.startswith(self.slug_prefix):
                continue
            seen_slugs.add(slug)
            state = self.upsert_market_from_row(row, now)
            if state:
                if state.slug not in self.markets:
                    self.markets[state.slug] = state
                    added += 1
                else:
                    current = self.markets[state.slug]
                    current.active = False if current.resolved else True
                    current.last_seen_discovery = now
                    current.market_end_ts = state.market_end_ts
                    current.market_end_dt = state.market_end_dt

        probe_hits = self.probe_timestamp_slugs(now)
        for row in probe_hits:
            slug = str(row.get("slug") or "")
            if not slug or slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            state = self.upsert_market_from_row(row, now)
            if state:
                if state.slug not in self.markets:
                    self.markets[state.slug] = state
                    added += 1
                else:
                    current = self.markets[state.slug]
                    current.active = False if current.resolved else True
                    current.last_seen_discovery = now
                    current.market_end_ts = state.market_end_ts
                    current.market_end_dt = state.market_end_dt

        # Keep prior active state for unseen unresolved markets. Discovery can be partial/transient,
        # and ended markets are already gated by should_poll_tick().
        self.last_discovery_at = now
        self.logger.info(
            "event=discover markets_seen=%d tracked=%d added=%d",
            len(seen_slugs),
            len(self.markets),
            added,
        )

    def probe_timestamp_slugs(self, now: dt.datetime) -> List[Dict[str, Any]]:
        hits: List[Dict[str, Any]] = []
        seen: set[str] = set()
        anchor = int(now.timestamp() // 900 * 900)
        for step in range(-self.probe_intervals_back, self.probe_intervals_forward + 1):
            ts_value = anchor + (step * 900)
            slug = f"{self.slug_prefix}{ts_value}"
            if slug in seen:
                continue
            seen.add(slug)
            try:
                market = self.fetch_market_by_slug(slug)
            except Exception:  # noqa: BLE001
                continue
            if not market:
                continue
            if str(market.get("active")).lower() != "true":
                continue
            hits.append(market)
        return hits

    def upsert_market_from_row(self, row: Dict[str, Any], now: dt.datetime) -> Optional[MarketState]:
        slug = str(row.get("slug") or "")
        outcomes = parse_json_list(row.get("outcomes"))
        token_ids = [str(item) for item in parse_json_list(row.get("clobTokenIds"))]
        if not token_ids:
            details = self.fetch_market_by_slug(slug)
            if details:
                outcomes = parse_json_list(details.get("outcomes")) or outcomes
                token_ids = [str(item) for item in parse_json_list(details.get("clobTokenIds"))]
                row = details
        if not token_ids:
            return None
        yes_index = self.resolve_yes_index(outcomes)
        yes_token_id = token_ids[yes_index] if yes_index < len(token_ids) else token_ids[0]
        no_token_id = token_ids[1 - yes_index] if len(token_ids) > 1 and yes_index in (0, 1) else None
        market_end_ts = str(row.get("endDate") or "")
        state = MarketState(
            slug=slug,
            market_id=str(row.get("id") or ""),
            condition_id=str(row.get("conditionId") or ""),
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
            market_end_ts=market_end_ts,
            market_end_dt=parse_iso_utc(market_end_ts),
            resolution_source=row.get("resolutionSource"),
            discovered_at=iso_utc(now),
            active=True,
            last_seen_discovery=now,
            resolved=slug in self.resolved_slugs,
        )
        store_entry = self.meta_store["markets"].setdefault(slug, {})
        resolved = bool(store_entry.get("resolved", False)) or (slug in self.resolved_slugs)
        store_entry.update(
            {
                "slug": slug,
                "market_id": state.market_id,
                "condition_id": state.condition_id,
                "yes_token_id": yes_token_id,
                "no_token_id": no_token_id,
                "market_end_ts": market_end_ts,
                "resolution_source": state.resolution_source,
                "discovered_at": store_entry.get("discovered_at", state.discovered_at),
                "active": False if resolved else True,
                "interval_open_price": None,
            }
        )
        state.resolved = resolved
        if resolved:
            state.active = False
        return state

    def fetch_btc_price(self) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        if self.price_source.lower() == "coinbase":
            payload = self.http.get_json("https://api.exchange.coinbase.com/products/BTC-USD/ticker")
            if not isinstance(payload, dict):
                return None, "coinbase", None
            price = parse_float(payload.get("price"))
            source_ts = payload.get("time")
            source_dt = parse_iso_utc(source_ts) if isinstance(source_ts, str) else None
            return price, "coinbase", iso_utc(source_dt) if source_dt else None
        if self.price_source.lower() == "binance":
            payload = self.http.get_json("https://api.binance.com/api/v3/ticker/price", params={"symbol": "BTCUSDT"})
            if not isinstance(payload, dict):
                return None, "binance", None
            price = parse_float(payload.get("price"))
            return price, "binance", iso_utc()
        raise ValueError(f"unsupported price source: {self.price_source}")

    def fetch_book_top(self, token_id: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        payload = self.http.get_json(
            f"{self.clob_base}/book",
            params={"token_id": token_id},
            extra_headers=self.clob_headers,
        )
        if not isinstance(payload, dict):
            return None, None, None, None
        bids = payload.get("bids")
        asks = payload.get("asks")
        top_bid = bids[0] if isinstance(bids, list) and bids else None
        top_ask = asks[0] if isinstance(asks, list) and asks else None
        bid = parse_float(top_bid.get("price") if isinstance(top_bid, dict) else None)
        bid_size = parse_float(top_bid.get("size") if isinstance(top_bid, dict) else None)
        ask = parse_float(top_ask.get("price") if isinstance(top_ask, dict) else None)
        ask_size = parse_float(top_ask.get("size") if isinstance(top_ask, dict) else None)
        return bid, bid_size, ask, ask_size

    def fetch_midpoint(self, token_id: str) -> Optional[float]:
        payload = self.http.get_json(
            f"{self.clob_base}/midpoint",
            params={"token_id": token_id},
            extra_headers=self.clob_headers,
        )
        if not isinstance(payload, dict):
            return None
        return parse_float(payload.get("mid"))

    def compute_vol_1m(self, state: MarketState, current_price: Optional[float], now: dt.datetime) -> Optional[float]:
        if current_price is None or current_price <= 0:
            return None
        state.price_window.append((now.timestamp(), current_price))
        cutoff = now.timestamp() - 60.0
        while state.price_window and state.price_window[0][0] < cutoff:
            state.price_window.popleft()
        if len(state.price_window) < 2:
            return None
        returns: List[float] = []
        previous = state.price_window[0][1]
        for _, px in list(state.price_window)[1:]:
            if previous <= 0 or px <= 0:
                previous = px
                continue
            returns.append(math.log(px / previous))
            previous = px
        if len(returns) < 2:
            return None
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
        return math.sqrt(variance)

    def write_tick(self, state: MarketState, tick: Dict[str, Any], now: dt.datetime) -> None:
        if self.dry_run:
            return
        date_folder = now.strftime("%Y-%m-%d")
        path = self.raw_dir / date_folder / f"{safe_slug_filename(state.slug)}.jsonl"
        append_jsonl(path, tick)

    def write_outcome(self, outcome: Dict[str, Any], now: dt.datetime) -> None:
        if self.dry_run:
            return
        path = self.outcomes_dir / f"{now.strftime('%Y-%m-%d')}.jsonl"
        append_jsonl(path, outcome)

    def resolve_outcome_label(self, market: Dict[str, Any]) -> Optional[str]:
        prices = [parse_float(item) for item in parse_json_list(market.get("outcomePrices"))]
        outcomes = [str(item) for item in parse_json_list(market.get("outcomes"))]
        if not prices or len(prices) != len(outcomes):
            return None
        max_index = max(range(len(prices)), key=lambda idx: prices[idx] if prices[idx] is not None else -1.0)
        winner_price = prices[max_index]
        if winner_price is None or winner_price < 0.99:
            return None
        winner = outcomes[max_index].strip().lower()
        if winner in ("up", "yes"):
            return "UP"
        if winner in ("down", "no"):
            return "DOWN"
        return winner.upper()

    def maybe_check_resolution(self, state: MarketState, now: dt.datetime) -> None:
        if state.resolved or state.slug in self.resolved_slugs:
            state.resolved = True
            state.active = False
            self.meta_store["markets"].setdefault(state.slug, {})["resolved"] = True
            self.meta_store["markets"][state.slug]["active"] = False
            return
        if state.last_resolution_check:
            if (now - state.last_resolution_check).total_seconds() < self.resolution_interval:
                return
        state.last_resolution_check = now
        market = self.fetch_market_by_slug(state.slug)
        if not market:
            return
        status = str(market.get("umaResolutionStatus") or "").strip().lower()
        if status != "resolved":
            return
        resolved = self.resolve_outcome_label(market)
        if not resolved:
            return
        resolved_ts = parse_iso_utc(market.get("closedTime")) or now
        outcome = {
            "slug": state.slug,
            "market_id": state.market_id,
            "condition_id": state.condition_id,
            "resolved": resolved,
            "resolved_ts": iso_utc(resolved_ts),
        }
        if state.slug not in self.resolved_slugs:
            self.write_outcome(outcome, now)
        self.resolved_slugs.add(state.slug)
        state.resolved = True
        state.active = False
        self.meta_store["markets"].setdefault(state.slug, {})["resolved"] = True
        self.meta_store["markets"][state.slug]["active"] = False
        self.meta_store["markets"][state.slug]["resolved_ts"] = outcome["resolved_ts"]
        self.logger.info("event=market_resolved slug=%s resolved=%s", state.slug, resolved)

    def should_poll_tick(self, state: MarketState, now: dt.datetime) -> bool:
        if state.resolved:
            return False
        if not state.active:
            return False
        if state.market_end_dt and now >= state.market_end_dt:
            return False
        return True

    def poll_once(self) -> None:
        loop_start = utc_now()
        if not self.last_discovery_at or (loop_start - self.last_discovery_at).total_seconds() >= self.discover_interval:
            self.discover_markets()
            self.save_metadata()

        price_value: Optional[float] = None
        price_source: Optional[str] = None
        price_ts: Optional[str] = None
        try:
            price_value, price_source, price_ts = self.fetch_btc_price()
        except Exception as exc:  # noqa: BLE001
            self.total_errors += 1
            self.logger.error("event=btc_price_error error=%s", exc)

        ticks_this_loop = 0
        for slug, state in list(self.markets.items()):
            try:
                now = utc_now()
                if self.should_poll_tick(state, now):
                    request_start = time.time()
                    bid, bid_size, ask, ask_size = self.fetch_book_top(state.yes_token_id)
                    midpoint = self.fetch_midpoint(state.yes_token_id)
                    latency_ms = int((time.time() - request_start) * 1000)
                    if midpoint is None:
                        midpoint = ((bid + ask) / 2.0) if bid is not None and ask is not None else None
                    spread = (ask - bid) if bid is not None and ask is not None else None
                    vol_1m = self.compute_vol_1m(state, price_value, now)
                    tick = {
                        "schema_version": 1,
                        "ts": iso_utc(now),
                        "slug": slug,
                        "market_id": state.market_id,
                        "condition_id": state.condition_id,
                        "yes_token_id": state.yes_token_id,
                        "market_end_ts": state.market_end_ts,
                        "bid": bid,
                        "bid_size": bid_size,
                        "ask": ask,
                        "ask_size": ask_size,
                        "mid": midpoint,
                        "spread": spread,
                        "btc_price": price_value,
                        "btc_price_source": price_source,
                        "btc_price_ts": price_ts,
                        "interval_open_price": state.interval_open_price,
                        "vol_1m": vol_1m,
                        "latency_ms": latency_ms,
                        "tick_seq": state.tick_seq,
                    }
                    self.write_tick(state, tick, now)
                    state.tick_seq += 1
                    self.meta_store["markets"].setdefault(slug, {})["tick_seq"] = state.tick_seq
                    ticks_this_loop += 1
                    self.total_ticks_written += 1

                if state.market_end_dt and now >= state.market_end_dt:
                    self.maybe_check_resolution(state, now)
                elif not state.active:
                    self.maybe_check_resolution(state, now)
            except Exception as exc:  # noqa: BLE001
                self.total_errors += 1
                self.logger.error("event=poll_market_error slug=%s error=%s", slug, exc)

        self.save_metadata()
        loop_elapsed = (utc_now() - loop_start).total_seconds()
        lag = max(0.0, loop_elapsed - self.poll_interval)
        self.worst_loop_lag_sec = max(self.worst_loop_lag_sec, lag)
        self.emit_heartbeat_if_due(ticks_this_loop)

    def emit_heartbeat_if_due(self, ticks_this_loop: int) -> None:
        now = utc_now()
        elapsed = (now - self.last_heartbeat_at).total_seconds()
        if elapsed < self.heartbeat_interval:
            return
        active_count = sum(1 for state in self.markets.values() if self.should_poll_tick(state, now))
        unresolved_after_end = sum(
            1
            for state in self.markets.values()
            if (not state.resolved and state.market_end_dt is not None and now >= state.market_end_dt)
        )
        self.logger.info(
            "event=heartbeat active_markets=%d unresolved_post_end=%d ticks_written_total=%d ticks_last_loop=%d errors_total=%d worst_lag_sec=%.3f",
            active_count,
            unresolved_after_end,
            self.total_ticks_written,
            ticks_this_loop,
            self.total_errors,
            self.worst_loop_lag_sec,
        )
        self.last_heartbeat_at = now
        self.worst_loop_lag_sec = 0.0

    def run(self) -> None:
        self.load_state()
        self.logger.info(
            "event=start dry_run=%s once=%s data_dir=%s interval_sec=%.2f discover_sec=%.1f resolution_sec=%.1f",
            self.dry_run,
            self.once,
            str(self.data_dir),
            self.poll_interval,
            self.discover_interval,
            self.resolution_interval,
        )
        while True:
            start_monotonic = time.monotonic()
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001
                self.total_errors += 1
                self.logger.error("event=loop_error error=%s", exc)
            if self.once:
                self.logger.info("event=exit_once ticks_written_total=%d", self.total_ticks_written)
                return
            elapsed = time.monotonic() - start_monotonic
            sleep_for = max(0.0, self.poll_interval - elapsed)
            time.sleep(sleep_for)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Polymarket BTC 15m data recorder")
    parser.add_argument("--data-dir", default=os.getenv("PM_DATA_DIR", "data"))
    parser.add_argument("--interval-sec", type=float, default=float(os.getenv("PM_INTERVAL_SEC", "2")))
    parser.add_argument(
        "--discover-interval-sec",
        type=float,
        default=float(os.getenv("PM_DISCOVER_INTERVAL_SEC", "60")),
    )
    parser.add_argument(
        "--resolution-interval-sec",
        type=float,
        default=float(os.getenv("PM_RESOLUTION_INTERVAL_SEC", "30")),
    )
    parser.add_argument(
        "--heartbeat-interval-sec",
        type=float,
        default=float(os.getenv("PM_HEARTBEAT_INTERVAL_SEC", "300")),
    )
    parser.add_argument(
        "--slug-prefix",
        default=os.getenv("PM_SLUG_PREFIX", "btc-updown-15m-"),
    )
    parser.add_argument(
        "--gamma-base",
        default=os.getenv("PM_GAMMA_BASE", "https://gamma-api.polymarket.com"),
    )
    parser.add_argument(
        "--clob-base",
        default=os.getenv("PM_CLOB_BASE", "https://clob.polymarket.com"),
    )
    parser.add_argument(
        "--price-source",
        default=os.getenv("PM_BTC_PRICE_SOURCE", "coinbase"),
        choices=["coinbase", "binance"],
    )
    parser.add_argument(
        "--discover-limit",
        type=int,
        default=int(os.getenv("PM_DISCOVER_LIMIT", "1000")),
    )
    parser.add_argument(
        "--discover-pages",
        type=int,
        default=int(os.getenv("PM_DISCOVER_PAGES", "8")),
    )
    parser.add_argument(
        "--probe-intervals-back",
        type=int,
        default=int(os.getenv("PM_PROBE_INTERVALS_BACK", "2")),
    )
    parser.add_argument(
        "--probe-intervals-forward",
        type=int,
        default=int(os.getenv("PM_PROBE_INTERVALS_FORWARD", "8")),
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=float,
        default=float(os.getenv("PM_REQUEST_TIMEOUT_SEC", "10")),
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.getenv("PM_MAX_RETRIES", "3")),
    )
    parser.add_argument(
        "--base-backoff-sec",
        type=float,
        default=float(os.getenv("PM_BASE_BACKOFF_SEC", "0.35")),
    )
    parser.add_argument("--gamma-auth-token", default=os.getenv("PM_GAMMA_AUTH_TOKEN", ""))
    parser.add_argument("--clob-auth-token", default=os.getenv("PM_CLOB_AUTH_TOKEN", ""))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log-level", default=os.getenv("PM_LOG_LEVEL", "INFO"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    recorder = Recorder(args)
    recorder.run()


if __name__ == "__main__":
    main()
