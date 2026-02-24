"""
analyze_simulation.py

OFFLINE analysis-only backtest simulator for Polymarket BTC 15m Up/Down datasets
captured by polymarket_data_capture.py.

This script never places real trades. It reads recorded JSONL tick data plus
recorded resolution outcomes, simulates a simple conservative strategy, and
writes trade-level and summary outputs under ./results/.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def parse_ts(ts: str) -> datetime:
    # Example: "2026-02-18T09:00:33.287123Z"
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def slug_to_interval_start_utc(slug: str) -> Optional[datetime]:
    # For btc-updown-15m-1771106400, the last token is a unix timestamp (UTC).
    try:
        ts_part = int(slug.rsplit("-", 1)[-1])
    except Exception:
        return None
    try:
        return datetime.fromtimestamp(ts_part, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def derive_market_end_ts(slug: str, row: Dict[str, Any]) -> Optional[datetime]:
    ts = row.get("market_end_ts")
    if isinstance(ts, str) and ts:
        try:
            return parse_ts(ts)
        except Exception:
            pass
    start = slug_to_interval_start_utc(slug)
    if start is None:
        return None
    return start + timedelta(minutes=15)


def load_outcomes(data_dir: str) -> Dict[str, Dict[str, Any]]:
    outcomes_dir = os.path.join(data_dir, "outcomes")
    paths = sorted(glob.glob(os.path.join(outcomes_dir, "*.jsonl")))

    out: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                slug = row.get("slug")
                resolved = row.get("resolved")
                resolved_ts = row.get("resolved_ts")

                if not slug or not resolved or not resolved_ts:
                    continue
                if resolved not in ("UP", "DOWN"):
                    continue
                try:
                    rts = parse_ts(resolved_ts)
                except Exception:
                    continue
                out[slug] = {
                    "slug": slug,
                    "market_id": row.get("market_id"),
                    "condition_id": row.get("condition_id"),
                    "resolved": resolved,
                    "resolved_ts": rts,
                }
    return out


def index_raw_files(data_dir: str) -> Dict[str, List[str]]:
    raw_glob = os.path.join(data_dir, "raw", "*", "*.jsonl")
    paths = glob.glob(raw_glob)
    out: Dict[str, List[str]] = {}
    for path in paths:
        slug = os.path.basename(path).replace(".jsonl", "")
        out.setdefault(slug, []).append(path)
    return out


def choose_raw_paths(
    paths_by_slug: Dict[str, List[str]],
    dedupe_by: str,
) -> Tuple[Dict[str, str], int]:
    chosen: Dict[str, str] = {}
    dup_skipped = 0

    for slug, paths in paths_by_slug.items():
        if not paths:
            continue
        if len(paths) > 1:
            dup_skipped += len(paths) - 1

        if dedupe_by == "first":
            chosen[slug] = paths[0]
            continue

        # default: mtime
        best_path = None
        best_mtime = None
        for p in paths:
            try:
                mtime = os.path.getmtime(p)
            except OSError:
                continue
            if best_mtime is None or mtime > best_mtime:
                best_mtime = mtime
                best_path = p
        if best_path is None:
            chosen[slug] = paths[0]
        else:
            chosen[slug] = best_path

    return chosen, dup_skipped


@dataclass(frozen=True)
class EntryCandidate:
    entry_ts: datetime
    side: str  # "SHORT_YES" or "LONG_YES"
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    spread: Optional[float]
    ttr_sec: Optional[float]


def scan_market_file_for_signals(
    path: str,
    slug: str,
    resolved_ts: datetime,
    high_conf: float,
    low_conf: float,
    early_time_sec: float,
    spread_max: float,
) -> Tuple[Optional[EntryCandidate], Optional[Dict[str, Any]], int]:
    """
    Returns:
      - earliest entry candidate by tick timestamp (<= resolved_ts), or None
      - last tick dict at-or-before resolved_ts, or None
      - number of malformed lines skipped
    """
    entry: Optional[EntryCandidate] = None
    last_tick: Optional[Dict[str, Any]] = None
    last_tick_ts: Optional[datetime] = None
    bad_lines = 0

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue

            ts_raw = row.get("ts")
            if not isinstance(ts_raw, str):
                continue
            try:
                tick_ts = parse_ts(ts_raw)
            except Exception:
                continue

            if tick_ts <= resolved_ts:
                if last_tick_ts is None or tick_ts >= last_tick_ts:
                    last_tick_ts = tick_ts
                    last_tick = row

            market_end_ts = derive_market_end_ts(slug, row)
            if market_end_ts is None:
                continue

            ttr_sec = (market_end_ts - tick_ts).total_seconds()
            if ttr_sec < 0:
                ttr_sec = 0.0

            bid = safe_float(row.get("bid"))
            ask = safe_float(row.get("ask"))
            mid = safe_float(row.get("mid"))
            spread = safe_float(row.get("spread"))

            if mid is None and bid is not None and ask is not None:
                mid = 0.5 * (bid + ask)
            if spread is None and bid is not None and ask is not None:
                spread = ask - bid

            if mid is None or spread is None:
                continue
            if spread < 0:
                continue

            if spread > spread_max:
                continue

            # Early-confidence fade: only act early in the interval.
            if ttr_sec <= early_time_sec:
                continue
            if tick_ts > resolved_ts:
                continue

            side: Optional[str] = None
            if mid >= high_conf:
                side = "SHORT_YES"
            elif mid <= low_conf:
                side = "LONG_YES"
            else:
                continue

            # Only consider signals we can trade under the conservative fill model:
            # - SHORT_YES uses bid
            # - LONG_YES uses ask
            fill_price: Optional[float]
            if side == "SHORT_YES":
                fill_price = bid
            else:
                fill_price = ask
            if fill_price is None:
                continue
            if not (0.0 <= float(fill_price) <= 1.0):
                continue

            candidate = EntryCandidate(entry_ts=tick_ts, side=side, bid=bid, ask=ask, mid=mid, spread=spread, ttr_sec=ttr_sec)

            if entry is None or candidate.entry_ts < entry.entry_ts:
                entry = candidate

    return entry, last_tick, bad_lines


def pnl_position_yes(
    *,
    side: str,
    outcome: str,
    stake: float,
    entry_price_yes: float,
    spread: float,
    fee_rate: float,
    half_spread_slippage_mult: float,
) -> Tuple[float, float]:
    """
    Position in YES-space with stake-sized notional payout.

    SHORT_YES:
      - If DOWN: profit = stake * entry_price_yes
      - If UP: loss    = -stake * (1 - entry_price_yes)

    LONG_YES:
      - If UP: profit  = stake * (1 - entry_price_yes)
      - If DOWN: loss  = -stake * entry_price_yes

    Fees: applied to positive PnL only (conservative).
    Slippage: subtract half-spread cost (configurable multiplier).
    """
    if side == "SHORT_YES":
        if outcome == "DOWN":
            pnl_gross = stake * entry_price_yes
        elif outcome == "UP":
            pnl_gross = -stake * (1.0 - entry_price_yes)
        else:
            raise ValueError(f"Unexpected outcome={outcome!r}")
    elif side == "LONG_YES":
        if outcome == "UP":
            pnl_gross = stake * (1.0 - entry_price_yes)
        elif outcome == "DOWN":
            pnl_gross = -stake * entry_price_yes
        else:
            raise ValueError(f"Unexpected outcome={outcome!r}")
    else:
        raise ValueError(f"Unexpected side={side!r}")

    pnl_gross -= (spread * stake * 0.5 * half_spread_slippage_mult)

    if pnl_gross > 0:
        pnl_net = pnl_gross * (1.0 - fee_rate)
    else:
        pnl_net = pnl_gross
    return pnl_gross, pnl_net


def compute_max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    return float(np.max(drawdowns))


def print_summary_table(summary: Dict[str, Any]) -> None:
    rows = [
        ("strategy", summary.get("strategy")),
        ("resolved_markets", summary.get("resolved_markets")),
        ("matched_markets", summary.get("matched_markets")),
        ("candidate_entries", summary.get("candidate_entries")),
        ("executed_trades", summary.get("executed_trades")),
        ("win_rate", summary.get("win_rate")),
        ("avg_pnl_per_trade", summary.get("avg_pnl_per_trade")),
        ("total_pnl", summary.get("total_pnl")),
        ("max_drawdown", summary.get("max_drawdown")),
    ]
    if "sharpe_like" in summary:
        rows.append(("sharpe_like", summary.get("sharpe_like")))

    width = max(len(k) for k, _ in rows)
    print("\nSummary:")
    for k, v in rows:
        if isinstance(v, float):
            if k in ("win_rate", "sharpe_like"):
                s = f"{v:.4f}"
            else:
                s = f"{v:.4f}"
        else:
            s = str(v)
        print(f"  {k:<{width}}  {s}")


def main() -> None:
    p = argparse.ArgumentParser(description="OFFLINE analysis-only Polymarket backtest simulator")
    p.add_argument("--data-dir", default="data", help='Data root (default: "data")')
    p.add_argument("--stake", type=float, default=10.0, help="Fixed $ stake per trade (default: 10)")
    p.add_argument("--fee-rate", type=float, default=0.02, help="Fee rate on winnings only (default: 0.02)")
    p.add_argument(
        "--high-conf",
        type=float,
        default=0.65,
        help="HIGH confidence threshold for SHORT_YES signal (default: 0.65)",
    )
    p.add_argument("--low-conf", type=float, default=0.35, help="LOW confidence threshold for LONG_YES signal (default: 0.35)")
    p.add_argument(
        "--early-time-sec",
        type=float,
        default=480.0,
        help="Require time remaining > this many seconds to consider entries (default: 480)",
    )
    p.add_argument("--spread-max", type=float, default=0.10, help="Max spread allowed for entry (default: 0.10)")
    p.add_argument(
        "--dedupe-by",
        choices=["mtime", "first"],
        default="mtime",
        help='How to dedupe multiple raw files for the same slug (default: "mtime")',
    )
    p.add_argument(
        "--half-spread-slippage-mult",
        type=float,
        default=1.0,
        help="Multiplier for half-spread slippage deduction (default: 1.0)",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args()

    strategy = "early_confidence_fade"
    verbose = bool(args.verbose)

    outcomes = load_outcomes(args.data_dir)
    resolved_slugs = list(outcomes.keys())
    if verbose:
        print(f"Loaded {len(resolved_slugs)} resolved markets from {os.path.join(args.data_dir, 'outcomes')}")

    raw_index = index_raw_files(args.data_dir)
    chosen_raw, dup_skipped = choose_raw_paths(raw_index, args.dedupe_by)
    if verbose:
        print(f"Indexed {sum(len(v) for v in raw_index.values())} raw files under {os.path.join(args.data_dir, 'raw')}")
        print(f"Skipped {dup_skipped} duplicate raw files (same slug) via dedupe-by={args.dedupe_by}")

    os.makedirs("results", exist_ok=True)

    trades: List[Dict[str, Any]] = []
    resolved_markets = len(outcomes)
    matched_markets = 0
    candidate_entries = 0
    executed_trades = 0
    bad_lines_total = 0
    missing_raw = 0

    for slug, outcome_row in outcomes.items():
        raw_path = chosen_raw.get(slug)
        if raw_path is None:
            missing_raw += 1
            continue

        resolved_ts = outcome_row["resolved_ts"]
        entry, last_tick, bad_lines = scan_market_file_for_signals(
            raw_path,
            slug=slug,
            resolved_ts=resolved_ts,
            high_conf=args.high_conf,
            low_conf=args.low_conf,
            early_time_sec=args.early_time_sec,
            spread_max=args.spread_max,
        )
        bad_lines_total += bad_lines

        if last_tick is not None:
            matched_markets += 1

        if entry is None:
            continue

        candidate_entries += 1

        spread = entry.spread
        if spread is None:
            if verbose:
                print(f"[skip] slug={slug} candidate entry missing spread")
            continue

        entry_price_yes: Optional[float]
        if entry.side == "SHORT_YES":
            entry_price_yes = entry.bid
        else:
            entry_price_yes = entry.ask

        if entry_price_yes is None:
            if verbose:
                print(f"[skip] slug={slug} candidate entry missing price for side={entry.side}")
            continue
        if not (0.0 <= float(entry_price_yes) <= 1.0):
            if verbose:
                print(f"[skip] slug={slug} candidate entry price out of range: {entry_price_yes}")
            continue

        pnl_gross, pnl_net = pnl_position_yes(
            side=entry.side,
            outcome=outcome_row["resolved"],
            stake=float(args.stake),
            entry_price_yes=float(entry_price_yes),
            spread=float(spread),
            fee_rate=float(args.fee_rate),
            half_spread_slippage_mult=float(args.half_spread_slippage_mult),
        )

        executed_trades += 1
        trades.append(
            {
                "slug": slug,
                "resolved_ts": iso_z(resolved_ts),
                "entry_ts": iso_z(entry.entry_ts),
                "side": entry.side,
                "entry_price_yes_bid": float(entry.bid) if entry.bid is not None else np.nan,
                "entry_price_yes_ask": float(entry.ask) if entry.ask is not None else np.nan,
                "spread": float(spread),
                "ttr_sec": float(entry.ttr_sec) if entry.ttr_sec is not None else np.nan,
                "outcome": outcome_row["resolved"],
                "pnl_gross": float(pnl_gross),
                "pnl_net": float(pnl_net),
                "fee_rate": float(args.fee_rate),
                "stake": float(args.stake),
            }
        )

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df["resolved_ts_dt"] = pd.to_datetime(trades_df["resolved_ts"], utc=True)
        trades_df = trades_df.sort_values("resolved_ts_dt").drop(columns=["resolved_ts_dt"])

    trades_csv_path = os.path.join("results", "trades.csv")
    # Use pandas for convenience, but enforce stable column order.
    trade_cols = [
        "slug",
        "resolved_ts",
        "entry_ts",
        "side",
        "entry_price_yes_bid",
        "entry_price_yes_ask",
        "spread",
        "ttr_sec",
        "outcome",
        "pnl_gross",
        "pnl_net",
        "fee_rate",
        "stake",
    ]
    if trades_df.empty:
        with open(trades_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=trade_cols)
            w.writeheader()
    else:
        trades_df.to_csv(trades_csv_path, index=False, columns=trade_cols)

    if executed_trades > 0:
        pnl_net_arr = trades_df["pnl_net"].to_numpy(dtype=float)
        equity = np.cumsum(pnl_net_arr)
        max_dd = compute_max_drawdown(equity)
        win_rate = float(np.mean(pnl_net_arr > 0))
        avg_pnl = float(np.mean(pnl_net_arr))
        total_pnl = float(np.sum(pnl_net_arr))

        # Sharpe-like metric on per-trade returns (net / stake)
        returns = pnl_net_arr / float(args.stake)
        sharpe_like: Optional[float]
        if returns.size >= 2 and float(np.std(returns, ddof=1)) > 0:
            sharpe_like = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(returns.size))
        else:
            sharpe_like = None
    else:
        max_dd = 0.0
        win_rate = 0.0
        avg_pnl = 0.0
        total_pnl = 0.0
        sharpe_like = None

    summary: Dict[str, Any] = {
        "strategy": strategy,
        "resolved_markets": int(resolved_markets),
        "matched_markets": int(matched_markets),
        "candidate_entries": int(candidate_entries),
        "executed_trades": int(executed_trades),
        "win_rate": float(win_rate),
        "avg_pnl_per_trade": float(avg_pnl),
        "total_pnl": float(total_pnl),
        "max_drawdown": float(max_dd),
        "fee_rate": float(args.fee_rate),
        "stake": float(args.stake),
        "high_conf": float(args.high_conf),
        "low_conf": float(args.low_conf),
        "early_time_sec": float(args.early_time_sec),
        "spread_max": float(args.spread_max),
        "dedupe_by": str(args.dedupe_by),
        "half_spread_slippage_mult": float(args.half_spread_slippage_mult),
        "dup_raw_files_skipped": int(dup_skipped),
        "missing_raw_for_outcome": int(missing_raw),
        "bad_jsonl_lines_skipped": int(bad_lines_total),
    }
    if sharpe_like is not None:
        summary["sharpe_like"] = float(sharpe_like)

    summary_path = os.path.join("results", "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print_summary_table(summary)
    print(f"\nWrote: {trades_csv_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
