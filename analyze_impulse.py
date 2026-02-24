"""
analyze_impulse.py

OFFLINE analysis-only script to detect overshoot / mean reversion in Polymarket mid
after fast BTC price moves.

Per tick i (ticks ordered by ts):
  1) BTC impulse: return over last lookback ticks (default 10 ~= 20 seconds)
     btc_impulse = (btc[i] - btc[i-lookback]) / btc[i-lookback]
  2) Probability move over same window:
     prob_move = mid[i] - mid[i-lookback]
  3) Reversion over next horizon ticks (default 10 ~= 20 seconds):
     future_move = mid[i+horizon] - mid[i]

Condition on |btc_impulse| > threshold (default 0.001 = 0.1%):
  - avg future_move when impulse positive
  - avg future_move when impulse negative
  - fraction of reversals: future_move tends to oppose prob_move (future_move * prob_move < 0)

This script never places real trades.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def index_raw_files(data_dir: str) -> Dict[str, List[str]]:
    raw_glob = os.path.join(data_dir, "raw", "*", "*.jsonl")
    paths = glob.glob(raw_glob)
    out: Dict[str, List[str]] = {}
    for path in paths:
        slug = os.path.basename(path).replace(".jsonl", "")
        out.setdefault(slug, []).append(path)
    return out


def choose_raw_paths(paths_by_slug: Dict[str, List[str]], dedupe_by: str) -> Tuple[Dict[str, str], int]:
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
        chosen[slug] = best_path or paths[0]

    return chosen, dup_skipped


@dataclass
class ImpulseStats:
    events: int = 0
    pos_n: int = 0
    pos_future_sum: float = 0.0
    neg_n: int = 0
    neg_future_sum: float = 0.0
    reversal_n: int = 0
    reversal_hits: int = 0

    def add(self, impulse: np.ndarray, prob_move: np.ndarray, future_move: np.ndarray, threshold: float) -> None:
        mask = np.isfinite(impulse) & np.isfinite(prob_move) & np.isfinite(future_move)
        if not np.any(mask):
            return
        impulse = impulse[mask]
        prob_move = prob_move[mask]
        future_move = future_move[mask]

        event_mask = np.abs(impulse) > threshold
        if not np.any(event_mask):
            return

        impulse = impulse[event_mask]
        prob_move = prob_move[event_mask]
        future_move = future_move[event_mask]

        self.events += int(impulse.size)

        pos = impulse > 0
        if np.any(pos):
            self.pos_n += int(np.count_nonzero(pos))
            self.pos_future_sum += float(np.sum(future_move[pos]))

        neg = impulse < 0
        if np.any(neg):
            self.neg_n += int(np.count_nonzero(neg))
            self.neg_future_sum += float(np.sum(future_move[neg]))

        # Reversal: future move tends to oppose the move that happened over the prior window.
        nonzero = prob_move != 0
        if np.any(nonzero):
            pm = prob_move[nonzero]
            fm = future_move[nonzero]
            self.reversal_n += int(pm.size)
            self.reversal_hits += int(np.count_nonzero(pm * fm < 0))


def analyze_file(path: str, *, lookback: int, horizon: int, threshold: float, stats: ImpulseStats, verbose: bool) -> Tuple[int, int, int]:
    """
    Returns: (ticks_used, events_added, bad_lines)
    """
    ts_list: List[datetime] = []
    btc_list: List[float] = []
    mid_list: List[float] = []

    bad_lines = 0
    out_of_order = False
    prev_ts: Optional[datetime] = None

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
                ts = parse_ts(ts_raw)
            except Exception:
                continue

            btc = safe_float(row.get("btc_price"))
            mid = safe_float(row.get("mid"))
            if btc is None or mid is None:
                continue
            if btc <= 0:
                continue
            if not (0.0 <= mid <= 1.0):
                continue

            if prev_ts is not None and ts < prev_ts:
                out_of_order = True
            prev_ts = ts

            ts_list.append(ts)
            btc_list.append(btc)
            mid_list.append(mid)

    ticks_used = len(btc_list)
    if ticks_used < (lookback + horizon + 1):
        return ticks_used, 0, bad_lines

    btc_arr = np.asarray(btc_list, dtype=float)
    mid_arr = np.asarray(mid_list, dtype=float)

    if out_of_order:
        idx = np.argsort(np.asarray([t.timestamp() for t in ts_list], dtype=float))
        btc_arr = btc_arr[idx]
        mid_arr = mid_arr[idx]
        if verbose:
            print(f"[warn] out-of-order ticks detected; sorted by ts: {path}")

    denom = btc_arr[:-lookback]
    btc_impulse = (btc_arr[lookback:] - denom) / denom  # length n-lookback; impulse[0] corresponds to i=lookback
    prob_move = mid_arr[lookback:] - mid_arr[:-lookback]  # same length, for i=lookback..n-1
    future_move = mid_arr[horizon:] - mid_arr[:-horizon]  # length n-horizon; future_move[i] corresponds to i

    m = btc_arr.size - horizon - lookback
    if m <= 0:
        return ticks_used, 0, bad_lines

    impulse_aligned = btc_impulse[:m]
    prob_move_aligned = prob_move[:m]
    future_move_aligned = future_move[lookback : lookback + m]

    before_events = stats.events
    stats.add(impulse_aligned, prob_move_aligned, future_move_aligned, float(threshold))
    events_added = stats.events - before_events

    return ticks_used, int(events_added), bad_lines


def main() -> None:
    p = argparse.ArgumentParser(description="OFFLINE analysis-only overshoot test: BTC impulse vs mid mean reversion")
    p.add_argument("--data-dir", default="data", help='Data root (default: "data")')
    p.add_argument("--lookback", type=int, default=10, help="Lookback ticks (default: 10 ~= 20s)")
    p.add_argument("--horizon", type=int, default=10, help="Future horizon ticks (default: 10 ~= 20s)")
    p.add_argument("--threshold", type=float, default=0.001, help="Impulse threshold (default: 0.001 = 0.1%)")
    p.add_argument(
        "--dedupe-by",
        choices=["mtime", "first"],
        default="mtime",
        help='How to dedupe multiple raw files for the same slug (default: "mtime")',
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args()

    if args.lookback <= 0 or args.horizon <= 0:
        raise SystemExit("--lookback and --horizon must be positive integers")

    raw_index = index_raw_files(args.data_dir)
    chosen_raw, dup_skipped = choose_raw_paths(raw_index, args.dedupe_by)
    paths = sorted(chosen_raw.values())

    stats = ImpulseStats()
    files = 0
    total_ticks = 0
    bad_lines_total = 0
    events_added_total = 0

    for path in paths:
        files += 1
        ticks_used, events_added, bad_lines = analyze_file(
            path,
            lookback=int(args.lookback),
            horizon=int(args.horizon),
            threshold=float(args.threshold),
            stats=stats,
            verbose=bool(args.verbose),
        )
        total_ticks += ticks_used
        bad_lines_total += bad_lines
        events_added_total += events_added

    avg_future_pos = (stats.pos_future_sum / stats.pos_n) if stats.pos_n > 0 else float("nan")
    avg_future_neg = (stats.neg_future_sum / stats.neg_n) if stats.neg_n > 0 else float("nan")
    reversal_frac = (stats.reversal_hits / stats.reversal_n) if stats.reversal_n > 0 else float("nan")

    print("\nImpulse / Overshoot Summary")
    print(f"  files_processed            {files}")
    print(f"  dup_raw_files_skipped      {dup_skipped}")
    print(f"  bad_jsonl_lines_skipped    {bad_lines_total}")
    print(f"  total_ticks_used           {total_ticks}")
    print(f"  qualifying_events          {stats.events}")
    print(f"  params                      lookback={args.lookback} horizon={args.horizon} threshold={args.threshold}")
    print(f"  avg_future_move|impulse>0  {avg_future_pos:.8f} (n={stats.pos_n})" if np.isfinite(avg_future_pos) else f"  avg_future_move|impulse>0  nan (n={stats.pos_n})")
    print(f"  avg_future_move|impulse<0  {avg_future_neg:.8f} (n={stats.neg_n})" if np.isfinite(avg_future_neg) else f"  avg_future_move|impulse<0  nan (n={stats.neg_n})")
    print(
        f"  reversal_fraction          {reversal_frac:.4f} (hits={stats.reversal_hits} n={stats.reversal_n})"
        if np.isfinite(reversal_frac)
        else f"  reversal_fraction          nan (hits={stats.reversal_hits} n={stats.reversal_n})"
    )

    print("\nConclusion:")
    min_events = 500
    overshoot = (
        stats.events >= min_events
        and np.isfinite(reversal_frac)
        and reversal_frac > 0.52
        and np.isfinite(avg_future_pos)
        and np.isfinite(avg_future_neg)
        and avg_future_pos < 0
        and avg_future_neg > 0
    )
    if overshoot:
        print("  Probability overshoots and mean reverts")
    else:
        print("  No overshoot detected")


if __name__ == "__main__":
    main()

