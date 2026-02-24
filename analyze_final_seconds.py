"""
analyze_final_seconds.py

OFFLINE analysis-only script to measure mid-price behavior in the final N seconds
before market resolution (default 30s).

For each resolved market:
  - Find last tick at-or-before resolved_ts => final_mid
  - Find tick at-or-before (resolved_ts - window_sec) => mid_before
  - delta_to_close = final_mid - mid_before

Groups:
  - high group: mid_before > 0.9
  - low group:  mid_before < 0.1

We check whether extreme probabilities move toward 0.5:
  - high group moves toward center if delta_to_close < 0
  - low group moves toward center if delta_to_close > 0

Prints averages and fraction moving toward center, plus a simple conclusion.

This script never places real trades.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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


def load_outcomes(data_dir: str) -> Dict[str, datetime]:
    outcomes_dir = os.path.join(data_dir, "outcomes")
    paths = sorted(glob.glob(os.path.join(outcomes_dir, "*.jsonl")))
    out: Dict[str, datetime] = {}
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                slug = row.get("slug")
                resolved_ts = row.get("resolved_ts")
                if not slug or not resolved_ts:
                    continue
                try:
                    out[slug] = parse_ts(resolved_ts)
                except Exception:
                    continue
    return out


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


def last_tick_mid_at_or_before(path: str, cutoff: datetime) -> Optional[Tuple[datetime, float]]:
    last_ts: Optional[datetime] = None
    last_mid: Optional[float] = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ts_raw = row.get("ts")
            if not isinstance(ts_raw, str):
                continue
            try:
                ts = parse_ts(ts_raw)
            except Exception:
                continue
            if ts > cutoff:
                continue
            mid = safe_float(row.get("mid"))
            if mid is None or not (0.0 <= mid <= 1.0):
                continue
            if last_ts is None or ts >= last_ts:
                last_ts = ts
                last_mid = float(mid)
    if last_ts is None or last_mid is None:
        return None
    return last_ts, last_mid


@dataclass
class FinalSecondsStats:
    high_n: int = 0
    high_sum: float = 0.0
    high_toward_center: int = 0
    low_n: int = 0
    low_sum: float = 0.0
    low_toward_center: int = 0


def main() -> None:
    p = argparse.ArgumentParser(description="OFFLINE analysis-only: mid behavior in final seconds before resolution")
    p.add_argument("--data-dir", default="data", help='Data root (default: "data")')
    p.add_argument("--window-sec", type=float, default=30.0, help="Seconds before resolution (default: 30)")
    p.add_argument("--mid-high", type=float, default=0.9, help="High group threshold on mid_before (default: 0.9)")
    p.add_argument("--mid-low", type=float, default=0.1, help="Low group threshold on mid_before (default: 0.1)")
    p.add_argument(
        "--dedupe-by",
        choices=["mtime", "first"],
        default="mtime",
        help='How to dedupe multiple raw files for the same slug (default: "mtime")',
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args()

    outcomes = load_outcomes(args.data_dir)
    raw_index = index_raw_files(args.data_dir)
    chosen_raw, dup_skipped = choose_raw_paths(raw_index, args.dedupe_by)

    stats = FinalSecondsStats()
    matched = 0
    missing_raw = 0
    missing_ticks = 0

    for slug, resolved_ts in outcomes.items():
        raw_path = chosen_raw.get(slug)
        if raw_path is None:
            missing_raw += 1
            continue

        final_tick = last_tick_mid_at_or_before(raw_path, resolved_ts)
        before_cutoff = resolved_ts - timedelta(seconds=float(args.window_sec))
        before_tick = last_tick_mid_at_or_before(raw_path, before_cutoff)

        if final_tick is None or before_tick is None:
            missing_ticks += 1
            continue

        matched += 1
        final_mid = final_tick[1]
        mid_before = before_tick[1]
        delta = final_mid - mid_before

        if mid_before > float(args.mid_high):
            stats.high_n += 1
            stats.high_sum += float(delta)
            if delta < 0:
                stats.high_toward_center += 1
        elif mid_before < float(args.mid_low):
            stats.low_n += 1
            stats.low_sum += float(delta)
            if delta > 0:
                stats.low_toward_center += 1

        if args.verbose and (mid_before > float(args.mid_high) or mid_before < float(args.mid_low)):
            print(f"[tick] slug={slug} mid_before={mid_before:.4f} final_mid={final_mid:.4f} delta={delta:+.5f}")

    high_avg = (stats.high_sum / stats.high_n) if stats.high_n > 0 else float("nan")
    low_avg = (stats.low_sum / stats.low_n) if stats.low_n > 0 else float("nan")
    high_frac_center = (stats.high_toward_center / stats.high_n) if stats.high_n > 0 else float("nan")
    low_frac_center = (stats.low_toward_center / stats.low_n) if stats.low_n > 0 else float("nan")

    # Combined "fraction moving toward center" across both groups.
    total_extreme_n = stats.high_n + stats.low_n
    total_center_hits = stats.high_toward_center + stats.low_toward_center
    total_center_frac = (total_center_hits / total_extreme_n) if total_extreme_n > 0 else float("nan")

    print("\nFinal Seconds Summary")
    print(f"  resolved_markets           {len(outcomes)}")
    print(f"  matched_markets            {matched}")
    print(f"  missing_raw_for_outcome    {missing_raw}")
    print(f"  missing_ticks              {missing_ticks}")
    print(f"  dup_raw_files_skipped      {dup_skipped}")
    print(f"  params                      window_sec={args.window_sec} mid_high={args.mid_high} mid_low={args.mid_low}")
    print(f"  avg_delta_high             {high_avg:.6f} (n={stats.high_n})" if np.isfinite(high_avg) else f"  avg_delta_high             nan (n={stats.high_n})")
    print(f"  avg_delta_low              {low_avg:.6f} (n={stats.low_n})" if np.isfinite(low_avg) else f"  avg_delta_low              nan (n={stats.low_n})")
    print(
        f"  frac_toward_center_high    {high_frac_center:.4f} (hits={stats.high_toward_center} n={stats.high_n})"
        if np.isfinite(high_frac_center)
        else f"  frac_toward_center_high    nan (hits={stats.high_toward_center} n={stats.high_n})"
    )
    print(
        f"  frac_toward_center_low     {low_frac_center:.4f} (hits={stats.low_toward_center} n={stats.low_n})"
        if np.isfinite(low_frac_center)
        else f"  frac_toward_center_low     nan (hits={stats.low_toward_center} n={stats.low_n})"
    )
    print(
        f"  frac_toward_center_total   {total_center_frac:.4f} (hits={total_center_hits} n={total_extreme_n})"
        if np.isfinite(total_center_frac)
        else f"  frac_toward_center_total   nan (hits={total_center_hits} n={total_extreme_n})"
    )

    print("\nConclusion:")
    # Heuristic: if extremes tend to move toward center and average deltas show pullback, call it a late premium.
    min_extreme = 200
    late_premium = (
        total_extreme_n >= min_extreme
        and np.isfinite(total_center_frac)
        and total_center_frac > 0.55
        and (np.isnan(high_avg) or high_avg < 0)
        and (np.isnan(low_avg) or low_avg > 0)
    )
    if late_premium:
        print("  Late certainty premium exists")
    else:
        print("  No late premium")


if __name__ == "__main__":
    main()

