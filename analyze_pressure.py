"""
analyze_pressure.py

OFFLINE analysis-only script to detect one-sided order pressure near market resolution.

For each tick within the last window_sec (default 120s) before resolved_ts:
  imbalance = (ask_size - bid_size) / (ask_size + bid_size)

Aggregates:
  - avg imbalance when mid > mid_high (default 0.9)
  - avg imbalance when mid < mid_low  (default 0.1)
  - fraction imbalance > 0.5
  - fraction imbalance < -0.5

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


@dataclass
class PressureStats:
    n: int = 0
    imb_sum: float = 0.0
    hi_n: int = 0
    hi_sum: float = 0.0
    lo_n: int = 0
    lo_sum: float = 0.0
    frac_pos_hits: int = 0
    frac_neg_hits: int = 0

    def add(self, imb: np.ndarray, mid: np.ndarray, *, mid_high: float, mid_low: float) -> None:
        mask = np.isfinite(imb) & np.isfinite(mid)
        if not np.any(mask):
            return
        imb = imb[mask]
        mid = mid[mask]

        self.n += int(imb.size)
        self.imb_sum += float(np.sum(imb))

        hi = mid > mid_high
        if np.any(hi):
            self.hi_n += int(np.count_nonzero(hi))
            self.hi_sum += float(np.sum(imb[hi]))

        lo = mid < mid_low
        if np.any(lo):
            self.lo_n += int(np.count_nonzero(lo))
            self.lo_sum += float(np.sum(imb[lo]))

        self.frac_pos_hits += int(np.count_nonzero(imb > 0.5))
        self.frac_neg_hits += int(np.count_nonzero(imb < -0.5))


def analyze_market_file(
    path: str,
    *,
    resolved_ts: datetime,
    window_sec: float,
    mid_high: float,
    mid_low: float,
    stats: PressureStats,
    verbose: bool,
) -> Tuple[int, int, int]:
    """
    Returns: (ticks_scanned, ticks_used, bad_lines)
    """
    ticks_scanned = 0
    bad_lines = 0
    imb_list: List[float] = []
    mid_list: List[float] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ticks_scanned += 1
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

            # Only consider ticks at-or-before resolution, within the final window.
            if tick_ts > resolved_ts:
                continue
            time_remaining_sec = (resolved_ts - tick_ts).total_seconds()
            if time_remaining_sec < 0 or time_remaining_sec > window_sec:
                continue

            bid_size = safe_float(row.get("bid_size"))
            ask_size = safe_float(row.get("ask_size"))
            mid = safe_float(row.get("mid"))
            if bid_size is None or ask_size is None or mid is None:
                continue
            denom = ask_size + bid_size
            if denom <= 0:
                continue

            imbalance = (ask_size - bid_size) / denom
            imb_list.append(float(imbalance))
            mid_list.append(float(mid))

    if not imb_list:
        return ticks_scanned, 0, bad_lines

    imb = np.asarray(imb_list, dtype=float)
    mid = np.asarray(mid_list, dtype=float)
    stats.add(imb, mid, mid_high=mid_high, mid_low=mid_low)
    if verbose:
        print(f"[ok] {os.path.basename(path)} used_ticks={imb.size}")
    return ticks_scanned, int(imb.size), bad_lines


def main() -> None:
    p = argparse.ArgumentParser(description="OFFLINE analysis-only: order book pressure near resolution")
    p.add_argument("--data-dir", default="data", help='Data root (default: "data")')
    p.add_argument("--window-sec", type=float, default=120.0, help="Seconds before resolution to analyze (default: 120)")
    p.add_argument("--mid-high", type=float, default=0.9, help="High mid threshold (default: 0.9)")
    p.add_argument("--mid-low", type=float, default=0.1, help="Low mid threshold (default: 0.1)")
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

    stats = PressureStats()
    files_total = 0
    files_matched = 0
    ticks_scanned_total = 0
    ticks_used_total = 0
    bad_lines_total = 0
    missing_raw = 0

    for slug, resolved_ts in outcomes.items():
        files_total += 1
        raw_path = chosen_raw.get(slug)
        if raw_path is None:
            missing_raw += 1
            continue
        files_matched += 1
        scanned, used, bad = analyze_market_file(
            raw_path,
            resolved_ts=resolved_ts,
            window_sec=float(args.window_sec),
            mid_high=float(args.mid_high),
            mid_low=float(args.mid_low),
            stats=stats,
            verbose=bool(args.verbose),
        )
        ticks_scanned_total += scanned
        ticks_used_total += used
        bad_lines_total += bad

    avg_imb = (stats.imb_sum / stats.n) if stats.n > 0 else float("nan")
    avg_hi = (stats.hi_sum / stats.hi_n) if stats.hi_n > 0 else float("nan")
    avg_lo = (stats.lo_sum / stats.lo_n) if stats.lo_n > 0 else float("nan")
    frac_pos = (stats.frac_pos_hits / stats.n) if stats.n > 0 else float("nan")
    frac_neg = (stats.frac_neg_hits / stats.n) if stats.n > 0 else float("nan")

    print("\nLate Pressure Summary")
    print(f"  resolved_markets           {len(outcomes)}")
    print(f"  matched_markets            {files_matched}")
    print(f"  missing_raw_for_outcome    {missing_raw}")
    print(f"  dup_raw_files_skipped      {dup_skipped}")
    print(f"  bad_jsonl_lines_skipped    {bad_lines_total}")
    print(f"  ticks_scanned_total        {ticks_scanned_total}")
    print(f"  ticks_used_total           {ticks_used_total}")
    print(f"  params                      window_sec={args.window_sec} mid_high={args.mid_high} mid_low={args.mid_low}")
    print(f"  avg_imbalance_all          {avg_imb:.6f}" if np.isfinite(avg_imb) else "  avg_imbalance_all          nan")
    print(
        f"  avg_imbalance|mid>high     {avg_hi:.6f} (n={stats.hi_n})" if np.isfinite(avg_hi) else f"  avg_imbalance|mid>high     nan (n={stats.hi_n})"
    )
    print(
        f"  avg_imbalance|mid<low      {avg_lo:.6f} (n={stats.lo_n})" if np.isfinite(avg_lo) else f"  avg_imbalance|mid<low      nan (n={stats.lo_n})"
    )
    print(f"  frac_imbalance>0.5         {frac_pos:.4f}" if np.isfinite(frac_pos) else "  frac_imbalance>0.5         nan")
    print(f"  frac_imbalance<-0.5        {frac_neg:.4f}" if np.isfinite(frac_neg) else "  frac_imbalance<-0.5        nan")

    print("\nConclusion:")
    # Imbalance > 0 implies larger ask_size than bid_size (sell-side dominates).
    # Imbalance < 0 implies larger bid_size than ask_size (buy-side dominates).
    dominance_eps = 0.05
    if np.isfinite(avg_imb) and avg_imb >= dominance_eps:
        print("  Late sellers dominate")
    elif np.isfinite(avg_imb) and avg_imb <= -dominance_eps:
        print("  Late buyers dominate")
    else:
        print("  Balanced")


if __name__ == "__main__":
    main()

