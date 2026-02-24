"""
analyze_lag.py

OFFLINE analysis-only script to measure whether Polymarket mid price lags BTC price changes.

For each market JSONL (ticks ordered by ts):
  - Compute BTC return over last k ticks (default k=5 ~= 10 sec at 2s polling)
  - Compute mid change over next future_n ticks (default future_n=3 ~= 6 sec)
  - Correlate: btc_move(t-k -> t) vs mid_move(t -> t+future_n)

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
class RunningCorr:
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0

    def add_batch(self, x: np.ndarray, y: np.ndarray) -> None:
        if x.size == 0:
            return
        self.n += int(x.size)
        self.sum_x += float(np.sum(x))
        self.sum_y += float(np.sum(y))
        self.sum_xx += float(np.sum(x * x))
        self.sum_yy += float(np.sum(y * y))
        self.sum_xy += float(np.sum(x * y))

    def corr(self) -> float:
        if self.n < 2:
            return float("nan")
        n = float(self.n)
        num = n * self.sum_xy - self.sum_x * self.sum_y
        den_x = n * self.sum_xx - self.sum_x * self.sum_x
        den_y = n * self.sum_yy - self.sum_y * self.sum_y
        if den_x <= 0 or den_y <= 0:
            return float("nan")
        return float(num / np.sqrt(den_x * den_y))


def analyze_file(
    path: str,
    *,
    k: int,
    future_n: int,
    threshold: float,
    corr_stats: RunningCorr,
    pos_sum: List[float],
    pos_n: List[int],
    neg_sum: List[float],
    neg_n: List[int],
    verbose: bool,
) -> Tuple[int, int, int]:
    """
    Returns: (ticks_used, pairs_added, bad_lines)
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

            if prev_ts is not None and ts < prev_ts:
                out_of_order = True
            prev_ts = ts

            ts_list.append(ts)
            btc_list.append(btc)
            mid_list.append(mid)

    ticks_used = len(btc_list)
    if ticks_used < (k + future_n + 1):
        return ticks_used, 0, bad_lines

    btc_arr = np.asarray(btc_list, dtype=float)
    mid_arr = np.asarray(mid_list, dtype=float)

    if out_of_order:
        idx = np.argsort(np.asarray([t.timestamp() for t in ts_list], dtype=float))
        btc_arr = btc_arr[idx]
        mid_arr = mid_arr[idx]
        if verbose:
            print(f"[warn] out-of-order ticks detected; sorted by ts: {path}")

    # Compute btc_ret for i in [k, n-1] and mid_move_future for i in [0, n-future_n-1]
    denom = btc_arr[:-k]
    btc_ret = (btc_arr[k:] - denom) / denom  # length n-k; btc_ret[0] corresponds to i=k
    mid_move_future = mid_arr[future_n:] - mid_arr[:-future_n]  # length n-future_n; mid_move_future[i] corresponds to i

    # Align on i in [k, n-future_n-1]
    m = btc_arr.size - future_n - k
    if m <= 0:
        return ticks_used, 0, bad_lines

    x = btc_ret[:m]
    y = mid_move_future[k:]

    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return ticks_used, 0, bad_lines

    x = x[mask]
    y = y[mask]

    corr_stats.add_batch(x, y)

    pos_mask = x > threshold
    if np.any(pos_mask):
        pos_sum[0] += float(np.sum(y[pos_mask]))
        pos_n[0] += int(np.count_nonzero(pos_mask))

    neg_mask = x < -threshold
    if np.any(neg_mask):
        neg_sum[0] += float(np.sum(y[neg_mask]))
        neg_n[0] += int(np.count_nonzero(neg_mask))

    return ticks_used, int(x.size), bad_lines


def main() -> None:
    p = argparse.ArgumentParser(description="OFFLINE analysis-only lag test: BTC moves vs Polymarket mid response")
    p.add_argument("--data-dir", default="data", help='Data root (default: "data")')
    p.add_argument("--k", type=int, default=5, help="Lookback ticks for BTC return (default: 5)")
    p.add_argument("--future-n", type=int, default=3, help="Future ticks for mid move (default: 3)")
    p.add_argument("--threshold", type=float, default=0.0005, help="BTC return threshold (default: 0.0005)")
    p.add_argument(
        "--dedupe-by",
        choices=["mtime", "first"],
        default="mtime",
        help='How to dedupe multiple raw files for the same slug (default: "mtime")',
    )
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = p.parse_args()

    if args.k <= 0 or args.future_n <= 0:
        raise SystemExit("--k and --future-n must be positive integers")

    raw_index = index_raw_files(args.data_dir)
    chosen_raw, dup_skipped = choose_raw_paths(raw_index, args.dedupe_by)
    paths = sorted(chosen_raw.values())

    corr_stats = RunningCorr()
    pos_sum = [0.0]
    pos_n = [0]
    neg_sum = [0.0]
    neg_n = [0]

    files = 0
    total_ticks = 0
    total_pairs = 0
    bad_lines_total = 0

    for path in paths:
        files += 1
        ticks_used, pairs_added, bad_lines = analyze_file(
            path,
            k=int(args.k),
            future_n=int(args.future_n),
            threshold=float(args.threshold),
            corr_stats=corr_stats,
            pos_sum=pos_sum,
            pos_n=pos_n,
            neg_sum=neg_sum,
            neg_n=neg_n,
            verbose=bool(args.verbose),
        )
        total_ticks += ticks_used
        total_pairs += pairs_added
        bad_lines_total += bad_lines

    corr = corr_stats.corr()
    pos_mean = (pos_sum[0] / pos_n[0]) if pos_n[0] > 0 else float("nan")
    neg_mean = (neg_sum[0] / neg_n[0]) if neg_n[0] > 0 else float("nan")

    print("\nLag Analysis Summary")
    print(f"  files_processed            {files}")
    print(f"  dup_raw_files_skipped      {dup_skipped}")
    print(f"  bad_jsonl_lines_skipped    {bad_lines_total}")
    print(f"  total_ticks_used           {total_ticks}")
    print(f"  total_pairs                {total_pairs}")
    print(f"  params                      k={args.k} future_n={args.future_n} threshold={args.threshold}")
    print(f"  corr(btc_ret, mid_move)    {corr:.6f}" if np.isfinite(corr) else "  corr(btc_ret, mid_move)    nan")
    print(
        f"  avg_mid_move|btc_ret>thr   {pos_mean:.8f} (n={pos_n[0]})"
        if np.isfinite(pos_mean)
        else f"  avg_mid_move|btc_ret>thr   nan (n={pos_n[0]})"
    )
    print(
        f"  avg_mid_move|btc_ret<-thr  {neg_mean:.8f} (n={neg_n[0]})"
        if np.isfinite(neg_mean)
        else f"  avg_mid_move|btc_ret<-thr  nan (n={neg_n[0]})"
    )

    # Simple heuristic conclusion.
    min_side_n = 50
    reacts = (
        np.isfinite(corr)
        and corr > 0.02
        and pos_n[0] >= min_side_n
        and neg_n[0] >= min_side_n
        and np.isfinite(pos_mean)
        and np.isfinite(neg_mean)
        and pos_mean > 0
        and neg_mean < 0
    )
    print("\nConclusion:")
    if reacts:
        print("  Polymarket reacts after BTC")
    else:
        print("  No lag detected")


if __name__ == "__main__":
    main()

