# analyze_calibration.py
import glob
import json
from datetime import datetime
import pandas as pd


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def load_outcomes(outcomes_glob: str = "data/outcomes/*.jsonl"):
    outcomes_map = {}
    resolved_ts_map = {}

    for path in glob.glob(outcomes_glob):
        with open(path, "r") as f:
            for line in f:
                row = json.loads(line)
                slug = row["slug"]
                outcomes_map[slug] = 1 if row["resolved"] == "UP" else 0
                resolved_ts_map[slug] = parse_ts(row["resolved_ts"])

    return outcomes_map, resolved_ts_map


def last_tick_at_or_before(path: str, cutoff_ts: datetime):
    last_valid = None
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            if parse_ts(row["ts"]) <= cutoff_ts:
                last_valid = row
    return last_valid


def main():
    outcomes, resolved_ts = load_outcomes()
    print(f"Loaded {len(outcomes)} resolved markets")

    records = []
    raw_paths = glob.glob("data/raw/*/*.jsonl")

    seen = set()
    dup_skipped = 0
    matched = 0

    for path in raw_paths:
        slug = path.split("/")[-1].replace(".jsonl", "")

        # Skip duplicate files for same slug
        if slug in seen:
            dup_skipped += 1
            continue
        seen.add(slug)

        if slug not in outcomes:
            continue

        cutoff = resolved_ts.get(slug)
        if cutoff is None:
            continue

        tick = last_tick_at_or_before(path, cutoff)
        if tick is None:
            continue

        matched += 1
        records.append({"slug": slug, "mid": float(tick["mid"]), "outcome": outcomes[slug]})

    df = pd.DataFrame(records)
    print(f"Matched {matched} unique markets with final midpoint (<= resolved_ts)")
    print(f"Skipped {dup_skipped} duplicate raw files (same slug)")

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    df["bin"] = pd.cut(df["mid"], bins=bins, include_lowest=True)

    calibration = df.groupby("bin").agg(
        predicted=("mid", "mean"),
        actual=("outcome", "mean"),
        count=("outcome", "size"),
    )

    print("\nCalibration Table:")
    print(calibration)

    calibration.reset_index().to_csv("calibration_table.csv", index=False)
    print("\nWrote: calibration_table.csv")


if __name__ == "__main__":
    main()
