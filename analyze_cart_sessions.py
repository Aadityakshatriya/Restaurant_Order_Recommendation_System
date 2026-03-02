from __future__ import annotations

from collections import Counter
import argparse
import math
from pathlib import Path

import pyarrow.parquet as pq


def _init_stats():
    return {"n": 0, "sum": 0.0, "sumsq": 0.0, "min": math.inf, "max": -math.inf}


def _update_stats(st, v):
    v = float(v)
    st["n"] += 1
    st["sum"] += v
    st["sumsq"] += v * v
    st["min"] = min(st["min"], v)
    st["max"] = max(st["max"], v)


def _summarize(st):
    if st["n"] == 0:
        return None
    mean = st["sum"] / st["n"]
    var = max(0.0, st["sumsq"] / st["n"] - mean * mean)
    return {"n": st["n"], "mean": mean, "std": math.sqrt(var), "min": st["min"], "max": st["max"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to cart_sessions.parquet")
    ap.add_argument("--batch-size", type=int, default=200_000)
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        raise SystemExit(f"Missing file: {p}")

    cols = [
        "session_id",
        "step",
        "split",
        "restaurant_id",
        "city",
        "hour",
        "meal_slot",
        "added",
        "user_id",
        "rest_price_tier",
        "rest_cuisine",
        "user_segment",
        "is_cold_start",
        "is_hot_weather",
        "weather_temp_c",
        "candidate_item_id",
        "candidate_price",
        "cart_total",
        "cart_size",
        "user_avg_order_value",
        "rest_rating",
        "rest_avg_margin",
        "revenue_weighted_label",
        "aov_lift_if_added",
    ]

    pf = pq.ParquetFile(str(p))

    row_count = 0
    pos_count = 0

    unique_sessions = set()
    unique_pos_sessions = set()
    unique_users = set()
    unique_restaurants = set()
    unique_candidate_items = set()

    # Grouping: a decision point is (session_id, step). Expect 1 positive row per group.
    group_size = Counter()
    group_pos = Counter()

    vc_pos = {k: Counter() for k in ["split", "city", "hour", "meal_slot", "rest_price_tier", "rest_cuisine", "user_segment"]}
    vc_pos.update({k: Counter() for k in ["is_cold_start", "is_hot_weather"]})

    num_cols = [
        "candidate_price",
        "cart_total",
        "cart_size",
        "user_avg_order_value",
        "rest_rating",
        "rest_avg_margin",
        "revenue_weighted_label",
        "aov_lift_if_added",
        "weather_temp_c",
    ]
    stats_pos = {c: _init_stats() for c in num_cols}

    night_hours = {22, 23, 0, 1, 2, 3, 4, 5}
    pos_night = 0

    for batch in pf.iter_batches(batch_size=args.batch_size, columns=cols):
        n = batch.num_rows
        row_count += n

        sids = batch.column(batch.schema.get_field_index("session_id")).to_pylist()
        steps = batch.column(batch.schema.get_field_index("step")).to_numpy(zero_copy_only=False)
        unique_sessions.update(sids)

        added = batch.column(batch.schema.get_field_index("added")).to_numpy(zero_copy_only=False)
        pos_mask = added == 1
        n_pos = int(pos_mask.sum())
        pos_count += n_pos

        # Uniques + grouping (all rows)
        uids = batch.column(batch.schema.get_field_index("user_id")).to_pylist()
        unique_users.update(uids)

        rest_ids = batch.column(batch.schema.get_field_index("restaurant_id")).to_pylist()
        unique_restaurants.update(rest_ids)

        cand_ids = batch.column(batch.schema.get_field_index("candidate_item_id")).to_pylist()
        unique_candidate_items.update(cand_ids)

        for sid, st in zip(sids, steps):
            key = f"{sid}|{int(st)}"
            group_size[key] += 1

        for sid, st, ispos in zip(sids, steps, pos_mask):
            if ispos:
                key = f"{sid}|{int(st)}"
                group_pos[key] += 1

        if not n_pos:
            continue

        hours = batch.column(batch.schema.get_field_index("hour")).to_numpy(zero_copy_only=False)
        for sid, h, ispos in zip(sids, hours, pos_mask):
            if not ispos:
                continue
            unique_pos_sessions.add(sid)
            if int(h) in night_hours:
                pos_night += 1

        # categorical counts
        for k in vc_pos:
            arr = batch.column(batch.schema.get_field_index(k)).to_pylist()
            if k == "hour":
                for v, ispos in zip(arr, pos_mask):
                    if ispos:
                        vc_pos[k][int(v)] += 1
            else:
                for v, ispos in zip(arr, pos_mask):
                    if ispos:
                        vc_pos[k][v] += 1

        # numeric stats (positives only)
        for c in num_cols:
            arr = batch.column(batch.schema.get_field_index(c)).to_numpy(zero_copy_only=False)
            st = stats_pos[c]
            for v, ispos in zip(arr, pos_mask):
                if not ispos:
                    continue
                if v is None:
                    continue
                if isinstance(v, float) and math.isnan(v):
                    continue
                _update_stats(st, v)

    print("rows", row_count)
    print("positives(added=1)", pos_count, "pos_rate", (pos_count / row_count if row_count else None))
    print("unique_sessions", len(unique_sessions))
    print("unique_pos_sessions", len(unique_pos_sessions))
    print("unique_users", len(unique_users))
    print("unique_restaurants", len(unique_restaurants))
    print("unique_candidate_items", len(unique_candidate_items))
    print("avg_candidates_per_decision", (row_count / pos_count if pos_count else None))
    print("night_positives(hour in 22-5)", pos_night, "night_pos_share", (pos_night / pos_count if pos_count else None))

    # Group diagnostics
    if group_size:
        sizes = list(group_size.values())
        print("\ndecisions(n_unique_session_step)", len(sizes))
        print("candidates_per_decision_min", min(sizes), "max", max(sizes), "mean", sum(sizes) / len(sizes))
        bad_pos = 0
        missing_pos = 0
        multi_pos = 0
        for k in group_size.keys():
            v = group_pos.get(k, 0)
            if v == 0:
                missing_pos += 1
                bad_pos += 1
            elif v > 1:
                multi_pos += 1
                bad_pos += 1
        print("decisions_with_pos_count_not_1", bad_pos, "(missing_pos", missing_pos, ")")
        if multi_pos:
            print("decisions_with_multiple_positives", multi_pos)

    print("\npositives_by_split", dict(vc_pos["split"]))
    print("positives_by_meal_slot", dict(vc_pos["meal_slot"]))
    print("positives_by_hour_top12", vc_pos["hour"].most_common(12))
    print("positives_by_city_top10", vc_pos["city"].most_common(10))
    print("positives_by_rest_price_tier", dict(vc_pos["rest_price_tier"]))
    print("positives_by_rest_cuisine_top10", vc_pos["rest_cuisine"].most_common(10))
    print("positives_by_user_segment", dict(vc_pos["user_segment"]))
    print("positives_by_cold_start", dict(vc_pos["is_cold_start"]))
    print("positives_by_is_hot_weather", dict(vc_pos["is_hot_weather"]))

    print("\npositive_numeric_stats")
    for c in num_cols:
        print(c, _summarize(stats_pos[c]))


if __name__ == "__main__":
    main()

