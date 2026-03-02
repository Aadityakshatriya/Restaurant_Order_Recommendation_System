from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pandas as pd


def _build_payload(parquet_path: Path, top_k: int) -> Dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    group_cols = ["session_id", "step"]
    largest_group = df.groupby(group_cols, observed=True).size().sort_values(ascending=False).index[0]
    group_df = df[(df["session_id"] == largest_group[0]) & (df["step"] == largest_group[1])].copy()
    return {
        "candidates": group_df.to_dict(orient="records"),
        "top_k": top_k,
    }


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((q / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Call /recommend with real candidates and measure latency.")
    parser.add_argument("--url", default="http://localhost:8001/recommend", help="Recommend endpoint URL")
    parser.add_argument(
        "--parquet",
        default="csao_data/cart_sessions_test.parquet",
        help="Parquet path for real candidate groups",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k recommendations to request")
    parser.add_argument("--runs", type=int, default=20, help="Number of timed calls")
    parser.add_argument("--timeout-sec", type=float, default=30.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    payload = _build_payload(parquet_path=parquet_path, top_k=args.top_k)
    latencies_ms: List[float] = []
    last_response: Dict[str, Any] = {}

    with httpx.Client(timeout=args.timeout_sec) as client:
        for i in range(args.runs):
            start = time.perf_counter()
            response = client.post(args.url, json=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            response.raise_for_status()
            latencies_ms.append(elapsed_ms)
            last_response = response.json()
            if i == 0:
                print(f"Warmup status={response.status_code}, latency_ms={elapsed_ms:.2f}")

    avg_ms = statistics.mean(latencies_ms)
    p50_ms = _percentile(latencies_ms, 50)
    p95_ms = _percentile(latencies_ms, 95)
    p99_ms = _percentile(latencies_ms, 99)

    print(f"Endpoint: {args.url}")
    print(f"Runs: {args.runs}")
    print(f"Candidates sent: {len(payload['candidates'])}")
    print(f"Top-k requested: {args.top_k}")
    print(f"Latency avg_ms={avg_ms:.2f} p50_ms={p50_ms:.2f} p95_ms={p95_ms:.2f} p99_ms={p99_ms:.2f}")
    print(f"Sample response: {last_response}")


if __name__ == "__main__":
    main()
