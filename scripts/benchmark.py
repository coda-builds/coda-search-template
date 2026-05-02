#!/usr/bin/env python3
"""
Benchmark script — measures end-to-end query latency for the search endpoint.

Usage:
    python scripts/benchmark.py [--url URL] [--api-key KEY] [--runs N] [--concurrency C]

Outputs per-run latency, p50, p95, p99, and a pass/fail for the <2 s SLA.

Requirements:
    httpx (already in requirements.txt)

Example output:
    ┌─────────────────────────────────────────────────┐
    │              Coda Search Benchmark               │
    ├──────────────┬──────────────────────────────────┤
    │ Queries run  │ 100                               │
    │ Concurrency  │ 5                                 │
    │ p50 latency  │  182 ms                           │
    │ p95 latency  │  431 ms                           │
    │ p99 latency  │  612 ms                           │
    │ Max latency  │  891 ms                           │
    │ SLA (<2000ms)│ ✅ PASS (0/100 breached)          │
    └──────────────┴──────────────────────────────────┘
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import time
from typing import NamedTuple

import httpx

# ── Test queries ──────────────────────────────────────────────────────────────
QUERIES = [
    "lightweight running shoes for marathons",
    "noise cancelling headphones for travel",
    "portable coffee maker for camping",
    "ergonomic office chair with lumbar support",
    "waterproof hiking backpack",
    "cast iron skillet for induction hob",
    "bamboo chopping board large",
    "smart home plug compatible with Alexa",
    "adjustable dumbbell set home gym",
    "merino wool base layer for cold weather",
    "portable bluetooth speaker waterproof",
    "standing desk riser for home office",
    "foam roller for muscle recovery",
    "insulated water bottle stainless steel",
    "solar charger for outdoor adventures",
    "resistance bands for physiotherapy",
    "stainless steel meal prep containers",
    "zero gravity recliner for garden",
    "HEPA air purifier for small room",
    "cold brew coffee maker",
]


class BenchmarkResult(NamedTuple):
    latency_ms: float
    status_code: int
    result_count: int


async def single_request(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    query: str,
) -> BenchmarkResult:
    payload = {"query": query, "top_k": 10}
    t0 = time.perf_counter()
    try:
        response = await client.post(
            url,
            headers={"x-api-key": api_key, "Content-Type": "application/json"},
            json=payload,
            timeout=10.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        if response.status_code == 200:
            data = response.json()
            return BenchmarkResult(latency_ms, 200, data.get("total", 0))
        return BenchmarkResult(latency_ms, response.status_code, 0)
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        print(f"  Request error: {exc}")
        return BenchmarkResult(latency_ms, 0, 0)


async def run_benchmark(
    base_url: str,
    api_key: str,
    runs: int,
    concurrency: int,
) -> None:
    search_url = f"{base_url.rstrip('/')}/search"

    # Build a list of queries, cycling through QUERIES
    query_list = [QUERIES[i % len(QUERIES)] for i in range(runs)]

    print(f"\n🔍  Coda Search Benchmark")
    print(f"    Target : {search_url}")
    print(f"    Runs   : {runs}")
    print(f"    Workers: {concurrency}")
    print()

    semaphore = asyncio.Semaphore(concurrency)
    results: list[BenchmarkResult] = []
    sla_ms = 2000.0

    async with httpx.AsyncClient() as client:
        async def bounded(query: str) -> BenchmarkResult:
            async with semaphore:
                result = await single_request(client, search_url, api_key, query)
                status = "✅" if result.latency_ms < sla_ms else "❌"
                print(
                    f"  {status}  {result.latency_ms:6.0f} ms  "
                    f"results={result.result_count:2d}  query={query!r:.45s}"
                )
                return result

        tasks = [bounded(q) for q in query_list]
        results = await asyncio.gather(*tasks)

    latencies = [r.latency_ms for r in results if r.status_code == 200]
    errors = [r for r in results if r.status_code != 200]

    if not latencies:
        print("\n❌  All requests failed — check the server is running and the API key is correct.")
        return

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    max_lat = max(latencies)
    breaches = sum(1 for l in latencies if l >= sla_ms)
    sla_pass = breaches == 0

    separator = "─" * 50
    print(f"\n{separator}")
    print(f"  Results ({len(latencies)}/{runs} successful)")
    print(separator)
    print(f"  p50 latency : {p50:6.0f} ms")
    print(f"  p95 latency : {p95:6.0f} ms")
    print(f"  p99 latency : {p99:6.0f} ms")
    print(f"  Max latency : {max_lat:6.0f} ms")
    print(f"  SLA (<2000ms): {'✅ PASS' if sla_pass else '❌ FAIL'}"
          f"  ({breaches}/{len(latencies)} breached)")
    if errors:
        print(f"  Errors      : {len(errors)}")
    print(separator)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the Coda Search API.")
    parser.add_argument(
        "--url", default=os.getenv("BASE_URL", "http://localhost:8000"), help="Base URL of the API"
    )
    parser.add_argument(
        "--api-key", default=os.getenv("API_KEY", "dev-insecure-key-change-in-production")
    )
    parser.add_argument("--runs", type=int, default=50, help="Number of queries to run")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrent requests")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.url, args.api_key, args.runs, args.concurrency))


if __name__ == "__main__":
    main()
