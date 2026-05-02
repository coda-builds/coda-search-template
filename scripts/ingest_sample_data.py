#!/usr/bin/env python3
"""
Ingest the sample product dataset into the database.

Usage:
    python scripts/ingest_sample_data.py

The script reads data/sample_products.json and sends it to the running API
via the POST /ingest/batch endpoint.  Make sure the server is running first:

    uvicorn app.main:app --reload

Or point it at a remote deployment by setting BASE_URL.
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-insecure-key-change-in-production")
BATCH_SIZE = 50   # Send up to 50 items per request
DATA_FILE = Path(__file__).parent.parent / "data" / "sample_products.json"


def main() -> None:
    if not DATA_FILE.exists():
        print(f"❌  Dataset not found: {DATA_FILE}")
        sys.exit(1)

    with open(DATA_FILE) as f:
        products = json.load(f)

    print(f"Loaded {len(products)} products from {DATA_FILE.name}")
    print(f"Target: {BASE_URL}")

    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}

    # Split into batches
    batches = [products[i : i + BATCH_SIZE] for i in range(0, len(products), BATCH_SIZE)]
    total_ingested = 0

    with httpx.Client(timeout=120.0) as client:
        for batch_num, batch in enumerate(batches, 1):
            payload = {"items": batch}
            t0 = time.perf_counter()
            response = client.post(
                f"{BASE_URL}/ingest/batch",
                headers=headers,
                json=payload,
            )
            elapsed = (time.perf_counter() - t0) * 1000

            if response.status_code == 201:
                data = response.json()
                total_ingested += data["ingested"]
                print(
                    f"  Batch {batch_num}/{len(batches)}: "
                    f"{data['ingested']} items ingested in {elapsed:.0f} ms"
                )
            else:
                print(f"  ❌  Batch {batch_num} failed: {response.status_code} — {response.text}")
                sys.exit(1)

    print(f"\n✅  Done. {total_ingested} items now in the database.")
    print(f"\nTry a search:\n")
    print(
        f'  curl -X POST {BASE_URL}/search \\\n'
        f'    -H "x-api-key: {API_KEY}" \\\n'
        f'    -H "Content-Type: application/json" \\\n'
        f'    -d \'{{"query": "lightweight running shoes", "top_k": 5}}\''
    )


if __name__ == "__main__":
    main()
