"""Fetch daily price history for sampled resolved markets from the CLOB API.

Reads data/processed/sample_markets.csv (built by 03_build_sample.py),
fetches /prices-history for the YES token of each market at daily fidelity,
and appends one JSON line per market to data/raw/price_histories.jsonl.
Skips markets already present (resumable). Uses a small thread pool.
"""
import json
from pathlib import Path
import os
import threading
import time
import urllib.request
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

BASE = "https://clob.polymarket.com/prices-history"
ROOT = Path(__file__).resolve().parents[1]
SAMPLE = f"{ROOT}/data/processed/sample_markets.csv"
OUT = f"{ROOT}/data/raw/price_histories.jsonl"
WORKERS = 12

lock = threading.Lock()
done_count = 0


def fetch_history(token_id, retries=6):
    params = urllib.parse.urlencode({
        "market": token_id,
        "interval": "max",
        "fidelity": 1440,
    })
    url = f"{BASE}?{params}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-script/1.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode()).get("history", [])
        except Exception as e:
            if attempt == retries - 1:
                return None
            time.sleep(min(2 ** attempt, 30))
    return None


def main():
    df = pd.read_csv(SAMPLE, dtype={"id": str, "yes_token": str})
    seen = set()
    if os.path.exists(OUT):
        with open(OUT) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["id"])
                except Exception:
                    pass
    todo = df[~df["id"].isin(seen)]
    total = len(todo)
    print(f"{len(seen)} already fetched, {total} to go", flush=True)

    out_f = open(OUT, "a")

    def work(row):
        global done_count
        hist = fetch_history(row.yes_token)
        rec = {"id": row.id, "n": len(hist) if hist is not None else -1,
               "history": [[int(h["t"]), round(float(h["p"]), 4)] for h in hist] if hist else []}
        with lock:
            out_f.write(json.dumps(rec) + "\n")
            done_count += 1
            if done_count % 500 == 0:
                out_f.flush()
                print(f"{done_count}/{total} fetched", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(work, todo.itertuples(index=False)))

    out_f.close()
    print(f"DONE: {done_count} new histories appended to {OUT}", flush=True)


if __name__ == "__main__":
    main()
