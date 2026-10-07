#!/usr/bin/env python3
"""
web/build.py — Generates the standalone, GitHub-Pages-ready web app.

This is the single source of truth for docs/index.html. It:
  1. Runs the real ingestion pipeline (src/ingest.py) over the bundled
     sample filing so the web demo ships with genuinely parsed, real data
     — not a hand-written fixture.
  2. Extracts structured metrics (src/metrics.py) and price-reaction
     events (src/price_analysis.py) the same way the CLI/Streamlit app does.
  3. Injects the result as a JSON payload into web/template.html, replacing
     the __DATA_JSON__ placeholder, and writes the result to docs/index.html.

Run it after changing the bundled sample data or the template:
    python web/build.py

Output is deterministic: chunk IDs are content-hashed (see src/ingest.py),
so re-running this with unchanged inputs produces a byte-identical file —
useful for CI to catch "docs/index.html is out of date" drift.
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.ingest import ingest_text_filing
from src.metrics import extract_key_metrics
from src.price_analysis import load_price_series, analyze_known_events

FILING_PATH = os.path.join(ROOT, "data", "filings", "AAPL_10K_FY2025.txt")
PRICES_PATH = os.path.join(ROOT, "data", "prices", "AAPL_prices.csv")
TEMPLATE_PATH = os.path.join(ROOT, "web", "template.html")
OUTPUT_PATH = os.path.join(ROOT, "docs", "index.html")

COMPANY = "AAPL"
FILING_TYPE = "10-K"
FISCAL_YEAR = "2025"
SOURCE_LABEL = "SEC EDGAR (real filing)"


def build_data_payload() -> dict:
    chunks = ingest_text_filing(FILING_PATH, COMPANY, FILING_TYPE, FISCAL_YEAR)
    chunk_data = [
        {"id": c.id, "text": c.text, "section": c.section, "chunk_type": c.chunk_type}
        for c in chunks
    ]

    with open(FILING_PATH, "r", encoding="utf-8", errors="ignore") as f:
        full_text = f.read()
    metrics = extract_key_metrics(full_text, COMPANY, FISCAL_YEAR)

    prices = load_price_series(PRICES_PATH)
    price_rows = prices.to_dict("records")
    for row in price_rows:
        row["date"] = str(row["date"].date())

    events = [r.__dict__ for r in analyze_known_events(prices)]

    return {
        "company": COMPANY,
        "filingType": FILING_TYPE,
        "fiscalYear": FISCAL_YEAR,
        "sourceLabel": SOURCE_LABEL,
        "chunks": chunk_data,
        "metrics": metrics.as_dict(),
        "prices": price_rows,
        "knownEvents": events,
    }


def inject_and_write(data: dict) -> None:
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    if "__DATA_JSON__" not in template:
        raise RuntimeError(
            f"{TEMPLATE_PATH} has no __DATA_JSON__ placeholder — did the template get corrupted?"
        )

    data_json = json.dumps(data, separators=(",", ":"), sort_keys=True)
    output = template.replace("__DATA_JSON__", data_json)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"Wrote {OUTPUT_PATH} ({len(output):,} bytes, {len(data['chunks'])} chunks)")


def validate(data: dict) -> None:
    """Lightweight sanity checks so a broken build fails loudly instead of
    shipping a blank or malformed demo."""
    assert len(data["chunks"]) > 0, "No chunks were produced from the sample filing."
    assert data["metrics"]["total_net_sales"], "Key metric extraction found nothing — check data/filings/."
    assert len(data["prices"]) > 0, "No price rows loaded — check data/prices/."
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        html = f.read()
    m = re.search(r"const SAMPLE = (\{.*?\});\n// ACTIVE", html, re.DOTALL)
    assert m, "Injected data block not found in the built HTML — injection likely failed."
    json.loads(m.group(1))  # raises if invalid JSON


if __name__ == "__main__":
    payload = build_data_payload()
    inject_and_write(payload)
    validate(payload)
    print("Build OK.")
