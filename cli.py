#!/usr/bin/env python3
"""
cli.py — Command-line interface for the Financial RAG Copilot.

Usage:
    python cli.py ingest --file data/filings/AAPL_10K_FY2025.txt \
        --company AAPL --filing-type 10-K --fiscal-year 2025

    python cli.py ask "What did Apple say about tariff risk?" --company AAPL

    python cli.py risk-summary --company AAPL

    python cli.py metrics --file data/filings/AAPL_10K_FY2025.txt \
        --company AAPL --fiscal-year 2025

    python cli.py price-reaction --csv data/prices/AAPL_prices.csv \
        --date 2025-09-19 --label "iPhone 17 launch"

    python cli.py known-events --csv data/prices/AAPL_prices.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.ingest import ingest_filing
from src.vectorstore import FilingVectorStore
from src.rag_engine import answer_question, summarize_risk_changes
from src.metrics import extract_key_metrics
from src.price_analysis import load_price_series, price_reaction_window, analyze_known_events

load_dotenv()
console = Console()


def cmd_ingest(args):
    chunks = ingest_filing(args.file, args.company, args.filing_type, args.fiscal_year)
    store = FilingVectorStore()
    n = store.add_chunks(chunks)
    console.print(
        f"[green]Ingested[/green] {n} chunks from {args.file} "
        f"({args.company} {args.filing_type} FY{args.fiscal_year}). "
        f"Collection now has {store.count()} total chunks."
    )


def cmd_ask(args):
    store = FilingVectorStore()
    result = answer_question(args.question, store, company=args.company, api_key=args.api_key)
    console.rule(f"[bold]{args.question}")
    console.print(result.answer)
    if result.sources:
        console.rule("Sources")
        for i, s in enumerate(result.sources, 1):
            m = s["metadata"]
            console.print(f"[{i}] {m['company']} {m['filing_type']} FY{m['fiscal_year']} — {m['section']} (distance={s['distance']:.3f})")


def cmd_risk_summary(args):
    store = FilingVectorStore()
    result = summarize_risk_changes(store, args.company, api_key=args.api_key)
    console.rule(f"[bold]Risk Summary — {args.company}")
    console.print(result.answer)


def cmd_metrics(args):
    with open(args.file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    km = extract_key_metrics(text, args.company, args.fiscal_year)
    table = Table(title=f"{args.company} FY{args.fiscal_year} Key Metrics")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in km.as_dict().items():
        table.add_row(k, str(v))
    console.print(table)
    print(json.dumps(km.as_dict(), indent=2))


def cmd_price_reaction(args):
    prices = load_price_series(args.csv)
    reaction = price_reaction_window(prices, args.date, args.label)
    console.print_json(data=reaction.__dict__)


def cmd_known_events(args):
    prices = load_price_series(args.csv)
    reactions = analyze_known_events(prices)
    table = Table(title="Known Event → Price Reaction")
    table.add_column("Event")
    table.add_column("Window")
    table.add_column("% Change")
    table.add_column("Max Drawdown %")
    table.add_column("Peak Volume")
    for r in reactions:
        table.add_row(
            r.event_label,
            f"{r.window_start} to {r.window_end}",
            f"{r.pct_change:+.2f}%" if r.pct_change is not None else "n/a",
            f"{r.max_drawdown_pct:.2f}%" if r.max_drawdown_pct is not None else "n/a",
            str(r.peak_volume) if r.peak_volume else "n/a",
        )
    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Financial RAG Copilot CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="Ingest a filing into the vector store")
    p.add_argument("--file", required=True)
    p.add_argument("--company", required=True)
    p.add_argument("--filing-type", required=True, choices=["10-K", "10-Q", "8-K", "transcript"])
    p.add_argument("--fiscal-year", required=True)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("ask", help="Ask a question over ingested filings")
    p.add_argument("question")
    p.add_argument("--company", default=None)
    p.add_argument("--api-key", default=None)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("risk-summary", help="Summarize risk factors for a company")
    p.add_argument("--company", required=True)
    p.add_argument("--api-key", default=None)
    p.set_defaults(func=cmd_risk_summary)

    p = sub.add_parser("metrics", help="Extract structured key metrics from a filing")
    p.add_argument("--file", required=True)
    p.add_argument("--company", required=True)
    p.add_argument("--fiscal-year", required=True)
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("price-reaction", help="Compute price reaction around a given date")
    p.add_argument("--csv", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--label", default="Event")
    p.set_defaults(func=cmd_price_reaction)

    p = sub.add_parser("known-events", help="Show price reactions for built-in known events")
    p.add_argument("--csv", required=True)
    p.set_defaults(func=cmd_known_events)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
