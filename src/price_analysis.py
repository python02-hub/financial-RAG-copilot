"""
price_analysis.py — Links qualitative disclosures (risk factors, legal
proceedings, product events mentioned in filings) to actual historical
stock price reactions around the relevant date.

This is the "why investment firms care" piece: an analyst reading a risk
factor about, say, a product launch or a regulatory fine wants to know
"did the market actually react to this, and by how much" without manually
cross-referencing a price chart.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd


@dataclass
class PriceReaction:
    event_label: str
    event_date: str
    window_start: str
    window_end: str
    start_close: float | None
    end_close: float | None
    pct_change: float | None
    max_drawdown_pct: float | None
    peak_volume: int | None
    note: str = ""


def load_price_series(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def price_reaction_window(
    prices: pd.DataFrame,
    event_date: str,
    label: str,
    days_before: int = 1,
    days_after: int = 5,
) -> PriceReaction:
    """Compute the price move in a window around a given event date.
    If the exact date isn't a trading day, snaps to nearest available
    dates inside the requested window."""
    ts = pd.Timestamp(event_date)
    window_start = ts - timedelta(days=days_before)
    window_end = ts + timedelta(days=days_after)

    window = prices[(prices["date"] >= window_start) & (prices["date"] <= window_end)]

    if window.empty:
        return PriceReaction(
            event_label=label,
            event_date=event_date,
            window_start=str(window_start.date()),
            window_end=str(window_end.date()),
            start_close=None,
            end_close=None,
            pct_change=None,
            max_drawdown_pct=None,
            peak_volume=None,
            note="No price data available in this window — extend the price dataset to cover this date.",
        )

    start_close = float(window.iloc[0]["close"])
    end_close = float(window.iloc[-1]["close"])
    pct_change = round((end_close - start_close) / start_close * 100, 2)
    running_max = window["close"].cummax()
    drawdown = ((window["close"] - running_max) / running_max * 100).min()
    peak_volume = int(window["volume"].max())

    return PriceReaction(
        event_label=label,
        event_date=event_date,
        window_start=str(window.iloc[0]["date"].date()),
        window_end=str(window.iloc[-1]["date"].date()),
        start_close=start_close,
        end_close=end_close,
        pct_change=pct_change,
        max_drawdown_pct=round(float(drawdown), 2),
        peak_volume=peak_volume,
    )


# Known events pulled from the ingested 10-K text — extend this as you
# ingest more filings / 8-Ks. Kept explicit (rather than auto-parsed dates)
# because filings often reference dates loosely ("beginning in Q2 2025")
# and analysts want control over exactly which date anchors the window.
KNOWN_EVENTS = {
    "iphone_17_launch": {
        "label": "iPhone 17 / iPhone Air / Apple Watch Series 11 launch (Item 7, Q4 FY2025 announcements)",
        "date": "2025-09-19",
    },
    "iphone_17_keynote": {
        "label": "iPhone 17 announcement keynote",
        "date": "2025-09-09",
    },
}


def analyze_known_events(prices: pd.DataFrame) -> list[PriceReaction]:
    return [
        price_reaction_window(prices, e["date"], e["label"])
        for e in KNOWN_EVENTS.values()
    ]
