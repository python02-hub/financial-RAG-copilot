"""
metrics.py — Extracts structured financial metrics from ingested filing text.

Rather than trying to parse every possible 10-K table format generically,
this targets the handful of tables buy-side analysts pull from every filing
(segment net sales, product/services net sales, gross margin, opex, tax
rate) using tolerant regex over the markdown-table chunks produced by
ingest.py. Returns tidy pandas DataFrames — ready for charting or export.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

NUM = r"[-+]?\$?[\d,]+(?:\.\d+)?%?"


def _clean_num(cell: str) -> float | None:
    cell = cell.strip()
    if not cell or cell in ("—", "-", "n/a", "N/A"):
        return None
    negative = cell.startswith("(") and cell.endswith(")")
    cell = cell.strip("()")
    cell = cell.replace("$", "").replace(",", "").replace("%", "")
    try:
        val = float(cell)
        return -val if negative else val
    except ValueError:
        return None


def parse_markdown_table(table_text: str) -> pd.DataFrame:
    """Parse a '| a | b | c |' markdown table (as produced by ingest.py or
    pdfplumber extraction) into a DataFrame. Skips the '---' separator row."""
    lines = [l.strip() for l in table_text.strip().split("\n") if l.strip().startswith("|")]
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(re.fullmatch(r"-{2,}", c) for c in cells if c):
            continue
        rows.append(cells)
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, *data = rows
    df = pd.DataFrame(data, columns=header)
    return df


def extract_all_tables(filing_text: str) -> list[pd.DataFrame]:
    """Find every markdown table block in a raw filing text file and parse
    each into a DataFrame, tagged with the line immediately preceding it
    (used as a caption) via df.attrs['caption']."""
    lines = filing_text.split("\n")
    tables = []
    buf: list[str] = []
    caption = ""
    for i, line in enumerate(lines):
        if line.strip().startswith("|"):
            buf.append(line)
        else:
            if len(buf) >= 2:
                df = parse_markdown_table("\n".join(buf))
                if not df.empty:
                    df.attrs["caption"] = caption
                    tables.append(df)
            buf = []
            if line.strip():
                caption = line.strip()
    if len(buf) >= 2:
        df = parse_markdown_table("\n".join(buf))
        if not df.empty:
            df.attrs["caption"] = caption
            tables.append(df)
    return tables


@dataclass
class KeyMetrics:
    company: str
    fiscal_year: str
    total_net_sales: float | None = None
    total_gross_margin_pct: float | None = None
    rd_expense: float | None = None
    sga_expense: float | None = None
    effective_tax_rate: float | None = None
    cash_and_marketable_securities: float | None = None

    def as_dict(self) -> dict:
        return self.__dict__


def extract_key_metrics(filing_text: str, company: str, fiscal_year: str) -> KeyMetrics:
    """Best-effort scrape of the handful of headline numbers analysts check
    first, using proximity regex against the raw text (robust to table
    formatting drift since it doesn't depend on exact column parsing)."""
    km = KeyMetrics(company=company, fiscal_year=fiscal_year)

    m = re.search(r"Total net sales\s*\|\s*([\d,]+)", filing_text)
    if m:
        km.total_net_sales = _clean_num(m.group(1))

    m = re.search(r"Total gross margin (?:percentage|%)\s*\|\s*([\d.]+)%", filing_text)
    if m:
        km.total_gross_margin_pct = float(m.group(1))

    m = re.search(r"Research and development\s*\|\s*([\d,]+)", filing_text)
    if m:
        km.rd_expense = _clean_num(m.group(1))

    m = re.search(r"Selling, general and administrative\s*\|\s*([\d,]+)", filing_text)
    if m:
        km.sga_expense = _clean_num(m.group(1))

    m = re.search(r"Effective tax rate\s*\|\s*([\d.]+)%", filing_text)
    if m:
        km.effective_tax_rate = float(m.group(1))

    m = re.search(r"cash.{0,40}marketable securities.{0,40}totaled\s*\$?([\d.]+)\s*billion", filing_text, re.IGNORECASE)
    if m:
        km.cash_and_marketable_securities = float(m.group(1)) * 1000  # normalize to $M

    return km
