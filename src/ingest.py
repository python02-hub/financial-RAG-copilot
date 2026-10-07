"""
ingest.py — Parses SEC filings (10-K/10-Q, .txt/.htm/.pdf) and earnings-call
transcripts into structured, section-aware chunks ready for embedding.

Handles:
  - Plain text / HTML-stripped filings (as produced by SEC EDGAR full-text)
  - PDF filings, including table extraction via pdfplumber
  - SEC "Item N." section header detection so every chunk carries its
    originating section (e.g. "Item 1A. Risk Factors") as metadata —
    this is what lets the RAG layer answer "what does the Risk Factors
    section say about X" instead of just "somewhere in this document".
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Optional


def _chunk_id(*parts: str) -> str:
    """Deterministic, content-derived chunk id (12 hex chars). Using a
    hash instead of uuid4 means re-ingesting the same filing produces the
    same ids every time — important for reproducible builds (see
    web/build.py) and for diffing what actually changed between runs."""
    h = hashlib.sha1("||".join(parts).encode("utf-8", errors="ignore"))
    return h.hexdigest()[:12]

# SEC 10-K / 10-Q item headers we recognize for section tagging.
SECTION_PATTERN = re.compile(
    r"^\s*(Item\s+\d+[A-C]?\.?\s*[-–—]?\s*.{0,80})$",
    re.IGNORECASE | re.MULTILINE,
)

TABLE_LINE_PATTERN = re.compile(r"\|.+\|")  # markdown-style pipe tables we saved


@dataclass
class Chunk:
    id: str
    text: str
    company: str
    filing_type: str
    fiscal_year: str
    section: str
    source_file: str
    chunk_type: str = "text"  # "text" | "table"
    metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "company": self.company,
            "filing_type": self.filing_type,
            "fiscal_year": self.fiscal_year,
            "section": self.section,
            "source_file": self.source_file,
            "chunk_type": self.chunk_type,
            **self.metadata,
        }


def _split_into_sections(raw_text: str) -> list[tuple[str, str]]:
    """Split filing text into (section_name, section_body) pairs using
    'Item N.' headers. Falls back to a single 'Full Document' section
    if no headers are found (e.g. earnings call transcripts)."""
    matches = list(SECTION_PATTERN.finditer(raw_text))
    if not matches:
        return [("Full Document", raw_text)]

    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        header = m.group(1).strip()
        body = raw_text[start:end]
        sections.append((header, body))
    return sections


def _chunk_text(text: str, max_chars: int = 1400, overlap: int = 200) -> list[str]:
    """Paragraph-aware sliding window chunker. Keeps tables intact where
    possible by never splitting inside a contiguous block of pipe-table
    lines."""
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        candidate = (buf + "\n\n" + para).strip() if buf else para
        if len(candidate) > max_chars and buf:
            chunks.append(buf.strip())
            # overlap: carry the tail of the previous buffer forward
            tail = buf[-overlap:] if len(buf) > overlap else buf
            buf = (tail + "\n\n" + para).strip()
        else:
            buf = candidate
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _extract_tables_from_text(section_body: str) -> list[str]:
    """Pull out contiguous markdown-pipe-table blocks as their own chunks,
    since tables carry structured financial data that benefits from being
    retrieved as a single unit rather than split mid-row."""
    lines = section_body.split("\n")
    tables = []
    current: list[str] = []
    for line in lines:
        if TABLE_LINE_PATTERN.search(line):
            current.append(line)
        else:
            if len(current) >= 2:
                tables.append("\n".join(current))
            current = []
    if len(current) >= 2:
        tables.append("\n".join(current))
    return tables


def ingest_text_filing(
    path: str,
    company: str,
    filing_type: str,
    fiscal_year: str,
) -> list[Chunk]:
    """Ingest a .txt (already-extracted) filing into section-aware,
    table-aware chunks."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    source_file = os.path.basename(path)
    chunks: list[Chunk] = []

    for section_name, body in _split_into_sections(raw):
        # 1. Extract tables as their own high-value chunks first
        for table_md in _extract_tables_from_text(body):
            table_text = f"[Table from section: {section_name}]\n{table_md}"
            chunks.append(
                Chunk(
                    id=_chunk_id(source_file, section_name, "table", table_text),
                    text=table_text,
                    company=company,
                    filing_type=filing_type,
                    fiscal_year=fiscal_year,
                    section=section_name,
                    source_file=source_file,
                    chunk_type="table",
                )
            )
        # 2. Strip tables out of the prose before chunking the narrative text,
        #    so numbers aren't duplicated awkwardly across chunk boundaries.
        prose = TABLE_LINE_PATTERN.sub("", body)
        for text_chunk in _chunk_text(prose):
            if len(text_chunk.strip()) < 40:
                continue
            chunks.append(
                Chunk(
                    id=_chunk_id(source_file, section_name, "text", text_chunk),
                    text=text_chunk,
                    company=company,
                    filing_type=filing_type,
                    fiscal_year=fiscal_year,
                    section=section_name,
                    source_file=source_file,
                    chunk_type="text",
                )
            )
    return chunks


def ingest_pdf_filing(
    path: str,
    company: str,
    filing_type: str,
    fiscal_year: str,
) -> list[Chunk]:
    """Ingest a PDF filing: extracts prose text page-by-page and detects
    tables with pdfplumber's table-finder, converting each detected table
    into a markdown chunk. This is the path used for filings that arrive
    as scanned/print-style PDFs rather than SEC EDGAR HTML."""
    import pdfplumber

    source_file = os.path.basename(path)
    chunks: list[Chunk] = []
    full_text_parts = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Tables
            for t_idx, table in enumerate(page.extract_tables() or []):
                if not table or len(table) < 2:
                    continue
                md_rows = []
                header = [c or "" for c in table[0]]
                md_rows.append("| " + " | ".join(header) + " |")
                md_rows.append("| " + " | ".join(["---"] * len(header)) + " |")
                for row in table[1:]:
                    row = [c or "" for c in row]
                    md_rows.append("| " + " | ".join(row) + " |")
                table_md = "\n".join(md_rows)
                table_text = f"[Table, page {page_num}]\n{table_md}"
                chunks.append(
                    Chunk(
                        id=_chunk_id(source_file, str(page_num), "table", str(t_idx), table_text),
                        text=table_text,
                        company=company,
                        filing_type=filing_type,
                        fiscal_year=fiscal_year,
                        section=f"Page {page_num}",
                        source_file=source_file,
                        chunk_type="table",
                        metadata={"page": page_num},
                    )
                )
            # Prose
            page_text = page.extract_text() or ""
            full_text_parts.append(f"\n\n--- Page {page_num} ---\n{page_text}")

    full_text = "".join(full_text_parts)
    for section_name, body in _split_into_sections(full_text):
        for text_chunk in _chunk_text(body):
            if len(text_chunk.strip()) < 40:
                continue
            chunks.append(
                Chunk(
                    id=_chunk_id(source_file, section_name, "pdf-text", text_chunk),
                    text=text_chunk,
                    company=company,
                    filing_type=filing_type,
                    fiscal_year=fiscal_year,
                    section=section_name,
                    source_file=source_file,
                    chunk_type="text",
                )
            )
    return chunks


def ingest_filing(
    path: str,
    company: str,
    filing_type: str,
    fiscal_year: str,
) -> list[Chunk]:
    """Dispatch to the right parser based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return ingest_pdf_filing(path, company, filing_type, fiscal_year)
    return ingest_text_filing(path, company, filing_type, fiscal_year)
