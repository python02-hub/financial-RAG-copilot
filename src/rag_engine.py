"""
rag_engine.py — Retrieval-augmented answer generation over ingested filings.

Retrieves relevant chunks from the vector store, builds a grounded prompt
that forces the model to cite which section/source each claim comes from,
and calls the Anthropic API for the final answer. If no API key is set,
falls back to a "retrieval-only" mode so the rest of the app still works
without a key (useful for testing the pipeline end-to-end).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .vectorstore import FilingVectorStore

CLAUDE_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an equity research assistant analyzing SEC filings \
(10-K/10-Q) and earnings call transcripts for a buy-side analyst.

Rules:
- Answer ONLY using the provided context chunks. If the context doesn't contain \
the answer, say so explicitly — never invent figures, dates, or claims.
- When you cite a fact, reference its section in brackets, e.g. [Item 1A. Risk \
Factors] or [Item 7. MD&A].
- For financial figures, be precise and include the fiscal year/period.
- Flag qualitative/tone shifts (e.g. new risk language, hedged forward-looking \
statements) when relevant to the question.
- Be concise and analytical, the way a research note would read — not \
conversational filler."""


@dataclass
class RAGAnswer:
    answer: str
    sources: list[dict]
    mode: str  # "generated" | "retrieval_only"


def _format_context(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        meta = r["metadata"]
        header = f"[{i}] {meta.get('company')} {meta.get('filing_type')} FY{meta.get('fiscal_year')} — {meta.get('section')}"
        blocks.append(f"{header}\n{r['text']}")
    return "\n\n---\n\n".join(blocks)


def answer_question(
    question: str,
    store: FilingVectorStore,
    company: str | None = None,
    n_results: int = 6,
    api_key: str | None = None,
) -> RAGAnswer:
    results = store.query(question, n_results=n_results, company=company)

    if not results:
        return RAGAnswer(
            answer="No relevant content found in the ingested filings for this query. "
            "Try ingesting more filings or rephrasing the question.",
            sources=[],
            mode="retrieval_only",
        )

    context = _format_context(results)
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        # Retrieval-only fallback: no LLM call, just return the ranked passages
        # so the app remains usable without an API key configured.
        preview = "\n\n".join(
            f"[{i+1}] ({r['metadata'].get('section')}) {r['text'][:400]}..."
            for i, r in enumerate(results)
        )
        return RAGAnswer(
            answer=(
                "ANTHROPIC_API_KEY is not set, so this is retrieval-only output "
                "(no generated answer). Top matching passages:\n\n" + preview
            ),
            sources=results,
            mode="retrieval_only",
        )

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    answer_text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return RAGAnswer(answer=answer_text, sources=results, mode="generated")


def summarize_risk_changes(
    store: FilingVectorStore,
    company: str,
    api_key: str | None = None,
) -> RAGAnswer:
    """Convenience query specialized for the most common buy-side ask:
    'what changed / what are the key risks this filing flags'."""
    return answer_question(
        question=(
            f"Summarize the most financially material risk factors for {company} "
            "in this filing. Group them into categories (macro, competitive, "
            "legal/regulatory, financial) and note any risk that appears newly "
            "emphasized (e.g. tariffs, AI-related risk, antitrust)."
        ),
        store=store,
        company=company,
        n_results=8,
        api_key=api_key,
    )
