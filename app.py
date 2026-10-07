"""
app.py — Streamlit UI for the Financial RAG Copilot.

Run with:  streamlit run app.py

Tabs:
  1. Ingest       — upload/point at a filing (.txt/.htm/.pdf), ingest into ChromaDB
  2. Ask          — chat-style Q&A over ingested filings, grounded + cited
  3. Metrics      — structured financial metrics pulled from a filing
  4. Price Reaction — link a disclosure/event date to actual stock price moves
"""
from __future__ import annotations

import os
import sys
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))

from src.ingest import ingest_filing
from src.vectorstore import FilingVectorStore
from src.rag_engine import answer_question, summarize_risk_changes
from src.metrics import extract_key_metrics
from src.price_analysis import load_price_series, price_reaction_window, analyze_known_events

load_dotenv()

st.set_page_config(page_title="Financial RAG Copilot", page_icon="📊", layout="wide")


@st.cache_resource
def get_store():
    return FilingVectorStore()


def sidebar_api_key() -> str | None:
    st.sidebar.header("Configuration")
    key = st.sidebar.text_input(
        "Anthropic API Key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Needed for generated answers. Without it, queries fall back to raw retrieval results.",
    )
    store = get_store()
    st.sidebar.metric("Chunks in vector store", store.count())
    companies = store.companies()
    if companies:
        st.sidebar.write("**Ingested companies:** " + ", ".join(companies))
    return key or None


st.title("📊 Financial RAG Copilot")
st.caption("SEC 10-K / 10-Q + earnings call analysis, grounded retrieval, price-reaction linking. Runs locally — ChromaDB + local embeddings, Claude for generation.")

api_key = sidebar_api_key()

tab_ingest, tab_ask, tab_metrics, tab_price = st.tabs(
    ["📥 Ingest", "💬 Ask", "📈 Metrics", "🔗 Price Reaction"]
)

# ---------------------------------------------------------------- Ingest ---
with tab_ingest:
    st.subheader("Ingest a filing")
    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload a filing (.txt, .htm, .pdf)", type=["txt", "htm", "html", "pdf"])
        company = st.text_input("Ticker / Company", value="AAPL")
        filing_type = st.selectbox("Filing type", ["10-K", "10-Q", "8-K", "transcript"])
        fiscal_year = st.text_input("Fiscal year", value="2025")
        if st.button("Ingest", type="primary"):
            if uploaded is None:
                st.error("Upload a file first.")
            else:
                suffix = os.path.splitext(uploaded.name)[1] or ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name
                with st.spinner("Parsing, chunking, and embedding..."):
                    chunks = ingest_filing(tmp_path, company, filing_type, fiscal_year)
                    store = get_store()
                    n = store.add_chunks(chunks)
                st.success(f"Ingested {n} chunks ({sum(1 for c in chunks if c.chunk_type=='table')} tables).")
                st.rerun()

    with col2:
        st.markdown("**Or load the bundled real sample data:**")
        st.caption("Apple Inc. FY2025 10-K, fetched from SEC EDGAR.")
        if st.button("Load sample: AAPL FY2025 10-K"):
            path = os.path.join("data", "filings", "AAPL_10K_FY2025.txt")
            with st.spinner("Parsing, chunking, and embedding..."):
                chunks = ingest_filing(path, "AAPL", "10-K", "2025")
                store = get_store()
                n = store.add_chunks(chunks)
            st.success(f"Ingested {n} chunks from the sample AAPL 10-K.")
            st.rerun()

# ------------------------------------------------------------------- Ask ---
with tab_ask:
    st.subheader("Ask a question over ingested filings")
    store = get_store()
    if store.count() == 0:
        st.info("No filings ingested yet — go to the Ingest tab first.")
    else:
        companies = store.companies()
        filter_company = st.selectbox("Restrict to company (optional)", ["Any"] + companies)
        question = st.text_area("Question", placeholder="What did Apple say about tariff risk and its supply chain?")
        colA, colB = st.columns(2)
        with colA:
            ask_clicked = st.button("Ask", type="primary")
        with colB:
            risk_clicked = st.button("Summarize key risk factors") if filter_company != "Any" else False

        if ask_clicked and question:
            with st.spinner("Retrieving and generating..."):
                result = answer_question(
                    question, store,
                    company=None if filter_company == "Any" else filter_company,
                    api_key=api_key,
                )
            st.markdown(f"**Mode:** `{result.mode}`")
            st.write(result.answer)
            with st.expander(f"Sources ({len(result.sources)})"):
                for i, s in enumerate(result.sources, 1):
                    m = s["metadata"]
                    st.markdown(f"**[{i}] {m['company']} {m['filing_type']} FY{m['fiscal_year']} — {m['section']}**")
                    st.text(s["text"][:500])

        if risk_clicked:
            with st.spinner("Summarizing risk factors..."):
                result = summarize_risk_changes(store, filter_company, api_key=api_key)
            st.markdown(f"**Mode:** `{result.mode}`")
            st.write(result.answer)

# --------------------------------------------------------------- Metrics ---
with tab_metrics:
    st.subheader("Structured financial metrics")
    default_path = os.path.join("data", "filings", "AAPL_10K_FY2025.txt")
    filing_path = st.text_input("Filing text path", value=default_path)
    m_company = st.text_input("Company", value="AAPL", key="m_company")
    m_year = st.text_input("Fiscal year", value="2025", key="m_year")
    if st.button("Extract metrics"):
        if os.path.exists(filing_path):
            with open(filing_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            km = extract_key_metrics(text, m_company, m_year)
            df = pd.DataFrame([km.as_dict()]).T.rename(columns={0: "Value"})
            st.table(df)

            chart_cols = ["total_net_sales", "rd_expense", "sga_expense"]
            chart_data = {k: v for k, v in km.as_dict().items() if k in chart_cols and v is not None}
            if chart_data:
                st.bar_chart(pd.Series(chart_data, name="$ Millions"))
        else:
            st.error(f"File not found: {filing_path}")

# ---------------------------------------------------------- Price Reaction ---
with tab_price:
    st.subheader("Link disclosures/events to price reactions")
    default_csv = os.path.join("data", "prices", "AAPL_prices.csv")
    csv_path = st.text_input("Price CSV path", value=default_csv)

    if os.path.exists(csv_path):
        prices = load_price_series(csv_path)
        st.line_chart(prices.set_index("date")["close"], height=250)

        st.markdown("#### Built-in known events (from ingested filing text)")
        reactions = analyze_known_events(prices)
        st.table(pd.DataFrame([r.__dict__ for r in reactions]))

        st.markdown("#### Custom event window")
        col1, col2, col3 = st.columns(3)
        with col1:
            event_date = st.date_input("Event date", value=pd.Timestamp("2025-09-19"))
        with col2:
            label = st.text_input("Event label", value="Custom event")
        with col3:
            days_after = st.number_input("Days after event", value=5, min_value=1, max_value=30)
        if st.button("Compute reaction"):
            reaction = price_reaction_window(prices, str(event_date), label, days_after=days_after)
            st.json(reaction.__dict__)
    else:
        st.error(f"File not found: {csv_path}")
