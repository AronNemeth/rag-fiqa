"""Streamlit demonstration interface for the FiQA financial Q&A RAG application."""

from typing import Any

import httpx
import streamlit as st

from rag_fiqa.config import (
    API_BASE_URL,
    MIN_RERANK_SCORE,
    RERANK_TOP_K_CHUNKS,
    RETRIEVAL_TOP_K_CHUNKS,
)

# -----------------------------------------------------------------------------
# Curated Demo Presets
# -----------------------------------------------------------------------------

DEMO_PRESETS: list[dict[str, str]] = [
    {
        "label": "Traditional vs. Roth IRA",
        "category": "Factual Grounding",
        "question": "What is the main difference between a traditional IRA and a Roth IRA?",
        "description": "Tests grounded answering with inline citations ([1], [2]).",
    },
    {
        "label": "Capital Gains on Mutual Funds",
        "category": "Conditional Tax Rules",
        "question": "When does an investor need to pay capital gains tax on mutual fund distributions?",
        "description": "Tests preservation of tax conditions and caveats from forum sources.",
    },
    {
        "label": "Baking Cookie Monster Cookies",
        "category": "Safe Abstention (Pre-LLM)",
        "question": "How do you bake Cookie Monster Cookies from Sesame Street?",
        "description": "Tests reranker score gate (< MIN_RERANK_SCORE) refusing before LLM.",
    },
    {
        "label": "Income Tax Abolished in 2015",
        "category": "Safe Abstention (False Premise)",
        "question": "Why did the US government completely abolish all federal income taxes in 2015?",
        "description": "Tests generator refusing to hallucinate when given a false premise.",
    },
]


# -----------------------------------------------------------------------------
# API Client Helper Functions (Decoupled for Unit Testing)
# -----------------------------------------------------------------------------


def check_api_health(base_url: str = API_BASE_URL, timeout: float = 3.0) -> dict[str, Any]:
    """Check health and connectivity of the FastAPI backend.

    Returns:
        A dictionary with 'available', 'status', and diagnostic details.
    """
    url = f"{base_url.rstrip('/')}/health"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url)
            if resp.status_code in (200, 503):
                payload = resp.json()
                return {
                    "available": True,
                    "status": payload.get("status", "unknown"),
                    "points_count": payload.get("vector_store", {}).get("points_count", 0),
                    "target_model": payload.get("ollama", {}).get("target_model", ""),
                    "model_available": payload.get("ollama", {}).get("model_available", False),
                }
            return {"available": False, "status": f"HTTP {resp.status_code}", "error": resp.text}
    except Exception as exc:
        return {"available": False, "status": "unreachable", "error": str(exc)}


def query_api(
    question: str,
    base_url: str = API_BASE_URL,
    retrieval_top_k: int | None = None,
    rerank_top_k: int | None = None,
    min_rerank_score: float | None = None,
    timeout: float = 90.0,
) -> dict[str, Any]:
    """Send a user question to the FastAPI /ask endpoint.

    Returns:
        A dictionary containing success flag and response data or error details.
    """
    url = f"{base_url.rstrip('/')}/ask"
    payload: dict[str, Any] = {"question": question}
    if retrieval_top_k is not None:
        payload["retrieval_top_k"] = retrieval_top_k
    if rerank_top_k is not None:
        payload["rerank_top_k"] = rerank_top_k
    if min_rerank_score is not None:
        payload["min_rerank_score"] = min_rerank_score

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=payload)
            if resp.status_code == 200:
                return {"success": True, "data": resp.json()}
            detail = (
                resp.json().get("detail", resp.text)
                if resp.headers.get("content-type") == "application/json"
                else resp.text
            )
            return {"success": False, "error": f"API Error ({resp.status_code}): {detail}"}
    except httpx.ConnectError:
        return {
            "success": False,
            "error": f"Could not connect to FastAPI server at {base_url}. Ensure `scripts/serve_api.py` is running.",
        }
    except httpx.TimeoutException:
        return {"success": False, "error": "Request timed out waiting for backend generation."}
    except Exception as exc:
        return {"success": False, "error": f"Unexpected client error: {exc}"}


# -----------------------------------------------------------------------------
# Streamlit Interface Rendering
# -----------------------------------------------------------------------------


def render_sidebar(api_base_url: str) -> tuple[str, int, int, float]:
    """Render sidebar configuration controls and return user settings."""
    st.sidebar.header("⚙️ Pipeline Configuration")
    api_url = st.sidebar.text_input("Backend API URL", value=api_base_url)

    # Health check display
    health = check_api_health(api_url)
    if health["available"] and health["status"] == "ok":
        st.sidebar.success(
            f"🟢 Backend Online\n- Index: {health['points_count']:,} chunks\n- Model: {health['target_model']}"
        )
    elif health["available"] and health["status"] == "degraded":
        st.sidebar.warning(f"🟡 Backend Degraded\n- Ollama model available: {health['model_available']}")
    else:
        st.sidebar.error("🔴 Backend Unreachable\nRun: `uv run python scripts/serve_api.py`")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Tunable Parameters")
    retrieval_k = st.sidebar.slider(
        "Retrieval Candidates (top-k)",
        min_value=5,
        max_value=100,
        value=RETRIEVAL_TOP_K_CHUNKS,
        step=5,
        help="Number of chunks retrieved from Qdrant via bi-encoder similarity.",
    )
    rerank_k = st.sidebar.slider(
        "Rerank Evidence (top-k)",
        min_value=1,
        max_value=15,
        value=RERANK_TOP_K_CHUNKS,
        step=1,
        help="Number of chunks passed into the LLM context after cross-encoder reranking.",
    )
    min_score = st.sidebar.number_input(
        "Min Rerank Score Threshold",
        value=float(MIN_RERANK_SCORE),
        step=0.5,
        format="%.1f",
        help="Relevance threshold below which the pipeline abstains before calling Ollama.",
    )
    return api_url, retrieval_k, rerank_k, float(min_score)


def set_preset_question(question: str) -> None:
    """Callback to populate the question input and trigger immediate execution."""
    st.session_state["question_input"] = question
    st.session_state["trigger_search"] = True


def render_presets() -> None:
    """Render one-click preset test buttons."""
    st.markdown("##### 💡 Preset Demonstration Queries")
    cols = st.columns(len(DEMO_PRESETS))
    for idx, preset in enumerate(DEMO_PRESETS):
        with cols[idx]:
            st.button(
                f"**{preset['label']}**\n\n*{preset['category']}*",
                key=f"preset_{idx}",
                help=preset["description"],
                use_container_width=True,
                on_click=set_preset_question,
                args=(preset["question"],),
            )


def render_answer_card(data: dict[str, Any]) -> None:
    """Render the generated answer, abstention status, and latency metrics."""
    st.markdown("### Answer")
    if data["abstained"]:
        reason_label = (
            "Reranker relevance score fell below threshold"
            if data["abstention_reason"] == "below_rerank_threshold"
            else "Evidence passages do not contain sufficient information"
        )
        st.warning(f"🛡️ **Pipeline Abstained ({data['abstention_reason']})**\n\n{reason_label}")
        st.info(data["answer"])
    else:
        card_style = (
            "background-color: #f8f9fa; padding: 18px; border-radius: 8px; "
            "border-left: 5px solid #0066cc; color: #1a1a1a; margin-bottom: 15px;"
        )
        st.markdown(
            f'<div style="{card_style}"><p style="font-size: 1.05rem; line-height: 1.6; margin: 0;">'
            f"{data['answer']}</p></div>",
            unsafe_allow_html=True,
        )

    if data.get("invalid_citations"):
        st.error(f"⚠️ Generator emitted invalid citation IDs: {data['invalid_citations']}")

    # Timings
    timings = data.get("timings", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Latency", f"{timings.get('total_ms', 0) / 1000:.2f} s")
    m2.metric("Dense Retrieval", f"{timings.get('retrieval_ms', 0):.0f} ms")
    m3.metric("Reranking", f"{timings.get('rerank_ms', 0):.0f} ms")
    m4.metric("LLM Generation", f"{timings.get('generation_ms', 0) / 1000:.2f} s")


def render_evidence_inspector(data: dict[str, Any]) -> None:
    """Render evidence passages with scores and citation labels."""
    context_chunks = data.get("context_chunks", [])
    cited_ids = {c["citation_id"] for c in data.get("citations", [])}
    if not context_chunks:
        return

    with st.expander(f"📚 Evidence Passages & Citation Inspector ({len(context_chunks)} chunks)", expanded=True):
        for chunk in context_chunks:
            cid = chunk["citation_id"]
            is_cited = cid in cited_ids
            badge = "✅ **CITED IN ANSWER**" if is_cited else "📄 **CONTEXT CANDIDATE**"
            border_color = "#28a745" if is_cited else "#cccccc"
            rerank_score_str = f"{chunk['rerank_score']:.2f}" if chunk.get("rerank_score") is not None else "N/A"
            retrieval_score_str = (
                f"{chunk['retrieval_score']:.2f}" if chunk.get("retrieval_score") is not None else "N/A"
            )
            title_suffix = f" ({chunk['title']})" if chunk.get("title") else ""

            header_html = (
                f"<span><strong>[{cid}] Document {chunk['parent_document_id']}</strong>{title_suffix}</span>"
                f"<span>{badge} | Rerank Score: <code>{rerank_score_str}</code> | "
                f"Cosine: <code>{retrieval_score_str}</code></span>"
            )
            box_style = f"border: 1px solid {border_color}; padding: 12px; border-radius: 6px; margin-bottom: 10px;"
            st.markdown(
                f'<div style="{box_style}">'
                f'<div style="display: flex; justify-content: space-between; margin-bottom: 6px;">{header_html}</div>'
                f'<div style="font-size: 0.95rem; color: #333333; line-height: 1.5;">{chunk["text"]}</div>'
                "</div>",
                unsafe_allow_html=True,
            )


def main() -> None:
    """Execute the Streamlit application."""
    st.set_page_config(page_title="FiQA Financial Q&A", page_icon="📈", layout="wide")

    api_url, retrieval_k, rerank_k, min_score = render_sidebar(API_BASE_URL)

    st.title("📈 FiQA Financial Q&A Assistant")
    st.caption("Grounded retrieval-augmented generation over 57,000+ FiQA financial documents with inline citations.")

    render_presets()
    st.markdown("---")

    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""

    # Question Input
    with st.form("query_form"):
        question = st.text_area(
            "Enter your financial question:",
            key="question_input",
            placeholder="e.g. How does dollar cost averaging protect an investor against market volatility?",
            height=80,
        )
        submitted = st.form_submit_button("🔍 Ask Assistant", use_container_width=True)

    auto_trigger = st.session_state.pop("trigger_search", False)
    query_text = (question or "").strip()
    should_run = (submitted or auto_trigger) and bool(query_text)
    if should_run:
        with st.spinner("Retrieving evidence, reranking, and generating grounded answer..."):
            result = query_api(
                query_text,
                base_url=api_url,
                retrieval_top_k=retrieval_k,
                rerank_top_k=rerank_k,
                min_rerank_score=min_score,
            )

        if not result["success"]:
            st.error(result["error"])
        else:
            render_answer_card(result["data"])
            render_evidence_inspector(result["data"])


if __name__ == "__main__":
    main()
