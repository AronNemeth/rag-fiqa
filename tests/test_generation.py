from typing import Any

import pytest

from rag_fiqa.generation import (
    ABSTENTION_PREFIX,
    extract_and_validate_citations,
    format_evidence_context,
    generate_answer,
    is_abstention,
)


class FakeOllamaClient:
    """Fake Ollama client returning a configurable message response."""

    def __init__(self, reply: str, *, should_fail: bool = False) -> None:
        self.reply = reply
        self.should_fail = should_fail
        self.last_call: dict[str, Any] | None = None

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        think: bool = False,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.should_fail:
            raise ConnectionError("Ollama daemon unreachable")
        self.last_call = {
            "model": model,
            "messages": messages,
            "think": think,
            "options": options,
        }
        return {"message": {"content": self.reply}}


def sample_chunks(top_score: float = 2.5) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": "c1",
            "parent_document_id": "doc_10",
            "original_document_ids": ["doc_10"],
            "title": "",
            "text": "Index funds typically charge lower expense ratios than actively managed funds.",
            "start_char": 0,
            "end_char": 78,
            "rerank_score": top_score,
            "rerank_rank": 1,
        },
        {
            "chunk_id": "c2",
            "parent_document_id": "doc_20",
            "original_document_ids": ["doc_20"],
            "title": "",
            "text": "ETFs trade intraday on exchanges like regular stocks.",
            "start_char": 0,
            "end_char": 53,
            "rerank_score": top_score - 1.0,
            "rerank_rank": 2,
        },
    ]


def test_format_evidence_context_numbers_passages() -> None:
    formatted = format_evidence_context(sample_chunks())
    assert "[1] (Document doc_10):" in formatted
    assert "[2] (Document doc_20):" in formatted
    assert "Index funds typically charge" in formatted


def test_extract_and_validate_citations_separates_valid_and_invalid() -> None:
    answer = "Index funds have low fees [1] and ETFs trade intraday [2][1], unlike [9] or [0]."
    valid, invalid = extract_and_validate_citations(answer, num_chunks=2)
    assert valid == [1, 2]
    assert invalid == [9, 0]


def test_generate_answer_returns_citations_and_disables_thinking() -> None:
    fake_client: Any = FakeOllamaClient(
        "Index funds have lower expense ratios [1], while ETFs trade intraday [2] [99]."
    )
    result = generate_answer(
        question="How do index funds and ETFs work?",
        chunks=sample_chunks(),
        client=fake_client,
    )

    assert not result["abstained"]
    assert result["abstention_reason"] is None
    assert [c["citation_id"] for c in result["citations"]] == [1, 2]
    assert [c["parent_document_id"] for c in result["citations"]] == ["doc_10", "doc_20"]
    assert result["invalid_citations"] == [99]
    assert fake_client.last_call["think"] is False
    assert fake_client.last_call["options"]["temperature"] == 0.0


def test_generate_answer_abstains_before_llm_when_empty_or_below_threshold() -> None:
    fake_client: Any = FakeOllamaClient("Should not be called")

    empty_res = generate_answer("What is EBITDA?", chunks=[], client=fake_client)
    assert empty_res["abstained"] is True
    assert empty_res["abstention_reason"] == "no_chunks"
    assert fake_client.last_call is None

    low_score_res = generate_answer(
        "What is EBITDA?",
        chunks=sample_chunks(top_score=-8.0),
        client=fake_client,
        min_rerank_score=-5.0,
    )
    assert low_score_res["abstained"] is True
    assert low_score_res["abstention_reason"] == "below_rerank_threshold"
    assert fake_client.last_call is None


def test_generate_answer_detects_model_abstention() -> None:
    fake_client: Any = FakeOllamaClient(
        f"{ABSTENTION_PREFIX} The passages do not mention 2026 Roth IRA contribution limits."
    )
    result = generate_answer(
        question="What is the 2026 Roth IRA limit?",
        chunks=sample_chunks(),
        client=fake_client,
    )
    assert result["abstained"] is True
    assert result["abstention_reason"] == "model_abstained"
    assert is_abstention(result["answer"])


def test_generate_answer_surfaces_ollama_connection_error() -> None:
    failing_client: Any = FakeOllamaClient("", should_fail=True)
    with pytest.raises(RuntimeError, match="Ollama generation failed"):
        generate_answer("What is an ETF?", chunks=sample_chunks(), client=failing_client)
