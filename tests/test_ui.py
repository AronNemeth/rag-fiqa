from unittest.mock import MagicMock, patch

import httpx

from rag_fiqa.ui import DEMO_PRESETS, check_api_health, query_api


def test_demo_presets_structure() -> None:
    assert len(DEMO_PRESETS) >= 4
    for preset in DEMO_PRESETS:
        assert "label" in preset
        assert "category" in preset
        assert "question" in preset
        assert len(preset["question"].strip()) > 10


def test_check_api_health_success() -> None:
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "status": "ok",
        "vector_store": {"points_count": 74283},
        "ollama": {"target_model": "qwen3:1.7b", "model_available": True},
    }

    with patch("httpx.Client.get", return_value=fake_response):
        result = check_api_health("http://localhost:8000")

    assert result["available"] is True
    assert result["status"] == "ok"
    assert result["points_count"] == 74283
    assert result["model_available"] is True


def test_check_api_health_degraded() -> None:
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 503
    fake_response.json.return_value = {
        "status": "degraded",
        "vector_store": {"points_count": 74283},
        "ollama": {"target_model": "qwen3:1.7b", "model_available": False},
    }

    with patch("httpx.Client.get", return_value=fake_response):
        result = check_api_health("http://localhost:8000")

    assert result["available"] is True
    assert result["status"] == "degraded"
    assert result["model_available"] is False


def test_check_api_health_connection_failure() -> None:
    with patch("httpx.Client.get", side_effect=httpx.ConnectError("Connection refused")):
        result = check_api_health("http://localhost:8000")

    assert result["available"] is False
    assert result["status"] == "unreachable"
    assert "error" in result


def test_query_api_success() -> None:
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "question": "What is an IRA?",
        "answer": "An IRA is an individual retirement account [1].",
        "abstained": False,
        "abstention_reason": None,
        "citations": [{"citation_id": 1, "parent_document_id": "doc_1"}],
        "timings": {"total_ms": 1200.0},
    }

    with patch("httpx.Client.post", return_value=fake_response):
        res = query_api("What is an IRA?", base_url="http://localhost:8000")

    assert res["success"] is True
    assert "IRA" in res["data"]["answer"]
    assert res["data"]["abstained"] is False


def test_query_api_http_error() -> None:
    fake_response = MagicMock(spec=httpx.Response)
    fake_response.status_code = 422
    fake_response.headers = {"content-type": "application/json"}
    fake_response.json.return_value = {"detail": "Question must not be empty."}

    with patch("httpx.Client.post", return_value=fake_response):
        res = query_api("", base_url="http://localhost:8000")

    assert res["success"] is False
    assert "422" in res["error"]
    assert "Question must not be empty" in res["error"]


def test_query_api_connection_error() -> None:
    with patch("httpx.Client.post", side_effect=httpx.ConnectError("Connection refused")):
        res = query_api("Sample question?", base_url="http://localhost:8000")

    assert res["success"] is False
    assert "Could not connect" in res["error"]


def test_query_api_timeout_error() -> None:
    request = httpx.Request("POST", "http://localhost:8000/ask")
    with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Timed out", request=request)):
        res = query_api("Sample question?", base_url="http://localhost:8000")

    assert res["success"] is False
    assert "timed out" in res["error"].lower()
