from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.deps import RateLimiter
from app.main import app

HEADERS = {"X-API-Key": "dev-key-123"}
BODY = {"prompt": "hello async world", "max_tokens": 5}


@pytest.fixture()
def client() -> Iterator[TestClient]:
    with TestClient(app) as test_client:  # runs the lifespan (pool, limiter, httpx client)
        yield test_client


def test_health(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_status_reports_pool(client: TestClient) -> None:
    data = client.get("/v1/status").json()
    assert data["status"] == "ok"
    assert data["db_pool_in_use"] == 0
    assert data["db_pool_size"] >= 1


def test_timing_headers(client: TestClient) -> None:
    res = client.get("/health")
    assert "X-Process-Time-Ms" in res.headers
    assert "X-Request-ID" in res.headers


def test_missing_api_key_is_401_json(client: TestClient) -> None:
    res = client.post("/v1/infer", json=BODY)
    assert res.status_code == 401
    assert res.json()["error"] == "http_error"


def test_infer_ok(client: TestClient) -> None:
    res = client.post("/v1/infer", json=BODY, headers=HEADERS)
    assert res.status_code == 200
    data = res.json()
    assert data["tokens"] == 5
    assert data["text"].startswith("This is a")


def test_model_regex_rejected(client: TestClient) -> None:
    res = client.post("/v1/infer", json={**BODY, "model": "Bad Model!"}, headers=HEADERS)
    assert res.status_code == 422
    body = res.json()
    assert body["error"] == "validation_error"
    assert body["detail"][0]["field"] == "model"


def test_blank_prompt_rejected(client: TestClient) -> None:
    res = client.post("/v1/infer", json={"prompt": "   "}, headers=HEADERS)
    assert res.status_code == 422


def test_embeddings(client: TestClient) -> None:
    res = client.post("/v1/embeddings", json={"texts": ["a", "b"]}, headers=HEADERS)
    assert res.status_code == 200
    data = res.json()
    assert data["dimensions"] == 8
    assert len(data["embeddings"]) == 2


def test_rate_limit_returns_429(client: TestClient) -> None:
    app.state.rate_limiter = RateLimiter(limit=2)
    for _ in range(2):
        assert client.post("/v1/infer", json=BODY, headers=HEADERS).status_code == 200
    res = client.post("/v1/infer", json=BODY, headers=HEADERS)
    assert res.status_code == 429
    assert "Retry-After" in res.headers


def test_sse_stream(client: TestClient) -> None:
    with client.stream("POST", "/v1/stream", json=BODY, headers=HEADERS) as res:
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")
        lines = list(res.iter_lines())
    assert lines.count("event: token") == 5
    assert "event: done" in lines


def test_chunked_stream(client: TestClient) -> None:
    with client.stream("POST", "/v1/stream-chunked", json=BODY, headers=HEADERS) as res:
        text = "".join(res.iter_text())
    assert text.strip().startswith("This is a")
