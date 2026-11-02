# Async AI Streaming Service

High-throughput asynchronous AI API built with FastAPI and Pydantic v2. It streams LLM tokens to clients in real time using Server-Sent Events (SSE) and HTTP chunked transfer, with API key auth, rate limiting, a pooled DB session dependency, and request telemetry.

The LLM and embedding backends are simulated, so the project runs with no API keys. Replace `src/app/llm.py` with real model calls to go further.

## Architecture

```
Client -> CORS + timing middleware -> Depends (API key -> rate limit -> DB session)
       -> Pydantic v2 validation -> async endpoint -> LLM stream (SSE / chunked)
```

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | liveness check |
| GET | `/v1/status` | uptime and DB pool usage |
| POST | `/v1/infer` | full (non-streaming) completion |
| POST | `/v1/embeddings` | text embeddings (CPU work offloaded with `asyncio.to_thread`) |
| POST | `/v1/stream` | token streaming over SSE (`text/event-stream`) |
| POST | `/v1/stream-chunked` | token streaming over chunked HTTP |

## Features
- Pydantic v2 models with field constraints, regex pattern and custom validators, and clear error messages
- Dependency injection with `Depends`: API key auth, sliding-window rate limiter, pooled DB session with automatic cleanup
- SSE streaming with client disconnect detection (`request.is_disconnected()`) to stop wasted generation
- `lifespan` context manager for startup/shutdown of the DB pool and a shared `httpx.AsyncClient`
- CORS, request timing middleware (`X-Process-Time-Ms`, `X-Request-ID`), structured JSON logs and JSON exception handlers
- Strict typing (mypy), Ruff, pre-commit hooks, pytest, `uv` lockfile

## Setup
```bash
uv sync
cp .env.example .env
uv run pre-commit install
```

## Run
```bash
uv run uvicorn app.main:app --reload --app-dir src
```
Interactive docs: http://127.0.0.1:8000/docs

## Try it
```bash
# normal completion
curl -X POST http://127.0.0.1:8000/v1/infer \
  -H "X-API-Key: dev-key-123" -H "Content-Type: application/json" \
  -d '{"prompt": "hello async world", "max_tokens": 8}'

# streaming (SSE)
curl -N -X POST http://127.0.0.1:8000/v1/stream \
  -H "X-API-Key: dev-key-123" -H "Content-Type: application/json" \
  -d '{"prompt": "stream me some tokens", "max_tokens": 10}'
```

## Tests and quality checks
```bash
uv run pytest
uv run ruff check .
uv run mypy
```

## Project structure
```
src/app/main.py      app, lifespan, middleware, exception handlers, routes
src/app/deps.py      API key auth, rate limiter, DB session dependency
src/app/db.py        simulated async connection pool
src/app/llm.py       simulated LLM stream and embeddings
src/app/schemas.py   Pydantic v2 request/response models
src/app/config.py    settings from environment variables
tests/test_api.py    pytest tests
```

## Author
**Ananya PM** - B.E., Bapuji Institute of Engineering and Technology, AI&ML, Semester 7, CGPA 7.17
[LinkedIn](https://linkedin.com/in/ananya-pm-40ab60427) | [GitHub](https://github.com/ANANYAPATIL1234)

## License
MIT
