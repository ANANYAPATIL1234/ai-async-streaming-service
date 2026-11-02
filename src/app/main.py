import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import aclosing, asynccontextmanager
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import load_settings
from app.db import DBPool
from app.deps import ApiKey, DBDep, RateLimiter
from app.llm import embed_texts, generate_completion, stream_tokens
from app.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    InferenceRequest,
    InferenceResponse,
    StatusResponse,
)

logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO, format="%(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared resources once at startup, close them at shutdown."""
    settings = load_settings()
    app.state.settings = settings
    app.state.rate_limiter = RateLimiter(settings.rate_limit_per_minute)
    app.state.db_pool = DBPool(settings.db_pool_size)
    app.state.started_at = time.monotonic()
    async with httpx.AsyncClient(timeout=10.0) as client:
        app.state.http_client = client  # reused for upstream LLM API calls
        yield


app = FastAPI(title="Async AI Streaming Service", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=load_settings().allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Request timing + request id + structured JSON access log."""
    start = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", uuid4().hex)
    request.state.request_id = request_id
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    response.headers["X-Request-ID"] = request_id
    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "ms": round(elapsed_ms, 2),
            }
        )
    )
    return response


def _error(status: int, error: str, detail: Any, request: Request) -> JSONResponse:
    body = {
        "error": error,
        "detail": detail,
        "request_id": getattr(request.state, "request_id", None),
    }
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)
    response = _error(exc.status_code, "http_error", exc.detail, request)
    if exc.headers:
        response.headers.update(exc.headers)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)
    problems = [
        {"field": ".".join(str(p) for p in err["loc"][1:]), "message": err["msg"]}
        for err in exc.errors()
    ]
    return _error(422, "validation_error", problems, request)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error")
    return _error(500, "internal_error", "Something went wrong", request)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/status")
async def status(request: Request) -> StatusResponse:
    pool: DBPool = request.app.state.db_pool
    return StatusResponse(
        status="ok",
        uptime_seconds=round(time.monotonic() - request.app.state.started_at, 2),
        db_pool_in_use=pool.in_use,
        db_pool_size=pool.size,
    )


@app.post("/v1/infer")
async def infer(
    body: InferenceRequest, request: Request, _key: ApiKey, db: DBDep
) -> InferenceResponse:
    await db.log_request(f"infer model={body.model}")
    delay = request.app.state.settings.token_delay_seconds
    tokens = await generate_completion(body.prompt, body.max_tokens, delay)
    return InferenceResponse(model=body.model, text="".join(tokens).strip(), tokens=len(tokens))


@app.post("/v1/embeddings")
async def embeddings(body: EmbeddingRequest, _key: ApiKey) -> EmbeddingResponse:
    vectors = await embed_texts(body.texts)
    return EmbeddingResponse(model=body.model, dimensions=len(vectors[0]), embeddings=vectors)


async def sse_events(request: Request, body: InferenceRequest) -> AsyncIterator[str]:
    """Server-Sent Events: each token is one `event: token` frame."""
    delay = request.app.state.settings.token_delay_seconds
    count = 0
    async with aclosing(stream_tokens(body.prompt, body.max_tokens, delay)) as tokens:
        async for token in tokens:
            if await request.is_disconnected():
                # client left: stop generating so we don't waste compute / token budget
                logger.info(json.dumps({"event": "client_disconnected", "tokens_sent": count}))
                return
            count += 1
            payload = json.dumps({"token": token, "index": count})
            yield f"event: token\ndata: {payload}\n\n"
    yield f"event: done\ndata: {json.dumps({'tokens': count})}\n\n"


@app.post("/v1/stream")
async def stream_sse(body: InferenceRequest, request: Request, _key: ApiKey) -> StreamingResponse:
    return StreamingResponse(
        sse_events(request, body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def raw_chunks(request: Request, body: InferenceRequest) -> AsyncIterator[str]:
    delay = request.app.state.settings.token_delay_seconds
    async with aclosing(stream_tokens(body.prompt, body.max_tokens, delay)) as tokens:
        async for token in tokens:
            if await request.is_disconnected():
                return
            yield token


@app.post("/v1/stream-chunked")
async def stream_chunked(
    body: InferenceRequest, request: Request, _key: ApiKey
) -> StreamingResponse:
    """Plain HTTP chunked transfer encoding (no SSE framing)."""
    return StreamingResponse(raw_chunks(request, body), media_type="text/plain")
