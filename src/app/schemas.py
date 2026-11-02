from typing import Any

from pydantic import BaseModel, Field, field_validator


class InferenceRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    # lowercase letters, digits, dots and dashes; 3-40 chars
    model: str = Field(default="demo-llm-1", pattern=r"^[a-z0-9][a-z0-9.\-]{2,39}$")
    max_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("prompt")
    @classmethod
    def prompt_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value.strip()


class InferenceResponse(BaseModel):
    model: str
    text: str
    tokens: int


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)
    model: str = Field(default="demo-embed-1", pattern=r"^[a-z0-9][a-z0-9.\-]{2,39}$")

    @field_validator("texts")
    @classmethod
    def texts_not_blank(cls, value: list[str]) -> list[str]:
        if any(not t.strip() for t in value):
            raise ValueError("texts must not contain blank strings")
        return value


class EmbeddingResponse(BaseModel):
    model: str
    dimensions: int
    embeddings: list[list[float]]


class StatusResponse(BaseModel):
    status: str
    uptime_seconds: float
    db_pool_in_use: int
    db_pool_size: int


class ErrorResponse(BaseModel):
    error: str
    detail: Any
    request_id: str | None = None
