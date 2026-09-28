"""TextRoute — schema-validated inter-agent messaging (Sections 4.5, 4.9)."""
from __future__ import annotations
import hashlib, datetime as dt
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError


class ProvenanceEntry(BaseModel):
    model: str
    weight: float = Field(ge=0.0, le=1.0)


class RoutedMessage(BaseModel):
    sender_role: Literal["planner", "coder", "reviewer", "ensemble"]
    receiver_role: Literal["planner", "coder", "reviewer", "user", "environment"]
    intent: Literal["inform", "request", "confirm"]
    priority: int = Field(ge=1, le=5)
    body: str = Field(min_length=1, max_length=8000)
    expects_reply: bool
    schema_version: str = "1.3"
    model_provenance: Optional[list[ProvenanceEntry]] = None


class SchemaViolationError(Exception):
    """Raised when a message fails TextRoute validation (blocked, not dispatched)."""


def validate(payload: dict) -> RoutedMessage:
    try:
        return RoutedMessage(**payload)
    except ValidationError as e:
        missing = [str(err["loc"][0]) for err in e.errors()]
        raise SchemaViolationError(
            f"TextRoute blocked message — invalid/missing fields: {missing}"
        ) from e


def enrich(msg: RoutedMessage, seq: int) -> dict:
    d = msg.model_dump()
    d["seq"] = seq
    d["timestamp"] = dt.datetime.now(dt.timezone.utc).isoformat()
    d["body_sha256"] = hashlib.sha256(msg.body.encode()).hexdigest()[:16]
    return d
