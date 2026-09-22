"""Request and response contracts exposed to the Java backend."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .config import settings


class TextRequest(BaseModel):
    """Raw Sinhala/English text to push through the text front-end."""

    text: str = Field(..., min_length=1, max_length=settings.max_text_chars)


class SynthesisRequest(TextRequest):
    """Everything the acoustic model needs for one utterance."""

    model: Optional[str] = Field(
        default=None,
        description="Registry key of the trained model; falls back to the configured default.",
    )
    speaker_id: int = Field(
        default=0,
        ge=0,
        description="Speaker index for multi-speaker checkpoints (ignored by single-speaker ones).",
    )
    speed: float = Field(
        default=1.0,
        gt=0.25,
        le=3.0,
        description="Speaking-rate multiplier; 1.0 is the pace the model was trained at.",
    )


class PhonemeResponse(BaseModel):
    status: str = "success"
    raw_input: str
    normalized_text: str
    ipa_sequence: str


class ModelInfo(BaseModel):
    key: str
    description: str
    is_default: bool
    is_loaded: bool
    num_speakers: Optional[int] = None
    sample_rate: Optional[int] = None
    checkpoint: Optional[str] = None


class ModelListResponse(BaseModel):
    status: str = "success"
    default_model: str
    models: List[ModelInfo]


class FrontendInfo(BaseModel):
    """Identity of the text front-end this service is running."""

    source: str = Field(description="'bundled' when the vendored copy is used.")
    status: str = Field(description="'clean', 'drifted' or 'unstamped'.")
    fingerprint: str
    recorded_fingerprint: Optional[str] = None
    synced_at: Optional[str] = None
    source_commit: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    auth_enabled: bool
    default_model: Optional[str] = None
    available_models: List[str] = Field(default_factory=list)
    loaded_models: List[str] = Field(default_factory=list)
    frontend: Optional[FrontendInfo] = None


class ErrorResponse(BaseModel):
    status: str = "error"
    error: str
    detail: Optional[str] = None
