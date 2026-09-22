"""
HTTP surface of the VoiceLK TTS Bridge.

The Java backend is the only intended client: it posts the answer text, gets WAV
bytes back, and owns everything that happens afterwards (object storage, the
database record, serving the clip to the browser). This service keeps no state
of its own beyond the models it holds in memory.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List
from urllib.parse import quote

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import __version__
from .config import settings
from .engine import BridgeError, engine
from .fingerprint import drift_report
from .schemas import (
    ErrorResponse,
    FrontendInfo,
    HealthResponse,
    ModelInfo,
    ModelListResponse,
    PhonemeResponse,
    SynthesisRequest,
    TextRequest,
)
from .security import require_api_key

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
LOGGER = logging.getLogger("voicelk.bridge")


def frontend_info() -> FrontendInfo:
    """Which text front-end is in use, and whether it still matches the stamped copy."""
    report = drift_report(settings.frontend_dir, settings.vendor_root)
    return FrontendInfo(
        source="bundled" if settings.uses_bundled_sources else "research-checkout",
        status=report.get("status", "unknown"),
        fingerprint=report.get("fingerprint", "unknown"),
        recorded_fingerprint=report.get("recorded_fingerprint"),
        synced_at=report.get("synced_at"),
        source_commit=report.get("source_commit"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    LOGGER.info("Starting %s v%s", settings.service_name, __version__)
    LOGGER.info("Models root: %s", settings.models_root)
    LOGGER.info("Registered models: %s", ", ".join(engine.model_keys) or "none")

    front = frontend_info()
    LOGGER.info(
        "Text front-end: %s, fingerprint %s (%s).",
        front.source,
        front.fingerprint,
        front.status,
    )
    if front.status == "drifted":
        # The phonemes this service produces no longer match the copy that was
        # taken from the research repository, and the models were trained on that one.
        LOGGER.warning(
            "The bundled text front-end has changed since it was synced (stamped %s). "
            "Pronunciation may no longer match what the models were trained on.",
            front.recorded_fingerprint,
        )

    if not settings.auth_enabled:
        LOGGER.warning("No API key configured — the bridge accepts unauthenticated requests.")
    if settings.preload_default_model:
        try:
            engine.warm_up()
        except BridgeError as exc:
            # A failed warm-up must not keep the service down: the health endpoint
            # still has to be reachable so the backend can report why TTS is off.
            LOGGER.error("Warm-up failed: %s", exc)
    yield
    LOGGER.info("Shutting down %s", settings.service_name)


app = FastAPI(
    title=settings.service_name,
    description="Sinhala/English code-switched text-to-speech for the VoiceLK backend.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(BridgeError)
async def bridge_error_handler(request: Request, exc: BridgeError) -> JSONResponse:
    LOGGER.warning("%s -> %s: %s", request.url.path, exc.status_code, exc)
    body = ErrorResponse(error=exc.__class__.__name__, detail=str(exc))
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


# ---------------------------------------------------------------------------
# Operational endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe — deliberately unauthenticated and free of model loading."""
    available: List[str] = engine.model_keys
    return HealthResponse(
        status="healthy" if available else "degraded",
        service=settings.service_name,
        version=__version__,
        auth_enabled=settings.auth_enabled,
        default_model=settings.default_model if available else None,
        available_models=available,
        loaded_models=engine.loaded_keys,
        frontend=frontend_info(),
    )


@app.get(
    "/api/v1/models",
    response_model=ModelListResponse,
    tags=["tts"],
    dependencies=[Depends(require_api_key)],
)
def list_models() -> ModelListResponse:
    """What the backend may ask for, and what is already warm."""
    infos: List[ModelInfo] = []
    for key in engine.model_keys:
        entry = engine.entry_for(key)
        loaded = engine.peek(key)
        infos.append(
            ModelInfo(
                key=key,
                description=entry.description,
                is_default=(key == settings.default_model),
                is_loaded=loaded is not None,
                num_speakers=loaded.num_speakers if loaded else None,
                sample_rate=loaded.sample_rate if loaded else None,
                checkpoint=loaded.checkpoint_name if loaded else entry.checkpoint,
            )
        )
    return ModelListResponse(default_model=settings.default_model, models=infos)


# ---------------------------------------------------------------------------
# Text and speech endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/phonemes",
    response_model=PhonemeResponse,
    tags=["tts"],
    dependencies=[Depends(require_api_key)],
)
def phonemes(request: TextRequest) -> PhonemeResponse:
    """Normalisation + grapheme-to-phoneme only, with no acoustic model involved."""
    result = engine.to_phonemes(request.text.strip())
    return PhonemeResponse(
        raw_input=result.raw_input,
        normalized_text=result.normalized_text,
        ipa_sequence=result.ipa_sequence,
    )


@app.post(
    "/api/v1/synthesize",
    tags=["tts"],
    dependencies=[Depends(require_api_key)],
    responses={
        200: {"content": {"audio/wav": {}}, "description": "The generated clip."},
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def synthesize(request: SynthesisRequest) -> Response:
    """
    Turn text into speech and return the raw WAV bytes.

    Everything the caller needs for its own bookkeeping — duration, how long the
    render took, which weights produced it — travels in response headers, so the
    body stays a plain audio file the backend can forward or store as is. The
    normalised text and the phoneme string are percent-encoded because header
    values cannot carry Sinhala or IPA characters directly.
    """
    result = engine.synthesize(
        text=request.text,
        model_key=request.model,
        speaker_id=request.speaker_id,
        speed=request.speed,
    )

    headers = {
        "X-Model-Key": result.model_key,
        "X-Model-Version": result.model_version,
        "X-Sample-Rate": str(result.sample_rate),
        "X-Audio-Duration": f"{result.duration_seconds:.3f}",
        "X-Processing-Time": f"{result.processing_seconds:.3f}",
        "X-Normalized-Text": quote(result.normalized_text),
        "X-Ipa-Sequence": quote(result.ipa_sequence),
        "Content-Disposition": 'inline; filename="voicelk.wav"',
    }
    if result.speaker_id is not None:
        headers["X-Speaker-Id"] = str(result.speaker_id)

    return Response(content=result.wav_bytes, media_type="audio/wav", headers=headers)


if __name__ == "__main__":  # pragma: no cover - convenience for local runs
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
