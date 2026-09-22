"""
The synthesis engine: text front-end plus the trained VITS acoustic models.

Loading a checkpoint costs several seconds and a few hundred megabytes, so each
model is loaded at most once per process and then reused. Every registered model
keeps its own lock because a single torch module is not safe to run from several
threads at once; different models can still synthesize in parallel.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional

from .bootstrap import ensure_ml_paths
from .config import ModelEntry, resolve_checkpoint, settings

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors — each carries the HTTP status the API layer should report
# ---------------------------------------------------------------------------

class BridgeError(RuntimeError):
    status_code = 500


class UnknownModelError(BridgeError):
    status_code = 404


class ModelLoadError(BridgeError):
    status_code = 503


class InvalidRequestError(BridgeError):
    status_code = 400


class SynthesisError(BridgeError):
    status_code = 500


# ---------------------------------------------------------------------------
# Heavy imports, done once and only when they are actually needed
# ---------------------------------------------------------------------------

_deps: Optional[SimpleNamespace] = None
_deps_lock = threading.Lock()


def load_dependencies() -> SimpleNamespace:
    """Import torch, the vendored TTS package and the text pipeline exactly once."""
    global _deps
    if _deps is not None:
        return _deps

    with _deps_lock:
        if _deps is not None:
            return _deps

        ensure_ml_paths()
        started = time.perf_counter()
        try:
            import torch
            from TTS.tts.configs.vits_config import VitsConfig
            from TTS.tts.models.vits import Vits
            from TTS.tts.utils.text.tokenizer import TTSTokenizer
            from TTS.utils.audio import AudioProcessor
            from pipeline import TextProcessingPipeline
        except Exception as exc:  # pragma: no cover - environment problem, not logic
            raise ModelLoadError(
                "Could not import the ML stack. Check that the bridge runs in the Python "
                f"environment that has torch and the NLP dependencies installed. Cause: {exc}"
            ) from exc

        _deps = SimpleNamespace(
            torch=torch,
            VitsConfig=VitsConfig,
            Vits=Vits,
            TTSTokenizer=TTSTokenizer,
            AudioProcessor=AudioProcessor,
            TextProcessingPipeline=TextProcessingPipeline,
        )
        LOGGER.info("ML stack imported in %.2fs.", time.perf_counter() - started)
        return _deps


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrontendResult:
    raw_input: str
    normalized_text: str
    ipa_sequence: str


@dataclass(frozen=True)
class SynthesisResult:
    wav_bytes: bytes
    sample_rate: int
    duration_seconds: float
    processing_seconds: float
    normalized_text: str
    ipa_sequence: str
    model_key: str
    model_version: str
    speaker_id: Optional[int]


# ---------------------------------------------------------------------------
# Text front-end (normalizer + code-switched G2P)
# ---------------------------------------------------------------------------

class TextFrontend:
    """Thread-safe wrapper around the project's normalizer/G2P pipeline."""

    def __init__(self) -> None:
        self._pipeline = None
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        self._get_pipeline()

    def _get_pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    deps = load_dependencies()
                    started = time.perf_counter()
                    self._pipeline = deps.TextProcessingPipeline()
                    LOGGER.info("Text pipeline ready in %.2fs.", time.perf_counter() - started)
        return self._pipeline

    def process(self, text: str) -> FrontendResult:
        pipeline = self._get_pipeline()
        try:
            with self._lock:
                result = pipeline.process(text)
        except Exception as exc:
            raise SynthesisError(f"Text processing failed: {exc}") from exc

        return FrontendResult(
            raw_input=result["raw_input"],
            normalized_text=result["normalized_text"],
            ipa_sequence=result["ipa_sequence"],
        )


# ---------------------------------------------------------------------------
# One loaded acoustic model
# ---------------------------------------------------------------------------

class LoadedModel:
    """A VITS checkpoint held in memory together with its tokenizer and audio processor."""

    def __init__(self, entry: ModelEntry) -> None:
        self.entry = entry
        self.lock = threading.Lock()

        deps = load_dependencies()
        config_path = entry.config_path
        if not config_path.is_file():
            raise ModelLoadError(
                f"Model '{entry.key}' has no config file next to its checkpoint — a checkpoint "
                "cannot be loaded reliably without the config from the same training run."
            )

        try:
            checkpoint_path = resolve_checkpoint(entry.directory, entry.checkpoint)
        except FileNotFoundError as exc:
            raise ModelLoadError(str(exc)) from exc

        started = time.perf_counter()
        try:
            config = deps.VitsConfig()
            config.load_json(str(config_path))

            audio_processor = deps.AudioProcessor.init_from_config(config)
            tokenizer, config = deps.TTSTokenizer.init_from_config(config)

            # speaker_manager stays None on purpose: speakers are addressed by raw
            # embedding index, exactly as the offline synthesis script does, because the
            # name->id map is not part of every training run.
            model = deps.Vits(config, audio_processor, tokenizer, speaker_manager=None)
            model.load_checkpoint(config, str(checkpoint_path), eval=True)
            model.eval()
        except Exception as exc:
            raise ModelLoadError(f"Loading model '{entry.key}' failed: {exc}") from exc

        self.config = config
        self.audio_processor = audio_processor
        self.tokenizer = tokenizer
        self.model = model
        self.checkpoint_name = checkpoint_path.name
        self.base_length_scale = float(getattr(model, "length_scale", 1.0))
        self.uses_speaker_embedding = bool(getattr(config.model_args, "use_speaker_embedding", False))
        self.num_speakers = int(getattr(config.model_args, "num_speakers", 0) or 0)
        self.sample_rate = int(audio_processor.sample_rate)

        LOGGER.info(
            "Model '%s' loaded in %.2fs (checkpoint %s, %d speaker(s), %d Hz).",
            entry.key,
            time.perf_counter() - started,
            self.checkpoint_name,
            self.num_speakers,
            self.sample_rate,
        )

    @property
    def version(self) -> str:
        """Human-readable identity of the weights, stored with every generated clip."""
        return f"{self.entry.key}:{self.entry.directory.name}:{self.checkpoint_name}"

    def synthesize(self, ipa_text: str, speaker_id: int, speed: float) -> tuple[bytes, float]:
        """Render an IPA string to WAV bytes; returns the bytes and the audio duration."""
        deps = load_dependencies()
        torch = deps.torch

        token_ids = self.tokenizer.text_to_ids(ipa_text)
        if not token_ids:
            raise InvalidRequestError(
                "The text produced no phonemes this model recognises — check that the input "
                "contains actual Sinhala or English words."
            )

        speaker_tensor = None
        if self.uses_speaker_embedding:
            if self.num_speakers and speaker_id >= self.num_speakers:
                raise InvalidRequestError(
                    f"Model '{self.entry.key}' has {self.num_speakers} speaker(s); "
                    f"speaker index {speaker_id} is out of range."
                )
            speaker_tensor = torch.LongTensor([speaker_id])

        tokens = torch.LongTensor(token_ids).unsqueeze(0)
        aux_input = {
            "x_lengths": None,
            "d_vectors": None,
            "language_ids": None,
            "durations": None,
            "speaker_ids": speaker_tensor,
        }

        with self.lock:
            previous_length_scale = self.model.length_scale
            # length_scale stretches the predicted durations, so a faster voice is a
            # smaller scale — the request asks for speed, the model wants length.
            self.model.length_scale = self.base_length_scale / speed
            try:
                with torch.no_grad():
                    outputs = self.model.inference(tokens, aux_input=aux_input)
            except Exception as exc:
                raise SynthesisError(f"Inference failed: {exc}") from exc
            finally:
                self.model.length_scale = previous_length_scale

        waveform = outputs["model_outputs"][0, 0].cpu().numpy()
        duration = float(len(waveform)) / float(self.sample_rate)

        buffer = io.BytesIO()
        # The audio processor writes through scipy, which accepts a file-like target,
        # so the clip never has to touch the disk.
        self.audio_processor.save_wav(waveform, buffer)
        return buffer.getvalue(), duration


# ---------------------------------------------------------------------------
# Engine — registry of models plus the shared text front-end
# ---------------------------------------------------------------------------

class SynthesisEngine:
    def __init__(self, entries: Dict[str, ModelEntry]) -> None:
        self._entries = entries
        self._models: Dict[str, LoadedModel] = {}
        self._load_locks: Dict[str, threading.Lock] = {key: threading.Lock() for key in entries}
        self.frontend = TextFrontend()

    # -- registry ----------------------------------------------------------

    @property
    def model_keys(self) -> List[str]:
        return [key for key, entry in self._entries.items() if entry.is_available]

    @property
    def loaded_keys(self) -> List[str]:
        return sorted(self._models)

    def entry_for(self, key: Optional[str]) -> ModelEntry:
        resolved = key or settings.default_model
        entry = self._entries.get(resolved)
        if entry is None or not entry.is_available:
            known = ", ".join(self.model_keys) or "none"
            raise UnknownModelError(f"Unknown model '{resolved}'. Available models: {known}.")
        return entry

    def peek(self, key: str) -> Optional[LoadedModel]:
        """The loaded model for a key, or None when it has not been loaded yet."""
        return self._models.get(key)

    def get(self, key: Optional[str] = None) -> LoadedModel:
        entry = self.entry_for(key)
        loaded = self._models.get(entry.key)
        if loaded is not None:
            return loaded

        with self._load_locks.setdefault(entry.key, threading.Lock()):
            loaded = self._models.get(entry.key)
            if loaded is None:
                loaded = LoadedModel(entry)
                self._models[entry.key] = loaded
        return loaded

    # -- work --------------------------------------------------------------

    def to_phonemes(self, text: str) -> FrontendResult:
        return self.frontend.process(text)

    def synthesize(
        self,
        text: str,
        model_key: Optional[str] = None,
        speaker_id: int = 0,
        speed: float = 1.0,
    ) -> SynthesisResult:
        cleaned = (text or "").strip()
        if not cleaned:
            raise InvalidRequestError("The text field is required and must not be empty.")

        started = time.perf_counter()
        model = self.get(model_key)
        front = self.frontend.process(cleaned)
        wav_bytes, duration = model.synthesize(front.ipa_sequence, speaker_id, speed)
        processing_seconds = time.perf_counter() - started

        LOGGER.info(
            "Synthesized %.2fs of audio in %.2fs with '%s'.",
            duration,
            processing_seconds,
            model.entry.key,
        )

        return SynthesisResult(
            wav_bytes=wav_bytes,
            sample_rate=model.sample_rate,
            duration_seconds=duration,
            processing_seconds=processing_seconds,
            normalized_text=front.normalized_text,
            ipa_sequence=front.ipa_sequence,
            model_key=model.entry.key,
            model_version=model.version,
            speaker_id=speaker_id if model.uses_speaker_embedding else None,
        )

    def warm_up(self) -> None:
        """Pay the import and load cost at startup instead of on the first request."""
        self.frontend.warm_up()
        if settings.default_model in self._entries:
            self.get(settings.default_model)


engine = SynthesisEngine(settings.models)
