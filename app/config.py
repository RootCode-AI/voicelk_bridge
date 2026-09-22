"""
Runtime configuration for the VoiceLK TTS Bridge.

Everything is driven by environment variables (loaded from the service's .env file
when python-dotenv is available) so the same code runs unchanged on a laptop and on
a server where the model files live somewhere else.

The service is self-contained: the text front-end and the TTS library it needs are
vendored beside the application code, so a deployment is one checkout plus the
checkpoints. Pointing VOICELK_ML_ROOT at a research checkout overrides the bundled
copies, which is convenient while developing the pipeline itself.

Nothing here imports torch or the vendored TTS package — this module stays cheap
so that `--reload` restarts and `/health` probes are fast.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# bridge/app/config.py -> bridge/ -> project root (the folder holding voicelk_ml, Models, ...)
BRIDGE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BRIDGE_DIR.parent

try:  # optional — the service runs fine without a .env file
    from dotenv import load_dotenv

    load_dotenv(BRIDGE_DIR / ".env")
except ImportError:  # pragma: no cover - dotenv is a convenience, not a requirement
    pass


# ---------------------------------------------------------------------------
# Small env helpers
# ---------------------------------------------------------------------------

def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return _env_str(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    raw = _env_str(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_list(name: str, default: List[str]) -> List[str]:
    raw = _env_str(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelEntry:
    """One trained VITS run the bridge is allowed to serve."""

    key: str
    directory: Path
    checkpoint: Optional[str] = None
    description: str = ""

    @property
    def config_path(self) -> Path:
        return self.directory / "config.json"

    @property
    def is_available(self) -> bool:
        return self.config_path.is_file()


_STEP_PATTERN = re.compile(r"(\d+)")


def _step_of(filename: str) -> int:
    """Training step encoded in a checkpoint filename (0 when there is none)."""
    matches = _STEP_PATTERN.findall(filename)
    return int(matches[-1]) if matches else 0


def resolve_checkpoint(directory: Path, explicit: Optional[str] = None) -> Path:
    """
    Pick the checkpoint to load from a training-run folder.

    Preference order: an explicitly configured filename, then `best_model.pth`,
    then the highest-step `best_model_*.pth`, then the highest-step
    `checkpoint_*.pth`. Speaker-map files are never treated as checkpoints.
    """
    if explicit:
        path = directory / explicit
        if not path.is_file():
            raise FileNotFoundError(f"Configured checkpoint '{explicit}' does not exist in '{directory}'.")
        return path

    candidates = [
        f.name
        for f in directory.glob("*.pth")
        if "speakers" not in f.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint (.pth) file found in '{directory}'.")

    if "best_model.pth" in candidates:
        return directory / "best_model.pth"

    best_models = [name for name in candidates if name.startswith("best_model")]
    pool = best_models or candidates
    pool.sort(key=_step_of)
    return directory / pool[-1]


def _discover_model_dir(models_root: Path, *patterns: str) -> Optional[Path]:
    """Newest folder under the models root matching any of the given patterns."""
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(d for d in models_root.glob(pattern) if d.is_dir())
    if not matches:
        return None
    matches.sort(key=lambda d: d.stat().st_mtime)
    return matches[-1]


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    vendor_root: Path
    ml_root: Optional[Path]
    models_root: Path
    models: Dict[str, ModelEntry]
    default_model: str
    api_key: str = ""
    max_text_chars: int = 1200
    preload_default_model: bool = False
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    service_name: str = "VoiceLK TTS Bridge"

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def available_models(self) -> Dict[str, ModelEntry]:
        return {key: entry for key, entry in self.models.items() if entry.is_available}

    @property
    def uses_bundled_sources(self) -> bool:
        """True when the service runs on its own vendored copy of the ML code."""
        return self.ml_root is None

    @property
    def frontend_dir(self) -> Path:
        """Folder holding the normaliser, the G2P engine and the lexicon in use."""
        if self.ml_root is not None:
            return self.ml_root / "model_engine"
        return self.vendor_root / "model_engine"

    @property
    def library_dir(self) -> Path:
        """Folder from which the TTS and trainer packages are imported."""
        if self.ml_root is not None:
            return self.ml_root / "model_training"
        return self.vendor_root


def _build_models(models_root: Path) -> Dict[str, ModelEntry]:
    """
    Registry of servable models.

    Each key can be pinned with VOICELK_MODEL_<KEY>_DIR / _CHECKPOINT; otherwise
    the newest matching training-run folder under the models root wins.
    """
    specs = [
        (
            "pathnirwana",
            ("voicelk_vits_pathnirwana*", "pathnirwana*"),
            "VITS trained on the Pathnirwana corpus (multi-speaker).",
        ),
        (
            "custom",
            ("voicelk_vits_custom*", "custom"),
            "VITS trained on the project's own curated corpus.",
        ),
    ]

    models: Dict[str, ModelEntry] = {}
    for key, patterns, description in specs:
        env_prefix = f"VOICELK_MODEL_{key.upper()}"
        configured = _env_str(f"{env_prefix}_DIR")
        directory = (
            Path(configured).expanduser().resolve()
            if configured
            else _discover_model_dir(models_root, *patterns)
        )
        if directory is None:
            continue
        models[key] = ModelEntry(
            key=key,
            directory=directory,
            checkpoint=_env_str(f"{env_prefix}_CHECKPOINT") or None,
            description=description,
        )
    return models


def _default_models_root() -> Path:
    """
    Where checkpoints live when nothing is configured.

    A folder inside the service wins — that is the deployment shape, where the
    checkpoints are copied or mounted next to the code. Otherwise the shared models
    folder of the development workspace is used, so a local checkout keeps working
    without any configuration.
    """
    bundled = BRIDGE_DIR / "models"
    # Only a folder that actually holds run directories counts; the folder itself
    # ships with nothing but its instructions.
    if bundled.is_dir() and any(child.is_dir() for child in bundled.iterdir()):
        return bundled
    workspace = PROJECT_ROOT / "Models"
    return workspace if workspace.is_dir() else bundled


def load_settings() -> Settings:
    # Set only when the service should run against a research checkout instead of
    # its own vendored copies.
    configured_ml_root = _env_str("VOICELK_ML_ROOT")
    ml_root = Path(configured_ml_root).expanduser().resolve() if configured_ml_root else None

    models_root = _env_path("VOICELK_MODELS_ROOT", _default_models_root())
    models = _build_models(models_root)

    default_model = _env_str("VOICELK_DEFAULT_MODEL", "pathnirwana")
    if default_model not in models and models:
        default_model = next(iter(models))

    return Settings(
        vendor_root=BRIDGE_DIR / "vendor",
        ml_root=ml_root,
        models_root=models_root,
        models=models,
        default_model=default_model,
        api_key=_env_str("VOICELK_BRIDGE_API_KEY"),
        max_text_chars=_env_int("VOICELK_MAX_TEXT_CHARS", 1200),
        preload_default_model=_env_bool("VOICELK_PRELOAD_MODEL", False),
        allowed_origins=_env_list("VOICELK_ALLOWED_ORIGINS", ["*"]),
        host=_env_str("VOICELK_BRIDGE_HOST", "0.0.0.0"),
        port=_env_int("VOICELK_BRIDGE_PORT", 8000),
        log_level=_env_str("VOICELK_LOG_LEVEL", "info"),
    )


settings = load_settings()
