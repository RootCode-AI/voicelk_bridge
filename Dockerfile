# VoiceLK TTS Bridge — CPU image.
#
# Checkpoints are deliberately NOT baked in: they are nearly a gigabyte each, they
# change independently of the code, and an image that carries them cannot be rebuilt
# cheaply. Mount them at /models instead (see the compose file).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VOICELK_MODELS_ROOT=/models \
    VOICELK_BRIDGE_HOST=0.0.0.0 \
    VOICELK_BRIDGE_PORT=8000

# libsndfile is what soundfile binds to; git is needed by some wheels' build steps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/bridge

# Dependencies first, so editing the service code does not reinstall torch.
COPY requirements.txt requirements-ml.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-ml.txt -r requirements.txt

# Application code plus the vendored speech sources.
COPY app ./app
COPY vendor ./vendor
COPY tools ./tools
COPY smoke_test.py ./

RUN useradd --create-home --uid 10001 voicelk \
    && mkdir -p /models \
    && chown -R voicelk:voicelk /srv/bridge /models
USER voicelk

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# One worker on purpose: every worker loads its own copy of the checkpoints, and a
# second copy costs more memory than it buys throughput on a small server.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
