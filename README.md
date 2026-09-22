# VoiceLK TTS Bridge

FastAPI microservice that turns Sinhala/English text into speech with the project's
trained VITS model, so the Spring Boot backend can voice an answer without hosting
Python itself.

```
Browser  ──►  Spring Boot backend  ──►  TTS Bridge  ──►  VITS checkpoint
                      │                      │
                      │                      └── normalizer + code-switched G2P
                      └── object storage + audio records
```

Responsibilities are split on purpose:

- **The bridge** only turns text into WAV bytes. No database, no storage credentials.
- **The backend** owns authentication, object storage and the database record.

## Self-contained by design

This service is deployable on its own: everything it needs to speak is inside the
checkout.

| What | Where | Why it is here |
| --- | --- | --- |
| Text front-end (normaliser, G2P, lexicon) | `vendor/model_engine/` | The phonemes must be produced exactly as they were during training |
| Coqui TTS library | `vendor/TTS/` | Shipped as source, not a pip package, like everywhere else in this project |
| Trainer package | `vendor/trainer/` | Imported by the model code |

**The research repository stays the source of truth** for the front-end: dataset
preparation, vocabulary building, training and evaluation all use the same code
there. The copy here is taken from it with the sync tool, never edited by hand.

Drift between the two is the one failure that hides itself — change a lexicon entry
after a model was trained and it is fed phonemes it never saw, so it simply
mispronounces words. Two things guard against it:

```bash
python tools/sync_from_ml.py --ml-root ../voicelk_ml          # refresh the copy
python tools/sync_from_ml.py --ml-root ../voicelk_ml --check  # fail if it is stale
```

The sync records a fingerprint of the front-end; the service logs it at startup and
reports it under `frontend` in the health response, with status `clean`, `drifted`
or `unstamped`.

## Checkpoints

Checkpoints are **not** in git — they are hundreds of megabytes each. Put each
training run (its `config.json` plus one checkpoint) in a folder under `models/`, or
anywhere else and set `VOICELK_MODELS_ROOT`. See `models/README.md`, and use
`tools/slim_checkpoint.py` to drop the optimizer state before copying a run to a
server.

Two model keys are registered — `pathnirwana` and `custom` — each resolving to the
newest matching run folder unless pinned through environment variables. Requests
choose between them.

## Running it

Copy `.env.example` to `.env` first and set at least `VOICELK_BRIDGE_API_KEY`.

**Server, with Docker**

```bash
MODELS_DIR=/srv/voicelk/models VOICELK_BRIDGE_API_KEY=… docker compose up -d --build
```

The image carries the code and the vendored sources; checkpoints are mounted at
`/models`, read-only. The port is published on loopback only, because the backend is
the single client.

**Server, without Docker**

```bash
./run_bridge.sh
```

Creates a virtual environment, installs the speech stack from the CPU wheel index and
starts the API. `deploy/voicelk-bridge.service` turns that into a systemd service.

**Windows workstation**

```powershell
.\start_bridge.ps1            # -Port, -Reload, -SkipInstall, -VenvPath
```

Reuses the research environment that already has torch installed.

Interactive API docs: <http://127.0.0.1:8000/docs>

## Checking it

```bash
python smoke_test.py                       # health, models, phonemes, one clip
python smoke_test.py --model custom --speed 0.9 --text "…"
```

It writes the generated clip next to the script so you can listen to it.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness, front-end fingerprint, which models are warm |
| `GET` | `/api/v1/models` | Registered models |
| `POST` | `/api/v1/phonemes` | Normalisation + G2P only |
| `POST` | `/api/v1/synthesize` | Text in, `audio/wav` bytes out |

Synthesis request body:

```json
{ "text": "…", "model": "pathnirwana", "speaker_id": 0, "speed": 1.0 }
```

The response body is the WAV file itself; the bookkeeping travels in headers:
`X-Model-Key`, `X-Model-Version`, `X-Sample-Rate`, `X-Audio-Duration`,
`X-Processing-Time`, `X-Speaker-Id`, plus percent-encoded `X-Normalized-Text` and
`X-Ipa-Sequence` (headers cannot carry Sinhala or IPA directly).

Errors are JSON with `status`, `error` and `detail`. Codes: `400` unusable text,
`401` bad API key, `404` unknown model key, `422` malformed body, `503` a checkpoint
that will not load.

## Configuration

| Variable | Meaning |
| --- | --- |
| `VOICELK_MODELS_ROOT` | Folder holding the training-run directories |
| `VOICELK_MODEL_<KEY>_DIR` / `_CHECKPOINT` | Pin a key to one run or file |
| `VOICELK_DEFAULT_MODEL` | Model used when a request does not name one |
| `VOICELK_PRELOAD_MODEL` | Load the default model at startup |
| `VOICELK_BRIDGE_API_KEY` | Shared secret expected in `X-API-Key` |
| `VOICELK_MAX_TEXT_CHARS` | Longest accepted input |
| `VOICELK_ML_ROOT` | Run against a research checkout instead of the bundled copy |
| `OMP_NUM_THREADS` | Torch thread count; keep it small on a shared VPS |

## Behaviour worth knowing

- The first synthesis is slow: importing torch and loading a checkpoint takes several
  seconds. `VOICELK_PRELOAD_MODEL=true` pays that at startup instead.
- Each model is loaded once per process and reused. One request at a time per model;
  different models can run concurrently.
- Run one worker. Every extra worker loads its own copy of the weights.
- No API key means an open service — acceptable on a development machine, nowhere else.
