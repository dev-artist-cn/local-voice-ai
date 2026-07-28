"""OpenAI-compatible TTS server backed by CosyVoice 2.

CosyVoice2 uses zero-shot voice cloning: it needs a reference audio clip
to copy the speaker's voice. The reference is set via COSYVOICE_PROMPT_WAV
and COSYVOICE_PROMPT_TEXT env vars.

Exposes only what livekit.plugins.openai.TTS needs:
  - POST /v1/audio/speech  → audio bytes
  - GET  /v1/models        → list of one model
  - GET  /health           → readiness probe
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

logger = logging.getLogger("cosyvoice")
logging.basicConfig(level=logging.INFO)

MODEL_DIR = os.getenv("COSYVOICE_MODEL_DIR", "")
PROMPT_WAV = os.getenv("COSYVOICE_PROMPT_WAV", "")
PROMPT_TEXT = os.getenv("COSYVOICE_PROMPT_TEXT", "希望你以后能够做的比我还好呦。")
DEVICE = os.getenv("DEVICE", "cuda")
FP16 = os.getenv("COSYVOICE_FP16", "1").lower() in ("1", "true", "yes")
SAMPLE_RATE = 24000

_model = None


def _load_model() -> None:
    global _model

    # Patch torchaudio.load: torchaudio 2.11+ requires torchcodec (ffmpeg),
    # which isn't available. Fall back to soundfile.
    import torchaudio

    def _patched_load(filepath, backend=None, **kwargs):
        data, sr = sf.read(filepath, dtype="float32")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return torch.from_numpy(data), sr

    torchaudio.load = _patched_load

    # CosyVoice repo must be on the path
    cosyvoice_repo = os.getenv("COSYVOICE_REPO", "/home/wei/CosyVoice")
    for p in [cosyvoice_repo, os.path.join(cosyvoice_repo, "third_party", "Matcha-TTS")]:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    from cosyvoice.cli.cosyvoice import CosyVoice2

    logger.info("loading CosyVoice2 from %s (fp16=%s)", MODEL_DIR, FP16)
    _model = CosyVoice2(MODEL_DIR, load_jit=False, fp16=FP16)
    logger.info("CosyVoice2 ready (sr=%d)", _model.sample_rate)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="CosyVoice2 TTS Server", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0


def _synthesize(text: str, speed: float) -> np.ndarray:
    if _model is None:
        raise RuntimeError("model not loaded")
    chunks = []
    for chunk in _model.inference_zero_shot(
        text, PROMPT_TEXT, PROMPT_WAV, stream=False, speed=speed
    ):
        audio = chunk["tts_speech"]
        if hasattr(audio, "cpu"):
            audio = audio.cpu().numpy()
        chunks.append(np.asarray(audio, dtype=np.float32).squeeze())
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def _encode(audio: np.ndarray, fmt: str) -> tuple[bytes, str]:
    fmt = (fmt or "mp3").lower()
    buf = io.BytesIO()
    if fmt in {"mp3", "opus", "aac", "flac"}:
        try:
            sf.write(buf, audio, SAMPLE_RATE, format=fmt.upper())
            return buf.getvalue(), f"audio/{fmt}"
        except Exception:
            buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "audio/wav"


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest) -> Response:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.input:
        raise HTTPException(status_code=400, detail="input is required")
    try:
        audio = _synthesize(req.input, float(req.speed or 1.0))
    except Exception as exc:
        logger.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    data, media_type = _encode(audio, req.response_format or "mp3")
    return Response(content=data, media_type=media_type)


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": "cosyvoice2",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "alibaba",
                }
            ],
        }
    )


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CosyVoice2 TTS Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8880)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
