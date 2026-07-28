"""OpenAI-compatible TTS server backed by Qwen3-TTS-1.7B-CustomVoice.

Uses preset speakers with instruction-based emotion control.
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

logger = logging.getLogger("qwen-tts")
logging.basicConfig(level=logging.INFO)

MODEL_NAME = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEFAULT_SPEAKER = os.getenv("QWEN_TTS_SPEAKER", "Vivian")
DEFAULT_LANGUAGE = os.getenv("QWEN_TTS_LANGUAGE", "Auto")
DEFAULT_INSTRUCT = os.getenv("QWEN_TTS_INSTRUCT", "")
SAMPLE_RATE = 24000

_model = None


def _load_model() -> None:
    global _model
    from qwen_tts import Qwen3TTSModel

    logger.info("loading %s (speaker=%s)", MODEL_NAME, DEFAULT_SPEAKER)
    _model = Qwen3TTSModel.from_pretrained(
        MODEL_NAME,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    logger.info("qwen3-tts ready (sr=%d)", SAMPLE_RATE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Qwen3-TTS Server", lifespan=lifespan)


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "mp3"
    speed: Optional[float] = 1.0


def _synthesize(text: str, voice: str) -> np.ndarray:
    if _model is None:
        raise RuntimeError("model not loaded")
    speaker = voice or DEFAULT_SPEAKER
    kwargs = dict(text=text, language=DEFAULT_LANGUAGE, speaker=speaker)
    if DEFAULT_INSTRUCT:
        kwargs["instruct"] = DEFAULT_INSTRUCT
    wavs, sr = _model.generate_custom_voice(**kwargs)
    audio = np.asarray(wavs[0], dtype=np.float32)
    return audio, sr


def _encode(audio: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    fmt = (fmt or "mp3").lower()
    buf = io.BytesIO()
    if fmt in {"mp3", "opus", "aac", "flac"}:
        try:
            sf.write(buf, audio, sr, format=fmt.upper())
            return buf.getvalue(), f"audio/{fmt}"
        except Exception:
            buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "audio/wav"


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest) -> Response:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    if not req.input:
        raise HTTPException(status_code=400, detail="input is required")
    try:
        audio, sr = _synthesize(req.input, req.voice or "")
    except Exception as exc:
        logger.exception("synthesis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    data, media_type = _encode(audio, sr, req.response_format or "mp3")
    return Response(content=data, media_type=media_type)


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": "qwen3-tts",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "qwen",
                }
            ],
        }
    )


@app.get("/health")
async def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": _model is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8880)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
