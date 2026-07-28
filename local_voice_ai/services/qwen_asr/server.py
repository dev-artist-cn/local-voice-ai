"""OpenAI-compatible STT server backed by Qwen3-ASR (multilingual, Chinese support).

Exposes the same interface as the nemotron/whisper services:
  - POST /v1/audio/transcriptions → {"text": ...} (or SSE deltas)
  - GET  /v1/models               → list of one model
  - GET  /health                  → readiness probe

The model is loaded once at startup and reused across requests.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

logger = logging.getLogger("qwen-asr")
logging.basicConfig(level=logging.INFO)

MODEL_NAME = os.getenv("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
DEVICE = os.getenv("DEVICE", "cuda")

_model = None


def _load_model() -> None:
    global _model
    from qwen_asr import Qwen3ASRModel

    device = "cuda:0" if DEVICE == "cuda" and torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    logger.info("loading %s (device=%s, dtype=%s)", MODEL_NAME, device, dtype)
    _model = Qwen3ASRModel.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        device_map=device,
        max_new_tokens=256,
    )
    logger.info("qwen3-asr model ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="Qwen3-ASR STT Server", lifespan=lifespan)


def _sse_generator(text: str):
    """Emit text as a single SSE delta + done event."""
    event = {"type": "transcript.text.delta", "delta": text}
    yield f"data: {json.dumps(event)}\n\n"
    done_event = {"type": "transcript.text.done", "text": text.strip()}
    yield f"data: {json.dumps(done_event)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(None),
    response_format: str | None = Form("json"),
    stream: str | None = Form(None),
    language: str | None = Form(None),
    temperature: str | None = Form(None),
    prompt: str | None = Form(None),
):
    del model, temperature, prompt

    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    is_stream = stream is not None and stream.lower() in ("true", "1", "yes")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    try:
        # Decode audio to numpy array
        data, sr = sf.read(io.BytesIO(audio_bytes))
        if data.ndim > 1:
            data = data[:, 0]  # mono
        data = np.asarray(data, dtype=np.float32)
    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Audio decode failed: {error}"
        ) from error

    try:
        # Map language names: "zh"/"chinese" → "Chinese", etc.
        lang = None
        if language:
            lang_map = {
                "zh": "Chinese", "chinese": "Chinese", "中文": "Chinese",
                "en": "English", "english": "English",
                "ja": "Japanese", "japanese": "Japanese",
                "ko": "Korean", "korean": "Korean",
            }
            lang = lang_map.get(language.lower(), language.capitalize())

        results = _model.transcribe(audio=[(data, sr)], language=lang)
        text = results[0].text.strip()
    except Exception as error:
        logger.exception("Transcription failed")
        raise HTTPException(
            status_code=500, detail=f"Transcription failed: {error}"
        ) from error

    if is_stream:
        return StreamingResponse(
            _sse_generator(text),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if response_format == "text":
        return PlainTextResponse(content=text)

    return JSONResponse(content={"text": text})


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": "qwen3-asr-1.7b",
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
    parser = argparse.ArgumentParser(description="Qwen3-ASR STT Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
