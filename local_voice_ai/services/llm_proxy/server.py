"""OpenAI-compatible LLM proxy that disables thinking on ollama.

Gemma 4 is a thinking model — it generates a long `reasoning` field before the
actual reply, adding seconds of dead air to voice conversations. Ollama 0.32's
OpenAI endpoint (/v1/chat/completions) doesn't support disabling thinking for
the gemma4 renderer, but the native endpoint (/api/chat) does via ``think:false``.

This proxy sits between the LiveKit ``openai.LLM`` plugin and ollama:
  LiveKit ─POST /v1/chat/completions──▶ proxy ─POST /api/chat (think:false)──▶ ollama

It translates streaming responses from ollama's JSON stream to OpenAI's SSE
format so the voice agent still gets token-by-token streaming for low latency.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("llm-proxy")
logging.basicConfig(level=logging.INFO)

BACKEND_URL = os.getenv("LLM_PROXY_BACKEND", "http://127.0.0.1:11434")
MODEL_MAP = {  # allow alias → real model name
    os.getenv("LLM_PROXY_MODEL_ALIAS", ""): os.getenv("LLM_PROXY_MODEL", ""),
}

app = FastAPI(title="LLM Thinking Proxy")


def _resolve_model(model: str) -> str:
    return MODEL_MAP.get(model, model)


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_ollama_as_openai(payload: dict) -> AsyncIterator[str]:
    """Convert ollama's /api/chat JSON stream into OpenAI SSE chunks."""
    created = int(time.time())
    model = payload["model"]

    # First chunk: role announcement (OpenAI convention)
    yield _sse({
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    })

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        async with client.stream(
            "POST",
            f"{BACKEND_URL}/api/chat",
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                content = msg.get("content", "")
                done = chunk.get("done", False)

                if content:
                    yield _sse({
                        "id": "chatcmpl-proxy",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }],
                    })

                if done:
                    yield _sse({
                        "id": "chatcmpl-proxy",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": chunk.get("done_reason", "stop"),
                        }],
                    })
                    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    body = await request.json()
    stream = body.get("stream", False)
    model = _resolve_model(body.get("model", ""))

    ollama_payload: dict[str, Any] = {
        "model": model,
        "messages": body["messages"],
        "stream": True,
        "think": False,  # ← the whole point
    }
    # Pass through supported sampling params
    for key in ("temperature", "top_p", "top_k", "max_tokens", "seed"):
        if key in body and body[key] is not None:
            ollama_payload[key] = body[key]
    if "tools" in body:
        ollama_payload["tools"] = body["tools"]
    if "tool_choice" in body:
        ollama_payload["tool_choice"] = body["tool_choice"]

    if stream:
        return StreamingResponse(
            _stream_ollama_as_openai(ollama_payload),
            media_type="text/event-stream",
        )

    # Non-streaming: collect then return one OpenAI-shaped response
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        resp = await client.post(
            f"{BACKEND_URL}/api/chat",
            json={**ollama_payload, "stream": False},
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})
        return JSONResponse({
            "id": "chatcmpl-proxy",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": msg.get("content", ""),
                },
                "finish_reason": data.get("done_reason", "stop"),
            }],
            "usage": {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            },
        })


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    """Forward ollama's model list in OpenAI shape (for readiness probes)."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
        resp = await client.get(f"{BACKEND_URL}/api/tags")
        resp.raise_for_status()
        data = resp.json()
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": m["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama",
            }
            for m in data.get("models", [])
        ],
    })


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Thinking Proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11435)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
