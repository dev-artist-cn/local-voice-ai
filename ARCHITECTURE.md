# Local Voice AI — Architecture

## Overview

A single Python supervisor process (`python -m local_voice_ai serve`) spawns and
supervises all child services needed for a real-time voice AI agent. The
supervisor is a mini-orchestrator — think Docker Compose, but in pure async
Python.

```
┌─ python -m local_voice_ai serve (parent/supervisor) ─────────────────┐
│                                                                      │
│  ┌─ child: livekit-server (Go binary) ────────────────────────────┐  │
│  │  WebRTC signaling & media relay (ports 7883/7881/7882)         │  │
│  │  Internal only — 127.0.0.1:7883, plain ws://                   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─ child: wss-proxy (Python, raw TCP) ───────────────────────────┐  │
│  │  TLS terminates WSS on 0.0.0.0:7880 → forwards to :7883        │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─ child: llama-server (C++ binary) [optional] ──────────────────┐  │
│  │  LLM inference, OpenAI-compatible /v1 endpoint (port 11434)    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─ child: nemotron server (Python uvicorn) ──────────────────────┐  │
│  │  STT — speech-to-text, OpenAI-compatible (port 8000)           │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─ child: kokoro server (Python uvicorn) ────────────────────────┐  │
│  │  TTS — text-to-speech, OpenAI-compatible (port 8880)           │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─ child: agent worker (Python, LiveKit Agents) ─────────────────┐  │
│  │  Orchestrator: wires STT → LLM → TTS pipeline together         │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌─ in-process: FastAPI (port 8083, HTTPS) ───────────────────────┐  │
│  │  Serves frontend + API endpoints for token minting & status    │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## File Map

| File | Role |
|------|------|
| `local_voice_ai/__main__.py` | Entry point. Parses CLI, builds `ChildSpec` list from `Config`, runs the supervisor |
| `local_voice_ai/supervisor.py` | Async process manager — spawns children, watches stdout/stderr, polls readiness URLs, restarts crashes |
| `local_voice_ai/config.py` | Single `Config` dataclass from env vars. Auto-decides which services to manage locally vs externally |
| `local_voice_ai/api.py` | FastAPI app: `POST /api/connection-details` (token minting), `GET /api/status` (readiness), `GET /*` (static frontend) |
| `local_voice_ai/agent.py` | LiveKit Agents worker — creates `AgentSession` wired with STT, LLM, TTS |
| `local_voice_ai/wakeword.py` | Optional "hey livekit" detection via ONNX model over sliding 2s audio window |
| `local_voice_ai/wss_proxy.py` | Raw TCP proxy: TLS-terminates WSS on :7880, forwards plain bytes to livekit-server :7883 |
| `local_voice_ai/services/nemotron/server.py` | FastAPI server wrapping NVIDIA Nemotron speech model (OpenAI-compatible STT) |
| `local_voice_ai/services/whisper/server.py` | Alternative STT: faster-whisper (smaller, CPU-friendly) |
| `local_voice_ai/services/kokoro/server.py` | FastAPI server wrapping Kokoro TTS (OpenAI-compatible) |
| `frontend/` | Next.js app (statically exported, served by FastAPI) |

---

## Data Flow During a Voice Call

```
Browser (mic) ──WebRTC──▶ livekit-server ──WebSocket──▶ Agent Worker
                                                           │
                    ┌──────────────────────────────────────┤
                    │  1. Audio frames arrive              │
                    │  2. STT: POST /v1/audio/transcriptions → nemotron (8000)
                    │  3. LLM: POST /v1/chat/completions   → ollama/llama.cpp (11434)
                    │  4. TTS: POST /v1/audio/speech       → kokoro (8880)
                    │  5. Audio frames sent back            │
                    └──────────────────────────────────────┤
                                                           ▼
Browser (speaker) ◀──WebRTC──◀ livekit-server ◀── Audio frames
```

---

## Supervisor (Process Manager)

### How it works

The `Supervisor` class manages a list of `ChildSpec` objects, each describing a
subprocess to spawn. For each child it:

1. **Spawns** the process via `asyncio.create_subprocess_exec()`
2. **Pumps** stdout/stderr through the parent logger with a `[name]` prefix
3. **Polls** a readiness URL (e.g. `GET /v1/models`) with exponential backoff
4. **Watches** for unexpected exits — if a child dies *after* becoming ready,
   it's restarted with linear backoff (2s → 4s → … → 10s cap, up to 5 retries)
5. If a child dies *before* becoming ready, that's a fatal error — the
   supervisor shuts everything down

### Shutdown

On SIGTERM/SIGINT, the supervisor sends SIGTERM to all children, waits up to
10s, then sends SIGKILL. The FastAPI server shuts down in parallel.

---

## The "Manage or Skip" Decision

Each service has a `manage_X` flag. The default logic is: **manage locally if
the URL points to localhost, otherwise skip**.

```python
manage_livekit = _is_loopback(livekit_url)  # True if 127.0.0.1/localhost
manage_llama   = _is_loopback(llama_base_url)
manage_stt     = _is_loopback(stt_base_url)
manage_tts     = _is_loopback(tts_base_url)
```

Override with explicit env vars: `MANAGE_LIVEKIT=1`, `MANAGE_LLAMA=0`, etc.

This means pointing any base URL at a remote endpoint automatically disables
the local child:

| Goal | Config |
|------|--------|
| Use LiveKit Cloud | `LIVEKIT_URL=wss://your-project.livekit.cloud` |
| Use OpenAI for LLM | `LLAMA_BASE_URL=https://api.openai.com/v1` |
| Use remote STT | `STT_BASE_URL=https://...` |
| Use remote TTS | `TTS_BASE_URL=https://...` |

---

## LiveKit ↔ Agent Worker Interface

There are **two separate channels**: a persistent WebSocket for real-time job
dispatch, and an HTTP health endpoint for monitoring.

### WebSocket Protocol (Protobuf over WS)

The worker connects to `ws://<host>:7880/agent` and speaks binary Protobuf
messages over a single persistent WebSocket.

**Connection setup:**

```
Agent Worker                           LiveKit Server
    │                                        │
    │── WS connect /agent ──────────────────▶│  (JWT auth in headers)
    │                                        │
    │── WorkerMessage{register} ────────────▶│  "I'm a JT_ROOM worker,
    │   .type = JT_ROOM                       │   agent_name = '',
    │   .agent_name = ''                      │   sdk_version = '1.6.7'"
    │   .version = '1.6.7'                    │
    │                                        │
    │◀── ServerMessage{register} ────────────│  "Registered as AW_F2RFMhhKocYV"
    │   .worker_id = 'AW_F2RFMhhKocYV'       │
    │                                        │
    │── WorkerMessage{update_worker} ────────▶│  "Status: AVAILABLE, load: 0.0"
    │   .status = WS_AVAILABLE               │  (re-sent every ~2.5s)
    │   .load = 0.0                           │
```

**Job dispatch when a user clicks "Start call":**

```
User's Browser                       LiveKit Server                    Agent Worker
    │                                       │                               │
    │─ POST /api/connection-details ───────▶│                               │
    │◀─ token, roomName ───────────────────│                               │
    │                                       │                               │
    │── join room via WebRTC ──────────────▶│                               │
    │                                       │── ServerMessage ─────────────▶│
    │                                       │   {availability}              │
    │                                       │   .job.id = '...'             │
    │                                       │   .job.room.name = '...'      │
    │                                       │                               │
    │                                       │◀── WorkerMessage ────────────│
    │                                       │   {availability}              │
    │                                       │   .job_id = '...'             │
    │                                       │   .available = true           │
    │                                       │                               │
    │                                       │── ServerMessage ─────────────▶│
    │                                       │   {assignment}                │
    │                                       │   .job = {...}                │
    │                                       │   .token = '<agent-jwt>'      │
    │                                       │                               │
    │                                       │◀── WorkerMessage ────────────│
    │                                       │   {update_job}                │
    │                                       │   .status = JS_RUNNING        │
    │                                       │                               │
    │◀──────── WebRTC audio/video ──────────│◀── audio frames ─────────────│
    │──────── WebRTC audio/video ──────────▶│── audio frames ─────────────▶│
```

**Message types:**

| Direction | Message (WorkerMessage) | When |
|-----------|------------------------|------|
| Worker → Server | `register` | Once on connect — declares job type, agent name, permissions |
| Worker → Server | `update_worker` | Every ~2.5s — status (AVAILABLE/FULL) + current load |
| Worker → Server | `availability` | Response to job offer — accept or decline |
| Worker → Server | `update_job` | Job lifecycle — RUNNING → SUCCESS/FAILED |
| Worker → Server | `ping` | Heartbeat |
| Worker → Server | `migrate_job` | Handoff a running job to another worker |

| Direction | Message (ServerMessage) | When |
|-----------|--------------------------|------|
| Server → Worker | `register` | Response with assigned worker ID |
| Server → Worker | `availability` | "Are you free for this job?" |
| Server → Worker | `assignment` | "Here's your job — room info + JWT to join" |
| Server → Worker | `termination` | "Kill this running job" |
| Server → Worker | `pong` | Heartbeat reply |

### HTTP Health Endpoint

The worker runs an HTTP server on port 8081 that LiveKit polls:

```
GET /         → 200 OK          (worker is healthy)
              → 503             (inference process dead or connection failed)

GET /worker   → JSON WorkerInfo  {
                  worker_type: "JT_ROOM",
                  agent_name: "",
                  active_jobs: 0,
                  sdk_version: "1.6.7",
                  worker_load: 0.0,
                  protocol_version: 17
                }
```

---

## STT → LLM → TTS Pipeline

Everything is orchestrated by `AgentActivity._pipeline_reply_task_impl()` in the
LiveKit Agents framework. The `agent.py` file just provides the configuration;
the framework does the rest.

### Phase 1: User speaks — STT runs continuously

```
Mic audio frames (via WebRTC)
      │
      ▼
┌──────────────────┐
│  _STTPipeline     │   Background pump task, runs forever
│  (audio_recog.)  │   Receives audio frames via Chan[AudioFrame]
│                  │   Calls openai.STT.__call__(audio_ch, settings)
│                  │   → POST /v1/audio/transcriptions (streaming)
│                  │   Yields SpeechEvent's into event_ch
└──────┬───────────┘
       │
       │  SpeechEvent(type=INTERIM_TRANSCRIPT, text="Hello,")
       │  SpeechEvent(type=INTERIM_TRANSCRIPT, text="Hello, how")
       │  SpeechEvent(type=FINAL_TRANSCRIPT,  text="Hello, how are you?")
       │
       ▼
  AudioRecognition
       │  Accumulates transcript in self._audio_transcript
       │  TurnDetector + VAD detect end-of-speech (user stopped talking)
       │
       ▼
  _commit_user_turn()
       │  Flushes STT buffer with silence frames
       │  Waits for final transcript (with timeout)
       │  Calls hooks.on_user_turn_completed()
       │
       ▼
  AgentActivity._on_user_turn_completed()
       │  Wraps transcript into llm.ChatMessage(role="user", text="...")
       │  Calls _generate_reply() → _pipeline_reply_task()
```

### Phase 2: Generate reply — LLM streams, TTS synthesizes in parallel

```
_pipeline_reply_task_impl()
│
├─▶ perform_llm_inference(node=agent.llm_node, chat_ctx=..., tools=...)
│   │
│   │  llm_node = self._llm.chat(chat_ctx, tools, settings)
│   │  → POST /v1/chat/completions (SSE streaming)
│   │
│   │  Returns: (_LLMGenerationData)
│   │    .text_ch     → Chan[str]           (streamed LLM text)
│   │    .function_ch → Chan[FunctionCall]  (tool calls)
│   │    .ttft        → float               (time-to-first-token metric)
│   │
│   │  AS EACH TEXT CHUNK ARRIVES FROM LLM:
│   │
│   │  ┌──────────────────────────────────────────────┐
│   │  │  _produce_segments()                          │
│   │  │                                              │
│   │  │  For each segment (split by FlushSentinel):  │
│   │  │                                              │
│   │  │  1. Start TTS: perform_tts_inference(...)    │
│   │  │     tts_node = self._tts.synthesize(input)   │
│   │  │     → POST /v1/audio/speech (streaming)     │
│   │  │     Returns: _TTSGenerationData              │
│   │  │       .audio_ch → Chan[AudioFrame]           │
│   │  │       .ttfb     → float (TTFB metric)        │
│   │  │                                              │
│   │  │  2. Forward LLM text → TTS input channel     │
│   │  │     For each chunk from LLM:                 │
│   │  │       tts_text_ch.send_nowait(chunk)         │
│   │  │       segment.text.send_nowait(chunk)        │
│   │  │                                              │
│   │  │  3. When LLM chunk is FlushSentinel:         │
│   │  │     Close TTS stream, start new one          │
│   │  └──────────────────────────────────────────────┘
│   │
│   └─▶ In parallel: perform_tool_executions()
│        Watches function_ch for tool calls
│        Executes them, feeds results back into chat_ctx
│        Can trigger nested _generate_reply() calls
│
├─▶ SpeechHandle.play()
│   │
│   │  For each _SpeechSegment:
│   │    1. Wait for TTS audio_ch to produce frames
│   │    2. Push AudioFrame to audio output (→ WebRTC → user's speaker)
│   │    3. Push timed text to transcription output (→ room transcript)
│   │
│   │  Can be interrupted at any point (user starts talking again)
│   │  → cancels remaining segments, triggers new STT → LLM → TTS cycle
```

### The Three Plugin Interfaces

Each service follows the **OpenAI-compatible protocol** via LiveKit's plugin
wrappers:

| Component | Plugin class | Service | Wire protocol |
|-----------|-------------|---------|---------------|
| STT | `openai.STT` | Nemotron on :8000 | `POST /v1/audio/transcriptions` (streaming multipart) |
| LLM | `openai.LLM` | Ollama/llama.cpp on :11434 | `POST /v1/chat/completions` (SSE streaming) |
| TTS | `openai.TTS` | Kokoro on :8880 | `POST /v1/audio/speech` (raw audio bytes stream) |

---

## Concurrency & Isolation

### Per-user isolation model

Every time a user clicks "Start call", the API generates a **random room name**:

```python
room_name = f"voice_assistant_room_{random.randint(0, 9999)}"
```

Each user gets their own LiveKit room. The LiveKit server dispatches one job per
room to the worker, and each job runs in its **own forked subprocess**:

```
User A → room_1234 → job_A → subprocess_1
User B → room_5678 → job_B → subprocess_2
User C → room_9012 → job_C → subprocess_3
```

Each subprocess has its own `AgentSession` with independent conversation state,
chat history, and turn detection. They share nothing.

### Shared backend services are stateless

The STT, LLM, and TTS servers are singletons shared by all users. They are
**stateless across requests** — each HTTP call is independent:

```
                    ┌── nemotron (port 8000) ◀── stateless: audio in → text out
subprocess_1 (user A)──┼── ollama   (port 11434) ◀── stateless: text in → text out
subprocess_2 (user B)──┼── kokoro   (port 8880) ◀── stateless: text in → audio out
subprocess_3 (user C)──┘
```

Conversation state (history, context, whose turn it is) lives entirely inside the
isolated job subprocess, in the `AgentSession` object.

### Process pool

The worker pre-forks idle subprocesses at startup:

```
"initializing process" pid=1848223
"initializing process" pid=1848227
... (one per CPU core)
```

Each is a pre-warmed container with VAD and turn-detector models loaded (shared
via copy-on-write memory). When all subprocesses are busy, the worker reports
`WS_FULL` to LiveKit.

### Bottleneck

The practical limit is LLM inference throughput. Ollama/llama.cpp processes
requests sequentially, so too many concurrent users cause queuing latency. But
conversations never interfere — each user's chat context is fully isolated.

---

## HTTPS / WSS Setup

LiveKit server v1.13.4 doesn't support native TLS (`--tls-cert`/`--tls-key`
flags don't exist yet). To serve the entire stack over HTTPS with a single
mkcert certificate, two things happen:

1. **uvicorn serves HTTPS** directly using the cert and key for the web frontend
   and API (port 8083).

2. **A raw TCP proxy** (`wss_proxy.py`) unwraps TLS on the LiveKit signaling
   port (7880) and forwards plain bytes to livekit-server on an internal port
   (7883). This works because LiveKit speaks its own signaling protocol over
   WebSocket — the proxy doesn't need to understand it.

### Port layout with HTTPS

| Port | Bound to | Service | Protocol |
|------|----------|---------|----------|
| **8083** | `0.0.0.0` | FastAPI web + API | **HTTPS** (uvicorn SSL) |
| **7880** | `0.0.0.0` | WSS proxy | **WSS** (raw TCP TLS unwrap) |
| **7883** | `127.0.0.1` | livekit-server (internal) | WS (plain, local only) |
| **8000** | `127.0.0.1` | Nemotron STT | HTTP (internal) |
| **8880** | `127.0.0.1` | Kokoro TTS | HTTP (internal) |
| **11434** | `*` | ollama LLM | HTTP |

### Connection flow

```
Browser                                This machine
───────                                ───────────
       │
       ├──https://100.126.67.88:8083──▶ uvicorn (SSL) → FastAPI
       │
       ├──wss://100.126.67.88:7880────▶ wss_proxy.py  → livekit-server :7883
       │   (TLS unwrapped here)         (raw TCP fwd)   (internal, plain ws)
       │
       └──WebRTC media───────────────▶ ports 7881-7882 (always DTLS encrypted)
```

The proxy is **protocol-agnostic** — it passes bytes through without interpreting
WebSocket frames. This avoids compatibility issues with LiveKit's non-standard
signaling handshake.

### Generating certs with mkcert

```bash
mkcert -install
mkcert -cert-file cert.pem -key-file key.pem 100.126.67.88 localhost 127.0.0.1 ::1
```

Env vars to enable:

```env
SSL_CERTFILE=./cert.pem
SSL_KEYFILE=./key.pem
LIVEKIT_EXTERNAL_URL=wss://100.126.67.88:7880
LIVEKIT_URL=ws://127.0.0.1:7883
LIVEKIT_BIND_PORT=7883
```

### Trusting the cert on client devices

Copy the mkcert root CA to your client:

```bash
# From the client
scp user@server:/home/user/.local/share/mkcert/rootCA.pem .
# macOS
sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain rootCA.pem
# Linux
sudo cp rootCA.pem /usr/local/share/ca-certificates/mkcert.crt && sudo update-ca-certificates
```

## Key Design Decisions

1. **Streaming all the way down** — STT streams partial transcripts in
   real-time, LLM streams text tokens as they're generated, TTS starts
   synthesizing *before the full LLM response is complete*
   (`preemptive_generation=True`). This is why voice responses feel instant.

2. **TTS segmentation** — If the LLM emits a `FlushSentinel` ("speak what you
   have so far"), the TTS stream is closed and a new one opened. This allows
   multi-sentence responses to begin playing while later sentences are still
   being generated.

3. **Tool calls are parallel** — While the LLM streams text for TTS, it may
   also emit function call deltas. These are routed to
   `perform_tool_executions()` which runs concurrently; results feed back into
   the chat context for subsequent LLM calls.

4. **Interruption** — If the user starts speaking during agent playback,
   `AudioRecognition` detects this (via VAD + turn detection), cancels
   remaining TTS segments, and starts a new STT→LLM→TTS cycle immediately.

5. **Web server starts first** — The FastAPI server boots before any children
   are ready. The frontend loads immediately and polls `/api/status` to show
   download progress during first boot.

6. **Offline-first** — The supervisor detects cached models and passes
   `--offline` to llama-server automatically. Combined with
   `HF_HUB_OFFLINE=1`, the entire stack works without internet after initial
   download.

7. **`LIVEKIT_EXTERNAL_URL`** — The agent worker connects to LiveKit via
   localhost (`LIVEKIT_URL`), but remote browsers get the external IP/label
   from `/api/connection-details`. This enables remote access while avoiding
   hairpin NAT issues.

---

## Environment Variables

The full configuration surface (see `.env` for current values):

| Variable | Default | Description |
|----------|---------|-------------|
| `WEB_PORT` | 8080 | FastAPI web server + frontend |
| `WEB_HOST` | 0.0.0.0 | Bind address for web server |
| `FRONTEND_DIR` | — | Path to Next.js static export |
| `SSL_CERTFILE` | — | TLS certificate for HTTPS (uvicorn + wss-proxy) |
| `SSL_KEYFILE` | — | TLS private key |
| `LIVEKIT_URL` | ws://127.0.0.1:7880 | LiveKit WebSocket URL (agent connection) |
| `LIVEKIT_EXTERNAL_URL` | — | LiveKit URL returned to browsers (for remote access, typically wss://) |
| `LIVEKIT_BIND_PORT` | 7880 | livekit-server listen port (use 7883 with wss-proxy) |
| `LIVEKIT_NODE_IP` | 127.0.0.1 | Advertised WebRTC ICE candidate IP |
| `LIVEKIT_API_KEY` | devkey | LiveKit API key |
| `LIVEKIT_API_SECRET` | secret | LiveKit API secret |
| `MANAGE_LIVEKIT` | auto | Force local LiveKit management |
| `LLAMA_BASE_URL` | http://127.0.0.1:11434/v1 | LLM endpoint |
| `LLAMA_MODEL` | gemma-4-e2b | Model name sent to LLM endpoint |
| `LLAMA_HF_REPO` | unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL | HF repo for llama.cpp |
| `LLAMA_OFFLINE` | auto | Force offline llama.cpp startup |
| `LLAMA_CTX_SIZE` | 16384 | LLM context window size |
| `MANAGE_LLAMA` | auto | Force local llama.cpp management |
| `STT_PROVIDER` | nemotron | `nemotron` or `whisper` |
| `STT_BASE_URL` | http://127.0.0.1:8000/v1 | STT endpoint |
| `STT_MODEL` | nemotron-speech-streaming | Model name for STT |
| `MANAGE_STT` | auto | Force local STT management |
| `TTS_BASE_URL` | http://127.0.0.1:8880/v1 | TTS endpoint |
| `TTS_VOICE` | af_nova | Voice for Kokoro TTS |
| `MANAGE_TTS` | auto | Force local TTS management |
| `HF_HUB_OFFLINE` | 0 | Use cached HF models only (no network) |
| `WAKE_WORD` | 0 | Enable "hey livekit" wake word detection |
| `WAKE_WORD_THRESHOLD` | 0.5 | Wake word sensitivity (0–1) |
| `DEVICE` | cpu | Device for torch: `cpu`, `cuda`, `mps` |
| `LOG_LEVEL` | INFO | Python logging level |
