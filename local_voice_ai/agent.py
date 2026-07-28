"""LiveKit Agents worker.

Moved verbatim from ``livekit_agent/src/agent.py``. The only change is that the
default base URLs are loopback (``127.0.0.1``) instead of Docker service names —
the supervisor spawns the inference children on loopback ports, so this is
correct for both single-image deployment and bare-metal local runs.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import openai, silero

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "你是一个乐于助人的语音AI助手。用户通过语音与你交互。"
                "你热情地用丰富的知识回答用户的问题。你的回答简洁明了，"
                "不使用任何表情符号、列表或其他特殊符号。"
                "你充满好奇心、友好，并且有幽默感。"
                "请始终用中文回复。"
            ),
        )

    @function_tool()
    async def multiply_numbers(
        self,
        context: RunContext,
        number1: int,
        number2: int,
    ) -> dict[str, Any]:
        """Multiply two numbers.

        Args:
            number1: The first number to multiply.
            number2: The second number to multiply.
        """
        return f"The product of {number1} and {number2} is {number1 * number2}."


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    # Lower min_silence_duration for faster turn detection (default 0.55s)
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.3)


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    # The job assignment URL uses LIVEKIT_NODE_IP (the external IP for WebRTC ICE).
    # The agent's Rust RTC engine can't complete the WebSocket handshake over the
    # tailscale/LAN IP, so rewrite the host to localhost for signaling.
    info = ctx._info  # RunningJobInfo
    logger.info("job assignment url: %s", info.url)
    if info.url:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(info.url)
        if parsed.hostname not in {"127.0.0.1", "localhost"}:
            info.url = urlunparse(parsed._replace(netloc=f"127.0.0.1:{parsed.port}"))

    llama_model = os.getenv("LLAMA_MODEL", "gemma-4-e2b")
    llama_base_url = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    llama_api_key = os.getenv("LLAMA_API_KEY", "no-key-needed")

    stt_provider = os.getenv("STT_PROVIDER", "nemotron").lower()
    if stt_provider == "whisper":
        default_stt_base_url = "http://127.0.0.1:8000/v1"
        default_stt_model = "Systran/faster-whisper-small"
    else:
        default_stt_base_url = "http://127.0.0.1:8000/v1"
        default_stt_model = "nemotron-speech-streaming"

    stt_base_url = os.getenv("STT_BASE_URL", default_stt_base_url)
    stt_model = os.getenv("STT_MODEL", default_stt_model)
    stt_api_key = os.getenv("STT_API_KEY", "no-key-needed")

    tts_base_url = os.getenv("TTS_BASE_URL", "http://127.0.0.1:8880/v1")
    tts_voice = os.getenv("TTS_VOICE", "af_nova")
    tts_api_key = os.getenv("TTS_API_KEY", "no-key-needed")

    logger.info(
        "agent session: stt=%s/%s llm=%s/%s tts=%s",
        stt_provider, stt_model, llama_base_url, llama_model, tts_base_url,
    )

    wake_word = os.getenv("WAKE_WORD", "").strip().lower() in {"1", "true", "yes", "on"}
    wake_word_model = os.getenv("WAKE_WORD_MODEL", "/app/models/wakeword/hey_livekit.onnx")
    wake_word_threshold = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))

    session = AgentSession(
        stt=openai.STT(base_url=stt_base_url, model=stt_model, api_key=stt_api_key),
        llm=openai.LLM(base_url=llama_base_url, model=llama_model, api_key=llama_api_key),
        # The model name selects the wire protocol the openai TTS plugin uses:
        # only {"tts-1", "tts-1-hd"} use the raw-audio-bytes stream that the
        # Kokoro server speaks. Any other name (e.g. "kokoro") routes the plugin
        # into the gpt-4o-mini-tts SSE reader, which parses Kokoro's binary audio
        # body as text, pushes zero frames, and raises "no audio frames were
        # pushed". Kokoro ignores the model field, so "tts-1" is purely a
        # protocol selector here.
        tts=openai.TTS(base_url=tts_base_url, model="tts-1", voice=tts_voice, api_key=tts_api_key),
        # VAD-only turn detection: the MultilingualModel turn detector has
        # a tokenizer initialization bug in the forked inference process.
        # VAD-only waits for silence after speech, slightly less smart but
        # reliable and lower latency since no failing inference is attempted.
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        min_endpointing_delay=0.2,
    )

    await session.start(agent=Assistant(), room=ctx.room)
    await ctx.connect()

    if wake_word:
        # Join deaf, wait for the wake phrase, then wake up and greet.
        from .wakeword import wait_for_wake_word

        session.input.set_audio_enabled(False)
        participant = await ctx.wait_for_participant()
        try:
            await wait_for_wake_word(participant, wake_word_model, wake_word_threshold)
        except Exception:
            # Fail open: a broken detector shouldn't brick the assistant.
            logger.exception("wake word detection failed; enabling audio input")
        session.input.set_audio_enabled(True)
        session.generate_reply(
            instructions=(
                "你刚被唤醒。简短地跟用户打个招呼，问问能帮什么忙。"
            )
        )
    else:
        # Speak first so the user knows the audio path works.
        session.generate_reply(
            instructions=(
                "用一句话热情地跟用户打招呼，邀请他们提问。请用中文。"
            )
        )


if __name__ == "__main__":
    cli.run_app(server)
