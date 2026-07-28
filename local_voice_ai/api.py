"""FastAPI app served from the supervisor process.

Three responsibilities:
  1. ``POST /api/auth/request-code`` + ``POST /api/auth/login`` — phone-OTP
     login (demo: no real SMS, code is hard-coded). Issues an HS256 access
     token the browser sends as ``Authorization: Bearer <token>``.
  2. ``POST /api/connection-details`` — mints a LiveKit access token.
     **Requires** a valid app access token (see above). This is the Python
     port of ``frontend/app/api/connection-details/route.ts``.
  3. ``GET /*`` — serves the statically-exported Next.js frontend, when
     ``Config.frontend_dir`` is set.

Readiness (``GET /api/status``) and liveness (``GET /healthz``) stay
unauthenticated: the boot splash polls status before login, and the Docker
healthcheck hits /healthz.
"""

from __future__ import annotations

import logging
import os
import random
import re
import sqlite3
import time
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import jwt
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from livekit import api as lk_api
from pydantic import BaseModel

from .config import Config

logger = logging.getLogger("api")


# --------------------------------------------------------------------------- #
# Phone-OTP auth helpers
# --------------------------------------------------------------------------- #


def _valid_phone(phone: str) -> bool:
    digits = re.sub(r"\D", "", phone)
    return 7 <= len(digits) <= 15


class UserStore:
    """Tiny SQLite-backed user table. Created lazily; safe to share across the
    FastAPI threadpool because each call opens its own connection."""

    def __init__(self, path: str) -> None:
        self._path = path
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    phone         TEXT PRIMARY KEY,
                    created_at    INTEGER NOT NULL,
                    last_login_at INTEGER NOT NULL
                )
                """
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def upsert(self, phone: str) -> None:
        now = int(time.time())
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO users (phone, created_at, last_login_at)
                VALUES (?, ?, ?)
                ON CONFLICT(phone) DO UPDATE SET last_login_at = excluded.last_login_at
                """,
                (phone, now, now),
            )


def create_access_token(phone: str, secret: str, ttl_seconds: int) -> str:
    now = int(time.time())
    payload = {"sub": phone, "iat": now, "exp": now + ttl_seconds}
    return jwt.encode(payload, secret, algorithm="HS256")


def verify_access_token(token: str, secret: str) -> str | None:
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None
    sub = payload.get("sub")
    return sub if isinstance(sub, str) else None


class RequestCodeBody(BaseModel):
    phone: str


class LoginBody(BaseModel):
    phone: str
    code: str


def _mint_livekit_token(cfg: Config, agent_name: str | None) -> dict[str, Any]:
    participant_name = "user"
    participant_identity = f"voice_assistant_user_{random.randint(0, 9999)}"
    room_name = f"voice_assistant_room_{random.randint(0, 9999)}"

    token = (
        lk_api.AccessToken(cfg.livekit_api_key, cfg.livekit_api_secret)
        .with_identity(participant_identity)
        .with_name(participant_name)
        .with_ttl(timedelta(minutes=15))
        .with_grants(
            lk_api.VideoGrants(
                room=room_name,
                room_join=True,
                can_publish=True,
                can_publish_data=True,
                can_subscribe=True,
            )
        )
    )

    if agent_name:
        token = token.with_room_config(
            lk_api.RoomConfiguration(agents=[lk_api.RoomAgentDispatch(agent_name=agent_name)])
        )

    return {
        "serverUrl": cfg.livekit_external_url or cfg.livekit_url,
        "roomName": room_name,
        "participantName": participant_name,
        "participantToken": token.to_jwt(),
    }


def build_app(
    cfg: Config,
    status_provider: Callable[[], list[dict[str, Any]]] | None = None,
) -> FastAPI:
    app = FastAPI(title="local-voice-ai", version="0.1.0")

    users = UserStore(cfg.auth_db_path)
    if cfg.auth_secret == Config.auth_secret:
        logger.warning(
            "AUTH_SECRET is the default value — set a strong AUTH_SECRET before production use."
        )

    bearer = HTTPBearer(auto_error=False)

    def require_auth(
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> str:
        """Decode the bearer token and return the authenticated phone number."""
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(401, "missing or malformed bearer token")
        phone = verify_access_token(credentials.credentials, cfg.auth_secret)
        if phone is None:
            raise HTTPException(401, "invalid or expired access token")
        return phone

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        """Per-child readiness, polled by the frontend's first-boot splash.

        The web server starts before the children are ready (first boot can
        spend a long time downloading model weights), so this is how the UI
        knows whether the stack is usable yet.
        """
        children = status_provider() if status_provider is not None else []
        return {
            "ready": all(c["ready"] for c in children),
            "children": children,
            # Lets the frontend hint "say the wake phrase" when enabled.
            "wake_word": cfg.wake_word,
        }

    @app.post("/api/auth/request-code")
    async def request_code(body: RequestCodeBody) -> dict[str, Any]:
        """Pretend to text an OTP. No SMS is ever sent — demo always succeeds."""
        phone = body.phone.strip()
        if not _valid_phone(phone):
            raise HTTPException(400, "invalid phone number")
        # In production: look up the user, throttle sends, integrate an SMS
        # provider, store a per-phone code + expiry, etc.
        logger.info("auth: (demo) OTP requested for %s", phone)
        return {"success": True, "message": "code sent"}

    @app.post("/api/auth/login")
    async def login(body: LoginBody) -> JSONResponse:
        phone = body.phone.strip()
        code = (body.code or "").strip()
        if not _valid_phone(phone):
            raise HTTPException(400, "invalid phone number")
        if code != cfg.auth_otp_code:
            raise HTTPException(401, "invalid code")

        # Create the user if they don't already exist, then stamp last login.
        users.upsert(phone)
        access_token = create_access_token(
            phone, cfg.auth_secret, cfg.auth_token_ttl_seconds
        )
        logger.info("auth: login ok for %s", phone)
        return JSONResponse(
            {"access_token": access_token, "token_type": "bearer", "phone": phone},
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/connection-details")
    async def connection_details(
        request: Request, phone: str = Depends(require_auth)
    ) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}

        agent_name: str | None = None
        try:
            agent_name = body.get("room_config", {}).get("agents", [{}])[0].get("agent_name")
        except (AttributeError, IndexError, TypeError):
            agent_name = None

        try:
            data = _mint_livekit_token(cfg, agent_name)
        except Exception as exc:
            logger.exception("token minting failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        logger.info("auth: minted LiveKit token for %s", phone)
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    if cfg.frontend_dir:
        # SPA-style: serve static export, falling back to index.html for unknown paths.
        static = StaticFiles(directory=cfg.frontend_dir, html=True)

        @app.get("/{path:path}")
        async def spa(path: str, request: Request) -> Any:
            try:
                return await static.get_response(path or "index.html", request.scope)
            except Exception:
                return FileResponse(f"{cfg.frontend_dir}/index.html")

    return app
