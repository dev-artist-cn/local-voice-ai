"""Raw TLS-terminating TCP proxy for livekit-server.

LiveKit uses its own signaling protocol over WebSocket, which the ``websockets``
library doesn't handle correctly (it expects a standard 101 upgrade).

This proxy does pure byte forwarding with no protocol interpretation:
  Browser ──WSS──▶ proxy (TLS unwrap) ──raw TCP──▶ livekit-server
"""

from __future__ import annotations

import asyncio
import logging
import os
import ssl

logger = logging.getLogger("wss-proxy")
logging.basicConfig(level=logging.INFO)

LISTEN_HOST = os.getenv("WSS_PROXY_LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.getenv("WSS_PROXY_LISTEN_PORT", "7880"))
BACKEND_HOST = os.getenv("WSS_PROXY_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("WSS_PROXY_BACKEND_PORT", "7883"))
CERTFILE = os.getenv("SSL_CERTFILE", "cert.pem")
KEYFILE = os.getenv("SSL_KEYFILE", "key.pem")


async def pipe(reader, writer, name):
    """Bidirectional byte pipe."""
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        pass
    except Exception:
        logger.debug("%s: pipe error", name, exc_info=True)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


async def handle(client_reader, client_writer):
    """Handle one TLS-terminated client, forwarding to backend over plain TCP."""
    try:
        backend_reader, backend_writer = await asyncio.open_connection(
            BACKEND_HOST, BACKEND_PORT
        )
    except Exception:
        logger.exception("failed to connect to backend %s:%d", BACKEND_HOST, BACKEND_PORT)
        client_writer.close()
        return

    await asyncio.gather(
        pipe(client_reader, backend_writer, "client→livekit"),
        pipe(backend_reader, client_writer, "livekit→client"),
    )


async def main():
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(CERTFILE, KEYFILE)

    server = await asyncio.start_server(
        handle, LISTEN_HOST, LISTEN_PORT, ssl=ssl_context
    )

    logger.info(
        "WSS proxy (raw TCP) listening on %s:%d → %s:%d",
        LISTEN_HOST, LISTEN_PORT, BACKEND_HOST, BACKEND_PORT,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
