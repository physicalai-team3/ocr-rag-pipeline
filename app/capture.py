"""실시간 영상 수신 (담당: B)

`wss://<host>:8443/ws/camera?sid=&t=`
- 연결 직후 session.verify(sid, t). 실패하면 close code 4401
- 연결되면 session.paired, 끊기면 session.disconnected
- 바이너리 메시지 = JPEG 바이트 그대로 → frame.received {bytes} → ocr.submit
  (회전·축소는 휴대폰 camera.js가 끝내서 보냄: 긴 변 1280px, 품질 0.8, 초당 2~3장)
- 텍스트 메시지 `{"type":"meta", ...}` = 휴대폰 설정(해상도, 회전 등). 로그로만 남김
- JPEG가 아니거나 너무 큰 프레임은 버리고 error {stage:"capture"}

SAVE_FRAMES=1 이면 받은 프레임을 captures/<sid>/<연결시각>/<순번>_<ts>.jpg 로 저장합니다 (C의 테스트 자료).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from . import events, netinfo, ocr, session

log = logging.getLogger(__name__)

router = APIRouter()

JPEG_SOI = b"\xff\xd8"
MAX_FRAME_BYTES = 4 * 1024 * 1024
CAPTURE_DIR = netinfo.ROOT / "captures"
CLOSE_REPLACED = 4000  # 같은 세션에 새 연결이 들어와 이전 연결을 끊을 때

# sid → 지금 프레임을 보내고 있는 연결.
# 늦게 끊긴 이전 연결이 session.disconnected를 보내 새 연결의 "연결됨"을 덮어쓰지 않게 구분합니다.
_active: dict[str, WebSocket] = {}


def save_enabled() -> bool:
    return os.getenv("SAVE_FRAMES") == "1"


def check_frame(data: bytes) -> str | None:
    """문제가 있으면 이유, 괜찮으면 None."""
    if len(data) > MAX_FRAME_BYTES:
        return f"frame too large ({len(data)} bytes)"
    if not data.startswith(JPEG_SOI):
        return "frame is not JPEG"
    return None


@router.websocket("/ws/camera")
async def camera_ws(ws: WebSocket, sid: str = "", t: str = "") -> None:
    await ws.accept()  # 커스텀 close code를 보내려면 먼저 accept해야 합니다
    if not session.verify(sid, t):
        await ws.close(code=4401, reason="invalid or expired session")
        return

    old = _active.get(sid)
    _active[sid] = ws
    if old is not None:
        try:
            await old.close(code=CLOSE_REPLACED, reason="replaced by new connection")
        except RuntimeError:
            pass
    await session.mark_paired(sid)

    save_dir = CAPTURE_DIR / sid / time.strftime("%Y%m%d-%H%M%S") if save_enabled() else None
    seq = 0
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                _on_text(sid, msg["text"])
                continue
            data = msg.get("bytes")
            if not data:
                continue
            problem = check_frame(data)
            if problem:
                await events.publish(sid, {"type": "error", "stage": "capture", "msg": problem})
                continue
            ts = events.now_ms()
            await events.publish(sid, {"type": "frame.received", "bytes": len(data)})
            if save_dir is not None:
                await run_in_threadpool(_save, save_dir, seq, ts, data)
            seq += 1
            await ocr.submit(sid, data, ts)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        if _active.get(sid) is ws:
            del _active[sid]
            await session.mark_disconnected(sid)


def _on_text(sid: str, text: str) -> None:
    try:
        meta = json.loads(text)
    except json.JSONDecodeError:
        return
    if isinstance(meta, dict) and meta.get("type") == "meta":
        log.info("camera meta sid=%s %s", sid, {k: v for k, v in meta.items() if k != "type"})


def _save(save_dir: Path, seq: int, ts: int, data: bytes) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / f"{seq:06d}_{ts}.jpg").write_bytes(data)
