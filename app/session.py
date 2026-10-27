"""QR 세션 (담당: A)

흐름
1. 대시보드가 `POST /api/sessions` → QR 표시 (`GET /api/sessions/{sid}/qr.svg`)
2. 휴대폰이 QR의 `https://<LAN IP>:8443/m?sid=&t=` 를 엶
3. 휴대폰이 `/ws/camera?sid=&t=` 로 접속 → B가 `session.verify(sid, t)` 호출
   실패하면 close code 4401
4. B가 `session.paired` / `session.disconnected` 이벤트를 보냄 → 대시보드 "연결됨" 표시

토큰 규칙
- QR은 10분 안에 처음 연결해야 함 (스캔 안 된 QR은 만료)
- 한 세션에 동시에 연결되는 휴대폰은 한 대. 같은 토큰으로 새 연결이 오면 새 연결이 이기고
  B(capture.py)가 이전 연결을 close code 4000으로 끊음
  (Wi-Fi가 잠깐 끊기면 서버가 이전 연결의 끊김을 알아채기 전에 휴대폰이 다시 붙기 때문)
- 처음 연결한 뒤로는 SESSION_TTL 동안 다시 붙을 수 있음
"""
from __future__ import annotations

import io
import secrets
import time
from dataclasses import dataclass, field

import segno
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from . import events, netinfo

PAIR_WINDOW_S = 10 * 60
SESSION_TTL_S = 2 * 60 * 60

router = APIRouter()


@dataclass
class Session:
    sid: str
    token: str
    created_at: float = field(default_factory=time.time)
    claimed_at: float | None = None
    status: str = "waiting"  # waiting | paired | disconnected

    @property
    def expires_at(self) -> float:
        if self.claimed_at is None:
            return self.created_at + PAIR_WINDOW_S
        return self.claimed_at + SESSION_TTL_S

    def mobile_url(self) -> str:
        return f"{netinfo.public_base_url()}/m?sid={self.sid}&t={self.token}"


_sessions: dict[str, Session] = {}


def _prune() -> None:
    now = time.time()
    for sid in [s.sid for s in _sessions.values() if s.expires_at + 60 < now]:
        del _sessions[sid]


def _check(sid: str, token: str) -> tuple[bool, str]:
    """토큰을 소모하지 않고 지금 접속 가능한지만 봅니다. (ok, 거부 이유)"""
    s = _sessions.get(sid)
    if s is None or not secrets.compare_digest(s.token, token or ""):
        return False, "invalid"
    if time.time() > s.expires_at:
        return False, "expired"
    return True, "ok"


def verify(sid: str, token: str) -> bool:
    """B의 /ws/camera가 연결 직후 호출. False면 close code 4401로 끊으세요.

    이미 연결된 기기가 있어도 토큰이 맞으면 True입니다. 이전 연결은 호출한 쪽이 끊어 주세요.
    """
    ok, _ = _check(sid, token)
    if ok:
        s = _sessions[sid]
        if s.claimed_at is None:
            s.claimed_at = time.time()
    return ok


async def mark_paired(sid: str, **extra: object) -> None:
    await events.publish(sid, {"type": "session.paired", **extra})


async def mark_disconnected(sid: str, **extra: object) -> None:
    await events.publish(sid, {"type": "session.disconnected", **extra})


def _on_event(ev: events.Event) -> None:
    s = _sessions.get(ev.get("sid") or "")
    if s is None:
        return
    if ev["type"] == "session.paired":
        s.status = "paired"
    elif ev["type"] == "session.disconnected":
        s.status = "disconnected"


events.add_listener(_on_event)


def state(sid: str) -> dict:
    s = _sessions.get(sid)
    if s is None or time.time() > s.expires_at:
        return {"session_id": sid, "status": "expired"}
    return {"session_id": sid, "status": s.status, "expires_at": int(s.expires_at * 1000)}


# ── HTTP ─────────────────────────────────────────────────────────────

@router.post("/api/sessions")
async def create_session() -> dict:
    _prune()
    s = Session(sid=secrets.token_urlsafe(8), token=secrets.token_urlsafe(16))
    _sessions[s.sid] = s
    return {
        "session_id": s.sid,
        "token": s.token,
        "mobile_url": s.mobile_url(),
        "qr_url": f"/api/sessions/{s.sid}/qr.svg",
        "expires_at": int(s.expires_at * 1000),
    }


@router.get("/api/sessions/{sid}")
async def get_session(sid: str, t: str | None = None) -> dict:
    """대시보드는 t 없이 상태만, 휴대폰은 t를 붙여 접속 가능 여부까지 확인합니다."""
    body = state(sid)
    if t is not None:
        ok, reason = _check(sid, t)
        body |= {"valid": ok, "reason": reason}
    return body


@router.get("/api/sessions/{sid}/qr.svg")
async def session_qr(sid: str) -> Response:
    s = _sessions.get(sid)
    if s is None:
        raise HTTPException(404, "session not found")
    buf = io.BytesIO()
    segno.make(s.mobile_url(), error="m").save(
        buf, kind="svg", scale=6, border=2, xmldecl=False, dark="#16181d", light="#ffffff"
    )
    return Response(buf.getvalue(), media_type="image/svg+xml", headers={"Cache-Control": "no-store"})


@router.get("/m", include_in_schema=False)
async def mobile_page() -> FileResponse:
    return FileResponse(netinfo.ROOT / "static" / "mobile.html")
