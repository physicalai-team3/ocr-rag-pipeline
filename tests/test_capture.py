"""② 영상 수신 테스트 (B). 실행: pip install pytest httpx && python -m pytest -q"""
import time

import time

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import capture, events
from app.main import app

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def seen(monkeypatch):
    got: list[dict] = []
    events.add_listener(got.append)
    yield got
    events._listeners.remove(got.append)


def new_session(client):
    s = client.post("/api/sessions").json()
    return s["session_id"], s["token"]


def test_rejects_bad_token(client):
    sid, _ = new_session(client)
    with client.websocket_connect(f"/ws/camera?sid={sid}&t=wrong") as ws:
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_text()
    assert e.value.code == 4401


def test_frames_flow_to_ocr_and_events(client, seen, monkeypatch):
    submitted = []

    async def fake_submit(sid, jpeg, ts):
        submitted.append((sid, jpeg, ts))

    monkeypatch.setattr(capture.ocr, "submit", fake_submit)
    sid, t = new_session(client)
    with client.websocket_connect(f"/ws/camera?sid={sid}&t={t}") as ws:
        ws.send_text('{"type":"meta","rotation":270}')
        ws.send_bytes(JPEG)
        ws.send_bytes(b"not a jpeg")
        ws.send_bytes(JPEG)
    types = [e["type"] for e in seen if e.get("sid") == sid]
    assert types == ["session.paired", "frame.received", "error", "frame.received", "session.disconnected"]
    assert [s[1] for s in submitted] == [JPEG, JPEG]
    err = next(e for e in seen if e["type"] == "error")
    assert err["stage"] == "capture"


def test_reconnect_after_disconnect_is_allowed(client, monkeypatch):
    async def noop(*a):
        pass

    monkeypatch.setattr(capture.ocr, "submit", noop)
    sid, t = new_session(client)
    with client.websocket_connect(f"/ws/camera?sid={sid}&t={t}") as ws:
        ws.send_bytes(JPEG)
    with client.websocket_connect(f"/ws/camera?sid={sid}&t={t}") as ws:
        ws.send_bytes(JPEG)
    assert client.get(f"/api/sessions/{sid}").json()["status"] == "disconnected"


def test_save_frames_per_connection(client, monkeypatch, tmp_path):
    async def noop(*a):
        pass

    monkeypatch.setattr(capture.ocr, "submit", noop)
    monkeypatch.setattr(capture, "CAPTURE_DIR", tmp_path)
    monkeypatch.setenv("SAVE_FRAMES", "1")
    sid, t = new_session(client)
    with client.websocket_connect(f"/ws/camera?sid={sid}&t={t}") as ws:
        ws.send_bytes(JPEG)
        ws.send_bytes(JPEG)
        # 저장은 스레드풀에서 하므로, 연결을 닫기 전에 끝날 때까지 기다린다
        deadline = time.monotonic() + 3
        while len(list(tmp_path.rglob("*.jpg"))) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
    files = sorted(tmp_path.rglob("*.jpg"))
    assert len(files) == 2
    assert {f.parent.parent.name for f in files} == {sid}
    assert files[0].name.startswith("000000_") and files[1].name.startswith("000001_")
    assert files[0].read_bytes() == JPEG
