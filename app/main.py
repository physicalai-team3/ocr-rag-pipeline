"""FastAPI 앱 (공용) — 라우터 등록만 합니다. 줄 추가만 하고 기존 줄은 고치지 않기."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import capture, events, netinfo, rag, session, status
from . import ocr

STATIC = netinfo.ROOT / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    events.bind_loop(asyncio.get_running_loop())
    await ocr.start()
    yield
    await ocr.stop()


app = FastAPI(title="OCR-RAG pipeline", lifespan=lifespan)

app.include_router(session.router)   # A ① QR
app.include_router(status.router)    # A ⑤ 저장 현황
app.include_router(capture.router)   # B ② 영상 전송
app.include_router(rag.router)       # D ⑥ RAG


@app.get("/", include_in_schema=False)
async def dashboard() -> FileResponse:
    return FileResponse(STATIC / "dashboard.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
