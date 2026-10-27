"""⑥ RAG 검색 — 담당 D.

합의사항(C ↔ D, A ↔ D)
  - 검색은 C 의 store.search(query: str, k: int) -> list[dict] 를 호출한다.
    dict: doc_id, card_no, title, text, score
  - POST /api/ask {"question"} -> {"answer", "sources": [{"card_no", "title"}]}
    (sources 의 text, score 는 화면에 근거 원문을 보여주기 위해 덧붙인 필드)
  - 출처 표기는 현황 피드와 같은 "#014 How Bread Fails"

main.py 에는 이미 `app.include_router(rag.router)` 가 등록돼 있다.

환경변수(전부 선택)
  RAG_TOP_K       기본 3
  RAG_MIN_SCORE   이 점수 미만 결과는 버린다. 기본 비활성. C 의 임베딩 모델 점수 분포를 보고 정한다.
  RAG_LLM         none(기본) | openai | anthropic
  RAG_LLM_MODEL   비우면 gpt-4o-mini / claude-haiku-4-5
  OPENAI_API_KEY / ANTHROPIC_API_KEY

LLM 이 없거나 실패하면 근거 원문에서 질문과 가장 많이 겹치는 문장을 그대로 답으로 쓴다
(대안표: "검색 결과 상위 3개 원문과 출처만 표시").

혼자 테스트: python -m app.rag  → http://127.0.0.1:8001  (C 의 store 대신 data/passages.json 원문 검색)
"""
from __future__ import annotations

import inspect
import logging
import os
import re
import time
from pathlib import Path
from typing import Callable, Optional

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

log = logging.getLogger("rag")
router = APIRouter()

TOP_K = int(os.getenv("RAG_TOP_K", "3"))
MIN_SCORE = os.getenv("RAG_MIN_SCORE")
LLM = os.getenv("RAG_LLM", "none")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "")
NOT_FOUND = "저장된 카드에서 관련 내용을 찾지 못했어요."

SearchFn = Callable[[str, int], list]
_search_override: Optional[SearchFn] = None  # 테스트·단독 실행용


def use_search(fn: Optional[SearchFn]) -> None:
    """C 의 store.search 대신 다른 검색 함수를 쓴다 (테스트·단독 실행용)."""
    global _search_override
    _search_override = fn


def _search_fn() -> SearchFn:
    if _search_override is not None:
        return _search_override
    from . import store  # C 담당. 뼈대 단계에서는 가짜 값을 돌려준다.

    return store.search


# ---------- 출처 ----------

def label(card_no, title: str) -> str:
    """현황 피드와 같은 출처 형식: #014 How Bread Fails"""
    try:
        return f"#{int(card_no):03d} {title}"
    except (TypeError, ValueError):
        return title or "제목 없음"


def _clean_hits(hits: list, k: int) -> list[dict]:
    """점수 0 이하 제거(뼈대와 동일) + RAG_MIN_SCORE 하한 + 같은 doc_id 는 한 번만."""
    out, seen = [], set()
    min_score = float(MIN_SCORE) if MIN_SCORE not in (None, "") else None
    for h in sorted(hits, key=lambda x: -float(x.get("score") or 0)):
        key = h.get("doc_id") or h.get("card_no") or h.get("title")
        if key in seen:
            continue
        score = float(h.get("score") or 0)
        if score <= 0 or (min_score is not None and score < min_score):
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= k:
            break
    return out


# ---------- 답변 ----------

SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided OCR excerpts of conveyor cards. "
    "Excerpts may contain OCR misreads. Never change numbers, units or negations: quote them exactly as written. "
    "If the excerpts do not contain the answer, reply exactly: NOT_FOUND. "
    "Cite cards like [#014]. Be concise. Answer in the same language as the question."
)

_STOP = set("the a an of is are what how in at to was for do does did on and or by it this that which who when where "
            "much many".split())

# 질문이 묻는 값의 종류 → 그 값이 들어 있을 문장의 단서 (카드 대부분이 숫자·단위가 답)
_ANSWER_HINTS = [
    (r"temperature|hot|cold|degree", r"°\s?C|degrees?"),
    (r"how far|distance|long is|length|wide|width|height|tall", r"\d\s?(km|kilomet|metres?|meters?|m\b|mm|miles?)"),
    (r"how long|duration|minutes|hours|time", r"\d\s?(seconds?|minutes?|hours?|days?|weeks?|months?|years?|s\b|ms\b)|\d{1,2}:\d{2}"),
    (r"cost|price|fee|pay|deposit|charge|£|\$|money", r"[£$€]\s?\d|\d\s?(pounds?|dollars?)"),
    (r"pressure", r"\d\s?(bar|hpa|psi|kpa)"),
    (r"torque", r"N·m|Nm"),
    (r"speed|fast|slow|knots", r"\d\s?(km/h|knots?|m/s|mph)"),
    (r"when|year|date", r"\b(1[5-9]\d\d|20\d\d)\b|\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December)"),
    (r"percent|%|rate|ratio", r"\d\s?%"),
    (r"phone|call|telephone|number", r"\d{3,}[\s-]\d{3,}"),
]


def _stem(w: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _words(s: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP}


def _body(h: dict) -> str:
    text, title = (h.get("text") or "").strip(), (h.get("title") or "").strip()
    return text[len(title):].lstrip() if title and text.startswith(title) else text


def _hint_patterns(question: str) -> list[str]:
    q = question.lower()
    return [ans for ask, ans in _ANSWER_HINTS if re.search(ask, q)]


def extractive_answer(question: str, hits: list[dict]) -> str:
    """LLM 없이 근거 문장 하나를 고른다.
    점수 = 질문 단어 겹침 + 묻는 값의 단위가 있으면 가산 + 숫자가 있으면 가산, 상위 카드일수록 가중.
    겹치는 문장이 없어도 검색 결과가 있으면 1위 카드의 첫 문장을 돌려준다(원문은 sources 로 표시)."""
    q = _words(question)
    hints = _hint_patterns(question)
    best, best_score = None, 0.0
    for rank, h in enumerate(hits):
        for sent in re.split(r"(?<=[.!?])\s+", _body(h)):
            s = float(len(q & _words(sent)))
            if s and any(re.search(p, sent, re.I) for p in hints):
                s += 1.5
            if s and re.search(r"\d", sent):
                s += 0.5
            s *= 1 + 0.5 / (rank + 1)
            if s > best_score:
                best, best_score = (sent.strip(), h), s
    if not best:
        h = hits[0]
        first = re.split(r"(?<=[.!?])\s+", _body(h))[0].strip()
        best = (first, h)
    sent, h = best
    return f"{sent} [{label(h.get('card_no'), h.get('title', ''))}]"


def _context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{label(h.get('card_no'), h.get('title', ''))}]\n{h.get('text', '')}" for h in hits)


def _llm_answer(question: str, hits: list[dict]) -> str:
    user = f"Excerpts:\n{_context(hits)}\n\nQuestion: {question}"
    if LLM == "openai":
        r = httpx.post("https://api.openai.com/v1/chat/completions", timeout=30,
                       headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                       json={"model": LLM_MODEL or "gpt-4o-mini", "temperature": 0,
                             "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": user}]})
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    elif LLM == "anthropic":
        r = httpx.post("https://api.anthropic.com/v1/messages", timeout=30,
                       headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                       json={"model": LLM_MODEL or "claude-haiku-4-5", "max_tokens": 400, "temperature": 0,
                             "system": SYSTEM_PROMPT, "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json()["content"])
    else:
        raise ValueError(f"unknown RAG_LLM: {LLM}")
    text = text.strip()
    return NOT_FOUND if text == "NOT_FOUND" else text


def make_answer(question: str, hits: list[dict]) -> tuple[str, str]:
    """(answer, mode) — mode 는 llm | extractive"""
    if not hits:
        return NOT_FOUND, "extractive"
    if LLM != "none":
        try:
            return _llm_answer(question, hits), "llm"
        except Exception as e:  # noqa: BLE001 — 대안: 원문 기반 답변으로 계속
            log.warning("LLM 실패, 원문 답변으로 대체: %s", type(e).__name__)
    return extractive_answer(question, hits), "extractive"


# ---------- API ----------

class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    k: Optional[int] = Field(default=None, ge=1, le=10)


async def ask(question: str, k: Optional[int] = None) -> dict:
    k = k or TOP_K
    t0 = time.perf_counter()
    search = _search_fn()
    # 임베딩은 무거우므로 스레드풀에서 (합의사항: 다른 기능을 멈추지 않게)
    raw = await search(question, k) if inspect.iscoroutinefunction(search) else await run_in_threadpool(search, question, k)
    hits = _clean_hits(list(raw or []), k)
    t1 = time.perf_counter()
    answer, mode = await run_in_threadpool(make_answer, question, hits)
    t2 = time.perf_counter()
    return {
        "answer": answer,
        "sources": [{"card_no": h.get("card_no"), "title": h.get("title"),
                     "text": h.get("text"), "score": round(float(h.get("score") or 0), 4)} for h in hits],
        "found": answer != NOT_FOUND,
        "mode": mode,
        "ms": {"search": round((t1 - t0) * 1000), "answer": round((t2 - t1) * 1000)},
    }


@router.post("/api/ask")
async def api_ask(body: AskIn):
    try:
        return await ask(body.question.strip(), body.k)
    except Exception as e:  # noqa: BLE001
        log.exception("ask failed")
        raise HTTPException(status_code=500, detail=f"검색 실패: {type(e).__name__}")


# ---------- 단독 실행 (C 의 store 없이) ----------

ROOT = Path(__file__).resolve().parents[1]


def passages_search(path: Path = ROOT / "data" / "passages.json") -> SearchFn:
    """data/passages.json 원문으로 만든 키워드(BM25) 검색. store.search 와 같은 dict 를 돌려준다."""
    import json
    import math

    docs = json.loads(path.read_text(encoding="utf-8"))
    toks = [re.findall(r"[a-z0-9]+", (d["title"] + " " + d["text"]).lower()) for d in docs]
    df: dict[str, int] = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    n, avg = len(docs), sum(map(len, toks)) / len(docs)

    def search(query: str, k: int) -> list[dict]:
        q = [w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in _STOP]
        scored = []
        for d, t in zip(docs, toks):
            s = 0.0
            for w in q:
                f = t.count(w)
                if f:
                    idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                    s += idf * f * 2.4 / (f + 1.4 * (0.25 + 0.75 * len(t) / avg))
            if s > 0:
                scored.append((s, d))
        scored.sort(key=lambda x: -x[0])
        return [{**d, "score": s} for s, d in scored[:k]]

    return search


if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    use_search(passages_search())
    dev = FastAPI(title="RAG dev")
    dev.include_router(router)
    dev.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @dev.get("/", response_class=HTMLResponse)
    def page():  # dashboard.html 은 A 담당이라, 같은 틀의 RAG 영역만 띄운다
        return ('<!doctype html><meta charset="utf-8"><title>RAG dev</title>'
                '<link rel="stylesheet" href="/static/css/common.css">'
                '<style>#zone-rag .zone-no{background:var(--violet)}</style>'
                '<body style="padding:24px;max-width:720px">'
                '<section class="zone" id="zone-rag"><div class="zone-head"><span class="zone-no">6</span>'
                '<h2>RAG 검색</h2></div></section><script src="/static/js/rag.js"></script>')

    uvicorn.run(dev, host="127.0.0.1", port=8001)
