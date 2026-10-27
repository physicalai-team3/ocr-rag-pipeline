"""⑥ RAG 테스트. C 의 실제 store 없이 data/passages.json 으로 돈다.

    pytest tests/test_rag.py -q
"""
import sys
import types

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import rag

SEARCH = rag.passages_search()


@pytest.fixture()
def client():
    rag.use_search(SEARCH)
    app = FastAPI()
    app.include_router(rag.router)
    yield TestClient(app)
    rag.use_search(None)


def ask(client, q):
    r = client.post("/api/ask", json={"question": q})
    assert r.status_code == 200, r.text
    return r.json()


# 최종 통합 테스트 체크리스트의 RAG 항목
@pytest.mark.parametrize("q, expect, card", [
    ("What water temperature kills yeast?", "46 °C", 14),
    ("What is the packaging capacity at Plant B per shift?", "6,000", 4),
    ("When did paper reach Baghdad?", "793", 9),
    ("What torque should part 41-B/7 be tightened to?", "24 N·m", 12),
])
def test_checklist_questions(client, q, expect, card):
    d = ask(client, q)
    assert expect in d["answer"]
    assert card in [s["card_no"] for s in d["sources"]]  # 순위는 C 의 검색 품질 몫
    assert f"[#{card:03d} " in d["answer"]  # 출처 형식 #014 How Bread Fails


def test_response_shape_matches_agreement(client):
    d = ask(client, "What water temperature kills yeast?")
    assert set(d) >= {"answer", "sources"}
    assert all({"card_no", "title"} <= set(s) for s in d["sources"])
    assert len(d["sources"]) <= rag.TOP_K


def test_not_found_and_validation(client):
    rag.use_search(lambda q, k: [])
    d = ask(client, "quantum chromodynamics")
    assert d["answer"] == rag.NOT_FOUND and d["found"] is False and d["sources"] == []
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_default_uses_c_store_search(monkeypatch):
    calls = []
    fake = types.ModuleType("app.store")
    fake.search = lambda q, k: calls.append((q, k)) or [
        {"doc_id": "014", "card_no": 14, "title": "How Bread Fails",
         "text": "How Bread Fails\nAbove 46 °C will kill it outright.", "score": 0.8},
        {"doc_id": "014", "card_no": 14, "title": "How Bread Fails", "text": "dup", "score": 0.7},
    ]
    monkeypatch.setitem(sys.modules, "app.store", fake)
    monkeypatch.setattr(sys.modules["app"], "store", fake, raising=False)
    rag.use_search(None)
    app = FastAPI()
    app.include_router(rag.router)
    d = TestClient(app).post("/api/ask", json={"question": "yeast temperature"}).json()
    assert calls == [("yeast temperature", 3)]
    assert len(d["sources"]) == 1  # 같은 doc_id 한 번만
    assert d["answer"].endswith("[#014 How Bread Fails]")


def test_async_search_and_min_score(client, monkeypatch):
    async def asearch(q, k):
        return [{"doc_id": "1", "card_no": 1, "title": "A", "text": "A alpha", "score": 0.9},
                {"doc_id": "2", "card_no": 2, "title": "B", "text": "B beta", "score": 0.1}]
    rag.use_search(asearch)
    monkeypatch.setattr(rag, "MIN_SCORE", "0.5")
    d = ask(client, "alpha")
    assert [s["card_no"] for s in d["sources"]] == [1]


def test_llm_failure_falls_back_to_extractive(client, monkeypatch):
    monkeypatch.setattr(rag, "LLM", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # KeyError → 원문 답변으로 대체
    d = ask(client, "What water temperature kills yeast?")
    assert d["mode"] == "extractive" and "46 °C" in d["answer"]


def test_through_main_app_without_indexing_reference_corpus(monkeypatch):
    """The D fixture stays a search double; C quality gates are never bypassed."""
    from app import main, store
    calls=[]
    def search(question,k):
        calls.append((question,k))
        return SEARCH(question,k)
    monkeypatch.setattr(store,'search',search)
    rag.use_search(None)
    with TestClient(main.app) as client:
        result=ask(client,'What water temperature kills yeast?')
    assert calls==[('What water temperature kills yeast?',3)]
    assert '46 °C' in result['answer']
    assert result['sources'] and all({'card_no','title'} <= set(row) for row in result['sources'])
    assert store.count()==0
