"""Fast deterministic contracts; real models run through verify_ocr_embedding.py."""
import pytest
from app import ocr,rag


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path,monkeypatch):
    monkeypatch.setenv('OCR_DATA_DIR',str(tmp_path))
    monkeypatch.delenv('OCR_MODEL_DIR',raising=False)
    monkeypatch.delenv('SUPABASE_SECRET_KEY',raising=False)
    monkeypatch.setattr(ocr,'prepare_models',lambda:None)
    monkeypatch.setattr(rag,'LLM','none')
    monkeypatch.setattr(rag,'MIN_SCORE',None)
    monkeypatch.setattr(rag,'TOP_K',3)
    rag.use_search(None)
    yield
    rag.use_search(None)
