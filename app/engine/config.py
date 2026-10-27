from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env')


@dataclass(frozen=True)
class Settings:
    version: str = '1.3.0-team1'
    ocr_backend: str = os.getenv('OCR_BACKEND', 'rapidocr')
    ocr_device: str = os.getenv('OCR_DEVICE', 'cpu')
    embedding_device: str = os.getenv('EMBEDDING_DEVICE', 'auto')
    cpu_threads: int = 4
    live_max_bytes: int = 4 * 1024 * 1024
    live_deadline_seconds: float = 1.0
    live_queue_seconds: float = .5
    live_ttl_seconds: float = 30.0
    sample_fps: float = 5.0
    candidates: int = 3
    min_observation_gap: float = 0.4
    min_line_confidence: float = 0.80
    min_words: int = 15
    embedding_model: str = 'BAAI/bge-m3'
    embedding_dimension: int = 1024
    embedding_limit: int = 512
    chunk_tokens: int = 350
    overlap_tokens: int = 50
    detection_model: str = 'PP-OCRv5_server_det'
    recognition_model: str = 'en_PP-OCRv5_mobile_rec'
    max_bytes: int = 200 * 1024 * 1024
    max_seconds: int = 1800
    max_dimension: int = 4096

    def fingerprint(self, manifest=None):
        payload = {'settings': asdict(self), 'models': manifest or {}}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


SETTINGS = Settings()


def data_dir():
    path = Path(os.getenv('OCR_DATA_DIR', str(ROOT / 'data' / 'runtime'))).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def model_dir():
    path = Path(os.getenv('OCR_MODEL_DIR', str(data_dir() / 'models'))).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
