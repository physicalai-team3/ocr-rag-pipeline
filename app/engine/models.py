import hashlib
import importlib.metadata
import os
import json
import sqlite3
from pathlib import Path
import re
import threading
import unicodedata
import numpy as np
from .config import SETTINGS, data_dir, model_dir

MODEL_LOCK = threading.RLock()
OCR_LOCK = threading.RLock()
EMBED_LOCK = threading.RLock()
_hash_cache = {}
_runtime = {}
_ocr = None
_embedder = None
_loaded_manifest_identity = None


def normalize(text):
    # NFC preserves distinctions in units and numbers; no case folding of body.
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def packages():
    result = {}
    for name in ('paddleocr', 'paddlepaddle', 'sentence-transformers', 'torch', 'opencv-python-headless', 'rapidocr', 'onnxruntime', 'onnxruntime-gpu', 'transformers', 'tokenizers'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = 'missing'
    return result


def ocr_model():
    global _ocr
    with OCR_LOCK:
        if _ocr is None:
            if SETTINGS.ocr_backend == 'rapidocr':
                _ocr = RapidAdapter()
                return _ocr
            if SETTINGS.ocr_backend != 'paddle' or SETTINGS.ocr_device != 'cpu':
                raise ValueError('OCR 프로필은 rapidocr cpu/cuda 또는 paddle cpu만 허용합니다.')
            _runtime['ocr'] = {'backend': 'paddle', 'device': 'cpu', 'precision': 'float32'}
            os.environ.setdefault('PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK', 'True')
            cache = model_dir()
            cache.mkdir(exist_ok=True)
            os.environ['PADDLE_PDX_CACHE_HOME'] = str(cache / 'paddlex')
            # Paddle's dataset import hardcodes expanduser('~') without a cache option.
            # Scope the redirect to initialization so libraries never write to user home.
            from unittest.mock import patch
            expanduser = os.path.expanduser
            with patch('os.path.expanduser', side_effect=lambda p: str(cache / 'paddle-home') if p == '~' else expanduser(p)):
                from paddleocr import PaddleOCR
                _ocr = PaddleOCR(text_detection_model_name=SETTINGS.detection_model,
                                 text_recognition_model_name=SETTINGS.recognition_model,
                                 use_doc_orientation_classify=False, use_doc_unwarping=False,
                                 use_textline_orientation=False, device='cpu', cpu_threads=SETTINGS.cpu_threads,
                                 enable_mkldnn=False)
        return _ocr


def embedding_model():
    global _embedder
    with EMBED_LOCK:
        if _embedder is None:
            import torch
            device = SETTINGS.embedding_device
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device not in {'cpu', 'cuda'} or (device == 'cuda' and not torch.cuda.is_available()):
                raise ValueError('임베딩 실행 장치를 사용할 수 없습니다.')
            torch.set_num_threads(SETTINGS.cpu_threads)
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(SETTINGS.embedding_model, device=device,
                                           cache_folder=str(model_dir() / 'hf'))
            model.max_seq_length = SETTINGS.embedding_limit
            if device == 'cuda':
                model.half()
            else:
                model.float()
            model.eval()
            _embedder = model
            _runtime['embedding'] = {'device': device, 'precision': 'float16' if device == 'cuda' else 'float32'}
        return _embedder


def recognize(image, region=None):
    height = image.shape[0]
    if region is not None:
        x, y, width, roi_height = region
        if not (x == 0 and y == 0 and width == image.shape[1] and 0 < roi_height <= height):
            raise ValueError('보조 ROI 좌표 검증 실패')
        image = image[:roi_height]
    with OCR_LOCK:
        output = list(ocr_model().predict(image))
    if not output:
        return {'text': '', 'lines': [], 'minimum_confidence': 0, 'bottom_margin': 0}
    data = dict(output[0])
    lines = []
    raw_lines, excluded_lines = [], []
    polygons = data.get('rec_polys', data.get('dt_polys', []))
    for text, confidence, polygon in zip(data.get('rec_texts', []), data.get('rec_scores', []), polygons):
        points = np.asarray(polygon, dtype=float)
        text = normalize(text)
        raw_line = {'text': text, 'confidence': float(confidence), 'polygon': points.tolist()}
        raw_lines.append(raw_line)
        if not text:
            continue
        # Source counter belongs to metadata, not a document heading. Moire may
        # also produce isolated punctuation in otherwise blank lower-card areas.
        if points[:, 0].min() > image.shape[1] * .78 and points[:, 1].mean() < height * .06:
            excluded_lines.append({**raw_line, 'reason': 'top-right card label'})
            continue
        if sum(character.isalnum() for character in text) < 1:
            excluded_lines.append({**raw_line, 'reason': 'isolated punctuation; retained in raw_lines'})
            continue
        lines.append({'text': text, 'confidence': float(confidence), 'polygon': points.tolist(),
                      'y': float(points[:, 1].mean()), 'x': float(points[:, 0].min())})
    # Cards contain a single column. A light row tolerance avoids tiny baseline jitter.
    lines.sort(key=lambda line: (round(line['y'] / 10), line['x']))
    maximum_y = max((max(p[1] for p in line['polygon']) for line in lines), default=image.shape[0])
    # Text ending at image edge is incomplete even if OCR emits punctuation.
    margin = (height - maximum_y) / height
    return {'text': '\n'.join(line['text'] for line in lines), 'lines': lines,
            'raw_lines': raw_lines, 'excluded_lines': excluded_lines,
            'input_region': region or [0, 0, image.shape[1], height],
            'mean_confidence': float(np.mean([line['confidence'] for line in lines])) if lines else 0,
            'minimum_confidence': min((line['confidence'] for line in lines), default=0),
            'bottom_margin': float(margin)}


def validate_vectors(vectors, dimension=1024):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != dimension or not np.isfinite(vectors).all():
        raise ValueError('임베딩 차원 또는 유한수 검증 실패')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=.002):
        raise ValueError('임베딩 단위 길이 검증 실패')
    return vectors


def split_text(text, tokenizer, maximum=350, overlap=50):
    if maximum <= overlap or overlap < 0:
        raise ValueError('invalid overlap')
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) + tokenizer.num_special_tokens_to_add(False) <= SETTINGS.embedding_limit:
        return [text]
    result = []
    start = 0
    while start < len(token_ids):
        end = min(start + maximum, len(token_ids))
        # Prefer a sentence boundary in the final quarter of the window.
        if end < len(token_ids):
            for boundary in range(end, max(start + maximum * 3 // 4, start + overlap), -1):
                candidate = tokenizer.decode(token_ids[start:boundary], skip_special_tokens=True)
                if candidate.rstrip().endswith(('.', '!', '?', ';')):
                    end = boundary
                    break
        chunk = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
        if len(tokenizer.encode(chunk, add_special_tokens=True)) > SETTINGS.embedding_limit:
            raise ValueError('분할 텍스트 토큰 한도 초과')
        result.append(chunk)
        if end == len(token_ids):
            break
        start = end - overlap
    return result


def embedding_identity():
    # Include actual weights, source representation, device and precision.
    embedding_model()
    manifest = model_manifest()
    payload = {key: manifest.get(key) for key in ('embedding_model', 'embedding_revision', 'files', 'pipeline_sources', 'runtime', 'packages')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def embed(texts):
    if not texts:
        return np.empty((0, SETTINGS.embedding_dimension), np.float32)
    if len(texts) > 10000:
        raise ValueError('임베딩 배치 입력 한도 초과')
    import torch
    with EMBED_LOCK:
        model = embedding_model()
        # Preserve exact supplied text: no case or whitespace rewriting for cache hits.
        unique = list(dict.fromkeys(texts))
        lengths = {text: len(model.tokenizer.encode(text, add_special_tokens=True)) for text in unique}
        if any(size > SETTINGS.embedding_limit for size in lengths.values()):
            raise ValueError('임베딩 입력이 512토큰을 초과합니다.')
        identity = embedding_identity()
        result, missing = {}, []
        with sqlite3.connect(data_dir() / 'embeddings.sqlite', timeout=30) as db:
            db.execute('create table if not exists embeddings (profile text, text_hash text, vector blob, primary key(profile,text_hash))')
            for text in unique:
                key = hashlib.sha256(text.encode()).hexdigest()
                row = db.execute('select vector from embeddings where profile=? and text_hash=?', (identity, key)).fetchone()
                try:
                    if row is None:
                        raise ValueError('cache miss')
                    result[text] = validate_vectors(np.frombuffer(row[0], dtype=np.float32).reshape(1, -1))[0].copy()
                except (ValueError, TypeError):
                    missing.append(text)
            missing.sort(key=lambda text: lengths[text])
            batch_size = 8 if _runtime.get('embedding', {}).get('device') == 'cuda' else 4
            with torch.inference_mode():
                for start in range(0, len(missing), batch_size):
                    batch = missing[start:start + batch_size]
                    vectors = model.encode(batch, batch_size=batch_size, normalize_embeddings=True,
                                           convert_to_numpy=True, show_progress_bar=False)
                    vectors = np.asarray(vectors, dtype=np.float32)
                    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                    vectors = validate_vectors(vectors)
                    for text, vector in zip(batch, vectors):
                        result[text] = vector
                        db.execute('insert or replace into embeddings values (?,?,?)',
                                   (identity, hashlib.sha256(text.encode()).hexdigest(), vector.tobytes()))
        return validate_vectors(np.stack([result[text] for text in texts]))


def document_chunks(text, tokenizer):
    canonical = normalize(text)
    chunks = split_text(canonical, tokenizer, SETTINGS.chunk_tokens, SETTINGS.overlap_tokens)
    if len(chunks) == 1:
        return chunks
    title = normalize(text.splitlines()[0])
    # Avoid silently truncating exceptionally long or malformed headings.
    if len(tokenizer.encode(title, add_special_tokens=False)) > 100:
        raise ValueError('제목이 지나치게 길어 문서 검토가 필요합니다.')
    output = [chunk if index == 0 else title + '\n' + chunk for index, chunk in enumerate(chunks)]
    if any(len(tokenizer.encode(chunk, add_special_tokens=True)) > SETTINGS.embedding_limit for chunk in output):
        raise ValueError('제목 포함 청크 토큰 한도 초과')
    return output


def model_manifest():
    manifest = {'runtime': dict(_runtime), 'packages': packages(), 'ocr_models': [SETTINGS.detection_model, SETTINGS.recognition_model],
                'embedding_model': SETTINGS.embedding_model, 'files': {}, 'pipeline_sources': {}}
    for filename in ('config.py', 'models.py', 'video.py', 'quality.py'):
        manifest['pipeline_sources'][filename] = hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
    for filename in ('ocr.py', 'store.py', 'remote.py'):
        manifest['pipeline_sources']['app/' + filename] = hashlib.sha256(Path(__file__).parents[1].joinpath(filename).read_bytes()).hexdigest()
    # Weight/config hashes are data provenance, not secrets. Hash only used model folders.
    roots = []
    paddle_cache = model_dir() / 'paddlex' / 'official_models'
    if SETTINGS.ocr_backend == 'paddle':
        for name in (SETTINGS.detection_model, SETTINGS.recognition_model):
            roots.append(paddle_cache / name)
    else:
        roots.append(model_dir() / 'rapidocr')
    if _embedder is not None:
        config = _embedder[0].auto_model.config
        manifest['embedding_revision'] = getattr(config, '_commit_hash', None)
        from huggingface_hub import try_to_load_from_cache
        for filename in ('model.safetensors', 'pytorch_model.bin', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'sentencepiece.bpe.model', 'sentence_bert_config.json', 'modules.json', '1_Pooling/config.json'):
            cached = try_to_load_from_cache(SETTINGS.embedding_model, filename, revision=manifest['embedding_revision'] or 'main',
                                           cache_dir=str(model_dir() / 'hf'))
            if isinstance(cached, str):
                roots.append(Path(cached))
    for root in roots:
        files = [root] if root.is_file() else list(root.rglob('*')) if root.exists() else []
        for path in files:
            if path.is_file() and path.suffix in {'.json', '.yml', '.yaml', '.pdparams', '.pdiparams', '.safetensors', '.bin', '.onnx', '.txt', '.model'}:
                manifest['files'][path.relative_to(model_dir()).as_posix()] = file_digest(path)

    return manifest


def runtime_fingerprint():
    """Identity of actually loaded weights, not just mutable model names."""
    global _loaded_manifest_identity
    with MODEL_LOCK:
        ocr_model()
        embedding_model()
        fingerprint = SETTINGS.fingerprint(model_manifest())
        if _loaded_manifest_identity is None:
            _loaded_manifest_identity = fingerprint
        elif fingerprint != _loaded_manifest_identity:
            raise ValueError('실행 중 모델 파일이 변경됐습니다. 서버를 재시작하고 다시 처리하세요.')
        return fingerprint


def file_digest(path):
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _hash_cache:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        after = path.stat()
        if (after.st_size, after.st_mtime_ns, after.st_ctime_ns) != key[1:]:
            raise ValueError('해시 계산 중 모델 파일 변경')
        _hash_cache.clear() if len(_hash_cache) > 2048 else None
        _hash_cache[key] = digest.hexdigest()
    return _hash_cache[key]


class RapidAdapter:
    def __init__(self):
        from rapidocr import RapidOCR, OCRVersion, ModelType, LangRec
        import onnxruntime as ort
        device = SETTINGS.ocr_device
        if device not in {'cpu', 'cuda'}:
            raise ValueError('OCR_DEVICE는 cpu 또는 cuda여야 합니다.')
        if device == 'cuda':
            import torch  # Load the matching CUDA/cuDNN runtime first on Windows.
            if not torch.cuda.is_available():
                raise ValueError('CUDA OCR을 사용할 수 없습니다.')
            ort.preload_dlls()
        self.engine = RapidOCR(params={
            'Global.model_root_dir': str(model_dir() / 'rapidocr'),
            'Global.use_cls': False, 'Global.text_score': 0.0,
            'Det.ocr_version': OCRVersion.PPOCRV5, 'Det.model_type': ModelType.SERVER,
            'Rec.ocr_version': OCRVersion.PPOCRV5, 'Rec.lang_type': LangRec.EN,
            'Rec.model_type': ModelType.MOBILE,
            'EngineConfig.onnxruntime.use_cuda': device == 'cuda',
            'EngineConfig.onnxruntime.intra_op_num_threads': SETTINGS.cpu_threads,
            'EngineConfig.onnxruntime.inter_op_num_threads': 1,
            'EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search': 'HEURISTIC'})
        # RapidOCR 3.10 loads lazily; resolve sessions before recording identity.
        det = self.engine._load_det_model().session.session
        rec = self.engine._load_rec_model().session.session
        providers = {'det': det.get_providers(), 'rec': rec.get_providers()}
        if device == 'cuda' and any('CUDAExecutionProvider' not in value for value in providers.values()):
            raise ValueError('CUDA provider 초기화 실패: CPU 모드를 명시적으로 선택하세요.')
        _runtime['ocr'] = {'backend': 'rapidocr', 'device': device, 'precision': 'float32', 'providers': providers}

    def predict(self, image):
        result = self.engine(image)
        return [{'rec_texts': result.txts or [], 'rec_scores': result.scores if result.scores is not None else [],
                 'rec_polys': result.boxes if result.boxes is not None else []}]
