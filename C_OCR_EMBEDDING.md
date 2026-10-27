# C OCR·임베딩 통합 명세

## 기준과 범위
팀main a21954dfad6a838f6513c63e136b3e4e7d2b841c에 기존 OCR-Embedding C 구현을 선택 반영한다. A의 QR/현황·B의 JPEG/4MB/재연결·D의 실제검색/원문답변/선택LLM 코드는 유지한다. main lifecycle과 C 의존성만 공용에 추가한다.

## 실행
```powershell
./scripts/setup_ocr.ps1 -PythonPath 'C:\path\python.exe' -Gpu
# .env: OCR_DEVICE=cuda, EMBEDDING_DEVICE=cuda
.venv/Scripts/python.exe -m app
```
CPU는 -Gpu 없이 설치하고 장치를 cpu로 설정한다. GPU검증조합은 torch2.11.0+cu128 / ONNX Runtime GPU1.24.4, RapidOCR3.10이다. ORT CPU/GPU 패키지는 함께 설치하지 않는다. 모델 로드·예열이 서버startup에서 끝난 뒤 QR/카메라 요청을 받는다.

## 처리와 팀 계약
- QR→기존 /ws/camera JPEG→ocr.submit(sid,jpeg,ts)→최신대기1/처리1→중앙카드 추적·간격0.4초이상3관측→품질gate→저장큐8→BGE-M3/SQLite→최신D store.search→원문답변/선택LLM.
- PP-OCRv5 server det + English mobile rec, ORT CPU/CUDA FP32. BGE-M3 dense1024, CPU FP32배치4/CUDA FP16배치8, 입력≤512토큰·긴문서350/겹침50.
- C는JPEG≤4MB·긴변≤1920을검사한다. B송신기본은1280px/품질0.8/2.5fps이며 이전1920px 측정과같지않다. 수신전WebSocket메모리제한은서버측별도범위다.
- 텍스트 불일치·최소15단어미달·낮은줄점수(<0.8)·하단잘림은 보류한다. 관측일치가 정답보장은아니다. 완료track도1초마다재판독한다. 같은위치 새본문은새track으로분리한다.
- doc.saved는로컬문서+벡터transaction완료다. 원격저장성공과구분한다. 기존 ocr.done/frame.dropped/doc.duplicate/error 이벤트형식을유지한다.
- profile+canonical본문SHA256 중복, 카드번호는로컬DB번호다. count/recent는전체모델이력, search는현재profile만. D의source번호와정답100장카드번호를혼동하지않는다.
- 실제가중치/토크나이저/Pooling/패키지/장치·정밀도/C코드를지문으로구분한다. 캐시는실입력별, 모델/프로필변경시서버재시작·재처리한다. dimension/finite/norm을검증하고손상캐시는다시계산한다.

## 검토·원격 저장
```powershell
python scripts/review_document.py
python scripts/review_document.py --candidate <ID> --text-file corrected.txt --confirm
python scripts/review_document.py --sync-pending
```
보류문서는data/runtime의후보DB와근거JPEG로남기며, 근거확인·수정텍스트·명시적확인·동일profile에서만승인한다. 원격은supabase/migrations/001_team_ocr.sql의추가형team_ocr_*테이블/RPC를사용한다.
.env의SUPABASE_URL/교체한SECRET_KEY/실제Auth OWNER_ID/선택DATASET으로설정한다. 비밀값은브라우저나채팅에넣지않는다. URL/owner/dataset별영수증을보존하고실패시로컬보존·100건단위재시도한다. 원격문서·벡터·근거메타데이터만연결했으며 영상/JPEG Storage업로드와실제SQL/RLS/RPC검증은남았다.

## 테스트와 실모델 확인
```powershell
python -m pip install -r requirements-test.txt
python -m pytest -q
python scripts/verify_ocr_embedding.py --video sample_input.mp4
python scripts/replay_video.py sample_input.mp4 --insecure-dev
```
빠른테스트는새임시DB·모델준비대역으로B/D/C계약을검증한다. D참조정답은검색대역에서만쓰며운영store에적재하지않는다. 실제모델검증은매실행새DB를강제하고Supabase/외부LLM을끄며기존모델캐시만재사용한다.
검증은이번sid의신규doc.saved/id/제목/본문/1024벡터norm/검색top1·D원문답변을확인한다. 선택sample영상의보류OCR벡터는검증용계산만하고색인/원격적재하지않는다. 실제폰촬영/무관질문검색품질/실Supabase는별도검증이다.

## 소스 명세
| source_name/type/provider | source_url | license_or_terms / usage |
|---|---|---|
| PP-OCRv5/model/Paddle | https://github.com/PaddlePaddle/PaddleOCR | Apache-2.0·검출/영어인식 |
| RapidOCR/asset/RapidAI | https://github.com/RapidAI/RapidOCR | Apache-2.0·ONNX어댑터 |
| ONNX Runtime/asset/Microsoft | https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html | MIT·CPU/CUDA |
| BGE-M3/model/BAAI | https://huggingface.co/BAAI/bge-m3 | MIT·다국어dense1024 |
| Supabase/API/Supabase | https://supabase.com/docs/guides/ai/vector-columns | 서비스약관·Postgres/pgvector |

request_method/parameters=로컬추론·최초가중치HTTPS GET·team_ocr_ingest POST(owner,dataset,profile,document,chunks). response_core_fields=text/lines/confidence/boxes·dense1024·doc_id영수증. spatial_fields=픽셀좌표만. storage_location=OCR_DATA_DIR 기본data/runtime, 모델OCR_MODEL_DIR 기본runtime/models, 원격team_ocr_*.
connected_code=app/ocr·store·engine·remote; join_key=streamUUID/track/profile+contentSHA/owner+dataset+doc_id. quality_gate=3관측·불일치보류·벡터차원/finite/norm·RLS/receipt. failure_policy=최신프레임생략·로컬보존·제한재시도·묵시적모델fallback금지. change_history=기존C검증코어를최신팀B/D에연결, 검증시RAG_LLM=none을import전에강제.

파일 갱신 시각: 2026-10-27 15:34:39 +09:00
