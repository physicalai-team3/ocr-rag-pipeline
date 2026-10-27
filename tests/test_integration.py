import asyncio
import json
import threading
import time
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from app import ocr,store,events

TEXT='Workshop Notice\nThe repair workshop opens every morning at eight and closes in the evening. Bring your equipment and keep the receipt.'


@pytest.fixture(autouse=True)
def isolate(tmp_path,monkeypatch):
    monkeypatch.setenv('OCR_DATA_DIR',str(tmp_path))
    monkeypatch.delenv('OCR_MODEL_DIR',raising=False)
    monkeypatch.delenv('SUPABASE_SECRET_KEY',raising=False)
    monkeypatch.setattr(ocr,'prepare_models',lambda:None)


@pytest.fixture
def models(monkeypatch):
    profile=['a'*64]
    monkeypatch.setattr(store,'runtime_fingerprint',lambda:profile[0])
    monkeypatch.setattr(store,'embedding_model',lambda:SimpleNamespace(tokenizer=None))
    monkeypatch.setattr(store,'document_chunks',lambda text,tokenizer:[text])
    def embed(texts):
        result=np.zeros((len(texts),1024),np.float32)
        for i,text in enumerate(texts):result[i,0 if 'eight' in text.lower() else 1]=1
        return result
    monkeypatch.setattr(store,'embed',embed)
    return profile


def doc(text=TEXT):return {'text':text,'sid':'session','status':'ready','ms':12,'provenance':{'source':'test'}}


def test_durable_vectors_dedup_and_profile_contract(models):
    one=doc();assert store.upsert(one)
    assert not store.upsert(doc(TEXT.replace('every morning','every  morning')))
    assert store.count()==1
    assert len(store.document(one['doc_id'])['chunks'][0]['embedding'])==1024
    assert store.search('eight',3)[0]['doc_id']==one['doc_id']
    models[0]='b'*64
    assert store.count()==1 and store.recent(6)[0]['doc_id']==one['doc_id']
    assert store.search('eight',3)==[] # historical count versus current profile search is explicit
    two=doc();assert store.upsert(two);assert two['doc_id']!=one['doc_id']
    assert store.search('eight',3)[0]['doc_id']==two['doc_id']
    assert store.count()==2
    for status in ('needs_review','partial'):
        bad=doc();bad['status']=status
        with pytest.raises(ValueError):store.upsert(bad)
    with pytest.raises(ValueError):store.search('x'*2001,3)


def test_failed_embedding_is_atomic(models,monkeypatch):
    def fail(texts):raise RuntimeError('model unavailable')
    monkeypatch.setattr(store,'embed',fail)
    with pytest.raises(RuntimeError):store.upsert(doc())
    assert store.count()==0


def test_remote_failure_preserves_local(models,monkeypatch):
    from app.remote import sync_document
    one=doc();store.upsert(one)
    assert sync_document(one['doc_id'])=='not_configured'
    monkeypatch.setenv('SUPABASE_URL','https://example.supabase.co')
    monkeypatch.setenv('SUPABASE_SECRET_KEY','test-only')
    monkeypatch.setenv('SUPABASE_OWNER_ID','00000000-0000-4000-8000-000000000001')
    import httpx
    client=httpx.Client
    transport=httpx.MockTransport(lambda request:httpx.Response(500,json={'error':'test'}))
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:client(transport=transport,**kwargs))
    assert sync_document(one['doc_id'])=='failed'
    assert store.count()==1 and store.recent(1)[0]['remote_status']=='failed'


def test_stream_valid_jpeg_three_observations_and_disconnect_isolation(monkeypatch):
    frame=np.full((400,300,3),240,np.uint8);_,jpeg=cv2.imencode('.jpg',frame)
    card={'box':[0,0,300,400],'image':frame,'signature':np.zeros((16,32)), 'centrality':1}
    monkeypatch.setattr(ocr,'runtime_fingerprint',lambda:'a'*64)
    monkeypatch.setattr(ocr,'detect_cards',lambda frame:[card])
    reading={'text':TEXT,'minimum_confidence':.99,'mean_confidence':.99,'bottom_margin':.2,'lines':[]}
    monkeypatch.setattr(ocr,'recognize',lambda *args:dict(reading))
    monkeypatch.setattr(ocr,'text_region',lambda frame:[0,0,300,400])
    processor=ocr.Processor();processor.reset('old');old_stream=processor.stream
    for index,seconds in enumerate((1,1.5,2)):
        output=processor.analyze(jpeg.tobytes(),index*500,seconds,old_stream)
    assert len(output[0][0].observations)==3
    assert ocr.finalize(output[0][0].observations)[0]=='ready'
    changed={**reading,'text':TEXT.replace('eight','nine')}
    monkeypatch.setattr(ocr,'recognize',lambda *args:dict(changed))
    replacement=processor.analyze(jpeg.tobytes(),3000,3.01,old_stream)
    assert replacement[0][0].number!=output[0][0].number
    assert len(replacement[0][0].observations)==1
    processor.reset('new')
    assert processor.stream is not old_stream and not processor.stream.tracks
    assert processor.stream.identifier!=old_stream.identifier
    with pytest.raises(Exception):processor.analyze(b'fake jpeg',0,3,processor.stream)
    _,huge=cv2.imencode('.jpg',np.zeros((64,1921,3),np.uint8))
    with pytest.raises(ValueError):processor.analyze(huge.tobytes(),0,3,processor.stream)


def test_latest_queue_immediate_return_and_generation_guard(monkeypatch):
    entered=threading.Event();release=threading.Event();seen=[]
    original_publish=events.publish
    async def publish(sid,event):seen.append((sid,event));return event
    monkeypatch.setattr(events,'publish',publish)
    def blocked(jpeg,ts,received,stream):
        entered.set();release.wait(3)
        observation={'ocr':{'text':TEXT,'minimum_confidence':.99,'bottom_margin':.3}}
        track=SimpleNamespace(number=0,observations=[observation]*3)
        return [(track,observation,'a'*64,stream.identifier)]
    async def run():
        processor=ocr.Processor();monkeypatch.setattr(processor,'analyze',blocked)
        await processor.start()
        try:
            start=time.perf_counter();await processor.submit('old',b'first',1)
            assert time.perf_counter()-start<.1
            for _ in range(100):
                if entered.is_set():break
                await asyncio.sleep(.01)
            assert entered.is_set()
            for n in range(10):await processor.submit('old',str(n).encode(),n+2)
            assert processor.pending[1]==b'9'
            assert sum(event.get('count',0) for _,event in seen if event['type']=='frame.dropped')==9
            assert processor.saves.maxsize==8
            processor.reset('new');release.set()
        finally:
            release.set();await processor.stop()
    asyncio.run(run())
    assert not any(event['type'] in ('ocr.done','doc.saved') for _,event in seen)
    assert not store.candidates()


def test_qr_capture_status_and_rag_contract(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    submitted=[]
    async def accept(sid,jpeg,ts):submitted.append((sid,jpeg,ts))
    monkeypatch.setattr(ocr,'submit',accept)
    with TestClient(app) as client:
        session=client.post('/api/sessions').json()
        assert client.get('/api/status').json()['count']==0
        with client.websocket_connect('/ws/dashboard?sid='+session['session_id']) as dashboard:
            assert dashboard.receive_json()['type']=='snapshot'
            with client.websocket_connect('/ws/camera?sid='+session['session_id']+'&t='+session['token']) as camera:
                _,jpeg=cv2.imencode('.jpg',np.zeros((64,64,3),np.uint8))
                camera.send_bytes(jpeg.tobytes())
                assert dashboard.receive_json()['type']=='session.paired'
            assert dashboard.receive_json()['type']=='session.disconnected'
        assert submitted and submitted[0][1]==jpeg.tobytes()
        result=client.post('/api/ask',json={'question':'workshop'}).json()
        assert set(result)>={'answer','sources','found','mode','ms'}


def test_remote_receipts_are_destination_scoped(models,monkeypatch):
    from app.remote import target_fingerprint
    monkeypatch.setenv('SUPABASE_URL','https://example.supabase.co')
    monkeypatch.setenv('SUPABASE_SECRET_KEY','test-only')
    monkeypatch.setenv('SUPABASE_OWNER_ID','00000000-0000-4000-8000-000000000001')
    one=doc();store.upsert(one)
    first=target_fingerprint();store.mark_remote(one['doc_id'],'synced',first)
    assert store.recent(1)[0]['remote_status']=='synced' and not store.pending_remote()
    monkeypatch.setenv('SUPABASE_DATASET','another')
    assert target_fingerprint()!=first
    assert store.recent(1)[0]['remote_status']=='pending'
    assert store.pending_remote()[0]['doc_id']==one['doc_id']
    for index in range(104):store.upsert(doc(TEXT+' Additional record '+str(index)))
    first_page=store.pending_remote();second_page=store.pending_remote(first_page[-1]['card_no'])
    assert len(first_page)==100 and len(second_page)==5


def test_model_preparation_precedes_worker_start(monkeypatch):
    calls=[]
    monkeypatch.setattr(ocr,'prepare_models',lambda:calls.append('models'))
    async def run():
        await ocr.start();calls.append('started')
        await ocr.stop()
    asyncio.run(run())
    assert calls==['models','started']


def test_concurrent_duplicate_commit_is_single_document(models):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(lambda _:store.upsert(doc()),range(12)))
    assert sum(results)==1 and store.count()==1
    assert len(store.document(store.recent(1)[0]['doc_id'])['chunks'])==1


def test_remote_success_uses_apikey_and_validated_vector_payload(models,monkeypatch):
    import httpx
    from app.remote import sync_document
    one=doc();store.upsert(one)
    monkeypatch.setenv('SUPABASE_URL','https://example.supabase.co')
    monkeypatch.setenv('SUPABASE_SECRET_KEY','test-only')
    monkeypatch.setenv('SUPABASE_OWNER_ID','00000000-0000-4000-8000-000000000001')
    seen=[]
    def handle(request):
        seen.append(request)
        assert request.headers['apikey']=='test-only' and 'authorization' not in request.headers
        payload=json.loads(request.content)['payload']
        assert len(payload['chunks'][0]['embedding'])==1024
        return httpx.Response(200,json=one['doc_id'])
    original=httpx.Client
    monkeypatch.setattr(httpx,'Client',lambda **kwargs:original(transport=httpx.MockTransport(handle),**kwargs))
    assert sync_document(one['doc_id'])=='synced' and seen
    assert store.recent(1)[0]['remote_status']=='synced'


def test_verification_isolates_preexisting_data_and_cloud(tmp_path,models,monkeypatch):
    from app.verification import isolated_settings
    import sqlite3,os
    one=doc();store.upsert(one)
    original=tmp_path/'pipeline.sqlite'
    monkeypatch.setenv('SUPABASE_SECRET_KEY','test-only')
    monkeypatch.setenv('RAG_LLM','openai')
    target=isolated_settings(tmp_path/'verification-project')
    assert target!=tmp_path and store.count()==0
    assert os.environ['SUPABASE_SECRET_KEY']==''
    assert os.environ['RAG_LLM']=='none'
    with sqlite3.connect(original) as db:assert db.execute('select count(*) from documents').fetchone()[0]==1


def test_real_store_contract_connects_to_latest_rag(models):
    from fastapi.testclient import TestClient
    from app.main import app
    one=doc();store.upsert(one)
    with TestClient(app) as client:
        result=client.post('/api/ask',json={'question':'Does the repair workshop open at eight?'}).json()
    assert result['found'] is True and result['mode']=='extractive'
    assert result['sources'][0]['title']==one['title']
    assert 'eight' in result['answer'].lower()
    assert result['sources'][0]['score']==1
    assert store.count()==1
