"""C: durable documents + real embeddings; preserves A/D's synchronous interface."""
from __future__ import annotations
import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
import time
import numpy as np
from .engine.config import data_dir
from .engine.models import (normalize, runtime_fingerprint, embedding_model,
                            document_chunks, embed, validate_vectors)


_SCHEMA_LOCK = threading.Lock()


@contextmanager
def connection():
    db = sqlite3.connect(data_dir() / 'pipeline.sqlite', timeout=30)
    db.row_factory = sqlite3.Row
    with _SCHEMA_LOCK:
        db.execute('pragma journal_mode=WAL')
        db.executescript('''
          create table if not exists documents (
            card_no integer primary key autoincrement, doc_id text unique not null,
            profile text not null, content_hash text not null, title text not null,
            text text not null, sid text not null, ms integer not null,
            provenance text not null, remote_status text not null, created_at integer not null);
          create table if not exists chunks (
            doc_id text not null, position integer not null, text text not null, vector blob not null,
            primary key(doc_id,position));
          create table if not exists remote_receipts (
            doc_id text not null, destination text not null, status text not null,
            updated_at integer not null, primary key(doc_id,destination));
          create table if not exists candidates (
            candidate_id text primary key, sid text not null, status text not null,
            text text not null, reason text not null, provenance text not null, created_at integer not null);
        ''')
    try:
        with db:
            yield db
    finally:
        db.close()


def upsert(doc: dict) -> bool:
    """True only after a new ready document and all vectors commit atomically."""
    text = doc.get('text', '').strip()
    if doc.get('status') != 'ready' or len(text.split()) < 15 or len(text) > 50000:
        raise ValueError('검토 전이거나 유효하지 않은 문서는 검색 DB에 저장할 수 없습니다.')
    profile = runtime_fingerprint()
    if doc.get('fingerprint', profile) != profile:
        raise ValueError('모델 프로필이 변경됐습니다.')
    digest = hashlib.sha256(normalize(text).encode()).hexdigest()
    identifier = hashlib.sha256((profile + ':' + digest).encode()).hexdigest()
    with connection() as db:
        existing = db.execute('select card_no from documents where doc_id=?', (identifier,)).fetchone()
    if existing:
        doc.update(doc_id=identifier, card_no=existing['card_no'])
        return False
    chunks = document_chunks(text, embedding_model().tokenizer)
    vectors = embed(chunks)
    if len(chunks) != len(vectors):
        raise ValueError('청크/벡터 개수 불일치')
    with connection() as db:
        db.execute('begin immediate')
        cursor = db.execute('''insert or ignore into documents
          (doc_id,profile,content_hash,title,text,sid,ms,provenance,remote_status,created_at)
          values (?,?,?,?,?,?,?,?,?,?)''', (identifier,profile,digest,text.splitlines()[0],text,
          doc['sid'],int(doc.get('ms',0)),json.dumps(doc.get('provenance',{}),ensure_ascii=False),
          'pending',int(time.time()*1000)))
        is_new = cursor.rowcount == 1
        if is_new:
            db.executemany('insert into chunks values (?,?,?,?)',
                          [(identifier,i,chunk,vector.tobytes()) for i,(chunk,vector) in enumerate(zip(chunks,vectors))])
        number = db.execute('select card_no from documents where doc_id=?',(identifier,)).fetchone()['card_no']
    doc.update(doc_id=identifier,card_no=number,title=text.splitlines()[0],preview=text[:120],chars=len(text))
    return is_new


def count() -> int:
    # A contract: global unique doc_id count, including historical model profiles.
    with connection() as db:
        return db.execute('select count(*) from documents').fetchone()[0]


def recent(n: int) -> list[dict]:
    from .remote import target_fingerprint
    target=target_fingerprint()
    with connection() as db:
        rows = db.execute('''select d.*,r.status as target_status from documents d
          left join remote_receipts r on r.doc_id=d.doc_id and r.destination=?
          order by d.card_no desc limit ?''', (target or '',max(0,min(n,100)))).fetchall()
    return [{**{key:row[key] for key in ('doc_id','card_no','title','ms')},
             'remote_status':row['target_status'] or ('pending' if target else 'not_configured'),
             'preview':row['text'][:120], 'chars':len(row['text'])} for row in rows]


def search(query: str, k: int) -> list[dict]:
    if not 1 <= len(query.strip()) <= 2000 or not 1 <= k <= 10:
        raise ValueError('질문/검색 개수 제한을 확인하세요.')
    # Empty databases do not download/load models for a dashboard RAG query.
    if not count():
        return []
    profile = runtime_fingerprint()
    with connection() as db:
        rows = db.execute('''select d.doc_id,d.card_no,d.title,d.text,c.vector from documents d
                             join chunks c on c.doc_id=d.doc_id where d.profile=?''',(profile,)).fetchall()
    if not rows:
        return []
    query_vector = embed([query])[0]
    vectors = validate_vectors(np.stack([np.frombuffer(row['vector'],dtype=np.float32) for row in rows]))
    scores = vectors @ query_vector
    seen, hits = set(), []
    for index in np.argsort(-scores):
        row=rows[int(index)]
        if row['doc_id'] in seen:
            continue
        seen.add(row['doc_id'])
        hits.append({**{key:row[key] for key in ('doc_id','card_no','title','text')},'score':float(scores[index])})
        if len(hits) == k:
            break
    return hits


def save_candidate(candidate_id, sid, status, text, reason, provenance):
    with connection() as db:
        db.execute('insert or replace into candidates values (?,?,?,?,?,?,?)',
                   (candidate_id,sid,status,text,reason,json.dumps(provenance,ensure_ascii=False),int(time.time()*1000)))


def candidates(limit=30):
    with connection() as db:
        return [dict(row) for row in db.execute('select * from candidates order by created_at desc limit ?', (limit,))]


def document(identifier):
    with connection() as db:
        row=db.execute('select * from documents where doc_id=?',(identifier,)).fetchone()
        if not row:return None
        chunks=db.execute('select * from chunks where doc_id=? order by position',(identifier,)).fetchall()
    return {**dict(row),'provenance':json.loads(row['provenance']),
            'chunks':[{'position':r['position'],'text':r['text'],
                       'embedding':validate_vectors(np.frombuffer(r['vector'],np.float32).reshape(1,-1))[0].tolist()} for r in chunks]}


def mark_remote(identifier, status, target=None):
    if status not in {'pending','synced','failed','not_configured'}:raise ValueError('invalid remote status')
    with connection() as db:
        db.execute('update documents set remote_status=? where doc_id=?',(status,identifier))
        if target:
            db.execute('insert or replace into remote_receipts values (?,?,?,?)',
                       (identifier,target,status,int(time.time()*1000)))


def pending_remote(after_card=0, limit=100):
    from .remote import target_fingerprint
    target=target_fingerprint() or ''
    with connection() as db:
        return [dict(row) for row in db.execute('''select d.doc_id,d.card_no from documents d
          left join remote_receipts r on r.doc_id=d.doc_id and r.destination=?
          where d.card_no>? and coalesce(r.status,'pending')<>'synced'
          order by d.card_no limit ?''',(target,after_card,max(1,min(limit,100))))]


def candidate(identifier):
    with connection() as db:
        row=db.execute('select * from candidates where candidate_id=?',(identifier,)).fetchone()
        return dict(row) if row else None
