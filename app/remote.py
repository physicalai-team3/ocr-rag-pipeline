"""Optional server-only Supabase persistence; local commits remain authoritative on failure."""
import os
import hashlib
import json
import uuid
import httpx
from . import store


def target_fingerprint():
    if not all(os.getenv(key) for key in ('SUPABASE_URL','SUPABASE_SECRET_KEY','SUPABASE_OWNER_ID')):
        return None
    value=[os.getenv('SUPABASE_URL','').rstrip('/'),os.getenv('SUPABASE_OWNER_ID',''),
           os.getenv('SUPABASE_DATASET','conveyor-v1')]
    return hashlib.sha256(json.dumps(value).encode()).hexdigest()


def sync_document(identifier):
    url=os.getenv('SUPABASE_URL','').rstrip('/')
    key=os.getenv('SUPABASE_SECRET_KEY','')
    owner=os.getenv('SUPABASE_OWNER_ID','')
    if not all((url,key,owner)):
        store.mark_remote(identifier,'not_configured')
        return 'not_configured'
    target=target_fingerprint()
    try:
        from urllib.parse import urlparse
        parsed=urlparse(url)
        if parsed.scheme!='https' or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Supabase HTTPS URL required')
        owner=str(uuid.UUID(owner))
        doc=store.document(identifier)
        if not doc:raise ValueError('missing local document')
        payload={'owner_id':owner,'dataset':os.getenv('SUPABASE_DATASET','conveyor-v1'),
                 'doc_id':doc['doc_id'],'profile':doc['profile'],'content_hash':doc['content_hash'],
                 'card_no':doc['card_no'],'title':doc['title'],'text':doc['text'],
                 'provenance':doc['provenance'],'chunks':doc['chunks']}
        # Secret API keys are sent only in apikey; they are not bearer JWTs.
        with httpx.Client(timeout=10,follow_redirects=False) as client:
            response=client.post(url+'/rest/v1/rpc/team_ocr_ingest',headers={'apikey':key},json={'payload':payload})
            response.raise_for_status()
            if response.json()!=identifier:raise ValueError('remote receipt mismatch')
        store.mark_remote(identifier,'synced',target)
        return 'synced'
    except Exception:
        store.mark_remote(identifier,'failed',target)
        return 'failed'
