"""Local-only explicit review and remote retry. Never reads evaluation passages."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import store
from app.engine.models import runtime_fingerprint
from app.remote import sync_document


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate');parser.add_argument('--text-file',type=Path)
    parser.add_argument('--confirm',action='store_true');parser.add_argument('--sync-pending',action='store_true')
    args=parser.parse_args()
    if args.sync_pending:
        cursor=0
        while page:=store.pending_remote(cursor):
            for row in page:print(row['doc_id'],sync_document(row['doc_id']))
            cursor=page[-1]['card_no']
        return
    candidates=store.candidates(100)
    if not args.candidate:
        print(json.dumps(candidates,ensure_ascii=False,indent=2));return
    selected=store.candidate(args.candidate)
    if not selected or not args.text_file or not args.confirm:
        parser.error('candidate, corrected text file, and --confirm are required')
    provenance=json.loads(selected['provenance'])
    if provenance['fingerprint']!=runtime_fingerprint():parser.error('model profile changed')
    text=args.text_file.read_text(encoding='utf-8-sig').strip()
    doc={'sid':selected['sid'],'text':text,'status':'ready','ms':0,
         'fingerprint':provenance['fingerprint'],'provenance':{**provenance,
         'review':{'previous_text':selected['text'],'previous_status':selected['status'],'confirmed':True}}}
    is_new=store.upsert(doc)
    print(json.dumps({'doc_id':doc['doc_id'],'new':is_new,'remote':sync_document(doc['doc_id'])}))


if __name__=='__main__':main()
