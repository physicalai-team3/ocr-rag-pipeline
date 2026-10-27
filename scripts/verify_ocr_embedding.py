import sys,time,json,os,textwrap
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.verification import isolated_settings
verification_directory=isolated_settings(Path(__file__).resolve().parents[1])
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--video',type=Path)
parser.add_argument('--long-side',type=int,choices=(1280,1920),default=1280)
parser.add_argument('--jpeg-quality',type=int,default=80)
args=parser.parse_args()
if not 50<=args.jpeg_quality<=100:parser.error('jpeg-quality must be 50..100')
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from fastapi.testclient import TestClient
from app.engine.models import runtime_fingerprint,model_manifest
from app.engine.config import data_dir
from app import store,events
from app.main import app
start=time.perf_counter();profile=None;cold=None
im=Image.new('RGB',(1920,1080),(25,25,25));draw=ImageDraw.Draw(im)
draw.rectangle((640,60,1280,1000),fill=(245,245,240))
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',29)
heading=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',32)
draw.text((680,110),'Workshop Opening Notice',font=heading,fill=(20,20,20))
body='The repair workshop opens every morning at eight and closes at six in the evening. Please bring your equipment and keep the receipt. Our team checks each machine carefully before returning it to the owner. Contact the front desk if you need help with your repair.'
y=190
for line in textwrap.wrap(body,36):draw.text((680,y),line,font=font,fill=(20,20,20));y+=43
def encode(frame):
 scale=min(1,args.long_side/max(frame.shape[:2]))
 frame=cv2.resize(frame,None,fx=scale,fy=scale)
 ok,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,args.jpeg_quality])
 if not ok:raise ValueError('JPEG encode failed')
 return jpeg
frame=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR);jpg=encode(frame)
records=[]
def listener(event):records.append(dict(event))
events.add_listener(listener)
with TestClient(app) as client:
 cold=time.perf_counter()-start;profile=runtime_fingerprint()
 session=client.post('/api/sessions').json()
 with client.websocket_connect('/ws/camera?sid='+session['session_id']+'&t='+session['token']) as socket:
  for index in range(5):
   socket.send_bytes(jpg.tobytes());deadline=time.monotonic()+10
   while sum(e['type']=='ocr.done' for e in records)<index+1 and time.monotonic()<deadline:time.sleep(.02)
   time.sleep(.5)
   if store.count():break
  deadline=time.monotonic()+10
  while store.count()<1 and time.monotonic()<deadline:time.sleep(.05)
  if store.count()!=1:
   print(json.dumps({'events':[e['type'] for e in records],'candidates':[{'status':c['status'],'observations':len(json.loads(c['provenance'])['observations'])} for c in store.candidates()]},ensure_ascii=False));raise RuntimeError('no accepted document')
  saved=store.recent(1)[0];document=store.document(saved['doc_id'])
  assert any(event['type']=='doc.saved' and event.get('sid')==session['session_id'] and event.get('doc_id')==saved['doc_id'] for event in records)
  assert saved['title']=='Workshop Opening Notice' and 'repair workshop opens' in document['text'].lower()
  vector=np.asarray(document['chunks'][0]['embedding'],np.float32)
  hits=store.search('What time does the repair workshop open?',3)
  rag=client.post('/api/ask',json={'question':'What time does the repair workshop open?'}).json()
  assert vector.shape==(1024,) and np.isfinite(vector).all() and abs(np.linalg.norm(vector)-1)<.002
  assert hits[0]['doc_id']==saved['doc_id'] and rag['sources']
  assert rag['found'] is True and rag['mode']=='extractive'
  assert 'eight' in rag['answer'].lower()
  assert rag['sources'][0]['title']==saved['title']
  result={'mode':'real synthetic JPEG through existing WebSocket, actual GPU models; isolated test DB',
   'capture':{'long_side':args.long_side,'jpeg_quality':args.jpeg_quality},'runtime':model_manifest()['runtime'],'cold_seconds':cold,'profile':profile,'documents':store.count(),
   'title':saved['title'],'ocr_text':document['text'],'vector_dimension':len(vector),'vector_norm':float(np.linalg.norm(vector)),
   'search_top1_title':hits[0]['title'],'search_similarity':hits[0]['score'],'rag_sources':rag['sources'],'rag_mode':rag['mode'],'rag_answer':rag['answer'],
   'remote_status':store.recent(1)[0]['remote_status'],
   'event_types':[e['type'] for e in records],
   'ocr_ms':[e['ms'] for e in records if e['type']=='ocr.done']}
if args.video:
 from app.ocr import Processor
 from app.engine.models import embed,document_chunks,embedding_model
 capture=cv2.VideoCapture(str(args.video))
 try:ok,source_frame=capture.read()
 finally:capture.release()
 if not ok:raise ValueError('Sample video cannot be decoded')
 source_jpeg=encode(source_frame)
 probe=Processor();probe.reset('verification-video')
 observation=probe.analyze(source_jpeg.tobytes(),0,time.monotonic(),probe.stream)
 if not observation:raise ValueError('No central card in sample video first frame')
 sample_text=observation[0][1]['ocr']['text']
 if not sample_text.strip():raise ValueError('No sample OCR text')
 chunks=document_chunks(sample_text,embedding_model().tokenizer)
 sample_vectors=embed(chunks)
 assert sample_vectors.shape[1]==1024 and np.isfinite(sample_vectors).all()
 assert store.count()==1
 result['sample_video']={'filename':args.video.name,'ocr_characters':len(sample_text),
    'vector_shape':list(sample_vectors.shape),'vector_norms':np.linalg.norm(sample_vectors,axis=1).tolist(),
    'persisted':False,'notice':'Provisional OCR embedded only for model verification; not indexed or uploaded.'}
(data_dir()/'model-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
