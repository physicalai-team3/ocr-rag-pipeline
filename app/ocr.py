"""C: latest-frame OCR worker; submit returns without waiting for inference."""
from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
import hashlib
import io
import time
import uuid
import cv2
import numpy as np
from PIL import Image
from . import events, store
from .engine.config import SETTINGS, data_dir
from .engine.models import recognize, normalize, runtime_fingerprint, embed
from .engine.quality import finalize
from .engine.video import detect_cards, text_region

MAX_JPEG = 4 * 1024 * 1024
MAX_DIMENSION = 1920
MAX_WAIT = .5


@dataclass
class Track:
    number: int
    box: list
    signature: np.ndarray
    seen: float
    velocity: float = 0
    observations: list = field(default_factory=list)
    last_ocr: float = -1e9
    queued: bool = False


@dataclass
class Stream:
    sid: str
    identifier: str = field(default_factory=lambda:uuid.uuid4().hex)
    tracks: list = field(default_factory=list)
    next_track: int = 0


class Processor:
    def __init__(self):
        self.pending = None
        self.wake = asyncio.Event()
        self.saves = asyncio.Queue(maxsize=8)
        self.tasks = []
        self.active_sid = None
        self.last_input = 0
        self.generation = 0
        self.stream = None
        self.closed = True

    async def start(self):
        self.closed = False
        self.tasks = [asyncio.create_task(self.worker()), asyncio.create_task(self.save_worker())]

    async def stop(self):
        # Do not cancel to_thread: wait for inference and accepted saves before shutdown.
        self.closed = True
        self.wake.set()
        await self.tasks[0]
        await self.saves.join()
        await self.saves.put(None)
        await self.tasks[1]

    def reset(self, sid=None):
        self.generation += 1
        self.active_sid = sid
        self.pending = None
        self.stream = Stream(sid) if sid else None

    async def submit(self, sid, jpeg, ts):
        if self.closed:
            await events.publish(sid, {'type':'error','stage':'ocr','msg':'OCR 작업자가 준비되지 않았습니다.'})
            return
        if not isinstance(jpeg,bytes) or not 0 < len(jpeg) <= MAX_JPEG:
            await events.publish(sid, {'type':'frame.dropped','count':1})
            await events.publish(sid, {'type':'error','stage':'ocr','msg':'JPEG 프레임은 4MB 이하여야 합니다.'})
            return
        now=time.monotonic()
        if sid != self.active_sid:
            if self.active_sid and now-self.last_input < 5:
                await events.publish(sid, {'type':'frame.dropped','count':1})
                return
            self.reset(sid)
        self.last_input=now
        if self.pending:
            await events.publish(self.pending[0], {'type':'frame.dropped','count':1})
        self.pending=(sid,jpeg,ts,now,self.generation)
        self.wake.set()

    async def worker(self):
        while not self.closed:
            await self.wake.wait()
            self.wake.clear()
            value,self.pending=self.pending,None
            if not value:
                continue
            sid,jpeg,ts,received,generation=value
            if time.monotonic()-received > MAX_WAIT:
                await events.publish(sid, {'type':'frame.dropped','count':1})
                continue
            try:
                started=time.perf_counter()
                output=await asyncio.to_thread(self.analyze,jpeg,ts,received,self.stream)
                if generation != self.generation:
                    continue
                duration=round((time.perf_counter()-started)*1000)
                for track,observation,profile,stream_id in output:
                    await events.publish(sid,{'type':'ocr.done','frame_ts':ts,
                        'chars':len(observation['ocr']['text']), 'conf':observation['ocr'].get('mean_confidence',0),
                        'ms':duration,'provisional':True,'late':time.monotonic()-received>1})
                    status,text,reason=finalize(track.observations)
                    candidate_id=hashlib.sha256(f'{sid}:{stream_id}:{track.number}'.encode()).hexdigest()
                    provenance={'observations':track.observations,'track_id':track.number,
                                'fingerprint':profile}
                    # Capture incomplete tracks too; only three observations can auto-complete.
                    if len(track.observations)<3:
                        status='needs_review'
                        reason='서로 다른 3개 관측을 확보하기 전입니다.'
                    await asyncio.to_thread(store.save_candidate,candidate_id,sid,status,text,reason,provenance)
                    if status=='ready' and not track.queued:
                        doc={'sid':sid,'text':text,'status':'ready','ms':duration,
                             'fingerprint':provenance['fingerprint'],'provenance':provenance}
                        try:
                            self.saves.put_nowait(doc)
                            track.queued=True
                        except asyncio.QueueFull:
                            await events.publish(sid,{'type':'error','stage':'store','msg':'저장 대기열이 가득 찼습니다. 검토 후보는 로컬에 보존했습니다.'})
            except Exception as error:
                # No raw exception or credential/URL enters the dashboard.
                await events.publish(sid,{'type':'error','stage':'ocr',
                    'msg':f'{type(error).__name__}: 프레임·모델 설치·실행 장치를 확인하세요.'})

    def analyze(self,jpeg,ts,received,stream):
        with Image.open(io.BytesIO(jpeg)) as header:
            if header.format!='JPEG' or max(header.size)>MAX_DIMENSION or min(header.size)<64:
                raise ValueError('invalid frame dimensions/format')
        frame=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
        if frame is None:raise ValueError('invalid JPEG')
        # Model initialization/hash verification stays off the event loop.
        profile=runtime_fingerprint()
        cards=detect_cards(frame)
        stream.tracks=[track for track in stream.tracks if received-track.seen<=1.2]
        matched=[];used=set()
        for card in cards:
            matches=[]
            for track in stream.tracks:
                if track.number in used:continue
                elapsed=received-track.seen
                distance=abs(card['box'][0]-(track.box[0]+track.velocity*elapsed))
                if distance < track.box[2]*.65 and abs(card['box'][2]/track.box[2]-1)<.35 and np.mean(np.abs(card['signature']-track.signature))<.20:
                    matches.append((distance,track))
            if matches:
                track=min(matches,key=lambda item:item[0])[1]
                elapsed=received-track.seen
                if elapsed>.01:track.velocity=(card['box'][0]-track.box[0])/elapsed
                track.box,track.signature,track.seen=card['box'],card['signature'],received
            else:
                if len(stream.tracks)>=32:continue
                track=Track(stream.next_track,card['box'],card['signature'],received)
                stream.next_track+=1;stream.tracks.append(track)
            used.add(track.number);matched.append((card,track))
        if not matched:return []
        card,track=max(matched,key=lambda item:item[0]['centrality'])
        completed=len(track.observations)>=3
        gap=1.0 if completed else SETTINGS.min_observation_gap
        if received-track.last_ocr + 1e-6 < gap:
            return []
        track.last_ocr=received
        original=recognize(card['image'])
        region=text_region(card['image']);alternative=None
        if region[3]<card['image'].shape[0]*.9:
            alternative=recognize(card['image'],region)
        elif original['minimum_confidence']<SETTINGS.min_line_confidence:
            gray=cv2.cvtColor(card['image'],cv2.COLOR_BGR2GRAY)
            alternative=recognize(cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR))
        chosen=original
        if alternative and alternative.get('mean_confidence',0)>original.get('mean_confidence',0) and len(normalize(alternative['text']))>=len(normalize(original['text']))*.7:
            chosen=alternative
        if completed:
            if normalize(chosen['text'])==normalize(track.observations[-1]['ocr']['text']):
                return [(track,track.observations[-1],profile,stream.identifier)]
            # Even a same-position/similar-signature card replacement starts new evidence.
            old=track
            track=Track(stream.next_track,card['box'],card['signature'],received,last_ocr=received)
            stream.next_track+=1
            stream.tracks.remove(old);stream.tracks.append(track)
        folder=data_dir()/'evidence'/stream.identifier
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/f'{track.number}-{len(track.observations)}.jpg'
        if not cv2.imwrite(str(path),card['image']):raise OSError('evidence write failed')
        observation={'timestamp':ts/1000,'box':card['box'],'image':path.relative_to(data_dir()).as_posix(),
                     'ocr':chosen,'original_ocr':original,'alternative_ocr':alternative,
                     'variant_disagreement':bool(alternative and normalize(alternative['text'])!=normalize(original['text']))}
        track.observations.append(observation)
        return [(track,observation,profile,stream.identifier)]

    async def save_worker(self):
        from .remote import sync_document
        while True:
            doc=await self.saves.get()
            try:
                if doc is None:return
                is_new=await asyncio.to_thread(store.upsert,doc)
                if is_new:
                    await events.publish(doc['sid'],{'type':'doc.saved',**{key:doc[key] for key in ('doc_id','card_no','title','preview','chars','ms')},'remote_status':'pending'})
                else:
                    await events.publish(doc['sid'],{'type':'doc.duplicate','doc_id':doc['doc_id']})
                remote=await asyncio.to_thread(sync_document,doc['doc_id'])
                if remote=='failed':
                    await events.publish(doc['sid'],{'type':'error','stage':'store.remote','msg':'Supabase 저장 실패. 로컬 결과를 보존했습니다.'})
            except Exception as error:
                await events.publish(doc['sid'] if doc else None,{'type':'error','stage':'store',
                    'msg':f'{type(error).__name__}: 저장 실패. 로컬 검토 후보를 확인하세요.'})
            finally:
                self.saves.task_done()


_processor=None


def prepare_models():
    runtime_fingerprint()
    recognize(np.full((400,300,3),255,np.uint8))
    embed(['Model warmup for document retrieval.'])


async def start():
    global _processor
    await asyncio.to_thread(prepare_models)
    _processor=Processor()
    await _processor.start()


async def stop():
    if _processor:await _processor.stop()


async def submit(sid: str,jpeg: bytes,ts: int) -> None:
    if _processor:await _processor.submit(sid,jpeg,ts)
    else:await events.publish(sid,{'type':'error','stage':'ocr','msg':'서버 작업자가 시작되지 않았습니다.'})


def _on_event(event):
    if _processor and event['type']=='session.disconnected' and event.get('sid')==_processor.active_sid:
        _processor.reset()


events.add_listener(_on_event)
