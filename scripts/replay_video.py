"""C integration test sender: real video JPEGs over the existing camera WebSocket."""
import argparse
import asyncio
import json
import ssl
import time
import urllib.request
from pathlib import Path
import cv2
from websockets.asyncio.client import connect


async def replay(args):
    # Dev self-signed TLS is opt-in, never the default for arbitrary hosts.
    tls=ssl._create_unverified_context() if args.insecure_dev else ssl.create_default_context()
    request=urllib.request.Request(args.server.rstrip('/')+'/api/sessions',method='POST')
    with urllib.request.urlopen(request,context=tls) as response:session=json.load(response)
    websocket=args.server.rstrip('/').replace('https://','wss://',1).replace('http://','ws://',1)
    capture=cv2.VideoCapture(str(args.video))
    try:
        url=f"{websocket}/ws/camera?sid={session['session_id']}&t={session['token']}"
        # Never print the token-bearing URL.
        async with connect(url,ssl=tls if websocket.startswith('wss:') else None,max_size=4*1024*1024) as ws:
            seconds=0
            while True:
                started=time.monotonic();capture.set(cv2.CAP_PROP_POS_MSEC,seconds*1000)
                ok,frame=capture.read()
                if not ok:break
                scale=min(1,1920/max(frame.shape[:2]));frame=cv2.resize(frame,None,fx=scale,fy=scale)
                _,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,90])
                await ws.send(jpeg.tobytes());seconds+=1/args.fps
                await asyncio.sleep(max(0,1/args.fps-(time.monotonic()-started)))
            await asyncio.sleep(2)
    finally:capture.release()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('video',type=Path)
    parser.add_argument('--server',default='https://localhost:8443')
    parser.add_argument('--fps',type=float,default=2.5);parser.add_argument('--insecure-dev',action='store_true')
    args=parser.parse_args()
    if not .5<=args.fps<=5:parser.error('fps must be between .5 and 5')
    asyncio.run(replay(args))
