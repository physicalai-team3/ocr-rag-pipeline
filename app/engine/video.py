"""Bright-card baseline. No ground-truth text is available to this module."""
from dataclasses import dataclass, field
import cv2
import numpy as np
from .config import SETTINGS


def selection_quality(sharpness, centrality):
    return float(centrality + .02 * np.tanh(max(0, sharpness) / 100))


def text_region(image):
    """Supplemental ROI only; full-card OCR remains mandatory evidence."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = max(30, min(150, float(np.percentile(gray, 75)) - 40))
    mask = (gray < threshold).astype(np.uint8) * 255
    mask[:, :int(w * .035)] = 0
    mask[:, int(w * .965):] = 0
    mask[:int(h * .025)] = 0
    connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 13), np.uint8))
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bottoms = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw >= w * .12 and 5 <= ch <= h * .08:
            bottoms.append(y + ch)
    if not bottoms:
        return [0, 0, w, h]
    bottom = min(h, max(max(bottoms) + max(32, int(h * .06)), int(h * .35)))
    return [0, 0, w, bottom]


def inspect_video(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError('영상 디코더가 파일을 열 수 없습니다.')
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        fps, frames = capture.get(cv2.CAP_PROP_FPS), capture.get(cv2.CAP_PROP_FRAME_COUNT)
        width, height = capture.get(3), capture.get(4)
        if not (np.isfinite(fps) and fps > 0 and frames > 0):
            raise ValueError('영상 길이 또는 프레임 정보를 확인할 수 없습니다.')
        seconds = frames / fps
        if seconds > SETTINGS.max_seconds or max(width, height) > SETTINGS.max_dimension:
            raise ValueError('영상 길이 30분 또는 해상도 4096 제한을 초과했습니다.')
        ok, frame = capture.read()
        if not ok:
            raise ValueError('영상 첫 프레임을 읽을 수 없습니다.')
        return {'fps': fps, 'frames': int(frames), 'duration': seconds,
                'width': frame.shape[1], 'height': frame.shape[0], 'rotation_applied': True}
    finally:
        capture.release()


def detect_cards(frame):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 100, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < w * .09 or cw > w * .65 or ch < h * .45 or cw / ch > 1.1:
            continue
        # Require both vertical sides. Bottom may legitimately leave the frame.
        if x < w * .012 or x + cw > w * .988 or y < 3 or y > h * .35:
            continue
        if cv2.contourArea(contour) / (cw * ch) < .65:
            continue
        epsilon = .015 * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        crop = frame[y:y + ch, x:x + cw].copy()
        transform = None
        if len(polygon) == 4 and y + ch < h - 3:
            sums = polygon.sum(axis=1)
            diffs = np.diff(polygon, axis=1).ravel()
            points = np.float32([polygon[np.argmin(sums)], polygon[np.argmin(diffs)],
                                 polygon[np.argmax(sums)], polygon[np.argmax(diffs)]])
            if len(np.unique(points, axis=0)) == 4:
                dst = np.float32([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]])
                transform = cv2.getPerspectiveTransform(points, dst)
                crop = cv2.warpPerspective(frame, transform, (cw, ch))
        # Score actual upper text band; lower-card moire must not dominate it.
        upper = cv2.cvtColor(crop[:max(40, int(ch * .5))], cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(upper, cv2.CV_64F).var())
        centrality = max(0, 1 - abs(x + cw / 2 - w / 2) / (w / 2))
        thumbnail = cv2.resize(upper, (32, 16)).astype(np.float32) / 255
        found.append({'box': [x, y, cw, ch], 'image': crop, 'quality': selection_quality(sharpness, centrality),
                      'sharpness': sharpness, 'centrality': centrality,
                      'signature': thumbnail, 'transform': transform.tolist() if transform is not None else None})
    return sorted(found, key=lambda card: card['box'][0])


@dataclass
class Track:
    id: int
    box: list
    last_time: float
    signature: np.ndarray
    velocity: float = 0.0
    observations: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    count: int = 0

    def add(self, card, timestamp):
        dt = timestamp - self.last_time
        if dt > 0:
            self.velocity = (card['box'][0] - self.box[0]) / dt
        self.box, self.last_time, self.signature = card['box'], timestamp, card['signature']
        self.count += 1
        observation = {k: v for k, v in card.items() if k != 'signature'}
        observation['timestamp'] = timestamp
        # Keep a short unfiltered reservoir: otherwise a new best frame every
        # 0.2s erases the older frame before it can become a 0.4s-spaced witness.
        reservoir = max(6, int(np.ceil(SETTINGS.sample_fps * SETTINGS.min_observation_gap * (SETTINGS.candidates - 1))) + 2)
        self.pending = (self.pending + [observation])[-reservoir:]
        pool = list({obs['timestamp']: obs for obs in self.observations + self.pending}.values())
        selected = []
        for slot in range(SETTINGS.candidates):
            eligible = [obs for obs in pool if all(abs(obs['timestamp'] - other['timestamp']) >= SETTINGS.min_observation_gap - 1e-8 for other in selected)]
            if not eligible:
                break
            key = (lambda obs: obs.get('sharpness', obs['quality'])) if slot == SETTINGS.candidates - 1 else (lambda obs: obs['quality'])
            selected.append(max(eligible, key=key))
        self.observations = selected


def extract_tracks(path, progress=lambda p: None):
    metadata = inspect_video(path)
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    tracks = []
    step = max(1, round(metadata['fps'] / SETTINGS.sample_fps))
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if timestamp <= 0:
                timestamp = index / metadata['fps']
            if timestamp > SETTINGS.max_seconds or max(frame.shape[:2]) > SETTINGS.max_dimension:
                raise ValueError('디코딩 중 영상 제한 초과')
            if index % step == 0:
                assigned = set()
                for card in detect_cards(frame):
                    x, y, width, height = card['box']
                    candidates = []
                    for track in tracks:
                        dt = timestamp - track.last_time
                        if track.id in assigned or dt < 0 or dt > .65:
                            continue
                        predicted = track.box[0] + track.velocity * dt
                        distance = abs(x - predicted) / max(width, track.box[2])
                        visual = float(np.mean(np.abs(card['signature'] - track.signature)))
                        if distance < .40 and abs(width / track.box[2] - 1) < .35 and visual < .20:
                            candidates.append((distance + visual, track))
                    track = min(candidates, key=lambda item: item[0])[1] if candidates else None
                    if track is None:
                        if len(tracks) >= 100:
                            raise ValueError('한 작업의 카드 추적 상한 100개를 초과했습니다. 영상을 나눠 처리하세요.')
                        track = Track(len(tracks), card['box'], timestamp, card['signature'])
                        tracks.append(track)
                    track.add(card, timestamp)
                    assigned.add(track.id)
                    arrays = {id(obs['image']): obs['image'] for item in tracks for obs in item.observations + item.pending}
                    memory = sum(image.nbytes for image in arrays.values())
                    if memory > 256 * 1024 * 1024:
                        raise ValueError('대표 이미지 메모리 상한 256MB를 초과했습니다. 영상을 나눠 처리하세요.')
                progress(min(1, index / metadata['frames']))
            index += 1
    finally:
        capture.release()
    if index < metadata['frames'] * .95:
        raise ValueError('영상이 중간에 디코딩 실패했습니다. 잘린 결과를 완료로 처리하지 않습니다.')
    return metadata, [track for track in tracks if track.observations]
