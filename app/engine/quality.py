from collections import Counter
from .config import SETTINGS
from .models import normalize


def best_observation(observations):
    longest = max(len(normalize(obs['ocr']['text'])) for obs in observations)
    covered = [obs for obs in observations if len(normalize(obs['ocr']['text'])) >= longest * .7]
    return max(covered, key=lambda obs: (obs['ocr']['bottom_margin'] >= .04,
               obs['ocr'].get('mean_confidence', obs['ocr']['minimum_confidence']),
               obs['ocr']['minimum_confidence']))


def finalize(observations):
    usable = [obs for obs in observations if len(normalize(obs['ocr']['text']).split()) >= SETTINGS.min_words]
    if not usable:
        return 'needs_review', '', '충분한 텍스트를 읽지 못했습니다.'
    counts = Counter(normalize(obs['ocr']['text']) for obs in usable)
    text, count = counts.most_common(1)[0]
    matching = [obs for obs in usable if normalize(obs['ocr']['text']) == text]
    selected = best_observation(usable)
    # Conservative gate: a 2:1 majority must not erase a conflicting number or negation.
    if len(counts) > 1:
        return 'needs_review', selected['ocr']['text'], '관측 간 텍스트 차이가 있습니다. 숫자·조건을 포함해 검토가 필요합니다.'
    if any(obs.get('variant_disagreement', False) for obs in observations):
        return 'needs_review', selected['ocr']['text'], '전체 카드와 보조 ROI 결과가 달라 검토가 필요합니다.'
    if any(obs['ocr']['bottom_margin'] < .04 for obs in matching):
        return 'partial', selected['ocr']['text'], '본문 하단이 잘렸을 가능성이 있습니다.'
    if count < 2:
        return 'needs_review', selected['ocr']['text'], '서로 다른 프레임의 텍스트 일치를 확보하지 못했습니다.'
    if any(obs['ocr']['minimum_confidence'] < SETTINGS.min_line_confidence for obs in matching):
        return 'needs_review', selected['ocr']['text'], '낮은 신뢰도 줄이 있습니다.'
    return 'ready', selected['ocr']['text'], '복수 프레임 일치·줄 점수·하단 여백 기준 통과. 실제 정답 보장은 아닙니다.'
