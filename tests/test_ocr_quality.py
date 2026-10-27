import numpy as np
import pytest
from app.engine.quality import finalize
from app.engine.video import Track, selection_quality, text_region


def observation(text, confidence, margin=.3):
    return {'ocr':{'text':text,'minimum_confidence':confidence,'mean_confidence':confidence,'bottom_margin':margin}}


def test_disagreement_displays_best_later_observation_without_approval():
    first='The source requires 24 units and must not exceed this amount during any operating period in this facility.'
    later=first.replace('source','procedure')
    status,text,_=finalize([observation(first,.81),observation(later,.99)])
    assert status=='needs_review' and text==later


def test_roi_disagreement_and_mixed_clipping_stay_unapproved():
    text='This source paragraph has enough words to provide a meaningful integrity check without any hidden missing sections.'
    a=observation(text,.99);b=observation(text,.99)
    a['variant_disagreement']=True
    assert finalize([a,b])[0]=='needs_review'
    assert finalize([observation(text,.99,.3),observation(text,.99,.01)])[0]=='partial'


def test_centrality_outweighs_extreme_moire_and_spacing_survives_rising_quality():
    assert selection_quality(40,.95)>selection_quality(100000,.5)
    track=Track(0,[0,0,100,200],0,np.zeros((16,32)))
    for i in range(11):
        card={'box':[i,0,100,200],'image':np.zeros((200,100,3),np.uint8),
              'signature':np.zeros((16,32)),'quality':i/10,'sharpness':i}
        track.add(card,i*.2)
    times=sorted(obs['timestamp'] for obs in track.observations)
    assert len(times)==3 and times[-1]==2
    assert all(b-a>=.4-1e-8 for a,b in zip(times,times[1:]))
    assert len(track.pending)<=6


def test_roi_does_not_use_fixed_top_half():
    import cv2
    image=np.full((1000,600,3),240,np.uint8)
    cv2.putText(image,'TITLE',(40,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    cv2.putText(image,'Last paragraph 46 C',(40,820),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
    region=text_region(image)
    assert region[3]>820
    assert text_region(np.full_like(image,240))==[0,0,600,1000]
