"""Conservative repeated-attempt comping from ASR words and a MIDI guide.

Only adjacent repeated words/phrases are candidates. Legitimate repeats found in
the supplied lyrics are protected. Edits mute discarded attempts in place so
the musical clock cannot move. Every edit is logged and reversible.
"""
from pathlib import Path
import json
import re
import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter1d


def normalize(word):
    return re.sub(r"[^a-z0-9']",'',word.lower())


def transcript_words(path):
    data=json.loads(Path(path).read_text())
    return [w for segment in data for w in segment.get('words',[])
            if normalize(w.get('word',''))]


def _components(mask):
    edges=np.diff(np.r_[False,mask,False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def _attempt_bounds(words,x,sr,pitch_time,f0):
    lo=max(0,float(words[0]['start'])-.08)
    hi=min(len(x)/sr,float(words[-1]['end'])+.25)
    voiced=(pitch_time>=lo)&(pitch_time<hi)&(f0>0)
    pieces=_components(binary_closing(voiced,structure=np.ones(9)))
    pieces=[(a,b) for a,b in pieces if (pitch_time[b-1]-pitch_time[a])>=.10]
    if not pieces:
        return None
    # ASR frequently assigns a long preceding rest to a word; choose its last
    # substantial voiced island, not the tail of the previous lyric.
    a,b=pieces[-1]
    vs,ve=float(pitch_time[a]),float(pitch_time[b-1])
    step=max(1,round(.01*sr))
    envelope=np.sqrt(np.mean(np.pad(x*x,(0,(-len(x))%step)).reshape(-1,step),axis=1))
    active=envelope>max(10**(-58/20),np.percentile(envelope,10)*8)
    # Include the noisy attack before a stop-consonant closure and its vowel.
    active=binary_closing(active,structure=np.ones(35))
    regions=_components(active)
    center=int(((vs+ve)/2)*sr/step)
    region=next(((a,b) for a,b in regions if a<=center<b),None)
    if region:
        start=max(lo,min(vs-.03,region[0]*step/sr-.04))
        end=min(hi,max(ve+.04,region[1]*step/sr+.025))
    else:
        start,end=max(lo,vs-.18),min(hi,ve+.12)
    return start,end


def plan_comp(x,sr,words,pitch_time,f0,guide_midi,lyrics=None):
    """Return discarded-attempt intervals; requires strong pitch-score evidence.

    guide_midi is aligned at pitch_time; no nearest-scale estimates are used.
    Supplying lyrics protects intended consecutive repetitions. Without lyrics,
    edits require a large improvement, and the audit labels this uncertainty.
    """
    tokens=[normalize(w['word']) for w in words]
    expected=[normalize(w) for w in lyrics.split()] if lyrics else []
    actual=69+12*np.log2(np.maximum(f0,1)/440)
    proposals=[]
    occupied=set()
    for size in range(6,0,-1):
        for i in range(len(words)-size*2+1):
            j=i+size
            if any(k in occupied for k in range(i,j+size)):
                continue
            a,b=tokens[i:j],tokens[j:j+size]
            if a!=b or not all(a) or sum(map(len,a))<3:
                continue
            if words[j]['start']-words[j-1]['end']>4:
                continue
            doubled=a+a
            if expected and any(expected[k:k+2*size]==doubled
                                for k in range(len(expected)-2*size+1)):
                continue
            bounds=[_attempt_bounds(ws,x,sr,pitch_time,f0)
                    for ws in (words[i:j],words[j:j+size])]
            if any(bd is None for bd in bounds):
                continue
            scores=[]
            for lo,hi in bounds:
                mask=(pitch_time>=lo)&(pitch_time<hi)&(f0>0)&np.isfinite(guide_midi)
                scores.append(float(np.median(abs(actual[mask]-guide_midi[mask])))
                              if mask.sum()>=8 else np.inf)
            keep=int(np.argmin(scores)); discard=1-keep
            improvement=scores[discard]-scores[keep]
            if not np.isfinite(improvement) or improvement<1.5 or scores[keep]>1.5:
                continue
            lo,hi=bounds[discard]
            # Protect the kept attempt including its consonant attack.
            klo,khi=bounds[keep]
            if lo<klo:
                hi=min(hi,klo-.03)
            else:
                lo=max(lo,khi+.03)
            if hi-lo<.1:
                continue
            proposals.append({'start':lo,'end':hi,'fade_seconds':.025,
                              'reason':'adjacent repeated attempt; stronger MIDI match retained',
                              'keep_interval':[klo,khi],'discarded_pitch_error_semitones':scores[discard],
                              'kept_pitch_error_semitones':scores[keep],
                              'lyrics_verified':bool(expected)})
            occupied.update(range(i,j+size))
    return sorted(proposals,key=lambda p:p['start'])


def apply_comp(x,sr,edits):
    result=np.asarray(x,dtype=float).copy()
    gain=np.ones(len(result))
    for edit in edits:
        a=max(0,round(edit['start']*sr)); b=min(len(result),round(edit['end']*sr))
        fade=min(round(edit.get('fade_seconds',.025)*sr),(b-a)//2)
        if b<=a:
            continue
        g=np.zeros(b-a)
        if fade:
            g[:fade]=.5+.5*np.cos(np.linspace(0,np.pi,fade))
            g[-fade:]=.5-.5*np.cos(np.linspace(0,np.pi,fade))
        gain[a:b]=np.minimum(gain[a:b],g)
    return result*gain
