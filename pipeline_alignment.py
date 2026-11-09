"""Register-aware, bounded note-sequence alignment for monophonic singing.

The MIDI is a timing/reference hypothesis, not proof of the intended melody.
Wrong notes are sequence substitutions; only pitch-consistent matches create
local timing anchors. The score endpoint is free for partial performances.
"""
from dataclasses import dataclass
import numpy as np
import parselmouth
from scipy.ndimage import gaussian_filter1d, median_filter


@dataclass
class Performance:
    times: np.ndarray
    f0: np.ndarray
    strength: np.ndarray


def analyze_pitch(audio, sr, step=.01):
    peak = np.max(np.abs(audio), initial=0)
    if peak < 1e-7 or len(audio) < sr*.25:
        raise ValueError('The recording has no usable vocal audio')
    sound = parselmouth.Sound(np.asarray(audio, dtype=float)*(.7/peak), sr)
    p = sound.to_pitch_ac(time_step=step, pitch_floor=55, pitch_ceiling=1000,
                         voicing_threshold=.5, octave_jump_cost=.4)
    return Performance(p.xs(), p.selected_array['frequency'], p.selected_array['strength'])


def _pc(value):
    return (np.asarray(value)+6) % 12-6


def _events(times, pitches, valid=None, minimum=.12, bridge=.16):
    """Stable pitch runs, preserving rests and never selecting by score fit."""
    times = np.asarray(times)
    pitches = np.asarray(pitches, dtype=float)
    dt = float(np.median(np.diff(times)))
    if valid is None:
        valid = np.isfinite(pitches)
    finite = np.flatnonzero(valid & np.isfinite(pitches))
    if not len(finite):
        return []
    filled = np.interp(times, times[finite], pitches[finite])
    smoothed = median_filter(filled, size=max(1, int(.07/dt)//2*2+1), mode='nearest')
    # Hysteresis avoids splitting a held, slightly detuned note at every
    # half-semitone quantizer boundary.
    runs = []
    start = finite[0]
    center = smoothed[start]
    last_good = start
    for i in range(start+1, len(times)):
        change = valid[i] and abs(smoothed[i]-center) > .72
        gap = times[i]-times[last_good] > bridge
        if change or gap:
            stop = last_good+1 if gap else i
            idx = np.arange(start, stop)
            idx = idx[valid[idx]]
            if len(idx) and times[stop-1]-times[start]+dt >= minimum:
                runs.append({'start':float(times[start]), 'end':float(times[stop-1]+dt),
                             'pitch':float(np.median(pitches[idx])), 'frames':len(idx)})
            if valid[i]:
                start = i
                center = smoothed[i]
            else:
                start = i+1
        if valid[i]:
            if start > i:
                start = i
                center = smoothed[i]
            last_good = i
            if not change:
                center = .97*center+.03*smoothed[i]
    if start < len(times):
        idx = np.arange(start, len(times))
        idx = idx[valid[idx]]
        if len(idx) and times[idx[-1]]-times[start]+dt >= minimum:
            runs.append({'start':float(times[start]), 'end':float(times[idx[-1]]+dt),
                         'pitch':float(np.median(pitches[idx])), 'frames':len(idx)})
    # A glide may briefly cross a new pitch then return to the same sustained
    # tone; merge only like-pitched neighbors across a genuinely short gap.
    result = []
    for run in runs:
        if result and run['start']-result[-1]['end'] <= bridge and abs(run['pitch']-result[-1]['pitch']) < .55:
            old = result[-1]
            total = old['frames']+run['frames']
            old['pitch'] = (old['pitch']*old['frames']+run['pitch']*run['frames'])/total
            old['frames'] = total
            old['end'] = run['end']
        else:
            result.append(run)
    return result


def _clock(rt, ref, st, sung):
    candidates = []
    first = float(rt[np.flatnonzero(np.isfinite(ref))[0]])
    finite = np.flatnonzero(np.isfinite(ref))
    continuous = np.interp(rt, rt[finite], ref[finite])
    def evaluate(speed, offset):
        mt = (st-offset)*speed
        predicted = np.interp(mt, rt, continuous)
        transpose = -12*round(float(np.median(predicted-sung))/12)
        error = _pc(sung-(predicted+transpose))
        cost = float(np.mean(np.minimum(abs(error), 2.5)))
        cost += .6*float(np.mean((mt < -.3) | (mt > rt[-1]+.3)))
        cost += .12*abs(np.log(speed))
        return cost, float(speed), float(offset), int(transpose)
    speeds = np.unique(np.r_[np.arange(.7,1.31,.025),.5,.6,1.4,1.5,1.75,2.])
    for speed in speeds:
        base = float(st[0]-first/speed)
        for delta in np.arange(-.6,.601,.05):
            candidates.append(evaluate(speed, base+delta))
    candidates.sort()
    _, speed, offset, _ = candidates[0]
    for rate in np.arange(max(.3,speed-.03),speed+.0301,.002):
        for delay in np.arange(offset-.15,offset+.151,.015):
            candidates.append(evaluate(rate, delay))
    candidates.sort()
    return candidates


def _sequence(source_events, reference_events, speed, offset, transpose):
    """Semi-global edit alignment; substitutions cost less than delete+insert.

    Duration/clock terms are weak evidence. They cannot turn a wrong pitch into
    a perfect match, and long reference tails cost nothing after the last event.
    """
    n, m = len(source_events), len(reference_events)
    costs = np.full((n+1,m+1), np.inf)
    previous = np.zeros((n+1,m+1), dtype=np.int8)
    costs[0,0] = 0
    for i in range(1,n+1):
        a=source_events[i-1]
        costs[i,0] = costs[i-1,0]+.95+1.1*min(a['end']-a['start'],3)
        previous[i,0] = 1
    for j in range(1,m+1):
        b=reference_events[j-1]
        costs[0,j] = costs[0,j-1]+.65+.25*min(b['end']-b['start'],3)
        previous[0,j] = 2
    for i, a in enumerate(source_events,1):
        amid = (a['start']+a['end'])/2
        adur = a['end']-a['start']
        for j, b in enumerate(reference_events,1):
            bmid = (b['start']+b['end'])/2
            bdur = b['end']-b['start']
            error = float(abs(_pc(a['pitch']-(b['pitch']+transpose))))
            # Huber-like clipped note error. A stable wrong note is still a
            # substitution, not permission to hold time on another note.
            raw_error=abs(a['pitch']-(b['pitch']+transpose))
            pitch_cost = min(error,2.1)+3*min(max(raw_error-5,0)/7,1)
            duration_cost = .13*min(abs(np.log(max(adur*speed,.03)/max(bdur,.03))),2.5)
            clock_error = bmid-(amid-offset)*speed
            clock_cost = .45*min((clock_error/8)**2,4)
            options = (costs[i-1,j-1]+pitch_cost+duration_cost+clock_cost,
                       costs[i-1,j]+.95+1.1*min(adur,3),
                       costs[i,j-1]+.65+.25*min(bdur,3))
            which = int(np.argmin(options))
            costs[i,j] = options[which]
            previous[i,j] = which
    # Free score endpoint: a prefix does not get stretched to the MIDI's end.
    endpoint = int(np.argmin(costs[n,1:]))+1
    i, j = n, endpoint
    matches, skipped_source, skipped_reference = [], [], []
    while i or j:
        action = previous[i,j]
        if i and j and action == 0:
            a, b = source_events[i-1], reference_events[j-1]
            matches.append((i-1,j-1,float(_pc(a['pitch']-(b['pitch']+transpose)))))
            i-=1; j-=1
        elif i and (not j or action == 1):
            skipped_source.append(i-1); i-=1
        else:
            skipped_reference.append(j-1); j-=1
    return list(reversed(matches)), list(reversed(skipped_source)), list(reversed(skipped_reference)), float(costs[n,endpoint]/max(n,1))


def _prefix_restart(source_events, reference_events):
    """Recognize a short abandoned opening followed by the same opening again.

    Require multiple following notes to confirm the second attempt; a lone
    repeated pitch is not sufficient evidence to discard anything.
    """
    count=min(6,len(reference_events))
    if count<4 or len(source_events)<count+2:
        return 0
    target=np.array([e['pitch'] for e in reference_events[:count]])
    def cost(start):
        values=np.array([e['pitch'] for e in source_events[start:start+count]])
        return np.minimum(abs(_pc(values-target)),3)
    initial=cost(0)
    if max(initial[:2])>1.2:
        return 0
    best=(float(np.mean(initial)),0)
    for start in range(2,min(7,len(source_events)-count+1)):
        if source_events[start]['start']-source_events[0]['start']>6:
            continue
        error=cost(start)
        if max(error[:2])>1.2 or np.sum(error<.8)<4:
            continue
        value=float(np.mean(error))
        if value<best[0]-.8:
            best=(value,start)
    return best[1]


def _bounded_map(duration, speed, offset, proposals, p):
    """Strictly increasing timing; voiced slopes .25..4, rest expansion <=8."""
    # Insert in confidence order. Later proposals cannot invalidate an already
    # accepted, better-evidenced local tempo relationship.
    accepted = []
    for item in sorted(proposals,key=lambda a:-a['confidence']):
        s, r = item['source_seconds'], item['midi_seconds']
        index = np.searchsorted([a['source_seconds'] for a in accepted],s)
        okay = True
        for other in accepted[max(0,index-1):index+1]:
            ds = s-other['source_seconds']; dr = r-other['midi_seconds']
            if abs(ds) < .16 or dr*ds <= 0:
                okay=False;break
            mask=(p.times>=min(s,other['source_seconds']))&(p.times<=max(s,other['source_seconds']))
            rest_fraction=float(np.mean(p.f0[mask]<=0)) if mask.any() else 0
            upper=8 if rest_fraction>.65 else 4
            if not .25 <= dr/ds <= upper:
                okay=False;break
        if okay:
            accepted.insert(index,item)
    grid=np.arange(0,duration+.005,.005)
    if len(accepted) < 2:
        return grid,(grid-offset)*speed,accepted
    sx=np.array([a['source_seconds'] for a in accepted])
    ry=np.array([a['midi_seconds'] for a in accepted])
    mt=np.interp(grid,sx,ry)
    left_rate=np.clip((ry[1]-ry[0])/(sx[1]-sx[0]),.25,4)
    right_rate=np.clip((ry[-1]-ry[-2])/(sx[-1]-sx[-2]),.25,4)
    mt[grid<sx[0]]=ry[0]+(grid[grid<sx[0]]-sx[0])*left_rate
    mt[grid>sx[-1]]=ry[-1]+(grid[grid>sx[-1]]-sx[-1])*right_rate
    # Smooth corrections rather than the whole clock, retaining positive end
    # slopes. Convolving bounded slopes preserves their bounds.
    correction=gaussian_filter1d(mt-(grid-offset)*speed,6,mode='nearest')
    mt=(grid-offset)*speed+correction
    assert np.all(np.diff(mt)>0)
    return grid,mt,accepted


def align_score(score, performance, track='auto'):
    """Return an inferred prefix alignment plus explicit uncertainty evidence."""
    p=performance
    valid=(p.f0>0)&(p.strength>.7)
    if valid.sum()<20:
        raise ValueError('Not enough clearly pitched singing to align automatically')
    rt=np.arange(0,score.duration+.02,.02)
    result=score.melody_at(rt,track_selection=track,hold_rests=False)
    ref=np.asarray(result.pitches,dtype=float)
    if not np.isfinite(ref).any():
        raise ValueError('The selected MIDI contains no usable melody')
    note=69+12*np.log2(np.maximum(p.f0,1)/440)
    src=_events(p.times,note,valid=valid)
    dst=_events(rt,ref,minimum=.10,bridge=.10)
    restart=_prefix_restart(src,dst)
    restart_regions=[]
    clock_valid=valid.copy()
    if restart:
        restart_regions=[{'source_start_seconds':src[0]['start'],
                          'source_end_seconds':src[restart]['start'],
                          'reason':'Repeated opening followed by a better matching continuation; possible abandoned first attempt.'}]
        clock_valid &= p.times>=src[restart]['start']
        src=src[restart:]
    voiced=np.flatnonzero(clock_valid)
    step=max(1,len(voiced)//1600)
    st=p.times[voiced[::step]]
    sung=69+12*np.log2(p.f0[voiced[::step]]/440)
    candidates=_clock(rt,ref,st,sung)
    best,speed,offset,transpose=candidates[0]
    warnings=[w.message for w in result.warnings]
    if restart_regions:
        warnings.append('A short opening restart was inferred; the earlier attempt is flagged for comping, not silently erased.')
    proposals=[];mismatches=[];skipped=[];sequence_cost=None
    if src and dst:
        matched,skipped,missing,sequence_cost=_sequence(src,dst,speed,offset,transpose)
        for si,ri,error in matched:
            a,b=src[si],dst[ri]
            if abs(error)>.8:
                mismatches.append({'source_start_seconds':a['start'],'source_end_seconds':a['end'],
                                   'source_pitch':a['pitch'],'reference_pitch':b['pitch']+transpose,
                                   'error_semitones':error,'midi_start_seconds':b['start']})
                continue
            # An onset is useful even when a singer holds it too long or too
            # briefly. Do not use its end as evidence for stretching a wrong
            # following note into the same reference pitch.
            duration=min(a['end']-a['start'],b['end']-b['start'])
            confidence=float(np.clip(1-abs(error)/1.4,.2,1))*min(1,duration/.3)
            proposals.append({'source_seconds':a['start'],'midi_seconds':b['start'],
                              'matching_error_semitones':abs(error),'confidence':confidence,
                              'source_event_index':si,'reference_event_index':ri})
        if skipped:
            warnings.append(f'{len(skipped)} source pitch events did not match the inferred score sequence; possible ornaments, restarts, or reference ambiguity.')
        if mismatches:
            warnings.append(f'{len(mismatches)} matched events disagree in pitch and were not used to retime the recording.')
    # If an ordinary clock already explains the performance, retain it rather
    # than letting a few similar chord tones create arbitrary phrase drift.
    # Small, pitch-consistent local note edits remain available for rubato.
    clock_preserved = best < 1.05
    if clock_preserved:
        proposals=[a for a in proposals if abs(a['midi_seconds']-(a['source_seconds']-offset)*speed) <= .35]
        for a in proposals:
            a['edit_seconds']=(a['midi_seconds']-(a['source_seconds']-offset)*speed)/speed
        for seconds in np.arange(0,float(p.times[-1])+.01,2):
            proposals.append({'source_seconds':float(seconds),'midi_seconds':float((seconds-offset)*speed),
                              'matching_error_semitones':None,'confidence':.05,'clock_anchor':True})
    source,mt,anchors=_bounded_map(float(p.times[-1]),speed,offset,proposals,p)
    distinct=[c for c in candidates if abs(c[1]-speed)>.08 or abs(c[2]-offset)>.3]
    margin=distinct[0][0]-best if distinct else 0
    anchor_fraction=sum(not a.get('clock_anchor',False) for a in anchors)/max(len(src),1)
    confidence=float(np.clip(.9-best*.25,.2,.95)*result.confidence*min(1,.55+anchor_fraction))
    if margin<.03:
        warnings.append('Several global clocks are similarly plausible; timing and melodic voice selection remain provisional.')
    if anchor_fraction<.35:
        warnings.append('Few source events provide trustworthy timing anchors; local timing may remain ambiguous.')
    if src and anchors:
        # Repeated or skipped passages are monotone hypotheses, not verified
        # lyrical alignment. Expose this rather than silently deleting attempts.
        residuals=[a['midi_seconds']-(a['source_seconds']-offset)*speed for a in anchors]
        if np.ptp(residuals)>2:
            warnings.append('Substantial nonlinear timing was inferred; review phrase boundaries and any repeated material.')
    return {'source_seconds':source,'midi_seconds':mt,'transpose_semitones':transpose,
            'speed':speed,'offset_seconds':offset,'confidence':confidence,
            'fit_cost':best,'alternative_margin':margin,'note_anchors':anchors,
            'melody':result,'warnings':warnings,'sequence_fit_cost':sequence_cost,
            'pitch_mismatches':mismatches,'unmatched_source_events':skipped,
            'restart_candidates':restart_regions,
            'unmatched_source_regions':[{'source_start_seconds':src[i]['start'],
                                          'source_end_seconds':src[i]['end'],
                                          'pitch':src[i]['pitch'],
                                          'reason':'Unmatched pitch event; do not assume a timing correction or silently delete it.'}
                                         for i in skipped],
            'alignment_method':'conservative_clock' if clock_preserved else 'bounded_monotone_note_sequence','source_event_count':len(src),
            'reference_event_count':len(dst),'inferred_score_end_seconds':float(mt[-1])}
