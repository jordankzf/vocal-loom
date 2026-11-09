"""Conservative, voice-led reduction of the supplied MIDI piano arrangement.

Only pitch classes present in each harmonic onset are used. Chords hold across
piano note-off gaps. No fixed-key or made-up fallback chords are injected.
"""
from pathlib import Path
from itertools import combinations
import json
import numpy as np
import mido


def load_note_events(path):
    if str(path).endswith('.json'):
        return json.loads(Path(path).read_text())
    midi=mido.MidiFile(path)
    t=0.; active={}; notes=[]
    for msg in midi:
        t+=msg.time
        if not hasattr(msg,'channel') or msg.channel==9: continue
        key=(msg.channel,getattr(msg,'note',-1))
        if msg.type=='note_on' and msg.velocity:
            if key in active:
                start,vel=active.pop(key)
                notes.append(dict(start=start,end=t,note=key[1],velocity=vel))
            active[key]=(t,msg.velocity)
        elif msg.type=='note_off' or (msg.type=='note_on' and msg.velocity==0):
            if key in active:
                start,vel=active.pop(key)
                notes.append(dict(start=start,end=t,note=key[1],velocity=vel))
    return sorted(notes,key=lambda n:(n['start'],n['note']))


def harmonic_snapshots(notes, transpose=-12, onset_tolerance=.015):
    groups=[]
    for note in notes:
        if not groups or note['start']-groups[-1][0]['start']>onset_tolerance:
            groups.append([])
        groups[-1].append(note)
    snaps=[]
    for group in groups:
        # A single or two-note ornament cannot replace the underlying harmony.
        pitches=sorted(set(n['note'] for n in group))
        if len(pitches)<3: continue
        onset=min(n['start'] for n in group)
        # The MIDI doubles roots at the bottom; remove only the actual octave double.
        if len(pitches)>3 and pitches[0]+12 in pitches:
            pitches=pitches[1:]
        # Sustain notes from a previous onset when there is genuine overlap.
        overlapping=[n['note'] for n in notes if n['start']<onset-.02 and n['end']>onset+.08]
        for pitch in overlapping:
            if pitch>=min(pitches) and pitch not in pitches:
                pitches.append(pitch)
        pitches=sorted(pitches)
        if len(pitches)>5:
            pitches=pitches[-5:]
        pool=sorted(set(n+transpose for n in pitches))
        snaps.append(dict(time=onset,pool=pool,source_notes=pitches))
    return snaps


def _choose(pool,lead,previous=None,ceiling=None):
    pcs=set(n%12 for n in pool)
    lower=max(33,int(round(lead))-9)
    upper=min(84,int(np.floor(lead))+24)
    if ceiling is not None: upper=min(upper,int(np.floor(ceiling)))
    lower=min(lower,upper-12)
    # Octave transpositions are allowed; changing the chord pitch classes is not.
    candidates=sorted(set(n+octave for n in pool for octave in (-24,-12,0,12,24)
                          if lower<=n+octave<=upper))
    if len(candidates)<3:
        candidates=sorted(set(n for n in range(lower,upper+1) if n%12 in pcs))
    guide=np.array([lead+3,lead+8,lead+12])
    best=None; best_cost=np.inf
    for c in combinations(candidates,3):
        c=np.array(c)
        intervals=np.diff(c)
        if max(intervals)>12: continue
        # Harmonic coverage first. Avoid unnecessarily repeating the melody PC.
        duplicate_pc=3-len(set(c%12))
        same_lead=np.sum(c%12==int(round(lead))%12)
        unison=np.sum(c==round(lead))
        cost=duplicate_pc*6 + same_lead*1.2 + unison*4
        cost+=.07*np.sum(abs(c-guide))
        cost+=.22*sum(min(abs(n-p) for p in pool) for n in c)
        cost+=2*sum(gap<3 for gap in intervals)
        if previous is not None:
            movements=abs(c-previous)
            cost+=.7*sum(movements)+.1*sum(np.maximum(movements-5,0)**2)
        if cost<best_cost:
            best,best_cost=c,cost
    if best is None:
        best=np.array(candidates[:3])
        if len(best)<3: raise ValueError('MIDI cannot support three harmonies')
    return best


def arrange(midi_times,lead_midi,notes_path,transpose=-12,
            return_audit=False):
    """Return (3,N) target notes, MIDI-clock timestamps plus natural-register lead.

    midi_times: monotonically increasing MIDI timeline for every target frame.
    lead_midi: corrected lead MIDI values in the desired natural vocal register.
    Gating/voicing remains the renderer's responsibility.
    """
    times=np.asarray(midi_times,dtype=float)
    lead=np.asarray(lead_midi,dtype=float)
    if times.shape!=lead.shape: raise ValueError('time and lead shapes differ')
    if np.any(np.diff(times)<0): raise ValueError('MIDI times must increase')
    snapshots=harmonic_snapshots(load_note_events(notes_path),transpose=transpose)
    if not snapshots: raise ValueError('No harmonic snapshots found')
    st=np.array([s['time'] for s in snapshots])
    ids=np.maximum(0,np.searchsorted(st,times,side='right')-1)
    result=np.empty((3,len(times)),dtype=float); audit=[]; previous=None
    for sid in np.unique(ids):
        mask=ids==sid
        valid=lead[mask & np.isfinite(lead)]
        guide=float(np.median(valid)) if len(valid) else 48.
        ceiling=(float(np.min(valid))+24) if len(valid) else guide+24
        selected=_choose(snapshots[sid]['pool'],guide,previous,ceiling=ceiling)
        result[:,mask]=selected[:,None]; previous=selected
        audit.append(dict(time=float(st[sid]),lead_guide=guide,
                          source_notes=snapshots[sid]['source_notes'],
                          shifted_pool=snapshots[sid]['pool'],harmony=selected.tolist()))
    if return_audit: return result,audit
    return result
