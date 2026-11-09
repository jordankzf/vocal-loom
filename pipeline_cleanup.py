"""Conservative, context-aware offline vocal de-clicking.

Only isolated short transients in unvoiced, otherwise quiet gaps are attenuated.
Voiced sound, sustained fricatives/aspirates, and ambiguous consonant attacks are
protected. This is a signal heuristic, not a phoneme/mouth-noise classifier.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d, binary_closing
from scipy.signal import find_peaks, butter, sosfiltfilt


def _pitch_voicing(audio, sample_rate):
    import parselmouth
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    peak = np.max(np.abs(mono), initial=0.0)
    if peak < 1e-10 or len(mono) < sample_rate * .1:
        return np.array([0., len(mono)/sample_rate]), np.array([False, False])
    sound = parselmouth.Sound(mono * (.6/peak), sample_rate)
    pitch = sound.to_pitch_ac(time_step=.005, pitch_floor=65, pitch_ceiling=900,
                              voicing_threshold=.5, octave_jump_cost=.4)
    return pitch.xs(), pitch.selected_array['frequency'] > 0


def cleanup_audio(audio, sample_rate, *, pitch_times=None, voiced=None,
                  protection_ms=55., attenuation_db=24., breath_reduction=.90):
    """Return ``(cleaned_audio, audit)`` without changing length or sample rate.

    External pitch/voicing may be passed in source-time coordinates. For stereo,
    analysis uses the mean signal and identical attenuation preserves the image.
    A smooth gain notch removes only the short candidate, not the gap or breath.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim not in (1, 2):
        raise ValueError('audio must be mono or samples-by-channels')
    if sample_rate < 8000 or not np.isfinite(x).all():
        raise ValueError('sample rate must be >=8000 and samples must be finite')
    if not 0 <= breath_reduction <= 1:
        raise ValueError('breath_reduction must be in 0..1')
    y = x.copy()
    mono = x.mean(axis=1) if x.ndim == 2 else x
    hop = max(1, round(sample_rate * .001))
    count = int(np.ceil(len(mono)/hop))
    if count < 50:
        return y, {'edits': [], 'candidates': [], 'reason': 'recording too short'}
    padded = np.pad(mono, (0, count*hop-len(mono)))
    rms = np.sqrt(np.mean(padded.reshape(count, hop)**2, axis=1))
    times = (np.arange(count)+.5)*hop/sample_rate
    floor = max(float(np.percentile(rms, 10)), 1e-7)
    peak_threshold = max(floor * 8, float(np.max(rms)) * .006)
    baseline = median_filter(rms, size=61, mode='nearest')
    if pitch_times is None or voiced is None:
        pitch_times, voiced = _pitch_voicing(x, sample_rate)
    pitch_times = np.asarray(pitch_times, dtype=float)
    voiced = np.asarray(voiced, dtype=bool)
    vt = pitch_times[voiced]
    if len(vt):
        ids = np.searchsorted(vt, times)
        left = vt[np.clip(ids-1, 0, len(vt)-1)]
        right = vt[np.clip(ids, 0, len(vt)-1)]
        distance = np.minimum(abs(times-left), abs(times-right))
    else:
        distance = np.full(count, np.inf)
    peaks, _ = find_peaks(rms, height=peak_threshold,
                          prominence=peak_threshold*.7, distance=3)
    gain = np.ones(len(mono))
    candidates = []
    occupied_until = -1
    for p in peaks:
        if p < occupied_until:
            continue
        amplitude = float(rms[p])
        threshold = max(floor * 3, float(baseline[p]) * 2.5, amplitude * .18)
        left = right = p
        while left > 0 and p-left < 35 and rms[left-1] > threshold:
            left -= 1
        while right+1 < count and right-p < 35 and rms[right+1] > threshold:
            right += 1
        width_ms = (right-left+1)*hop/sample_rate*1000
        ring = np.r_[rms[max(0,left-55):max(0,left-8)],
                     rms[min(count,right+9):min(count,right+56)]]
        background = max(floor, float(np.median(ring)) if len(ring) else floor)
        isolation = amplitude/background
        context_level = max(float(np.percentile(ring, 75)) if len(ring) else floor, floor)
        peak_distance = float(distance[left:right+1].min())
        approved = (width_ms <= 18 and isolation >= 7 and
                    peak_distance >= protection_ms/1000 and
                    context_level < amplitude*.30)
        # Extended hiss/aspiration beside the impulse indicates a consonant.
        context_busy = float(np.mean(ring > amplitude*.25)) if len(ring) else 0.
        approved = bool(approved and context_busy < .35)
        reason = ('isolated_short_unvoiced_gap_transient' if approved else
                  'protected_voiced_or_consonant_context')
        if width_ms > 25 or isolation < 3:
            continue
        start = max(0, left*hop-round(.002*sample_rate))
        end = min(len(mono), (right+1)*hop+round(.003*sample_rate))
        event = {'start':start/sample_rate, 'end':end/sample_rate,
                 'peak_seconds':float(times[p]), 'core_width_ms':width_ms,
                 'isolation_ratio':isolation,
                 'nearest_voicing_seconds':peak_distance if np.isfinite(peak_distance) else None,
                 'context_busy_fraction':context_busy,
                 'action':'attenuate' if approved else 'retain', 'reason':reason}
        if approved:
            # 2ms attack/3ms release preserve continuity and leave waveform residual.
            length = end-start
            attack = min(round(.002*sample_rate), length//3)
            release = min(round(.003*sample_rate), length//3)
            bottom = 10**(-attenuation_db/20)
            notch = np.full(length, bottom)
            notch[:attack] = bottom+(1-bottom)*(.5+.5*np.cos(np.linspace(0,np.pi,attack)))
            notch[-release:] = bottom+(1-bottom)*(.5-.5*np.cos(np.linspace(0,np.pi,release)))
            gain[start:end] = np.minimum(gain[start:end], notch)
            event['attenuation_db'] = float(attenuation_db)
            occupied_until = right+5
        candidates.append(event)
    breath_edits=[]
    if breath_reduction and len(vt):
        # Breaths occupy lower-energy aperiodic noise between sung phrases.
        # S/F and aspirated onsets are protected by spectral and distance tests.
        high = sosfiltfilt(butter(3, min(4500, sample_rate*.35),
                          fs=sample_rate, btype='highpass',output='sos'), mono)
        high=np.pad(high,(0,count*hop-len(high)))
        high_rms=np.sqrt(np.mean(high.reshape(count,hop)**2,axis=1))
        high_ratio=median_filter(high_rms/(rms+floor),size=11,mode='nearest')
        voice_frames=np.interp(times,pitch_times,voiced.astype(float))>.8
        vocal_level=float(np.median(rms[voice_frames])) if voice_frames.any() else float(np.max(rms))
        quiet_breath=(distance>=.120)&(rms>floor*3)&(rms<max(floor*20,vocal_level*.28))&(high_ratio<.58)
        # Join only tiny holes inside noise; preserve ≥120ms around voicing.
        quiet_breath=binary_closing(quiet_breath,iterations=8)&(distance>=.120)&(high_ratio<.65)
        edges=np.diff(np.r_[False,quiet_breath,False].astype(int))
        starts,ends=np.flatnonzero(edges==1),np.flatnonzero(edges==-1)
        breath_gain_frames=np.ones(count)
        for left,right in zip(starts,ends):
            if (right-left)*hop/sample_rate<.045:
                continue
            start=max(0,left*hop);end=min(len(mono),right*hop)
            if end<=start: continue
            # Smooth the entire gain envelope, avoiding rectangular gates.
            breath_gain_frames[left:right]=1-breath_reduction
            breath_edits.append({'start':start/sample_rate,'end':end/sample_rate,
                                'reduction':float(breath_reduction),
                                'reason':'quiet_aperiodic_gap_noise_outside_120ms_voicing_protection'})
        breath_gain_frames=gaussian_filter1d(breath_gain_frames,.012*sample_rate/hop,mode='nearest')
        # A strict unity mask on actual voicing prevents breath cleanup changing vowels.
        breath_gain_frames[distance<.100]=1.
        # A final 3ms smoothing avoids jumps where protection meets attenuation.
        breath_gain_frames=gaussian_filter1d(breath_gain_frames,.003*sample_rate/hop,mode='nearest')
        breath_gain=np.interp(np.arange(len(mono))/sample_rate,times,breath_gain_frames)
        gain*=breath_gain
    y *= gain if x.ndim == 1 else gain[:, None]
    edits = [e for e in candidates if e['action'] == 'attenuate']
    audit = {'method':'short-transient/quiet-gap/voicing-context heuristic',
             'sample_rate':int(sample_rate), 'duration_seconds':len(mono)/sample_rate,
             'noise_floor_rms':floor, 'protection_ms':float(protection_ms),
             'attenuation_db':float(attenuation_db), 'edits':edits,
             'breath_reduction':float(breath_reduction), 'breath_edits':breath_edits,
             'candidate_count':len(candidates), 'candidates':candidates,
             'edited_sample_fraction':float(np.mean(gain < .999)),
             'duration_preserved':len(y)==len(x),
             'caveat':'Ambiguous clicks adjoining consonants are retained to protect words; no heuristic can guarantee phoneme identity.'}
    return y, audit


def declick(audio, sample_rate, **kwargs):
    """Alias for the pipeline's source cleanup stage."""
    return cleanup_audio(audio, sample_rate, **kwargs)


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path
    import soundfile as sf
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('--output',default='output/comp-analysis/source-cleaned.wav')
    parser.add_argument('--audit',default='output/comp-analysis/mouth-noise.json')
    args=parser.parse_args()
    data,sr=sf.read(args.input,dtype='float64')
    cleaned,report=cleanup_audio(data,sr)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    sf.write(args.output,cleaned,sr,subtype='PCM_24')
    Path(args.audit).parent.mkdir(parents=True,exist_ok=True)
    Path(args.audit).write_text(json.dumps(report,indent=2))
    print(json.dumps({'edits':len(report['edits']),
                      'edited_sample_fraction':report.get('edited_sample_fraction'),
                      'output':args.output},indent=2))
