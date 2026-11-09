"""Local recording + MIDI + lyrics -> cleaned, tuned and arranged stereo WAV."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
from scipy.signal import butter,sosfiltfilt

from pipeline_midi import load_score
from pipeline_alignment import analyze_pitch,align_score
from pipeline_cleanup import cleanup_audio
from pipeline_comp import plan_comp,apply_comp
from pipeline_transcript import read_lyrics,transcribe,words_from_segments,lyric_coverage
from arrangement import arrange
from duet_part import choose_duet


def progress(message):
    print(message,flush=True)


def atomic_json(path,data):
    p=Path(path); temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(p)


def atomic_wav(path,audio,sr):
    p=Path(path); temp=p.with_name(p.stem+'.partial.wav')
    sf.write(temp,audio,sr,subtype='PCM_24')
    temp.replace(p)


def input_fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_vocal(path,*,highpass=True):
    audio,sr=sf.read(path,dtype='float64',always_2d=True)
    if not len(audio) or sr<16000 or not np.isfinite(audio).all():
        raise ValueError('Use a finite, nonempty vocal WAV/FLAC at 16 kHz or higher')
    if len(audio)/sr>1200:
        raise ValueError('Split recordings longer than 20 minutes into songs')
    warnings=[]
    if audio.shape[1]>1:
        mid=audio.mean(axis=1)
        if np.sqrt(np.mean(mid*mid))<.25*np.sqrt(np.mean(audio*audio)):
            # Avoid destructive phase cancellation in unusual stereo input.
            channel=np.argmax(np.mean(audio*audio,axis=0))
            mono=audio[:,channel]
            warnings.append('Stereo phase cancellation detected; strongest channel used.')
        else:
            mono=mid
    else:
        mono=audio[:,0]
    peak=float(np.max(abs(mono)))
    if peak<1e-6:
        raise ValueError('The recording is silent')
    if np.mean(abs(mono)>=.999)>1e-5:
        warnings.append('The source contains clipping; tuning cannot restore lost waveform detail.')
    if highpass:
        mono=sosfiltfilt(butter(3,60,fs=sr,btype='highpass',output='sos'),mono)
    # Match working level independent of microphone gain, while preserving headroom.
    gain=min(20,.65/max(np.quantile(abs(mono),.9995),1e-7))
    mono*=gain
    if np.max(abs(mono))>.95:
        mono*=.95/np.max(abs(mono))
    return mono,sr,{'sample_rate':sr,'duration_seconds':len(mono)/sr,
                    'channels':audio.shape[1],'source_peak':peak,'working_gain':gain,
                    'warnings':warnings}


def reduce_stationary_noise(audio,sr):
    """Mild Wiener suppression only when a measurable stationary floor warrants it."""
    from scipy.signal import stft,istft
    _,_,spectrum=stft(audio,fs=sr,nperseg=1024,noverlap=768)
    power=abs(spectrum)**2
    energy=np.mean(power,axis=0)
    quiet=energy<=np.percentile(energy,10)
    noise=np.median(power[:,quiet],axis=1)
    noise_rms=float(np.sqrt(np.percentile(np.convolve(audio*audio,
                         np.ones(256)/256,mode='valid')[::256],10)))
    floor_db=20*np.log10(max(noise_rms,1e-12))
    signal_rms=float(np.sqrt(np.mean(audio*audio)))
    snr=20*np.log10(max(signal_rms,1e-12)/max(noise_rms,1e-12))
    applied=floor_db>-58 and snr<30
    if applied:
        from scipy.ndimage import gaussian_filter
        gain=np.sqrt(np.maximum(.20,1-noise[:,None]/np.maximum(power,1e-18)))
        gain=gaussian_filter(gain,sigma=(1,1.5))
        _,clean=istft(spectrum*gain,fs=sr,nperseg=1024,noverlap=768)
        clean=clean[:len(audio)]
        if len(clean)<len(audio):
            clean=np.pad(clean,(0,len(audio)-len(clean)))
    else:
        clean=audio.copy()
    return clean,{'applied':bool(applied),'estimated_floor_dbfs':float(floor_db),
                  'estimated_snr_db':float(snr),'method':'conservative stationary Wiener suppression'}


def run(args):
    from pipeline_dsp import render_vocal
    started=time.time()
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    work=out/'stages'; work.mkdir(exist_ok=True)
    final=out/'finished.wav'
    if Path(args.recording).resolve()==final:
        raise ValueError('Choose an output folder that does not overwrite your recording')
    atomic_json(out/'status.json',{'status':'running','stage':'input analysis'})
    progress('Reading recording, MIDI and lyrics...')
    x,sr,source_report=read_vocal(args.recording,highpass=args.processing!='dry')
    lyrics=read_lyrics(args.lyrics) if args.lyrics else None
    score=load_score(args.midi)
    atomic_json(work/'score.json',score.to_dict())
    performance=analyze_pitch(x,sr)
    track='auto' if args.melody_track=='auto' else int(args.melody_track)
    progress('Matching the sung melody and musical clock...')
    alignment=align_score(score,performance,track)
    if args.alignment:
        fixed=np.load(args.alignment)
        ss,mm=fixed['source_seconds'],fixed['midi_seconds']
        if len(ss)<2 or ss.shape!=mm.shape or not np.isfinite(ss).all() or not np.isfinite(mm).all() or np.any(np.diff(ss)<=0) or np.any(np.diff(mm)<=0):
            raise ValueError('The supplied alignment needs finite, strictly increasing source and MIDI clocks')
        alignment['source_seconds']=fixed['source_seconds']
        alignment['midi_seconds']=fixed['midi_seconds']
        alignment['clock_source']='supplied alignment map'
        alignment['warnings']=[w.message for w in alignment['melody'].warnings]
        alignment['warnings'].append('A supplied alignment map overrides automatic clock matching.')
        alignment['duration_ratio_range']=[float(np.min(np.gradient(mm,ss))),
                                             float(np.max(np.gradient(mm,ss)))]
    np.savez(work/'alignment.npz',source_seconds=alignment['source_seconds'],
             midi_seconds=alignment['midi_seconds'])
    align_report={k:v for k,v in alignment.items()
                  if k not in ('source_seconds','midi_seconds','melody')}
    atomic_json(work/'alignment.json',align_report)
    source_midi=np.interp(performance.times,alignment['source_seconds'],alignment['midi_seconds'])
    guide=score.melody_at(source_midi,track_selection=track,hold_rests=True,
                           transpose=alignment['transpose_semitones']).pitches
    transcript_path=work/'transcript.json'
    cache=Path(__file__).resolve().parent/'.models'
    if args.transcript:
        segments=json.loads(Path(args.transcript).read_text(encoding='utf-8'))
    elif args.no_asr:
        segments=[]
        alignment['warnings'].append('Transcription disabled; automatic repeated-attempt detection skipped.')
    else:
        key=input_fingerprint(args.recording)
        cache_meta=work/'transcript-source.json'
        prior=json.loads(cache_meta.read_text()) if cache_meta.exists() else {}
        if transcript_path.exists() and prior.get('sha256')==key:
            segments=json.loads(transcript_path.read_text())
            progress('Using cached local phrase analysis')
        else:
            model='small.en' if args.language=='en' else 'small'
            segments=transcribe(x,sr,cache,model_name=model,language=args.language,progress=progress)
            atomic_json(cache_meta,{'sha256':key,'language':args.language,'model':model})
    atomic_json(transcript_path,segments)
    progress('Removing mouth clicks, reducing breaths and checking repeated attempts...')
    cleaned,cleanup=cleanup_audio(x,sr,pitch_times=performance.times,
                                  voiced=performance.f0>0,breath_reduction=args.breath_reduction)
    edits=plan_comp(x,sr,words_from_segments(segments),performance.times,performance.f0,guide,lyrics)
    cleaned=apply_comp(cleaned,sr,edits)
    cleaned,nr=reduce_stationary_noise(cleaned,sr)
    atomic_json(work/'cleanup.json',cleanup)
    atomic_json(work/'take-comp.json',edits)
    atomic_json(work/'noise-reduction.json',nr)
    atomic_wav(work/'cleaned-recording.wav',cleaned,sr)
    # Keep a small consonant lead-in and sung release, but not stop/handling noises.
    valid=(performance.f0>0)&(performance.strength>.7)
    first=max(0,float(performance.times[valid][0])-.32)
    last=min(len(x)/sr,float(performance.times[valid][-1])+.38)
    a,b=int(first*sr),int(last*sr)
    cleaned=cleaned[a:b]
    performance=analyze_pitch(cleaned,sr,step=.005)
    source_times=performance.times+first
    midi_times=np.interp(source_times,alignment['source_seconds'],alignment['midi_seconds'])
    lead_result=score.melody_at(midi_times,track_selection=track,hold_rests=True,
                                transpose=alignment['transpose_semitones'])
    lead=lead_result.pitches
    notes=[{'start':n.start,'end':n.end,'note':n.pitch,'velocity':n.velocity}
           for n in score.melodic_notes]
    notes_file=work/'arrangement-notes.json'; atomic_json(notes_file,notes)
    arrangement_audit=[]
    if args.mode=='solo':
        harmonies=None
    elif args.mode=='duet':
        harmonies,arrangement_audit,duet_warnings=choose_duet(midi_times,lead,
            notes_path=notes_file,transpose=alignment['transpose_semitones']+12)
        alignment['warnings'].extend(duet_warnings)
    else:
        try:
            harmonies,arrangement_audit=arrange(midi_times,lead,notes_path=notes_file,
                       transpose=alignment['transpose_semitones']+12,return_audit=True)
        except ValueError:
            # A melody-only score supplies no chord information. Octave doubling
            # is honest; inventing thirds here can create the wrong harmony.
            harmonies=np.vstack((lead+12,lead-12,lead))
            alignment['warnings'].append('No usable chord information: octave/unison arrangement used.')
    atomic_json(work/'arrangement.json',arrangement_audit)
    output_times=performance.times if args.processing=='dry' else midi_times
    progress(f'Rendering corrected {args.mode} vocal arrangement...')
    rendered=render_vocal(cleaned,sr,performance,output_times,lead,harmonies,
                          mode=args.mode,pitch_strength=args.pitch_strength,
                          lead_mode=args.lead_mode,harmony_follow=args.harmony_follow,
                          pitch_style=args.pitch_style,harmony_engine=args.harmony_engine,
                          progress=progress)
    progress('Exporting 24-bit stems and finished stereo master...')
    for name,audio in rendered.stems.items():
        atomic_wav(work/(name+'.wav'),audio,sr)
    np.savez(work/'automation.npz',**rendered.automation)
    if args.processing=='dry':
        import pyloudnorm as pyln
        from scipy.signal import resample_poly
        dry=rendered.stems['lead'].astype(float)
        loudness=pyln.Meter(sr).integrated_loudness(dry)
        gain=10**((-20-loudness)/20)
        peak=np.max(abs(resample_poly(dry,4,1)))
        gain=min(gain,10**(-1.1/20)/max(peak,1e-12))
        dry*=gain
        atomic_wav(final,dry,sr)
        rendered.report['dry_export']={'processing':'tuned lead only; constant playback gain',
            'eq':False,'compression':False,'reverb':False,'timing_edits':False,
            'channels':1,'duration_seconds':len(dry)/sr,'gain':float(gain),
            'loudness_lufs':float(pyln.Meter(sr).integrated_loudness(dry))}
        rendered.report.update({'processing':'dry','channels':1,
            'finished_duration_seconds':len(dry)/sr,
            'loudness_lufs':float(pyln.Meter(sr).integrated_loudness(dry)),
            'true_peak_4x_dbfs':float(20*np.log10(max(peak*gain,1e-12)))})
    else:
        atomic_wav(final,rendered.audio,rendered.sample_rate)
    report={'status':'complete','output':str(final),'mode':args.mode,
            'source':source_report,'source_sha256':input_fingerprint(args.recording),
            'midi_sha256':input_fingerprint(args.midi),'melody':{
               'method':lead_result.method,'confidence':lead_result.confidence,
               'tracks':list(lead_result.selected_tracks)},
            'alignment':align_report,'source_trim_seconds':[first,last],
            'comp_edits':edits,'click_edits':len(cleanup['edits']),
            'breath_edits':len(cleanup.get('breath_edits',[])),'noise_reduction':nr,
            'lyrics':lyric_coverage(segments,lyrics) if lyrics and segments else None,
            'render':rendered.report,'warnings':list(dict.fromkeys(source_report['warnings']+
                  alignment['warnings']+rendered.report.get('warnings',[]))),
            'elapsed_seconds':time.time()-started}
    atomic_json(out/'report.json',report)
    atomic_json(out/'status.json',{'status':'complete','output':str(final)})
    progress(f'Finished: {final}')
    for warning in report['warnings']:
        progress('Review note: '+warning)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    r=commands.add_parser('render',help='Run the complete local production chain')
    r.add_argument('--recording',required=True)
    r.add_argument('--midi',required=True)
    r.add_argument('--lyrics')
    r.add_argument('--mode',choices=['solo','duet','vocoder'],default='duet')
    r.add_argument('--processing',choices=['dry','full'],default='full',
                   help='Tuned dry solo or full production chain')
    r.add_argument('--pitch-style',choices=['auto','transparent','tight'],default='auto')
    r.add_argument('--harmony-engine',choices=['auto','psola','world'],default='auto')
    r.add_argument('--out',required=True)
    r.add_argument('--melody-track',default='auto')
    r.add_argument('--pitch-strength',type=float,default=1.)
    r.add_argument('--lead-mode',choices=['original','tuned'],default='tuned',
                   help='Keep original lead pitch/phrasing or tune/retime it')
    r.add_argument('--harmony-follow',choices=['score','lead'],default='score',
                   help='Keep MIDI pitch targets or move overlays with the untouched lead')
    r.add_argument('--breath-reduction',type=float,default=.95)
    r.add_argument('--language',default='en')
    r.add_argument('--transcript',help='Optional precomputed local ASR JSON')
    r.add_argument('--alignment',help='Optional reviewed source/MIDI timing NPZ')
    r.add_argument('--no-asr',action='store_true',help='Skip phrase/retake detection explicitly')
    args=p.parse_args(argv)
    if args.harmony_follow=='lead':
        args.lead_mode='original'
    if args.processing=='dry':
        args.mode='solo'
        args.lead_mode='tuned'
    if args.pitch_style=='auto':
        args.pitch_style='tight' if args.mode=='vocoder' else 'transparent'
    if args.harmony_engine=='auto':
        args.harmony_engine='psola' if args.pitch_style=='transparent' else 'world'
    if not 0<=args.pitch_strength<=1 or not 0<=args.breath_reduction<=1:
        p.error('Pitch strength and breath reduction must be between 0 and 1')
    try:
        run(args)
    except KeyboardInterrupt:
        progress('Cancelled. Original input files were not modified.')
        return 130
    except Exception as exc:
        try:
            Path(args.out).mkdir(parents=True,exist_ok=True)
            atomic_json(Path(args.out)/'status.json',{'status':'failed','error':str(exc)})
        except OSError:
            pass  # The original error may itself be an unwritable destination.
        progress(f'Could not finish: {exc}')
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
