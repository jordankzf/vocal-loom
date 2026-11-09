"""Render a technical piano-roll video from a mastered WAV and voice automation.

Build dependency: Pillow. Media encoding: FFmpeg with libx264 and libmp3lame.
The audio is encoded directly from the supplied master; no DSP is applied here.
No source recording, lyrics, or score are copied into the output directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import soundfile as sf

WIDTH, HEIGHT, FPS = 1280, 720, 24
BACKGROUND = '#0c121c'
GRID = '#233041'
MUTED = '#8c9cb0'
WHITE = '#edf3fb'
COLORS = [WHITE, '#42c7bb', '#a89ae8', '#e6b860']
VOICE_NAMES = ['MELODY', 'HARMONY 1', 'HARMONY 2', 'HARMONY 3']
LEFT, RIGHT, TOP, BOTTOM = 107, 1224, 192, 590
PLAYHEAD = 562
PIXELS_PER_SECOND = 72
PITCH_MIN, PITCH_MAX = 36, 68
ROW = (BOTTOM-TOP)/(PITCH_MAX-PITCH_MIN+1)


@lru_cache(maxsize=24)
def font(size, bold=False, mono=False):
    candidates = ([r'C:\Windows\Fonts\consola.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'] if mono else
                  ([r'C:\Windows\Fonts\seguisb.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'] if bold else
                   [r'C:\Windows\Fonts\segoeui.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']))
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def note_name(midi):
    return ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][int(round(midi))%12]+str(int(round(midi))//12-1)


def timestamp(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds//60:02d}:{seconds%60:02d}'


def load_voices(path, cleanup=None):
    data=np.load(path)
    ts=data['output_time']
    if 'target_midi' in data:
        notes=data['target_midi']
    else:
        lead=data['effective_lead_midi']
        harmonies=data['effective_harmony_midi'] if 'effective_harmony_midi' in data else data['harmony_midi']
        notes=np.vstack((lead,harmonies))
    if notes.ndim!=2 or len(notes)>4:
        raise ValueError('Visualization supports one to four rendered vocal parts')
    activity=data['source_f0']>0
    if cleanup:
        edits=json.loads(Path(cleanup).read_text()).get('discarded_attempts',[])
        for edit in edits:
            a,b=edit['master_interval']
            activity[(ts>=a)&(ts<=b)]=False
    segments=[]
    dt=float(np.median(np.diff(ts)))
    for voice in range(len(notes)):
        values=notes[voice]
        active=activity&np.isfinite(values)
        quant=np.rint(np.nan_to_num(values)).astype(int)
        begin=None
        for i in range(len(ts)+1):
            continuing=(i<len(ts) and active[i] and begin is not None and
                        quant[i]==quant[begin] and (i==0 or ts[i]-ts[i-1]<dt*3))
            if begin is not None and not continuing:
                end=ts[i-1]+dt/2
                start=ts[begin]-dt/2
                if end-start>=.025:
                    segments.append({'voice':voice,'start':max(0,float(start)),
                                     'end':float(end),'pitch':int(quant[begin])})
                begin=None
            if i<len(ts) and active[i] and begin is None:
                begin=i
    return ts,notes,activity,segments


def render_frame(now, duration, title, credit, ts, notes, activity, segments, waveform):
    image=Image.new('RGB',(WIDTH,HEIGHT),BACKGROUND)
    draw=ImageDraw.Draw(image)
    draw.text((52,30),'VOCAL LOOM',font=font(29,bold=True),fill=WHITE)
    draw.text((52,72),'MIDI VOCAL HARMONIZER',font=font(12,mono=True),fill=MUTED)
    draw.text((430,31),title,font=font(24,bold=True),fill=WHITE)
    draw.text((430,68),credit,font=font(14),fill=MUTED)
    draw.text((1063,37),f'{timestamp(now)} / {timestamp(duration)}',font=font(19,mono=True),fill=WHITE)
    draw.line((52,105,1224,105),fill=GRID,width=1)

    index=int(np.clip(np.searchsorted(ts,now),0,len(ts)-1))
    sounding=bool(activity[index] and abs(ts[index]-now)<.03)
    for v in range(len(notes)):
        x=52+v*303
        draw.rectangle((x,126,x+4,156),fill=COLORS[v])
        draw.text((x+14,123),VOICE_NAMES[v],font=font(11,mono=True),fill=MUTED)
        label=note_name(notes[v,index]) if sounding and np.isfinite(notes[v,index]) else '--'
        draw.text((x+14,139),label,font=font(20,mono=True),fill=COLORS[v])
    draw.text((LEFT,174),'PITCH',font=font(10,mono=True),fill=MUTED)
    draw.text((PLAYHEAD+8,174),'NOW',font=font(10,mono=True),fill=WHITE)
    draw.text((RIGHT-129,174),'RENDERED TARGETS',font=font(10,mono=True),fill=MUTED)

    for pitch in range(PITCH_MIN,PITCH_MAX+1):
        cy=BOTTOM-(pitch-PITCH_MIN+.5)*ROW
        black=pitch%12 in (1,3,6,8,10)
        draw.rectangle((LEFT,cy-ROW/2,RIGHT,cy+ROW/2),fill='#101a27' if black else '#131f2e')
        if pitch%12==0:
            draw.line((LEFT,cy+ROW/2,RIGHT,cy+ROW/2),fill='#35465c',width=1)
        else:
            draw.line((LEFT,cy+ROW/2,RIGHT,cy+ROW/2),fill='#1c2b3c',width=1)
        draw.text((56,cy-7),note_name(pitch),font=font(10,mono=True),fill='#9dafc3' if not black else '#596c83')
    start_time=now-(PLAYHEAD-LEFT)/PIXELS_PER_SECOND
    end_time=now+(RIGHT-PLAYHEAD)/PIXELS_PER_SECOND
    for second in range(max(0,math.ceil(start_time)),math.ceil(end_time)+1):
        x=round(PLAYHEAD+(second-now)*PIXELS_PER_SECOND)
        if LEFT<=x<=RIGHT:
            draw.line((x,TOP,x,BOTTOM),fill='#2c3b4e' if second%2==0 else '#1e2d40',width=1)
            if second%2==0:
                draw.text((x-17,596),timestamp(second),font=font(10,mono=True),fill=MUTED)
    # Place voices in separate vertical sublanes when an octave/unison doubles.
    for segment in segments:
        if segment['end']<start_time or segment['start']>end_time:
            continue
        v=segment['voice']
        x1=max(LEFT,round(PLAYHEAD+(segment['start']-now)*PIXELS_PER_SECOND))
        x2=min(RIGHT,round(PLAYHEAD+(segment['end']-now)*PIXELS_PER_SECOND))
        if x2<=x1:
            continue
        cy=BOTTOM-(segment['pitch']-PITCH_MIN+.5)*ROW
        if not TOP<=cy<=BOTTOM:
            raise ValueError('A rendered pitch lies outside the visualization range')
        active=segment['start']<=now<segment['end']
        color=COLORS[v]
        if segment['end']<now:
            rgb=tuple(int(color[k:k+2],16) for k in (1,3,5))
            color=tuple(round(c*.48) for c in rgb)
        lane=cy+(-2.6,-.9,.9,2.6)[v]
        draw.rounded_rectangle((x1,lane-2,x2,lane+2),radius=2,fill=color)
        if active:
            draw.rectangle((max(x1,PLAYHEAD-2),lane-3,min(x2,PLAYHEAD+3),lane+3),fill=COLORS[v])
    draw.line((PLAYHEAD,TOP-2,PLAYHEAD,BOTTOM+3),fill='#f3f7fc',width=2)
    draw.polygon(((PLAYHEAD-5,TOP-7),(PLAYHEAD+5,TOP-7),(PLAYHEAD,TOP-1)),fill=WHITE)

    wave_top,wave_bottom=633,672
    draw.line((LEFT,wave_top,RIGHT,wave_top),fill=GRID,width=1)
    for col in range(0,RIGHT-LEFT,3):
        t=now+(LEFT+col-PLAYHEAD)/PIXELS_PER_SECOND
        if t<0 or t>=duration:
            continue
        amplitude=float(np.interp(t,waveform[:,0],waveform[:,1]))
        h=1+min(15,amplitude*25)
        color='#8295ad' if LEFT+col<=PLAYHEAD else '#3d5069'
        draw.line((LEFT+col,653-h,LEFT+col,653+h),fill=color,width=1)
    draw.line((PLAYHEAD,wave_top+1,PLAYHEAD,wave_bottom),fill=WHITE,width=1)
    count=len(notes)-1
    parts=f'1 MELODY + {count} '+('HARMONY' if count==1 else 'HARMONIES')
    draw.text((52,691),parts+'  /  PROCESSED COVER PERFORMANCE',font=font(10,mono=True),fill=MUTED)
    state='REVERB TAIL' if now>ts[-1] else ('VOICED' if sounding else 'REST / CONSONANTS')
    draw.text((1028,691),state,font=font(10,mono=True),fill=MUTED)
    return image


def main():
    global PITCH_MIN,PITCH_MAX,ROW
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio',required=True)
    p.add_argument('--automation',required=True)
    p.add_argument('--cleanup-audit')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--title',default='Hide and Seek')
    p.add_argument('--credit',default='Imogen Heap / cover performed by jordankzf')
    p.add_argument('--snapshots-only',action='store_true')
    args=p.parse_args()
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        raise SystemExit('FFmpeg is required to encode the demo')
    audio,sr=sf.read(args.audio,always_2d=True)
    duration=len(audio)/sr
    ts,notes,activity,segments=load_voices(args.automation,args.cleanup_audit)
    PITCH_MIN=max(0,int(np.floor(np.nanmin(notes)))-2)
    PITCH_MAX=min(127,int(np.ceil(np.nanmax(notes)))+2)
    ROW=(BOTTOM-TOP)/(PITCH_MAX-PITCH_MIN+1)
    hop=round(sr*.025)
    padded=np.pad(np.mean(audio**2,axis=1),(0,(-len(audio))%hop)).reshape(-1,hop)
    rms=np.sqrt(np.mean(padded,axis=1))
    waveform=np.column_stack(((np.arange(len(rms))+.5)*hop/sr,np.minimum(1,rms/.22)))
    for sec in [4.,22.,78.,97.,137.,140.]:
        frame=render_frame(sec,duration,args.title,args.credit,ts,notes,activity,segments,waveform)
        frame.save(out/f'preview-{int(sec):03d}.png')
    render_frame(78.,duration,args.title,args.credit,ts,notes,activity,segments,waveform).save(out/'poster.png')
    if args.snapshots_only:
        print('Preview frames created',flush=True)
        return
    mp3=out/'hide-and-seek-demo.mp3'
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-i',args.audio,
        '-map_metadata','-1','-c:a','libmp3lame','-q:a','2','-id3v2_version','3',
        '-metadata','title=Hide and Seek (Vocal Loom cover demo)',
        '-metadata','artist=jordankzf','-metadata','comment=Song by Imogen Heap. Processed cover performance.',str(mp3)],check=True)
    video=out/'hide-and-seek-notes.mp4'
    command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-f','rawvideo',
        '-pix_fmt','rgb24','-s',f'{WIDTH}x{HEIGHT}','-r',str(FPS),'-i','-',
        '-i',args.audio,'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','medium',
        '-b:v','330k','-maxrate','360k','-bufsize','720k','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','144k','-movflags','+faststart','-map_metadata','-1',
        '-t',str(duration),str(video)]
    started=time.time()
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for i in range(math.ceil(duration*FPS)):
            frame=render_frame(i/FPS,duration,args.title,args.credit,ts,notes,activity,segments,waveform)
            process.stdin.write(frame.tobytes())
            if i%(FPS*15)==0:
                print(f'Visualization {i/FPS:.0f}/{duration:.0f}s',flush=True)
        process.stdin.close()
        code=process.wait()
        if code:
            raise RuntimeError(f'FFmpeg exited with status {code}')
    except BaseException:
        process.kill();process.wait()
        raise
    if video.stat().st_size>=10_000_000:
        raise RuntimeError('Video exceeds the 10 MB GitHub free-plan attachment limit')
    report={'title':args.title,'credit':args.credit,'audio_source_sha256':hashlib.sha256(Path(args.audio).read_bytes()).hexdigest(),
            'duration_seconds':duration,'width':WIDTH,'height':HEIGHT,'fps':FPS,
            'visualization':'Rendered pitch targets from four voice-control lanes; activity gated by source voicing and approved comp edits',
            'voices':len(notes),'note_segments':len(segments),'pitches_midi':[int(np.nanmin(notes)),int(np.nanmax(notes))],
            'soundtrack_processing':'Encoding only; approved master used directly without DSP',
            'mp3_bytes':mp3.stat().st_size,'mp4_bytes':video.stat().st_size,'elapsed_seconds':time.time()-started}
    (out/'media.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
