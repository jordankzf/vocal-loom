"""Create a tiny, newly written synthetic example; no user media is included."""
from pathlib import Path
import argparse
import hashlib
import json

import mido
import numpy as np
import soundfile as sf


def create_fixture(directory, sample_rate=24000):
    out=Path(directory)
    out.mkdir(parents=True,exist_ok=True)
    pitches=[57,59,61,64,62]
    starts=.25+np.arange(len(pitches))*.60
    duration=float(starts[-1]+.50+.25)
    x=np.zeros(round(duration*sample_rate),dtype=np.float64)
    for start,note in zip(starts,pitches):
        length=round(.50*sample_rate)
        t=np.arange(length)/sample_rate
        hz=440*2**((note+.35-69)/12)
        vowel=.12*np.sin(2*np.pi*hz*t)+.045*np.sin(2*np.pi*2*hz*t)+.02*np.sin(2*np.pi*3*hz*t)
        fade=round(.025*sample_rate)
        vowel[:fade]*=np.sin(np.linspace(0,np.pi/2,fade))**2
        vowel[-fade:]*=np.cos(np.linspace(0,np.pi/2,fade))**2
        pos=round(start*sample_rate)
        x[pos:pos+length]=vowel
    audio=out/'synthetic-vocal.wav'
    sf.write(audio,x,sample_rate,subtype='PCM_24')
    midi=mido.MidiFile(type=1,ticks_per_beat=480)
    meta=mido.MidiTrack([mido.MetaMessage('track_name',name='Synthetic demonstration'),
                         mido.MetaMessage('set_tempo',tempo=500000)])
    melody=mido.MidiTrack([mido.MetaMessage('track_name',name='Vocal melody')])
    last=0
    for start,note in zip(starts,pitches):
        on=round(start*960);off=round((start+.50)*960)
        melody.append(mido.Message('note_on',channel=0,note=note,velocity=90,time=on-last))
        melody.append(mido.Message('note_off',channel=0,note=note,velocity=0,time=off-on))
        last=off
    midi.tracks.extend([meta,melody])
    score=out/'synthetic-score.mid';midi.save(score)
    lyrics=out/'fictional-lyrics.txt'
    lyrics.write_text('Lanterns shimmer over quiet water\n',encoding='utf-8')
    manifest={'description':'New synthetic harmonic vowels, no human voice or copyrighted song',
              'sample_rate':sample_rate,'duration_seconds':duration,'pitches_midi':pitches,
              'source_detuning_cents':35.,'starts_seconds':starts.tolist(),
              'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [audio,score,lyrics]}}
    (out/'fixture.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return {'audio':audio,'midi':score,'lyrics':lyrics,'manifest':manifest}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',nargs='?',default='examples/synthetic')
    args=parser.parse_args()
    fixture=create_fixture(args.directory)
    print(f'Created synthetic example in {Path(args.directory).resolve()}')
