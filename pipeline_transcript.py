"""Local transcription and plain-text lyric hints; lyric text is never executed."""
from pathlib import Path
import json
import re
from math import gcd
import numpy as np
from scipy.signal import resample_poly


def read_lyrics(path):
    text=Path(path).read_text(encoding='utf-8-sig')
    lines=[]
    for line in text.splitlines():
        line=re.sub(r'^\s*\[[^\]]*\]\s*','',line).strip()
        if re.fullmatch(r'[\d\s:,.\- >]+',line or ' '):
            continue
        if line:
            lines.append(line)
    if not lines:
        raise ValueError('The lyrics file is empty after removing headings/timestamps')
    return '\n'.join(lines)


def transcribe(audio,sr,cache_dir,*,model_name='small.en',language='en',progress=print):
    from faster_whisper import WhisperModel
    progress('Transcribing the take locally for phrase and repeated-attempt checks...')
    factor=gcd(sr,16000)
    mono=np.asarray(audio,dtype=np.float32)
    if mono.ndim==2:
        mono=mono.mean(axis=1)
    samples=resample_poly(mono,16000//factor,sr//factor).astype(np.float32)
    peak=np.max(abs(samples),initial=0)
    if peak>0:
        samples*=min(4,.8/peak)
    model=WhisperModel(model_name,device='cpu',compute_type='int8',cpu_threads=4,
                       download_root=str(Path(cache_dir).resolve()))
    segments,_=model.transcribe(samples,language=None if language=='auto' else language,
                                word_timestamps=True,beam_size=5,vad_filter=False,
                                condition_on_previous_text=False)
    result=[]
    for s in segments:
        result.append({'start':s.start,'end':s.end,'text':s.text,
                       'words':[{'start':w.start,'end':w.end,'word':w.word,
                                 'probability':w.probability} for w in s.words]})
        progress(f'Phrase analysis through {s.end:.0f} seconds')
    return result


def words_from_segments(segments):
    return [w for s in segments for w in s.get('words',[])]


def lyric_coverage(segments,lyrics):
    """An alignment hint, not an accuracy score; ASR is imperfect on singing."""
    from difflib import SequenceMatcher
    def tokens(text):
        return re.findall(r"[\w']+",text.lower())
    heard=tokens(' '.join(s.get('text','') for s in segments))
    expected=tokens(lyrics)
    matches=SequenceMatcher(None,expected,heard,autojunk=False).get_matching_blocks()
    return {'matched_tokens':sum(m.size for m in matches),'reference_tokens':len(expected),
            'transcribed_tokens':len(heard),
            'warning':'Automatic transcription is a phrase/comping aid, not an exact lyric verifier.'}
