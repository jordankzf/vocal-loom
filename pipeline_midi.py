"""Reusable MIDI score inspection and cautious melody/harmony extraction.

Times are always seconds, pitches are untransposed MIDI note numbers, and rests
are NaN unless a caller explicitly requests note holds. Instrumental piano MIDI
does not establish a vocal melody: automatic top-line extraction reports that
ambiguity instead of presenting a heuristic as a verified transcription.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterable

import mido
import numpy as np


@dataclass(frozen=True)
class ScoreWarning:
    code: str
    message: str
    track: int | None = None


@dataclass(frozen=True)
class Note:
    start: float
    end: float
    pitch: int
    velocity: int
    channel: int
    track: int

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_drum(self) -> bool:
        return self.channel == 9


@dataclass(frozen=True)
class TempoChange:
    tick: int
    seconds: float
    microseconds_per_beat: int

    @property
    def bpm(self) -> float:
        return 60_000_000 / self.microseconds_per_beat


@dataclass(frozen=True)
class TrackInfo:
    index: int
    name: str
    note_count: int
    drum_note_count: int
    pitch_min: int | None
    pitch_max: int | None
    pitch_median: float | None
    channels: tuple[int, ...]
    programs: tuple[int, ...]
    maximum_polyphony: int
    monophonic_fraction: float
    sounding_seconds: float


@dataclass(frozen=True)
class HarmonySnapshot:
    start: float
    end: float
    pitches: tuple[int, ...]


@dataclass
class MelodyResult:
    pitches: np.ndarray
    confidence: float
    method: str
    selected_tracks: tuple[int, ...]
    warnings: tuple[ScoreWarning, ...]

    def to_dict(self) -> dict:
        """A JSON-safe result; rests become null, not nonstandard JSON NaN."""
        return {
            "pitches": [float(v) if np.isfinite(v) else None for v in self.pitches],
            "confidence": self.confidence,
            "method": self.method,
            "selected_tracks": list(self.selected_tracks),
            "warnings": [asdict(w) for w in self.warnings],
        }


@dataclass(frozen=True)
class Score:
    path: str
    midi_type: int
    ticks_per_beat: int
    duration: float
    notes: tuple[Note, ...]
    tracks: tuple[TrackInfo, ...]
    tempo_changes: tuple[TempoChange, ...]
    time_signatures: tuple[tuple[float, int, int], ...]
    warnings: tuple[ScoreWarning, ...]

    @property
    def melodic_notes(self) -> tuple[Note, ...]:
        return tuple(n for n in self.notes if not n.is_drum)

    def melody_at(self, times, **kwargs) -> MelodyResult:
        return melody_at(self, times, **kwargs)

    def harmonic_snapshots(self, **kwargs) -> tuple[HarmonySnapshot, ...]:
        return harmonic_snapshots(self, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)


def _polyphony(notes: Iterable[Note]) -> tuple[int, float, float]:
    events = []
    for note in notes:
        events.extend(((note.start, 1), (note.end, -1)))
    # Releases precede new attacks at equal times; legato boundaries are not chords.
    events.sort(key=lambda e: (e[0], e[1]))
    maximum = count = 0
    sounding = mono = 0.0
    previous = 0.0
    for seconds, change in events:
        duration = seconds - previous
        if count:
            sounding += duration
            if count == 1:
                mono += duration
        count += change
        maximum = max(maximum, count)
        previous = seconds
    return maximum, mono / sounding if sounding else 0.0, sounding


def load_score(path: str | Path) -> Score:
    """Read a type-0/1 PPQN MIDI, including tempo changes and sustain.

    Note-on velocity zero is always a release. Channel-10 percussion is retained
    for inspection and excluded from melody/harmony extraction. CC64 sustain is
    channel-wide, including controllers placed on a separate MIDI track. CC120,
    CC121 and CC123 are honored. Unreleased notes close at the score's end with
    a warning. Type-2 independent timelines and SMPTE division are rejected.
    """
    midi = mido.MidiFile(path)
    if midi.type not in (0, 1):
        raise ValueError("Type-2 MIDI contains independent timelines; use a type-0/1 score")
    if midi.ticks_per_beat <= 0:
        raise ValueError("SMPTE MIDI timing is unsupported; export a PPQN MIDI")
    events = []
    names = []
    programs = []
    warnings = []
    end_tick = 0
    for track_index, track in enumerate(midi.tracks):
        tick = 0
        names.append(track.name or f"Track {track_index}")
        programs.append(tuple(sorted({m.program for m in track if m.type == "program_change"})))
        for order, msg in enumerate(track):
            tick += msg.time
            if tick < 0:
                raise ValueError("Negative MIDI event time")
            events.append((tick, track_index, order, msg))
        end_tick = max(end_tick, tick)
    events.sort(key=lambda e: e[:3])

    # Resolve one shared tempo map. Changes at a tick affect the following interval.
    tempos = {0: 500_000}
    explicit_tempos = {}
    for tick, track, _, msg in events:
        if msg.type == "set_tempo":
            if msg.tempo <= 0:
                raise ValueError("MIDI tempo must be positive")
            if tick in explicit_tempos and explicit_tempos[tick] != msg.tempo:
                warnings.append(ScoreWarning("conflicting_tempo", "Conflicting tempo events at one tick; the last event wins", track))
            explicit_tempos[tick] = tempos[tick] = msg.tempo
    tempo_ticks = sorted(tempos)
    tempo_seconds = [0.0]
    for previous, tick in zip(tempo_ticks, tempo_ticks[1:]):
        tempo_seconds.append(tempo_seconds[-1] + mido.tick2second(tick - previous, midi.ticks_per_beat, tempos[previous]))

    def seconds_at(tick):
        index = bisect_right(tempo_ticks, tick) - 1
        return tempo_seconds[index] + mido.tick2second(tick - tempo_ticks[index], midi.ticks_per_beat, tempos[tempo_ticks[index]])

    duration = seconds_at(end_tick)
    held = defaultdict(deque)
    sustained = defaultdict(list)
    pedal = set()
    notes = []
    unmatched = defaultdict(int)

    def close(started, end):
        start, velocity, channel, track, pitch = started
        if end > start:
            notes.append(Note(start, end, pitch, velocity, channel, track))

    def release(key, seconds):
        if not held[key]:
            # Some exporters write note-offs on another track sharing the channel.
            alternatives = [k for k, values in held.items() if values and k[1:] == key[1:]]
            if len(alternatives) == 1:
                key = alternatives[0]
            else:
                unmatched[key[0]] += 1
                return
        started = held[key].popleft()
        if key[1] in pedal:
            sustained[key[1]].append(started)
        else:
            close(started, seconds)

    signatures = []
    for tick, track, _, msg in events:
        seconds = seconds_at(tick)
        if msg.type == "time_signature":
            signatures.append((seconds, msg.numerator, msg.denominator))
        if not hasattr(msg, "channel"):
            continue
        channel = msg.channel
        if msg.type in ("note_on", "note_off"):
            key = track, channel, msg.note
            if msg.type == "note_on" and msg.velocity:
                held[key].append((seconds, msg.velocity, channel, track, msg.note))
            else:
                release(key, seconds)
        elif msg.type == "control_change":
            if msg.control == 64:
                if msg.value >= 64:
                    pedal.add(channel)
                else:
                    pedal.discard(channel)
                    for started in sustained.pop(channel, []):
                        close(started, seconds)
            elif msg.control in (120, 123):
                for key in list(held):
                    if key[1] == channel:
                        while held[key]:
                            started = held[key].popleft()
                            if msg.control == 123 and channel in pedal:
                                sustained[channel].append(started)
                            else:
                                close(started, seconds)
                if msg.control == 120:
                    for started in sustained.pop(channel, []):
                        close(started, seconds)
            elif msg.control == 121:
                pedal.discard(channel)
                for started in sustained.pop(channel, []):
                    close(started, seconds)
    dangling = sum(len(v) for v in held.values()) + sum(len(v) for v in sustained.values())
    for values in [*held.values(), *sustained.values()]:
        for started in values:
            close(started, duration)
    if dangling:
        warnings.append(ScoreWarning("unreleased_notes", f"Closed {dangling} unreleased/sustained notes at the score end"))
    for track, count in unmatched.items():
        warnings.append(ScoreWarning("unmatched_note_off", f"Ignored {count} note-off events with no matching note-on", track))
    notes.sort(key=lambda n: (n.start, n.track, n.pitch, n.end))
    tracks = []
    for index, name in enumerate(names):
        all_notes = [n for n in notes if n.track == index]
        tonal = [n for n in all_notes if not n.is_drum]
        pitches = [n.pitch for n in tonal]
        maximum, mono, sounding = _polyphony(tonal)
        tracks.append(TrackInfo(index, name, len(tonal), len(all_notes) - len(tonal),
                                min(pitches) if pitches else None, max(pitches) if pitches else None,
                                float(np.median(pitches)) if pitches else None,
                                tuple(sorted({n.channel for n in all_notes})), programs[index],
                                maximum, mono, sounding))
    if not any(not n.is_drum for n in notes):
        warnings.append(ScoreWarning("no_pitched_notes", "No pitched notes are available for vocal melody or harmony"))
    changes = tuple(TempoChange(tick, seconds, tempos[tick]) for tick, seconds in zip(tempo_ticks, tempo_seconds))
    return Score(str(Path(path).resolve()), midi.type, midi.ticks_per_beat, duration,
                 tuple(notes), tuple(tracks), changes, tuple(signatures), tuple(warnings))


def _track_selection(score: Score, selection) -> tuple[tuple[int, ...], float, str, tuple[ScoreWarning, ...]]:
    available = [track for track in score.tracks if track.note_count]
    if not available:
        return (), 0.0, "no_melody", (ScoreWarning("no_pitched_notes", "Score has no nonpercussion notes"),)
    if selection != "auto":
        if isinstance(selection, (int, np.integer)):
            indices = (int(selection),)
        else:
            indices = tuple(int(v) for v in selection)
        valid = {t.index for t in available}
        if not indices or any(i not in valid for i in indices):
            raise ValueError(f"Select a nonempty pitched track from {sorted(valid)}")
        mono = min(score.tracks[i].monophonic_fraction for i in indices)
        warnings = () if len(indices) == 1 and mono >= .9 else (
            ScoreWarning("polyphonic_override", "Selected tracks are polyphonic; their highest active note is a melody heuristic"),)
        return indices, .95 if not warnings else .55, "selected_track", warnings

    ranked = []
    for track in available:
        name = track.name.lower()
        named = bool(re.search(r"\b(melody|vocal|vocals|lead|voice|soprano|sing)\b", name))
        bass = bool(re.search(r"\b(bass|drum|percussion|accompaniment|chords?)\b", name))
        score_value = 3 * track.monophonic_fraction + min(track.note_count, 40) / 40
        score_value += 3 if named else 0
        score_value -= 3 if bass else 0
        score_value += np.clip((track.pitch_median - 48) / 24, -1, 1)
        ranked.append((float(score_value), named, track))
    ranked.sort(key=lambda entry: entry[0], reverse=True)
    best_score, named, best = ranked[0]
    if best.monophonic_fraction >= .9 and best.note_count >= 2:
        confidence = .9 if named else .8
        warnings = [ScoreWarning("unverified_melody", "Automatic track selection is inferred from musical structure; verify it against the intended vocal line", best.index)]
        if best.pitch_median < 48:
            confidence = min(confidence, .55)
            warnings.append(ScoreWarning("low_register_track", "The selected monophonic track may be a bass line", best.index))
        if len(ranked) > 1 and best_score - ranked[1][0] < .5:
            confidence = min(confidence, .65)
            warnings.append(ScoreWarning("ambiguous_melody_track", "Multiple tracks are plausible melodies; an explicit track selection is recommended"))
        return (best.index,), confidence, "automatic_monophonic_track", tuple(warnings)
    # A named polyphonic vocal track is preferable to adding an entire arrangement.
    indices = (best.index,) if named else tuple(track.index for track in available)
    return indices, .4, "automatic_polyphonic_topline", (
        ScoreWarning("instrumental_melody_ambiguous", "No clear monophonic melody track was found. The highest active instrumental note may be accompaniment or a piano ornament; verify or override the melody track"),)


def melody_at(score: Score, times, track_selection="auto", *, hold_rests: bool = False,
              transpose: int = 0) -> MelodyResult:
    """Sample a melody at arbitrary seconds; return notes plus confidence/warnings.

    An explicit track index or iterable overrides automatic selection. Polyphonic
    tracks use the highest active note. ``hold_rests=True`` carries the previous
    note through internal/final rests, but never backward before its first attack.
    No automatic vocal-register transposition or key correction is performed.
    """
    sample_times = np.asarray(times, dtype=float)
    if sample_times.ndim != 1 or not np.isfinite(sample_times).all():
        raise ValueError("times must be a finite one-dimensional array")
    indices, confidence, method, warnings = _track_selection(score, track_selection)
    snapshots = harmonic_snapshots(score, tracks=indices, hold_rests=hold_rests)
    pitches = np.full(sample_times.shape, np.nan)
    if snapshots:
        starts = [s.start for s in snapshots]
        values=[]
        previous=None
        selected_notes=[n for n in score.melodic_notes if n.track in indices]
        for snapshot in snapshots:
            value=max(snapshot.pitches) if snapshot.pitches else None
            # A released top note can leave only an old accompaniment tail.
            # Do not turn that held bass/chord into a new sung melody attack.
            if hold_rests and method=='automatic_polyphonic_topline' and previous is not None and value is not None and value<previous:
                fresh=any(n.pitch==value and abs(n.start-snapshot.start)<1e-6
                          for n in selected_notes)
                if not fresh:
                    value=previous
            if value is not None:
                previous=value
            values.append(value)
        for i, seconds in enumerate(sample_times):
            index = bisect_right(starts, seconds) - 1
            if index >= 0:
                snapshot = snapshots[index]
                if seconds < snapshot.end and snapshot.pitches:
                    pitches[i] = values[index] + transpose
                elif hold_rests and seconds >= snapshot.end and snapshot.pitches:
                    pitches[i] = values[index] + transpose
    if confidence < .6:
        warnings += (ScoreWarning("review_required", "This melody guide is provisional and requires musical verification"),)
    return MelodyResult(pitches, confidence, method, indices, score.warnings + warnings)


def harmonic_snapshots(score: Score, *, tracks: Iterable[int] | None = None,
                       hold_rests: bool = False, min_notes: int = 1) -> tuple[HarmonySnapshot, ...]:
    """Return exact active pitched-note sets with boundaries in seconds.

    Sustain is already resolved by ``load_score``. Octave doublings remain intact;
    simultaneous identical pitches collapse. No chord labels or missing thirds
    are invented. With ``min_notes>1`` sparse moments are treated as rests, and
    ``hold_rests=True`` carries the most recent qualifying set across such gaps.
    """
    if min_notes < 1:
        raise ValueError("min_notes must be positive")
    selected = None if tracks is None else set(int(v) for v in tracks)
    if selected is not None and any(i < 0 or i >= len(score.tracks) for i in selected):
        raise ValueError("Track index outside score")
    events = defaultdict(list)
    for note in score.notes:
        if note.is_drum or (selected is not None and note.track not in selected):
            continue
        events[note.start].append((note.pitch, 1))
        events[note.end].append((note.pitch, -1))
    if not events:
        return ()
    events.setdefault(score.duration, [])
    points = sorted(events)
    counts = defaultdict(int)
    last = ()
    snapshots = []
    for start, end in zip(points, points[1:]):
        for pitch, delta in events[start]:
            counts[pitch] += delta
        active = tuple(sorted(p for p, count in counts.items() if count > 0))
        if len(active) < min_notes:
            active = last if hold_rests else ()
        else:
            last = active
        if end <= start:
            continue
        if snapshots and snapshots[-1].pitches == active:
            previous = snapshots[-1]
            snapshots[-1] = HarmonySnapshot(previous.start, end, active)
        else:
            snapshots.append(HarmonySnapshot(start, end, active))
    return tuple(snapshots)
