"""Choose one higher duet part from real score tones, with smooth voice leading."""
from __future__ import annotations

import numpy as np

from arrangement import harmonic_snapshots, load_note_events


def _interval_cost(interval):
    # Thirds lead, sixths provide the other preferred close vocal relationship.
    preferred = ((3., 0.), (4., 0.), (8., .4), (9., .4))
    nearest = min(cost + abs(interval-distance)*1.25 for distance, cost in preferred)
    if abs(interval-5) < .6 or abs(interval-7) < .6:
        nearest = min(nearest, 1.65)
    if interval > 12:
        nearest += 4.0  # compound thirds are a continuity escape, not the default
    return nearest


def choose_duet(midi_times, lead_midi, notes_path, transpose=-12):
    """Return ``(targets, audit, warnings)`` with targets shaped ``(1, N)``.

    Targets come only from pitch classes in held MIDI harmonic snapshots. Notes
    normally sit 3–12 semitones above the lead; up to 16 is available when it
    meaningfully avoids a large leap. Thirds/sixths are preferred, not imposed
    when the score lacks those notes. A dynamic program balances the whole line
    across melody and chord changes. The first available third is preferred at
    the opening. NaN lead rests stay muted. A score without harmonic snapshots
    falls back to an octave and emits an explicit warning.
    """
    times = np.asarray(midi_times, dtype=float)
    lead = np.asarray(lead_midi, dtype=float)
    if times.ndim != 1 or lead.shape != times.shape or not np.isfinite(times).all():
        raise ValueError("MIDI times and lead must share a one-dimensional timeline")
    if np.any(np.diff(times) < 0) or np.isinf(lead).any():
        raise ValueError("MIDI times must increase and lead pitches may be finite or NaN")
    output = np.full((1, len(times)), np.nan)
    warnings = []
    if not len(times):
        return output, [], warnings
    snapshots = harmonic_snapshots(load_note_events(notes_path), transpose=transpose)
    if not snapshots:
        output[0] = np.where(np.isfinite(lead), lead+12, np.nan)
        warnings.append("No polyphonic chord part was found in the MIDI; the duet uses an octave above the lead.")
        return output, [{"method": "octave_fallback", "interval_semitones": 12}], warnings
    starts = np.array([snapshot["time"] for snapshot in snapshots])
    ids = np.maximum(0, np.searchsorted(starts, times, side="right")-1)
    # A selected note stays steady through each scored melody/chord segment.
    rounded = np.where(np.isfinite(lead), np.round(lead), -999)
    changes = np.r_[0, np.flatnonzero((np.diff(ids) != 0) | (np.diff(rounded) != 0))+1, len(times)]
    segments = []
    for first, last in zip(changes, changes[1:]):
        if not np.isfinite(lead[first]):
            continue
        section = lead[first:last]
        snapshot = snapshots[ids[first]]
        pcs = {int(note) % 12 for note in snapshot["pool"]}
        low = int(np.ceil(np.max(section)+3))
        normal_high = int(np.floor(np.min(section)+12))
        high = int(np.floor(np.min(section)+16))
        candidates = np.array([note for note in range(low, high+1) if note % 12 in pcs], dtype=float)
        if not len(candidates):
            raise ValueError("The MIDI harmony cannot provide an upper duet note within the supported range")
        center = float(np.median(section))
        normal = candidates[candidates <= normal_high]
        # Extensions are permitted only where they can continue an earlier
        # voice more smoothly; their vertical cost is deliberately substantial.
        local = np.array([_interval_cost(note-center) for note in candidates])
        if last-first > 1:
            duration = max(.05, float(times[last-1]-times[first]))
        else:
            duration = .05
        weight = min(2., max(.4, np.sqrt(duration)))
        segments.append({"first": first, "last": last, "lead": center,
                         "candidates": candidates, "local": local*weight,
                         "snapshot": snapshot, "pitch_classes": sorted(pcs)})
    if not segments:
        return output, [], warnings
    costs = []
    parents = []
    for index, segment in enumerate(segments):
        candidates = segment["candidates"]
        local = segment["local"]
        if index == 0:
            intervals = candidates-segment["lead"]
            thirds = np.flatnonzero((intervals >= 2.5) & (intervals <= 4.5))
            initial = local.copy()
            if len(thirds):
                # Honor a clear opening third rather than trading it away for
                # accompaniment convenience many phrases later.
                initial[np.setdiff1d(np.arange(len(initial)), thirds)] = np.inf
            costs.append(initial)
            parents.append(np.full(len(candidates), -1))
            continue
        previous = segments[index-1]["candidates"]
        movement = abs(candidates[:, None]-previous[None, :])
        transition = .34*movement + .08*np.maximum(movement-5, 0)**2
        total = transition+costs[-1][None, :]
        best = np.argmin(total, axis=1)
        costs.append(local+total[np.arange(len(candidates)), best])
        parents.append(best)
    chosen = [0]*len(segments)
    chosen[-1] = int(np.argmin(costs[-1]))
    for index in range(len(segments)-1, 0, -1):
        chosen[index-1] = int(parents[index][chosen[index]])
    audit = []
    for segment, choice in zip(segments, chosen):
        note = float(segment["candidates"][choice])
        first, last = segment["first"], segment["last"]
        output[0, first:last] = note
        audit.append({"midi_start_seconds": float(times[first]),
                      "midi_end_seconds": float(times[last-1]),
                      "lead_midi": segment["lead"], "harmony_midi": note,
                      "interval_semitones": note-segment["lead"],
                      "source_chord_pitch_classes": segment["pitch_classes"],
                      "source_notes": segment["snapshot"]["source_notes"]})
    extended = sum(entry["interval_semitones"] > 12 for entry in audit)
    if extended:
        warnings.append(f"The duet uses {extended} wider intervals, up to a tenth, to preserve score tones and voice continuity.")
    return output, audit, warnings
