"""Continuous offline vocal tuning, retiming, harmonies, and stereo mastering.

The orchestration layer owns file I/O and cleanup. All arrays here are local,
floating-point audio; write returned stems/master as PCM_24 when exporting WAV.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any

import numpy as np
import parselmouth
from parselmouth.praat import call
import pyworld as pw
import pyloudnorm as pyln
from pedalboard import Compressor, HighpassFilter, PeakFilter, Pedalboard, Reverb
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import resample_poly


@dataclass
class RenderResult:
    audio: np.ndarray
    sample_rate: int
    stems: dict[str, np.ndarray]
    automation: dict[str, np.ndarray]
    report: dict[str, Any]


def hertz(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


def fit_length(audio, length):
    x = np.asarray(audio, dtype=float).reshape(-1)
    return np.pad(x, (0, max(0, length - len(x))))[:length]


def _resample(audio, original_rate, target_rate):
    common = gcd(int(original_rate), int(target_rate))
    return resample_poly(audio, int(target_rate) // common, int(original_rate) // common)


def _finite_fill(values, times, fallback):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.any():
        return np.interp(times, times[finite], values[finite])
    return np.broadcast_to(fallback, values.shape).astype(float).copy()


def _pitch_bounds(f0):
    voiced = f0[f0 > 0]
    if not len(voiced):
        raise ValueError("The recording has no clearly pitched singing to tune")
    p5, p95 = np.percentile(voiced, [5, 95])
    return float(np.clip(p5 * .72, 40, 150)), float(np.clip(p95 * 1.65, 300, 1800))


def _observed_lead(f0, times, strength=None):
    """Robust actual tuning, including cents and genuinely wrong sung notes.

    Reject low-periodicity frames, bridge gaps with the nearest reliable frame,
    then remove isolated tracker spikes with a ~20 ms median and 10 ms Gaussian.
    Gaps are filled for target interpolation only; harmony gates stay closed.
    """
    valid = f0 > 0
    if strength is not None:
        strength = np.asarray(strength, dtype=float)
        if strength.shape != f0.shape or not np.isfinite(strength).all():
            raise ValueError("Pitch confidence must share the finite pitch timeline")
        valid &= strength >= .6
    if not valid.any():
        raise ValueError("Not enough periodic singing to make harmonies follow the lead")
    good = np.flatnonzero(valid)
    insertion = np.searchsorted(times[good], times)
    right = np.clip(insertion, 0, len(good)-1)
    left = np.clip(insertion-1, 0, len(good)-1)
    nearest = np.where(abs(times-times[good[left]]) <= abs(times-times[good[right]]), good[left], good[right])
    observed = 69+12*np.log2(f0[nearest]/440)
    step = float(np.median(np.diff(times)))
    kernel = max(1, round(.020/step)) | 1
    observed = median_filter(observed, size=kernel, mode="nearest")
    observed = gaussian_filter1d(observed, max(.1, .010/step), mode="nearest")
    return observed, valid


def _transparent_targets(sung, reference, target_valid, voiced, times, strength=None):
    """Tune note centres while preserving the measured within-note trajectory.

    One robust offset is applied per reference/source-note plateau. This leaves
    vibrato and scoops intact instead of replacing the trajectory with MIDI.
    """
    step = float(np.median(np.diff(times)))
    reliable = voiced.copy()
    if strength is not None:
        confidence = np.asarray(strength, dtype=float)
        if confidence.shape != sung.shape:
            raise ValueError("Pitch confidence must share the pitch timeline")
        reliable &= np.isfinite(confidence) & (confidence >= .6)
    labels = np.where(target_valid, np.round(reference), -999)
    boundaries = set(np.r_[0, np.flatnonzero(np.diff(labels) != 0)+1, len(times)].tolist())
    # A sustained actual wrong note inside one reference note needs its own
    # centre. Short vibrato half-cycles and tracker blips do not create notes.
    trend = median_filter(sung, size=max(1, round(.09/step)) | 1, mode="nearest")
    plateau = np.round(trend)
    runs = np.r_[0, np.flatnonzero(np.diff(plateau) != 0)+1, len(times)]
    previous = None
    for first, last in zip(runs, runs[1:]):
        if (last-first)*step < .18 or np.mean(reliable[first:last]) < .65:
            continue
        value = float(np.median(trend[first:last]))
        if previous is not None and abs(value-previous) >= 1.5:
            boundaries.add(int(first))
        previous = value
    # A long silence separates repetitions of the same scored note.
    silence_runs = np.r_[0, np.flatnonzero(np.diff(voiced.astype(int)) != 0)+1, len(times)]
    for first, last in zip(silence_runs, silence_runs[1:]):
        if not voiced[first] and (last-first)*step >= .18:
            boundaries.update((int(first), int(last)))
    points = sorted(boundaries)
    offsets = np.zeros(len(times))
    audit = []
    for first, last in zip(points, points[1:]):
        if not target_valid[first] or last <= first:
            continue
        trim = min(round(.065/step), (last-first)//5)
        region = np.arange(first+trim, last-trim)
        good = region[reliable[region]]
        if len(good) < 3:
            good = np.arange(first, last)[voiced[first:last]]
        if not len(good):
            continue
        center = float(np.median(sung[good]))
        intended = float(np.median(reference[first:last]))
        shift = intended-center
        if abs(shift) < .18:
            shift = 0.0  # avoid processing already-in-tune note centres
        offsets[first:last] = shift
        audit.append({"source_start": float(times[first]), "source_end": float(times[last-1]),
                      "measured_center_midi": center, "target_center_midi": intended,
                      "correction_semitones": shift})
    # Smooth correction decisions, not the sung trajectory. About 50 ms around
    # a boundary avoids hard jumps without flattening the singer's vibrato.
    offsets = gaussian_filter1d(offsets, max(.1, .018/step), mode="nearest")
    offsets[~target_valid] = 0
    return sung+offsets, offsets, audit


def praat_voice(sound, manipulation, source_times, target_midi, voiced, duration_ratio):
    """Resynthesize one whole voice using source pulse marks, never short windows."""
    tier = call("Create PitchTier", "corrected_vocal", 0, sound.duration)
    valid = voiced & np.isfinite(target_midi)
    if not valid.any():
        raise ValueError("No voiced target points are available")
    for seconds, hz in zip(source_times[valid], hertz(target_midi[valid])):
        call(tier, "Add point", float(seconds), float(hz))
    call([tier, manipulation], "Replace pitch tier")
    duration = call("Create DurationTier", "corrected_timing", 0, sound.duration)
    # Include the endpoints so the integrated duration does not inherit an
    # arbitrary first/last sample extension. A point every 20 ms is sufficient.
    stride = max(1, round(.02 / np.median(np.diff(source_times))))
    times = np.unique(np.r_[0., source_times[::stride], sound.duration])
    rates = np.interp(times, source_times, duration_ratio)
    for seconds, ratio in zip(times, rates):
        call(duration, "Add point", float(seconds), float(ratio))
    call([duration, manipulation], "Replace duration tier")
    return call(manipulation, "Get resynthesis (overlap-add)").values[0]


def master(mix, sr, *, target_lufs=-17.0, highpass_hz=60.0, reverb_wet=.065):
    """Return finite stereo audio and measured loudness/4x true-peak metrics."""
    mix = np.asarray(mix, dtype=float)
    if mix.ndim != 2 or mix.shape[1] != 2 or not np.isfinite(mix).all():
        raise ValueError("Master input must be finite stereo audio")
    if len(mix) < 2 or np.max(abs(mix), initial=0) < 1e-10:
        return np.zeros_like(mix, dtype=np.float32), {"loudness_lufs": None, "true_peak_4x_dbfs": None}
    chain = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=highpass_hz),
        PeakFilter(cutoff_frequency_hz=380, gain_db=-1.6, q=.65),
        PeakFilter(cutoff_frequency_hz=2500, gain_db=.7, q=.7),
        Compressor(threshold_db=-23, ratio=2.5, attack_ms=12, release_ms=120),
        Reverb(room_size=.38, damping=.67, wet_level=reverb_wet,
               dry_level=1-reverb_wet, width=.9),
    ])
    y = chain(mix.T.astype(np.float32), sr).T.astype(float)
    block_size = min(.4, max(.01, len(y)/sr*.8))
    meter = pyln.Meter(sr, block_size=block_size)
    measured = float(meter.integrated_loudness(y))
    if np.isfinite(measured):
        y *= 10**(np.clip(target_lufs-measured, -35, 35)/20)
    ceiling = 10**(-1.1/20)
    true_peak = float(np.max(abs(resample_poly(y, 4, 1, axis=0)), initial=0))
    if true_peak > ceiling:
        y *= ceiling/true_peak
    fade = min(len(y), round(.30*sr))
    y[-fade:] *= np.linspace(1, 0, fade)[:, None]
    if not np.isfinite(y).all():
        raise ValueError("Mastering produced nonfinite audio")
    final_lufs = float(meter.integrated_loudness(y))
    true_peak = float(np.max(abs(resample_poly(y, 4, 1, axis=0)), initial=0))
    metrics = {"loudness_lufs": final_lufs if np.isfinite(final_lufs) else None,
               "true_peak_4x_dbfs": 20*np.log10(true_peak) if true_peak > 0 else None}
    return y.astype(np.float32), metrics


def _interpolate_frames(frames, source_grid, lookup):
    """Linear frame interpolation, preserving float64 contiguous WORLD inputs."""
    index = np.interp(lookup, source_grid, np.arange(len(source_grid)))
    left = np.floor(index).astype(int)
    right = np.minimum(left+1, len(source_grid)-1)
    fraction = (index-left)[:, None]
    return np.ascontiguousarray(frames[left]*(1-fraction)+frames[right]*fraction)


def render_vocal(audio, sr, performance, output_times, lead_midi, harmony_midi=None,
                 mode="solo", pitch_strength=1.0, progress=None,
                 lead_mode="tuned", harmony_follow="score",
                 pitch_style="transparent", harmony_engine="auto") -> RenderResult:
    """Render a cleaned mono performance on a corrected clock.

    ``performance`` provides 1-D ``times`` and ``f0`` arrays (seconds, Hz).
    ``output_times`` gives the corrected time for each source pitch frame; it
    must increase strictly. The clock is extrapolated to audio endpoints and
    normalized so the mapped source start is zero. ``lead_midi`` has one pitch
    per frame; NaN rests preserve the source voice. ``harmony_midi`` is (K,N),
    with NaN meaning muted. Supply one voice for duet and three for vocoder.

    ``pitch_strength`` blends correction in semitones. Solo/duet retain a small,
    bounded amount of fast natural vibrato near the intended note; the stack
    uses tighter pitch. Continuous TD-PSOLA tunes/retimes the lead, while WORLD
    interpolates spectral frames to the same timeline for harmony resynthesis.
    Optional ``progress(message)`` reports stage changes to the caller.
    ``lead_mode='original'`` bypasses lead pitch/timing resynthesis completely:
    its stem is an exact copy of the cleaned input samples. The output clock
    becomes the original recording clock, and harmony contours remain associated
    with their supplied source performance times. Master processing still applies.
    ``harmony_follow='lead'`` requires an original lead. It shifts every harmony
    by actual sung MIDI pitch minus intended lead MIDI pitch, including cents
    and large sung-note errors; the original lead is never retuned to the stack.
    """
    def notify(message):
        if progress:
            progress(message)

    if mode not in {"solo", "duet", "vocoder"}:
        raise ValueError("mode must be solo, duet, or vocoder")
    if lead_mode not in {"tuned", "original"}:
        raise ValueError("lead_mode must be tuned or original")
    if harmony_follow not in {"score", "lead"}:
        raise ValueError("harmony_follow must be score or lead")
    if harmony_follow == "lead" and lead_mode != "original":
        raise ValueError("Lead-following harmonies require lead_mode='original'")
    if pitch_style not in {"transparent", "tight"}:
        raise ValueError("pitch_style must be transparent or tight")
    if harmony_engine == "auto":
        harmony_engine = "psola" if pitch_style == "transparent" else "world"
    if harmony_engine not in {"psola", "world"}:
        raise ValueError("harmony_engine must be psola, world, or auto")
    if not 0 <= pitch_strength <= 1:
        raise ValueError("pitch_strength must be between 0 and 1")
    sr = int(sr)
    x = np.asarray(audio, dtype=float)
    if sr < 16000 or x.ndim != 1 or len(x) < sr*.25 or not np.isfinite(x).all():
        raise ValueError("Use finite mono audio of at least 0.25 seconds at 16 kHz or higher")
    if np.max(abs(x), initial=0) < 1e-8:
        raise ValueError("The recording is silent")
    times = np.asarray(performance.times, dtype=float)
    f0 = np.asarray(performance.f0, dtype=float)
    output_times = np.asarray(output_times, dtype=float)
    lead = np.asarray(lead_midi, dtype=float)
    if times.ndim != 1 or len(times) < 3 or any(a.shape != times.shape for a in (f0, output_times, lead)):
        raise ValueError("Pitch, target and clock arrays must share one frame timeline")
    if not np.isfinite(times).all() or not np.isfinite(f0).all() or not np.isfinite(output_times).all():
        raise ValueError("Pitch and clock arrays must be finite")
    if np.any(f0 < 0) or np.any(np.diff(times) <= 0) or np.any(np.diff(output_times) <= 0):
        raise ValueError("Pitch must be nonnegative and source/output clocks must increase")
    duration = len(x)/sr
    if times[0] < 0 or times[-1] > duration+.001:
        raise ValueError("Pitch frame times lie outside the supplied recording")
    voiced = f0 > 0
    floor, ceiling = _pitch_bounds(f0)
    step = float(np.median(np.diff(times)))
    if lead_mode == "original":
        # Targets are already sampled on the performance's source timeline.
        # Ignore timing edits while preserving where the harmony changes occur.
        output_times = times.copy()
    ratio = gaussian_filter1d(np.gradient(output_times, times), max(.5, .025/step))
    if np.min(ratio) < .25 or np.max(ratio) > 4:
        raise ValueError("The timing map requests an excessive local stretch")
    origin = output_times[0] - times[0]*ratio[0]
    clock = output_times-origin
    target_duration = float(clock[-1]+(duration-times[-1])*ratio[-1])
    length = round(target_duration*sr)
    if length < 1:
        raise ValueError("The corrected duration is empty")

    sung = 69+12*np.log2(np.maximum(f0, 1)/440)
    sung = np.interp(times, times[voiced], sung[voiced])
    reference = np.where(np.isfinite(lead), lead, sung)
    note_center_audit = []
    if pitch_style == "transparent":
        transparent, offsets, note_center_audit = _transparent_targets(
            sung, reference, np.isfinite(lead), voiced, times, getattr(performance, "strength", None))
        effective_lead = sung+pitch_strength*offsets
    else:
        smoothed = gaussian_filter1d(reference, max(.35, .006/step))
        residual = sung-gaussian_filter1d(sung, max(.75, .075/step))
        residual = np.clip(residual, -.25, .25)
        near_target = abs(sung-reference) < 1.25
        natural = .32*residual*near_target if mode != "vocoder" else 0
        effective_lead = sung+pitch_strength*(smoothed+natural-sung)
    effective_lead[~np.isfinite(lead)] = sung[~np.isfinite(lead)]
    source_bypass = (lead_mode == "original" or
                     (np.max(abs(effective_lead[voiced]-sung[voiced]), initial=0) < 1e-10
                      and np.max(abs(ratio-1), initial=0) < 1e-8))
    sound = manipulation = None
    if source_bypass:
        notify("Keeping your original lead pitch and phrasing…")
        effective_lead = sung.copy()
        lead_audio = x.copy()
        stems = {"lead": lead_audio}
    else:
        notify("Correcting the lead vocal and timing…")
        sound = parselmouth.Sound(x, sampling_frequency=sr)
        manipulation = call(sound, "To Manipulation", min(.005, step), floor, ceiling)
        lead_audio = fit_length(praat_voice(sound, manipulation, times, effective_lead, voiced, ratio), length)
        stems = {"lead": lead_audio.astype(np.float32)}
    requested = {"solo": 0, "duet": 1, "vocoder": 3}[mode]
    harmonies = np.empty((0, len(times)))
    reference_harmonies = np.empty((0, len(times)))
    observed_lead = sung.copy()
    follow_offset = np.zeros(len(times))
    harmony_voiced = voiced.copy()
    warnings = []
    if np.any(voiced & ~np.isfinite(lead)):
        warnings.append("Sung frames without a MIDI melody target retain their original pitch.")
    if requested:
        harmonies = np.asarray(harmony_midi, dtype=float)
        if harmonies.ndim == 1:
            harmonies = harmonies[None, :]
        if harmonies.ndim != 2 or harmonies.shape[1] != len(times) or harmonies.shape[0] < requested:
            raise ValueError(f"{mode} mode needs {requested} harmony target contours")
        harmonies = harmonies[:requested].copy()
        reference_harmonies = harmonies.copy()
        if np.any(np.isinf(harmonies)):
            raise ValueError("Harmony targets may contain pitches or NaN rests, not infinity")
        if harmony_follow == "lead":
            observed_lead, harmony_voiced = _observed_lead(f0, times, getattr(performance, "strength", None))
            follow_offset = np.where(np.isfinite(lead), observed_lead-lead, 0.)
            # Do not limit this displacement to small intonation fixes: the
            # user's wrong sung notes also become the stack's local reference.
            harmonies += follow_offset[None, :]
        elif pitch_style == "transparent" and lead_mode == "tuned":
            # Expression is measured relative to the CORRECTED lead's intended
            # note centre, not the cancelled experiment's raw singing errors.
            expressive = np.where(np.isfinite(lead), effective_lead-reference, 0.)
            harmonies += expressive[None, :]
        notify("Preparing the harmony voices…")
        if harmony_engine == "psola":
            if sound is None:
                sound = parselmouth.Sound(x, sampling_frequency=sr)
                manipulation = call(sound, "To Manipulation", min(.005, step), floor, ceiling)
            audio_time = np.arange(length)/sr
            for index, target in enumerate(harmonies):
                notify(f"Rendering natural harmony {index+1} of {requested}...")
                active = np.isfinite(target)
                if not np.any(active & harmony_voiced):
                    stems[f"harmony_{index+1}"] = np.zeros(length, dtype=np.float32)
                    continue
                target_filled = _finite_fill(target, times, effective_lead)
                # Continuous source pulse marks preserve vocal timbre. Only
                # scored harmony changes receive a gentle transition.
                target_filled = gaussian_filter1d(target_filled, max(.1, .004/step))
                y = fit_length(praat_voice(sound, manipulation, times, target_filled,
                                          harmony_voiced, ratio), length)
                gate = gaussian_filter1d(np.interp(audio_time, clock,
                    (active & harmony_voiced).astype(float)), .010*sr)
                stems[f"harmony_{index+1}"] = (y*gate).astype(np.float32)
        else:
            fs = 24000
            frame_period = 10.0
            xw = np.ascontiguousarray(_resample(x, sr, fs), dtype=np.float64)
            tw = np.arange(int(len(xw)/fs/(frame_period/1000))+1)*(frame_period/1000)
            analysis_f0 = np.where(harmony_voiced, hertz(observed_lead), 0.) if harmony_follow == "lead" else f0
            fw = np.interp(tw, times, analysis_f0)
            fw[np.interp(tw, times, harmony_voiced.astype(float)) < .8] = 0
            fw = np.ascontiguousarray(fw)
            # Adaptive low-F0 analysis. D4C must explicitly match CheapTrick's FFT.
            world_floor = min(90., max(40., floor))
            spectral = pw.cheaptrick(xw, fw, tw, fs, f0_floor=world_floor)
            aperiodic = pw.d4c(xw, fw, tw, fs, fft_size=(spectral.shape[1]-1)*2)
            ot = np.arange(round(target_duration/(frame_period/1000))+1)*(frame_period/1000)
            source_lookup = np.interp(ot, np.r_[0., clock, target_duration], np.r_[0., times, duration])
            spectral_out = _interpolate_frames(spectral, tw, source_lookup)
            aperiodic_out = _interpolate_frames(aperiodic, tw, source_lookup)
            del spectral, aperiodic
            voiced_out = np.interp(ot, clock, harmony_voiced.astype(float)) > .65
            audio_time = np.arange(length)/sr
            for index, target in enumerate(harmonies):
                notify(f"Rendering harmony {index+1} of {requested}…")
                active = np.isfinite(target)
                target_filled = _finite_fill(target, times, effective_lead)
                smooth_target = gaussian_filter1d(target_filled, max(.35, .006/step))
                note_out = np.interp(ot, clock, smooth_target)
                target_hz = np.ascontiguousarray(hertz(note_out))
                target_hz[~voiced_out] = 0
                bins = np.arange(spectral_out.shape[1])
                factor = (1.045 if mode == "duet" else (1.015, 1.04, 1.065)[index])
                locations = bins/factor
                left = np.floor(locations).astype(int)
                right = np.minimum(left+1, len(bins)-1)
                fraction = locations-left
                envelope = np.ascontiguousarray(spectral_out[:, left]*(1-fraction)+spectral_out[:, right]*fraction)
                y = pw.synthesize(target_hz, envelope, aperiodic_out, fs, frame_period=frame_period)
                y = fit_length(_resample(y, fs, sr), length)
                # WORLD f0=0 is unvoiced noise, not silence. Keep one lead's
                # consonants and breaths rather than multiplying them in the stack.
                gate_points = (harmony_voiced & active).astype(float)
                gate = gaussian_filter1d(np.interp(audio_time, clock, gate_points), .010*sr)
                stems[f"harmony_{index+1}"] = (y*gate).astype(np.float32)
                del envelope
    notify("Balancing the voices and finishing the WAV…")
    mix = np.zeros((length, 2))
    gains = {"solo": [1.0], "duet": [.83, .43], "vocoder": [.72, .34, .32, .30]}[mode]
    pans = {"solo": [0.], "duet": [-.08, .32], "vocoder": [0., -.65, .65, .15]}[mode]
    for index, stem in enumerate(stems.values()):
        mix[:, 0] += stem*gains[index]*np.sqrt((1-pans[index])/2)
        mix[:, 1] += stem*gains[index]*np.sqrt((1+pans[index])/2)
    tail = round(1.5*sr)
    mix = np.pad(mix, ((0, tail), (0, 0)))
    mastered, metrics = master(mix, sr, highpass_hz=max(35., min(65., floor)),
                               reverb_wet=.055 if mode == "solo" else .075)
    if any(not np.isfinite(stem).all() for stem in stems.values()):
        raise ValueError("Harmony synthesis produced nonfinite audio")
    report = {"mode": mode, "sample_rate": sr, "channels": 2, "recommended_wav_subtype": "PCM_24",
              "source_duration_seconds": duration, "corrected_vocal_duration_seconds": target_duration,
              "finished_duration_seconds": len(mastered)/sr, "pitch_strength": float(pitch_strength),
              "source_pitch_floor_hz": floor, "source_pitch_ceiling_hz": ceiling,
              "duration_ratio_range": [float(min(ratio)), float(max(ratio))],
              "clock_origin_shift_seconds": float(origin), "harmony_count": requested,
              "lead_mode": lead_mode,
              "pitch_style": pitch_style, "harmony_engine": harmony_engine,
              "lead_processing_bypassed": bool(source_bypass),
              "note_center_corrections": note_center_audit,
              "harmony_follow": harmony_follow,
              "harmony_follow_offset_range_semitones": [float(min(follow_offset)), float(max(follow_offset))],
              "observed_lead_midi_range": [float(min(observed_lead)), float(max(observed_lead))],
              "harmony_target_audit": "automation contains reference_harmony_midi and effective_harmony_midi",
              "lead_samples_preserved": lead_mode == "original" and np.array_equal(stems["lead"], x),
              "lead_timing_preserved": lead_mode == "original",
              "pitch_method": (f"original lead samples; {harmony_engine} harmony resynthesis" if lead_mode == "original"
                               else f"{pitch_style} note-centre TD-PSOLA lead; {harmony_engine} harmony resynthesis"),
              "warnings": warnings, "finite": True, **metrics}
    automation = {"source_time": times, "output_time": clock, "source_f0": f0,
                  "reference_lead_midi": lead, "effective_lead_midi": effective_lead,
                  "observed_lead_midi": observed_lead, "harmony_follow_offset": follow_offset,
                  "reference_harmony_midi": reference_harmonies,
                  "effective_harmony_midi": harmonies,
                  "harmony_midi": harmonies, "duration_ratio": ratio}
    return RenderResult(mastered, sr, stems, automation, report)
