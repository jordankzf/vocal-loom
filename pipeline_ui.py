"""Local desktop front end for vocal_pipeline.py; no audio is uploaded."""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


WORKSPACE = Path(__file__).resolve().parent
ENGINE = WORKSPACE / "vocal_pipeline.py"
MODES = {"Tuned solo": "solo", "Tuned dry solo": "solo", "Duet harmony": "duet", "Vocal stack": "vocoder"}
PITCH_STYLES = {"Natural": "transparent", "Tight / electronic": "tight"}


def build_command(recording, midi, lyrics, mode, output, track="auto",
                  pitch_strength=1.0, breath_reduction=.5, lead_mode="tuned", harmony_follow="score",
                  processing="full", pitch_style="transparent", harmony_engine="auto"):
    """Build an argument list, never a shell command or interpolated script."""
    if harmony_follow == "lead":
        lead_mode = "original"
    if processing == "dry":
        mode, lead_mode, harmony_follow = "solo", "tuned", "score"
    return [sys.executable, "-u", str(ENGINE), "render",
            "--recording", str(recording), "--midi", str(midi),
            "--lyrics", str(lyrics), "--mode", mode, "--out", str(output),
            "--melody-track", str(track), "--pitch-strength", str(float(pitch_strength)),
            "--breath-reduction", str(float(breath_reduction)), "--lead-mode", lead_mode,
            "--harmony-follow", harmony_follow, "--processing", processing,
            "--pitch-style", pitch_style, "--harmony-engine", harmony_engine]


def wav_inventory(folder):
    result = {}
    if not folder.exists():
        return result
    for path in folder.rglob("*.wav"):
        try:
            stat = path.stat()
            result[path.resolve()] = stat.st_mtime_ns, stat.st_size
        except OSError:
            pass
    return result


def find_finished_wav(folder, before):
    """Prefer changed final mixes over stems, and never open an older result."""
    after = wav_inventory(folder)
    changed = [path for path, stat in after.items() if before.get(path) != stat and stat[1] > 44]
    if not changed:
        return None
    def rank(path):
        relative = path.relative_to(folder.resolve())
        stem = any(part.lower() in {"stems", "production", "intermediate", "cache"}
                   for part in relative.parts[:-1])
        final = any(word in path.stem.lower() for word in ("finished", "final", "master"))
        return not stem, final, len(relative.parts) == 1, after[path][0]
    return max(changed, key=rank)


class VocalPipelineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("VocalLoom")
        self.root.geometry("940x860")
        self.root.minsize(780, 780)
        self.root.configure(background="#f4f6f9")
        self.events = queue.Queue()
        self.cancel_requested = threading.Event()
        self.process = None
        self.worker = None
        self.finished_file = None
        self.close_requested = False
        self.running = False
        self.recording = tk.StringVar()
        self.midi = tk.StringVar()
        self.lyrics = tk.StringVar()
        self.output = tk.StringVar(value=str(WORKSPACE / "output" / "songs"))
        self.mode = tk.StringVar(value="Tuned solo")
        self.track = tk.StringVar(value="auto")
        self.pitch_strength = tk.DoubleVar(value=1.0)
        self.breath_reduction = tk.DoubleVar(value=.5)
        self.preserve_lead = tk.BooleanVar(value=False)
        self.pitch_style = tk.StringVar(value="Natural")
        self.status = tk.StringVar(value="Choose your recording, MIDI and lyrics to begin.")
        self.controls = []
        self._layout()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(80, self._poll_events)

    def _layout(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background="#f4f6f9")
        style.configure("TLabel", background="#f4f6f9", foreground="#233043", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 25, "bold"), foreground="#182335")
        style.configure("Muted.TLabel", foreground="#5f6e80", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 7))
        style.configure("Accent.TButton", font=("Segoe UI", 11, "bold"), padding=(18, 9))
        style.configure("TEntry", padding=7, font=("Segoe UI", 10))
        style.configure("TCombobox", padding=5, font=("Segoe UI", 10))
        outer = ttk.Frame(self.root, padding=(26, 20))
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="VocalLoom", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Your voice. A finished track.", style="Muted.TLabel").pack(anchor="w", pady=(2, 17))
        inputs = ttk.Frame(outer)
        inputs.pack(fill="x")
        inputs.columnconfigure(1, weight=1)
        self._file_row(inputs, 0, "Dry vocal recording", self.recording, "recording")
        self._file_row(inputs, 1, "Song MIDI", self.midi, "midi")
        self._file_row(inputs, 2, "Lyrics", self.lyrics, "lyrics")
        self._file_row(inputs, 3, "Save finished track to", self.output, "output")
        ttk.Label(outer, text="Use a fresh, isolated vocal take for each song. Files stay on this computer.",
                  style="Muted.TLabel").pack(anchor="w", pady=(4, 15))

        settings = ttk.Frame(outer)
        settings.pack(fill="x")
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)
        ttk.Label(settings, text="Sound").grid(row=0, column=0, sticky="w", padx=(0, 12))
        mode = ttk.Combobox(settings, textvariable=self.mode, values=list(MODES), state="readonly", width=20)
        mode.grid(row=0, column=1, sticky="ew", padx=(0, 25))
        mode.bind("<<ComboboxSelected>>", self._mode_changed)
        ttk.Label(settings, text="Melody track").grid(row=0, column=2, sticky="w", padx=(0, 12))
        track = ttk.Entry(settings, textvariable=self.track, width=14)
        track.grid(row=0, column=3, sticky="ew")
        self.controls.extend((mode, track))
        ttk.Label(settings, text="Auto selects the most likely vocal part. Enter a track number to choose it yourself.",
                  style="Muted.TLabel", wraplength=820).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 11))
        self._slider(settings, 2, "Pitch correction", self.pitch_strength,
                     "Less", "More")
        self._slider(settings, 3, "Breath reduction", self.breath_reduction,
                     "Keep", "Reduce")
        preserve = ttk.Checkbutton(settings, text="Keep my original lead pitch and phrasing",
                                    variable=self.preserve_lead)
        ttk.Label(settings, text="Tuning style").grid(row=4, column=0, sticky="w", pady=(8, 0))
        pitch_style = ttk.Combobox(settings, textvariable=self.pitch_style, values=list(PITCH_STYLES),
                                  state="readonly", width=22)
        pitch_style.grid(row=4, column=1, columnspan=3, sticky="ew", pady=(8, 0))
        self.controls.append(pitch_style)
        preserve.grid(row=5, column=0, columnspan=4, sticky="w", pady=(8, 0))
        self.preserve_checkbox = preserve
        self.controls.append(preserve)

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(17, 12))
        self.render_button = ttk.Button(actions, text="Make my track", style="Accent.TButton", command=self._render)
        self.render_button.pack(side="left")
        self.cancel_button = ttk.Button(actions, text="Stop", command=self._cancel, state="disabled")
        self.cancel_button.pack(side="left", padx=9)
        self.open_button = ttk.Button(actions, text="Open finished WAV", command=self._open_finished, state="disabled")
        self.open_button.pack(side="right")
        self.progress = ttk.Progressbar(outer, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 9))
        ttk.Label(outer, textvariable=self.status, wraplength=830).pack(anchor="w", pady=(0, 8))
        self.log = ScrolledText(outer, height=11, state="disabled", wrap="word", font=("Consolas", 9),
                                background="#ffffff", foreground="#314155", relief="solid", borderwidth=1,
                                padx=10, pady=8)
        self.log.pack(fill="both", expand=True)

    def _file_row(self, parent, row, label, variable, kind):
        ttk.Label(parent, text=label, width=21).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=5)
        button = ttk.Button(parent, text="Browse…", command=lambda: self._browse(kind))
        button.grid(row=row, column=2, padx=(10, 0), pady=5)
        self.controls.extend((entry, button))

    def _slider(self, parent, row, label, variable, low, high):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        slider_frame = ttk.Frame(parent)
        slider_frame.grid(row=row, column=1, columnspan=3, sticky="ew", pady=5)
        slider_frame.columnconfigure(1, weight=1)
        ttk.Label(slider_frame, text=low, style="Muted.TLabel", width=7).grid(row=0, column=0, sticky="w")
        slider = ttk.Scale(slider_frame, from_=0, to=1, variable=variable)
        slider.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(slider_frame, text=high, style="Muted.TLabel", width=7).grid(row=0, column=2, sticky="e")
        self.controls.append(slider)

    def _browse(self, kind):
        if kind == "output":
            path = filedialog.askdirectory(parent=self.root, title="Choose a folder for the finished track",
                                           initialdir=self.output.get() if Path(self.output.get()).exists() else WORKSPACE)
        else:
            types = {
                "recording": [("Audio recordings", "*.wav *.flac *.aif *.aiff *.mp3 *.m4a *.ogg"), ("All files", "*.*")],
                "midi": [("MIDI files", "*.mid *.midi"), ("All files", "*.*")],
                "lyrics": [("Text files", "*.txt *.md"), ("All files", "*.*")],
            }
            path = filedialog.askopenfilename(parent=self.root, title=f"Choose {kind}", initialdir=WORKSPACE,
                                              filetypes=types[kind])
        if path:
            getattr(self, kind).set(path)

    def _append(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _mode_changed(self, event=None):
        dry = self.mode.get() == "Tuned dry solo"
        if dry:
            self.preserve_lead.set(False)
        self.preserve_checkbox.configure(state="disabled" if dry or self.running else "normal")

    def _busy(self, busy):
        self.running = busy
        for control in self.controls:
            if isinstance(control, ttk.Combobox):
                control.configure(state="disabled" if busy else "readonly")
            else:
                control.configure(state="disabled" if busy else "normal")
        self.render_button.configure(state="disabled" if busy else "normal")
        self.cancel_button.configure(state="normal" if busy else "disabled")
        self.open_button.configure(state="normal" if not busy and self.finished_file else "disabled")
        if busy:
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.configure(mode="determinate", value=0)
            self._mode_changed()

    def _render(self):
        if self.running:
            return
        if not ENGINE.is_file():
            messagebox.showerror("Pipeline is unavailable", f"The processing program is missing:\n{ENGINE}", parent=self.root)
            return
        paths = {}
        for kind, label in (("recording", "vocal recording"), ("midi", "MIDI file"), ("lyrics", "lyrics file")):
            raw = getattr(self, kind).get().strip()
            if not raw or not Path(raw).expanduser().is_file():
                messagebox.showerror("Choose your files", f"Choose an existing {label}.", parent=self.root)
                return
            paths[kind] = Path(raw).expanduser().resolve()
        if paths["midi"].suffix.lower() not in {".mid", ".midi"}:
            messagebox.showerror("Choose a MIDI file", "The song MIDI should end in .mid or .midi.", parent=self.root)
            return
        track = self.track.get().strip().lower()
        if track != "auto" and (not track.isdigit() or int(track) < 0):
            messagebox.showerror("Choose the melody track", "Use auto or a track number starting at 0.", parent=self.root)
            return
        output_raw = self.output.get().strip()
        if not output_raw:
            messagebox.showerror("Choose an output folder", "Choose where to save the finished track.", parent=self.root)
            return
        output = Path(output_raw).expanduser().resolve()
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("Cannot use this folder", str(exc), parent=self.root)
            return
        command = build_command(paths["recording"], paths["midi"], paths["lyrics"], MODES[self.mode.get()], output,
                                track, self.pitch_strength.get(), self.breath_reduction.get(),
                                "original" if self.preserve_lead.get() else "tuned",
                                "score", processing="dry" if self.mode.get() == "Tuned dry solo" else "full",
                                pitch_style=PITCH_STYLES[self.pitch_style.get()], harmony_engine="auto")
        self.finished_file = None
        self.cancel_requested.clear()
        self.status.set("Starting your track…")
        self._append(f"\n{self.mode.get()} · {paths['recording'].name}\nSaving to {output}")
        self._busy(True)
        self.worker = threading.Thread(target=self._run_worker, args=(command, output), daemon=True)
        self.worker.start()

    def _run_worker(self, command, output):
        before = wav_inventory(output)
        try:
            kwargs = {"cwd": str(WORKSPACE), "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
                      "text": True, "encoding": "utf-8", "errors": "replace", "bufsize": 1}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            environment = os.environ.copy()
            environment["PYTHONIOENCODING"] = "utf-8"
            self.process = subprocess.Popen(command, env=environment, **kwargs)
            if self.cancel_requested.is_set():
                self._terminate_process()
            for line in self.process.stdout:
                self.events.put(("log", line))
            returncode = self.process.wait()
            if self.cancel_requested.is_set():
                self.events.put(("cancelled", None))
            elif returncode:
                self.events.put(("error", f"The render stopped (exit code {returncode}). Review the details above."))
            else:
                finished = find_finished_wav(output, before)
                if finished:
                    self.events.put(("done", finished))
                else:
                    self.events.put(("error", "Processing ended without a new finished WAV. Review the details above."))
        except Exception as exc:
            self.events.put(("error", f"Could not complete the track: {exc}"))
        finally:
            self.process = None

    def _terminate_process(self):
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "nt":
                process.terminate()
            else:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _cancel(self):
        if not self.running:
            return
        self.cancel_requested.set()
        self.cancel_button.configure(state="disabled")
        self.status.set("Stopping the render…")
        threading.Thread(target=self._terminate_process, daemon=True).start()

    def _poll_events(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "log":
                    # JSON progress is optional; plain engine output remains useful.
                    try:
                        parsed = json.loads(value)
                    except (ValueError, TypeError):
                        parsed = None
                    if isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
                        value = parsed["message"]
                        if parsed.get("stage"):
                            self.status.set(value)
                    self._append(value)
                elif kind == "done":
                    self.finished_file = value
                    self.status.set(f"Your track is ready: {value.name}")
                    self._append(f"Finished: {value}")
                    self._busy(False)
                    self.progress.configure(value=100)
                elif kind == "cancelled":
                    self.status.set("Render stopped. You can change the settings and try again.")
                    self._append("Stopped by you. The original recording is unchanged.")
                    self._busy(False)
                elif kind == "error":
                    self.status.set("The track could not be finished. See the details below.")
                    self._append(value)
                    self._busy(False)
                    if not self.close_requested:
                        messagebox.showerror("Track not finished", value, parent=self.root)
        except queue.Empty:
            pass
        if self.close_requested and not self.running:
            self.root.destroy()
            return
        self.root.after(80, self._poll_events)

    def _open_finished(self):
        if not self.finished_file or not self.finished_file.is_file():
            messagebox.showerror("File unavailable", "The finished WAV is no longer in its output folder.", parent=self.root)
            return
        try:
            if os.name == "nt":
                os.startfile(str(self.finished_file))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(self.finished_file)])
            else:
                subprocess.Popen(["xdg-open", str(self.finished_file)])
        except OSError as exc:
            messagebox.showerror("Could not open WAV", str(exc), parent=self.root)

    def _close(self):
        if self.running:
            if not messagebox.askyesno("Stop this render?", "The track is still rendering. Stop it and close VocalLoom?", parent=self.root):
                return
            self.close_requested = True
            self._cancel()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    VocalPipelineApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
