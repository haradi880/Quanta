"""Tkinter desktop shell backed exclusively by the Orchestrator contract."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from pathlib import Path
from uuid import uuid4

import tkinter as tk
from tkinter import messagebox, ttk

from core.orchestrator import process_job
from core.schemas import (
    AuthBlock,
    CallbackConfig,
    InterfaceType,
    JobEnvelope,
    JobMode,
    ModelSource,
    SystemPrompt,
)


def session_path() -> Path:
    root = Path(
        os.environ.get("HARADIBOTS_CACHE_ROOT", Path.home() / ".haradibots" / "cache")
    ).expanduser()
    return root / "session.jwt"


def load_session_jwt() -> str:
    token = os.environ.get("HARADIBOTS_SESSION_JWT")
    if token:
        return token
    try:
        return session_path().read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError("no local GUI session JWT is available") from exc


def build_envelope(model: str, mode: str, jwt_token: str) -> JobEnvelope:
    source = (
        ModelSource(local_path=model)
        if Path(model).exists()
        else ModelSource(repo_id=model)
    )
    return JobEnvelope(
        schema_version="3.0",
        job_id=uuid4(),
        auth=AuthBlock(jwt_token=jwt_token),
        interface=InterfaceType.GUI,
        mode=JobMode(mode),
        model_source=source,
        hardware_override=None,
        quantization_override=None,
        cluster_config=None,
        validation_prompts=None,
        system_prompt=SystemPrompt(preset_id="default"),
        telemetry_interval_ms=1000,
        callbacks=CallbackConfig(
            progress_channel="tkinter_queue",
            completion_channel="tkinter_queue",
        ),
    )


class HaradiBotsApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HaradiBots")
        self.events: queue.Queue[object] = queue.Queue()
        self.model = tk.StringVar()
        self.mode = tk.StringVar(value="auto")
        self.state = tk.StringVar(value="IDLE")
        self.telemetry = tk.StringVar(value="No telemetry")
        self._build()
        self.root.after(75, self._drain_events)

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=12)
        frame.grid(sticky="nsew")
        ttk.Label(frame, text="Model repository or local path").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(frame, textvariable=self.model, width=60).grid(
            row=1, column=0, columnspan=3, sticky="ew"
        )
        ttk.Radiobutton(frame, text="Auto", variable=self.mode, value="auto").grid(
            row=2, column=0, sticky="w"
        )
        ttk.Radiobutton(
            frame, text="Manual", variable=self.mode, value="manual"
        ).grid(row=2, column=1, sticky="w")
        self.submit = ttk.Button(frame, text="Run", command=self._submit)
        self.submit.grid(row=2, column=2, sticky="e")
        ttk.Label(frame, textvariable=self.state).grid(
            row=3, column=0, columnspan=3, sticky="w"
        )
        ttk.Label(frame, textvariable=self.telemetry).grid(
            row=4, column=0, columnspan=3, sticky="w"
        )
        self.results = tk.Text(frame, width=80, height=20, state="disabled")
        self.results.grid(row=5, column=0, columnspan=3, sticky="nsew")

    def _submit(self) -> None:
        if not self.model.get().strip():
            messagebox.showerror("HaradiBots", "Enter a model repository or path.")
            return
        try:
            envelope = build_envelope(
                self.model.get().strip(),
                self.mode.get(),
                load_session_jwt(),
            )
        except (RuntimeError, ValueError) as exc:
            messagebox.showerror("HaradiBots", str(exc))
            return
        self.submit.state(["disabled"])
        threading.Thread(
            target=lambda: asyncio.run(self._run(envelope)),
            daemon=True,
        ).start()

    async def _run(self, envelope: JobEnvelope) -> None:
        try:
            async for event in process_job(envelope):
                self.events.put(event)
        except Exception as exc:
            self.events.put(exc)
        finally:
            self.events.put(None)

    def _drain_events(self) -> None:
        while True:
            try:
                item = self.events.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self.submit.state(["!disabled"])
            elif isinstance(item, Exception):
                self.state.set(f"ERROR: {item}")
            else:
                state = item.payload.get("state", item.event_type.value)
                self.state.set(str(state))
                self.telemetry.set(json.dumps(item.telemetry, sort_keys=True))
                self.results.configure(state="normal")
                self.results.insert("end", item.model_dump_json() + "\n")
                self.results.see("end")
                self.results.configure(state="disabled")
        self.root.after(75, self._drain_events)


def main() -> None:
    root = tk.Tk()
    HaradiBotsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
