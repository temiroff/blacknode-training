"""Managed OpenPI VLA training jobs."""
from __future__ import annotations

import atexit
import threading
from dataclasses import asdict
from typing import Any

from .vla_openpi import OpenPIProvider, VLATrainConfig


class VLAJob:
    def __init__(self, config: VLATrainConfig) -> None:
        self.config = config
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.thread: threading.Thread | None = None
        self.status: dict[str, Any] = {
            "kind": "blacknode.vla-training-job",
            "schema_version": 1,
            "run_id": config.run_id,
            "provider": "openpi",
            "architecture": "pi05",
            "backend": "jax",
            "method": "lora",
            "phase": "queued",
            "running": False,
            "progress": 0.0,
            "step": 0,
            "steps": config.steps,
            "metrics": {},
            "model": {},
            "model_path": "",
            "error": "",
            "config": asdict(config) | {"runner_path": ""},
        }

    def start(self) -> dict[str, Any]:
        with self.lock:
            if self.thread and self.thread.is_alive():
                return self.snapshot()
            self.status.update({"phase": "preparing", "running": True, "error": ""})
            self.thread = threading.Thread(
                target=self._run,
                name=f"blacknode-openpi-{self.config.run_id}",
                daemon=True,
            )
            self.thread.start()
            return self.snapshot()

    def _run(self) -> None:
        provider = OpenPIProvider()
        try:
            prepared = provider.prepare(self.config)
            with self.lock:
                self.status.update({"phase": "training", "running": True})
            result = provider.train(prepared, self._event, self.stop_event)
            model = provider.export(prepared, result)
            with self.lock:
                self.status.update({
                    "phase": "completed",
                    "running": False,
                    "progress": 1.0,
                    "step": result.final_step,
                    "metrics": {**dict(self.status.get("metrics") or {}), **result.metrics},
                    "model": model,
                    "model_path": str(model["path"]),
                })
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.status.update({
                    "phase": "stopped" if self.stop_event.is_set() else "failed",
                    "running": False,
                    "error": str(exc),
                })

    def _event(self, event: dict[str, Any]) -> None:
        with self.lock:
            if event.get("type") == "progress":
                progress = float(event.get("progress") or 0) / 100.0
                self.status["progress"] = min(0.99, max(0.0, progress))
                self.status["step"] = int(event.get("step") or self.status["step"])
            elif event.get("type") == "metric":
                name = str(event.get("name") or "")
                if name:
                    self.status.setdefault("metrics", {})[name] = float(event.get("value") or 0.0)
                self.status["step"] = int(event.get("step") or self.status["step"])

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        with self.lock:
            if self.status.get("running"):
                self.status["phase"] = "stopping"
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            value = dict(self.status)
            value["metrics"] = dict(value.get("metrics") or {})
            value["model"] = dict(value.get("model") or {})
            value["config"] = dict(value.get("config") or {})
            return value


_jobs: dict[str, VLAJob] = {}
_lock = threading.RLock()


def start_job(config: VLATrainConfig) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(config.run_id)
        if job is None or (not job.snapshot().get("running") and job.config != config):
            job = VLAJob(config)
            _jobs[config.run_id] = job
        return job.start()


def job_status(run_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(run_id)
        if job is None:
            return {
                "kind": "blacknode.vla-training-job",
                "schema_version": 1,
                "run_id": run_id,
                "phase": "idle",
                "running": False,
                "progress": 0.0,
                "step": 0,
                "steps": 0,
                "metrics": {},
                "model": {},
                "model_path": "",
                "error": "",
            }
        return job.snapshot()


def stop_job(run_id: str) -> dict[str, Any]:
    with _lock:
        job = _jobs.get(run_id)
        return job.stop() if job else job_status(run_id)


def stop_all() -> None:
    with _lock:
        for job in _jobs.values():
            job.stop()


atexit.register(stop_all)
