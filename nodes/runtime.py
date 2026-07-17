"""Managed background training jobs and checkpoint inference."""
from __future__ import annotations

import atexit
import base64
import html
import json
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from . import data
from .model import ActionChunkingConfig, ActionChunkingTransformer, masked_l1_loss

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class TrainingConfig:
    run_id: str
    dataset_path: str
    output_dir: str
    device: str = "auto"
    steps: int = 5000
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    chunk_size: int = 32
    hidden_dim: int = 256
    attention_heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 2
    validation_fraction: float = 0.1
    eval_every: int = 250
    checkpoint_every: int = 1000
    seed: int = 42
    resume: bool = False


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _device(requested: str) -> Any:
    if torch is None:
        raise RuntimeError("torch is required for training")
    requested = str(requested or "auto").lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cuda, or cpu")
    return torch.device(requested)


def _torch_load(path: Path, device: Any) -> dict[str, Any]:
    assert torch is not None
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch before weights_only support
        return torch.load(path, map_location=device)


class TrainingJob:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"blacknode-training-{config.run_id}")
        self.phase = "starting"
        self.step = 0
        self.train_loss: float | None = None
        self.validation_loss: float | None = None
        self.best_validation_loss: float | None = None
        self.checkpoint = ""
        self.error = ""
        self.started_at = _now()
        self.started_ns = time.time_ns()
        self.ended_at = ""
        self.ended_ns = 0
        self.actual_device = ""
        self.dataset_summary: dict[str, Any] = {}
        self.logs: deque[str] = deque(maxlen=30)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _log(self, message: str) -> None:
        with self.lock:
            self.logs.append(f"{time.strftime('%H:%M:%S')} {message}")

    def status(self) -> dict[str, Any]:
        with self.lock:
            elapsed = max(0.0, ((self.ended_ns or time.time_ns()) - self.started_ns) / 1e9)
            return {
                "kind": "blacknode.training-job",
                "schema_version": 1,
                "run_id": self.config.run_id,
                "phase": self.phase,
                "running": self.thread.is_alive(),
                "stop_requested": self.stop_event.is_set(),
                "step": self.step,
                "steps": self.config.steps,
                "progress": min(1.0, self.step / max(1, self.config.steps)),
                "train_loss": self.train_loss,
                "validation_loss": self.validation_loss,
                "best_validation_loss": self.best_validation_loss,
                "checkpoint": self.checkpoint,
                "output_dir": self.config.output_dir,
                "device": self.actual_device or self.config.device,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "elapsed_seconds": elapsed,
                "error": self.error,
                "dataset": dict(self.dataset_summary),
                "logs": list(self.logs),
            }

    def _save_checkpoint(
        self,
        output: Path,
        model: Any,
        optimizer: Any,
        model_config: ActionChunkingConfig,
        statistics: dict[str, list[float]],
        summary: dict[str, Any],
        train_indexes: list[int],
        validation_indexes: list[int],
    ) -> Path:
        assert torch is not None
        checkpoint = output / f"checkpoint-{self.step:08d}.pt"
        temp = checkpoint.with_suffix(".pt.tmp")
        payload = {
            "kind": "blacknode.action-chunking-checkpoint",
            "schema_version": 1,
            "created_at": _now(),
            "step": self.step,
            "model_config": model_config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "statistics": statistics,
            "dataset": {
                key: summary[key]
                for key in ("path", "episode_count", "total_frames", "state_dim", "action_dim", "joint_names", "cameras", "fps")
            },
            "split": {"train_episode_indexes": train_indexes, "validation_episode_indexes": validation_indexes},
            "training_config": asdict(self.config),
            "metrics": {
                "train_loss": self.train_loss,
                "validation_loss": self.validation_loss,
                "best_validation_loss": self.best_validation_loss,
            },
        }
        torch.save(payload, temp)
        temp.replace(checkpoint)
        _atomic_json(output / "latest.json", {
            "checkpoint": str(checkpoint),
            "step": self.step,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "updated_at": _now(),
        })
        with self.lock:
            self.checkpoint = str(checkpoint)
        return checkpoint

    def _evaluate(self, model: Any, loader: Any, device: Any, maximum_batches: int = 20) -> float | None:
        if loader is None:
            return None
        assert torch is not None
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                prediction = model(batch["qpos"].to(device), batch["images"].to(device))
                loss = masked_l1_loss(prediction, batch["actions"].to(device), batch["is_pad"].to(device))
                losses.append(float(loss.detach().cpu()))
                if batch_index + 1 >= maximum_batches:
                    break
        model.train()
        return sum(losses) / len(losses) if losses else None

    def _run(self) -> None:
        train_dataset = None
        validation_dataset = None
        try:
            data.require_dependencies(needs_torch=True)
            assert torch is not None
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
            summary = data.inspect_dataset(self.config.dataset_path)
            with self.lock:
                self.dataset_summary = summary
            files = data.episode_files(Path(summary["path"]))
            train_indexes, validation_indexes = data.split_episodes(
                len(files), self.config.validation_fraction, self.config.seed
            )
            statistics = data.compute_statistics(files, train_indexes)
            camera_names = list(summary["cameras"])
            train_dataset = data.HDF5ActionChunkDataset(
                files, train_indexes, camera_names=camera_names,
                chunk_size=self.config.chunk_size, statistics=statistics,
            )
            validation_dataset = (
                data.HDF5ActionChunkDataset(
                    files, validation_indexes, camera_names=camera_names,
                    chunk_size=self.config.chunk_size, statistics=statistics,
                ) if validation_indexes else None
            )
            generator = torch.Generator().manual_seed(self.config.seed)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.config.batch_size, shuffle=True,
                num_workers=0, pin_memory=torch.cuda.is_available(), drop_last=False, generator=generator,
            )
            validation_loader = (
                torch.utils.data.DataLoader(validation_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
                if validation_dataset is not None else None
            )
            model_config = ActionChunkingConfig(
                state_dim=int(summary["state_dim"]), action_dim=int(summary["action_dim"]),
                camera_count=len(camera_names), chunk_size=self.config.chunk_size,
                hidden_dim=self.config.hidden_dim, attention_heads=self.config.attention_heads,
                encoder_layers=self.config.encoder_layers, decoder_layers=self.config.decoder_layers,
            )
            device = _device(self.config.device)
            model = ActionChunkingTransformer(model_config).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
            output = Path(self.config.output_dir).expanduser().resolve()
            if output.exists() and not self.config.resume:
                raise FileExistsError(f"output_dir already exists: {output}; choose another path or enable resume")
            output.mkdir(parents=True, exist_ok=True)
            run_manifest = {
                "kind": "blacknode.training-run", "schema_version": 1,
                "created_at": self.started_at, "config": asdict(self.config),
                "dataset": summary, "statistics": statistics,
                "split": {"train_episode_indexes": train_indexes, "validation_episode_indexes": validation_indexes},
                "model_config": model_config.to_dict(),
            }
            if self.config.resume:
                checkpoints = sorted(output.glob("checkpoint-*.pt"))
                if not checkpoints:
                    raise ValueError(f"resume requested but no checkpoints exist in {output}")
                restored = _torch_load(checkpoints[-1], device)
                if restored.get("model_config") != model_config.to_dict():
                    raise ValueError("checkpoint model configuration does not match this training run")
                restored_dataset = dict(restored.get("dataset") or {})
                for key in ("episode_count", "total_frames", "state_dim", "action_dim", "joint_names", "cameras", "fps"):
                    if restored_dataset.get(key) != summary.get(key):
                        raise ValueError(f"checkpoint dataset {key} does not match the current dataset")
                if dict(restored.get("split") or {}) != {
                    "train_episode_indexes": train_indexes,
                    "validation_episode_indexes": validation_indexes,
                }:
                    raise ValueError("checkpoint episode split does not match the current dataset")
                restored_statistics = dict(restored.get("statistics") or {})
                for key, values in statistics.items():
                    if key not in restored_statistics or not np.allclose(restored_statistics[key], values, rtol=1e-6, atol=1e-7):
                        raise ValueError(f"checkpoint normalization statistic {key} does not match")
                model.load_state_dict(restored["model_state"])
                optimizer.load_state_dict(restored["optimizer_state"])
                self.step = int(restored["step"])
                self.checkpoint = str(checkpoints[-1])
                self._log(f"resumed checkpoint at step {self.step}")
            _atomic_json(output / "run.json", run_manifest)
            with self.lock:
                self.phase = "training"
                self.actual_device = str(device)
            self._log(f"training on {device} with {len(train_dataset)} samples")
            iterator = iter(train_loader)
            model.train()
            while self.step < self.config.steps and not self.stop_event.is_set():
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch["qpos"].to(device), batch["images"].to(device))
                loss = masked_l1_loss(prediction, batch["actions"].to(device), batch["is_pad"].to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite training loss at step {self.step + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                with self.lock:
                    self.step += 1
                    value = float(loss.detach().cpu())
                    self.train_loss = value if self.train_loss is None else self.train_loss * 0.95 + value * 0.05
                should_evaluate = self.step % max(1, self.config.eval_every) == 0 or self.step == self.config.steps
                if should_evaluate:
                    validation_loss = self._evaluate(model, validation_loader, device)
                    with self.lock:
                        self.validation_loss = validation_loss
                        if validation_loss is not None and (
                            self.best_validation_loss is None or validation_loss < self.best_validation_loss
                        ):
                            self.best_validation_loss = validation_loss
                    self._log(f"step {self.step}: train={self.train_loss:.5f}, val={validation_loss}")
                should_checkpoint = self.step % max(1, self.config.checkpoint_every) == 0 or self.step == self.config.steps
                if should_checkpoint:
                    self._save_checkpoint(
                        output, model, optimizer, model_config, statistics, summary,
                        train_indexes, validation_indexes,
                    )
            if self.step and (not self.checkpoint or not self.checkpoint.endswith(f"{self.step:08d}.pt")):
                self._save_checkpoint(
                    output, model, optimizer, model_config, statistics, summary,
                    train_indexes, validation_indexes,
                )
            with self.lock:
                self.phase = "stopped" if self.stop_event.is_set() else "completed"
            self._log(self.phase)
        except Exception as exc:  # noqa: BLE001 - recorded in structured job status
            with self.lock:
                self.phase = "failed"
                self.error = f"{type(exc).__name__}: {exc}"
            self._log(self.error)
        finally:
            if train_dataset is not None:
                train_dataset.close()
            if validation_dataset is not None:
                validation_dataset.close()
            with self.lock:
                self.ended_at = _now()
                self.ended_ns = time.time_ns()


_jobs: dict[str, TrainingJob] = {}
_jobs_lock = threading.RLock()


def start_job(config: TrainingConfig) -> dict[str, Any]:
    with _jobs_lock:
        current = _jobs.get(config.run_id)
        if current and current.thread.is_alive():
            raise RuntimeError(f"training run {config.run_id!r} is already active")
        job = TrainingJob(config)
        _jobs[config.run_id] = job
        job.start()
        return job.status()


def stop_job(run_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(run_id)
    if job is None:
        raise ValueError(f"training run {run_id!r} was not found")
    job.stop()
    return job.status()


def job_status(run_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(run_id)
    if job is None:
        return {
            "kind": "blacknode.training-job", "schema_version": 1, "run_id": run_id,
            "phase": "not_started", "running": False, "step": 0, "steps": 0,
            "progress": 0.0, "train_loss": None, "validation_loss": None,
            "best_validation_loss": None, "checkpoint": "", "output_dir": "",
            "device": "", "error": "", "logs": [],
        }
    return job.status()


def checkpoint_info(checkpoint_path: str | Path) -> dict[str, Any]:
    data.require_dependencies(needs_torch=True)
    assert torch is not None
    path = Path(str(checkpoint_path or "").strip()).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")
    payload = _torch_load(path, torch.device("cpu"))
    if payload.get("kind") != "blacknode.action-chunking-checkpoint":
        raise ValueError(f"unsupported checkpoint: {path}")
    return {
        "kind": payload["kind"], "schema_version": int(payload.get("schema_version") or 0),
        "path": str(path), "step": int(payload["step"]),
        "model_config": dict(payload["model_config"]),
        "dataset": dict(payload["dataset"]), "statistics": dict(payload["statistics"]),
        "metrics": dict(payload.get("metrics") or {}),
    }


def preview(checkpoint_path: str | Path, dataset_path: str | Path, episode_index: int, frame_index: int, device_name: str) -> dict[str, Any]:
    data.require_dependencies(needs_torch=True)
    assert torch is not None and np is not None and data.h5py is not None
    checkpoint = Path(str(checkpoint_path or "").strip()).expanduser().resolve()
    payload = _torch_load(checkpoint, _device(device_name))
    if payload.get("kind") != "blacknode.action-chunking-checkpoint":
        raise ValueError("checkpoint is not a Blacknode action chunking checkpoint")
    summary = data.inspect_dataset(dataset_path)
    trained = payload["dataset"]
    for key in ("state_dim", "action_dim", "joint_names", "cameras"):
        if summary[key] != trained[key]:
            raise ValueError(f"dataset {key} does not match checkpoint")
    files = data.episode_files(Path(summary["path"]))
    if episode_index < 0 or episode_index >= len(files):
        raise IndexError(f"episode_index must be between 0 and {len(files) - 1}")
    device = _device(device_name)
    model_config = ActionChunkingConfig.from_dict(payload["model_config"])
    model = ActionChunkingTransformer(model_config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    stats = payload["statistics"]
    with data.h5py.File(files[episode_index], "r") as handle:
        frame_count = int(handle["action"].shape[0])
        if frame_index < 0 or frame_index >= frame_count:
            raise IndexError(f"frame_index must be between 0 and {frame_count - 1}")
        qpos = np.asarray(handle["observations/qpos"][frame_index], dtype=np.float32)
        target_action = np.asarray(handle["action"][frame_index], dtype=np.float32)
        qpos = (qpos - np.asarray(stats["qpos_mean"], dtype=np.float32)) / np.asarray(stats["qpos_std"], dtype=np.float32)
        images = np.stack([
            np.asarray(handle[f"observations/images/{name}"][frame_index], dtype=np.float32).transpose(2, 0, 1) / 255.0
            for name in summary["cameras"]
        ])
    with torch.no_grad():
        normalized = model(
            torch.from_numpy(qpos).unsqueeze(0).to(device),
            torch.from_numpy(images).unsqueeze(0).to(device),
        )[0].cpu().numpy()
    actions = normalized * np.asarray(stats["action_std"], dtype=np.float32) + np.asarray(stats["action_mean"], dtype=np.float32)
    return {
        "kind": "blacknode.policy-preview", "schema_version": 1,
        "episode_index": episode_index, "frame_index": frame_index,
        "joint_names": list(summary["joint_names"]),
        "action": actions[0].astype(float).tolist(),
        "target_action": target_action.astype(float).tolist(),
        "absolute_error": np.abs(actions[0] - target_action).astype(float).tolist(),
        "action_chunk": actions.astype(float).tolist(),
        "device": str(device), "motion_commanded": False,
    }


def dashboard(status: dict[str, Any]) -> str:
    phase = str(status.get("phase") or "unknown").upper()
    step = int(status.get("step") or 0)
    steps = int(status.get("steps") or 0)
    progress = max(0.0, min(1.0, float(status.get("progress") or 0.0)))
    train_loss = status.get("train_loss")
    validation_loss = status.get("validation_loss")
    error = html.escape(str(status.get("error") or ""))
    color = "#22c55e" if phase == "COMPLETED" else "#ef4444" if phase == "FAILED" else "#f97316"
    width = 520
    fill = int(472 * progress)
    def metric(value: Any) -> str:
        return "—" if value is None else f"{float(value):.5f}"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="220" viewBox="0 0 {width} 220">
<rect width="100%" height="100%" rx="18" fill="#111827"/>
<circle cx="30" cy="34" r="7" fill="{color}"/><text x="48" y="40" fill="#f9fafb" font-family="sans-serif" font-size="19" font-weight="700">ACT TRAINING · {phase}</text>
<text x="24" y="76" fill="#9ca3af" font-family="sans-serif" font-size="13">STEP</text><text x="24" y="99" fill="#f9fafb" font-family="monospace" font-size="20">{step} / {steps}</text>
<text x="250" y="76" fill="#9ca3af" font-family="sans-serif" font-size="13">TRAIN LOSS</text><text x="250" y="99" fill="#f9fafb" font-family="monospace" font-size="20">{metric(train_loss)}</text>
<text x="390" y="76" fill="#9ca3af" font-family="sans-serif" font-size="13">VAL LOSS</text><text x="390" y="99" fill="#f9fafb" font-family="monospace" font-size="20">{metric(validation_loss)}</text>
<rect x="24" y="122" width="472" height="14" rx="7" fill="#374151"/><rect x="24" y="122" width="{fill}" height="14" rx="7" fill="{color}"/>
<text x="24" y="166" fill="#d1d5db" font-family="sans-serif" font-size="13">Prediction-only training package · never commands robot motion</text>
<text x="24" y="194" fill="#fca5a5" font-family="sans-serif" font-size="12">{error[:75]}</text>
</svg>'''
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def _shutdown() -> None:
    with _jobs_lock:
        jobs = list(_jobs.values())
    for job in jobs:
        job.stop()


atexit.register(_shutdown)
