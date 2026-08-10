"""Provider-neutral VLA contracts and the guarded OpenPI process provider."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
OPENPI_BASE_MODEL = "gs://openpi-assets/checkpoints/pi05_base/params"
_STEP = re.compile(r"Step\s+(\d+)\s*:\s*(.*)")
_METRIC = re.compile(r"([a-zA-Z0-9_.-]+)=([-+0-9.eE]+)")


@dataclass(frozen=True)
class VLAProviderCapabilities:
    models: tuple[str, ...]
    backends: tuple[str, ...]
    methods: tuple[str, ...]
    supports_resume: bool


@dataclass(frozen=True)
class VLATrainConfig:
    run_id: str
    dataset: dict[str, Any]
    output_dir: str
    steps: int = 5000
    batch_size: int = 8
    action_horizon: int = 10
    action_mode: str = "absolute_joint"
    learning_rate: float = 5e-5
    save_interval: int = 1000
    seed: int = 42
    resume: bool = True
    overwrite: bool = False
    runner_path: str = ""


@dataclass(frozen=True)
class PreparedRun:
    config: VLATrainConfig
    output_dir: Path
    request_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_dir: Path
    norm_stats_path: Path
    final_step: int
    metrics: dict[str, float]
    inference: dict[str, Any]


class TrainingEventSink(Protocol):
    def __call__(self, event: dict[str, Any]) -> None: ...


class VLAProvider(Protocol):
    name: str
    capabilities: VLAProviderCapabilities

    def validate(self, config: VLATrainConfig) -> None: ...
    def prepare(self, config: VLATrainConfig) -> PreparedRun: ...
    def train(
        self,
        run: PreparedRun,
        events: TrainingEventSink,
        stop_event: threading.Event,
    ) -> TrainingResult: ...
    def export(self, run: PreparedRun, result: TrainingResult) -> dict[str, Any]: ...


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    if not slug:
        raise ValueError("run_id is required")
    return slug


class OpenPIProvider:
    name = "openpi"
    capabilities = VLAProviderCapabilities(
        models=("pi05",),
        backends=("jax",),
        methods=("lora",),
        supports_resume=True,
    )

    def validate(self, config: VLATrainConfig) -> None:
        _safe_slug(config.run_id)
        dataset = dict(config.dataset or {})
        if dataset.get("kind") != "blacknode.dataset-source":
            raise ValueError("dataset must be a blacknode.dataset-source")
        uri = str(dataset.get("uri") or dataset.get("metadata", {}).get("source_uri") or "").strip()
        if not uri:
            raise ValueError("dataset source URI is required")
        if uri.startswith("hf://") and not str(dataset.get("revision") or "").strip():
            raise ValueError("remote datasets require an immutable revision")
        if not 1 <= int(config.steps) <= 10_000_000:
            raise ValueError("steps must be between 1 and 10000000")
        if not 1 <= int(config.batch_size) <= 1024:
            raise ValueError("batch_size must be between 1 and 1024")
        if not 1 <= int(config.action_horizon) <= 256:
            raise ValueError("action_horizon must be between 1 and 256")
        if config.action_mode not in {"absolute_joint", "delta"}:
            raise ValueError("action_mode must be absolute_joint or delta")
        if float(config.learning_rate) <= 0:
            raise ValueError("learning_rate must be positive")

    def prepare(self, config: VLATrainConfig) -> PreparedRun:
        self.validate(config)
        output = Path(config.output_dir).expanduser().resolve()
        if output.exists() and config.overwrite:
            raise FileExistsError(
                "OpenPI output already exists; choose a fresh output directory or resume it"
            )
        output.mkdir(parents=True, exist_ok=True)
        request_path = output / "training_config.json"
        payload = {
            "kind": "blacknode.vla-train-request",
            "schema_version": 1,
            "provider": "openpi",
            "model": "pi05",
            "backend": "jax",
            "method": "lora",
            "openpi_commit": OPENPI_COMMIT,
            "base_model": OPENPI_BASE_MODEL,
            **asdict(config),
        }
        payload.pop("runner_path", None)
        _atomic_json(request_path, payload)
        return PreparedRun(
            config=config,
            output_dir=output,
            request_path=request_path,
            metrics_path=output / "metrics.jsonl",
        )

    def train(
        self,
        run: PreparedRun,
        events: TrainingEventSink,
        stop_event: threading.Event,
    ) -> TrainingResult:
        runner = str(
            run.config.runner_path
            or os.environ.get("BLACKNODE_OPENPI_RUNNER", "")
            or Path(__file__).resolve().parents[1] / "openpi_runner.py"
        )
        command = [sys.executable, runner, "--request", str(run.request_path)]
        environment = os.environ.copy()
        environment.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
        process = subprocess.Popen(
            command,
            cwd=run.output_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=environment,
        )
        lines: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            try:
                for line in iter(process.stdout.readline, ""):
                    lines.put(line)
            finally:
                process.stdout.close()
                lines.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        ended = False
        while not ended:
            if stop_event.is_set() and process.poll() is None:
                process.terminate()
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                ended = True
                continue
            text = line.rstrip("\r\n")
            print(text, flush=True)
            self._handle_line(run, text, events)
        return_code = process.wait(timeout=30)
        reader.join(timeout=2)
        if stop_event.is_set():
            raise RuntimeError("OpenPI training stopped by operator")
        if return_code:
            raise RuntimeError(f"OpenPI training exited with code {return_code}")
        result_path = run.output_dir / "openpi-result.json"
        if not result_path.is_file():
            raise RuntimeError("OpenPI runner did not produce openpi-result.json")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        checkpoint = Path(str(payload.get("checkpoint_dir") or "")).resolve()
        norm_stats = Path(str(payload.get("norm_stats_path") or "")).resolve()
        if run.output_dir != checkpoint and run.output_dir not in checkpoint.parents:
            raise ValueError("OpenPI checkpoint path escapes the training output")
        if not checkpoint.is_dir() or not norm_stats.is_file():
            raise FileNotFoundError("OpenPI checkpoint or normalization statistics are missing")
        metrics = {
            str(key): float(value) for key, value in dict(payload.get("metrics") or {}).items()
        }
        if run.metrics_path.is_file():
            for raw_line in run.metrics_path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "metric" and event.get("name"):
                    metrics[str(event["name"])] = float(event.get("value") or 0.0)
        return TrainingResult(
            checkpoint_dir=checkpoint,
            norm_stats_path=norm_stats,
            final_step=int(payload.get("final_step") or run.config.steps),
            metrics=metrics,
            inference=dict(payload.get("inference") or {}),
        )

    def _handle_line(self, run: PreparedRun, line: str, events: TrainingEventSink) -> None:
        if line.startswith("BLACKNODE_VLA_EVENT "):
            try:
                event = json.loads(line.removeprefix("BLACKNODE_VLA_EVENT "))
            except json.JSONDecodeError:
                return
            if isinstance(event, dict):
                self._record_event(run, event, events)
            return
        match = _STEP.search(line)
        if not match:
            return
        step = int(match.group(1))
        progress = min(99, round(100 * step / max(1, run.config.steps)))
        self._record_event(run, {"type": "progress", "progress": progress, "step": step}, events)
        for name, raw in _METRIC.findall(match.group(2)):
            self._record_event(
                run,
                {"type": "metric", "name": name, "value": float(raw), "step": step},
                events,
            )

    def _record_event(
        self,
        run: PreparedRun,
        event: dict[str, Any],
        events: TrainingEventSink,
    ) -> None:
        payload = {**event, "recorded_at": time.time()}
        with run.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        if event.get("type") in {"progress", "metric"}:
            cloud = dict(event)
            cloud.pop("recorded_at", None)
            print(f"BLACKNODE_CLOUD_EVENT {json.dumps(cloud, separators=(',', ':'))}", flush=True)
        events(event)

    def export(self, run: PreparedRun, result: TrainingResult) -> dict[str, Any]:
        model_dir = run.output_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_archive = model_dir / "adapter-checkpoint.tar.gz"
        temporary = checkpoint_archive.with_suffix(checkpoint_archive.suffix + ".tmp")

        def normalize_tar(info: tarfile.TarInfo) -> tarfile.TarInfo:
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            return info

        with temporary.open("wb") as compressed:
            with gzip.GzipFile(filename="", mode="wb", fileobj=compressed, mtime=0) as stream:
                with tarfile.open(fileobj=stream, mode="w") as archive:
                    archive.add(
                        result.checkpoint_dir,
                        arcname="checkpoint",
                        recursive=True,
                        filter=normalize_tar,
                    )
        temporary.replace(checkpoint_archive)
        norm_target = model_dir / "norm_stats.json"
        norm_target.write_bytes(result.norm_stats_path.read_bytes())
        config_target = model_dir / "training_config.json"
        config_target.write_bytes(run.request_path.read_bytes())
        metrics_target = model_dir / "metrics.jsonl"
        metrics_target.write_bytes(run.metrics_path.read_bytes() if run.metrics_path.is_file() else b"")
        dataset = dict(run.config.dataset or {})
        source_uri = str(dataset.get("uri") or dataset.get("metadata", {}).get("source_uri") or "")
        source_revision = str(dataset.get("revision") or dataset.get("metadata", {}).get("source_revision") or "")
        digest = _sha256_file(checkpoint_archive)
        model_id = "vla-" + hashlib.sha256(
            f"{source_uri}\0{source_revision}\0{run.config.run_id}\0{result.final_step}".encode("utf-8")
        ).hexdigest()[:24]
        manifest = {
            "kind": "blacknode.vla-model",
            "schema_version": 1,
            "model_id": model_id,
            "provider": "openpi",
            "architecture": "pi05",
            "backend": "jax",
            "base_model": OPENPI_BASE_MODEL,
            "base_model_revision": OPENPI_COMMIT,
            "dataset": {"uri": source_uri, "revision": source_revision},
            "training_method": "lora",
            "step": result.final_step,
            "seed": run.config.seed,
            "action_horizon": run.config.action_horizon,
            "action_mode": run.config.action_mode,
            "checkpoint": checkpoint_archive.name,
            "checkpoint_sha256": digest,
            "normalization": norm_target.name,
            "normalization_sha256": _sha256_file(norm_target),
            "training_config_sha256": _sha256_file(config_target),
            "metrics_sha256": _sha256_file(metrics_target),
            "metrics": result.metrics,
            "inference": result.inference,
            "job_id": os.environ.get("BLACKNODE_CLOUD_JOB_ID", ""),
            "owner": os.environ.get("BLACKNODE_CLOUD_ORGANIZATION_ID", ""),
            "physical_motion_authorized": False,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _atomic_json(model_dir / "manifest.json", manifest)
        published = model_dir
        if os.environ.get("BLACKNODE_CLOUD_JOB_ID"):
            published = (Path.cwd() / "model").resolve()
            published.mkdir(parents=True, exist_ok=True)
            for source in model_dir.iterdir():
                if source.is_file():
                    shutil.copy2(source, published / source.name)
        return {**manifest, "path": str(published)}
