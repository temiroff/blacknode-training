"""Outcome-producing VLA training nodes."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Dict, Enum, Float, Int, Text, node

from . import vla_runtime
from .vla_openpi import OpenPIProvider, VLATrainConfig

_CATEGORY = "Training"


def _run_id(value: Any) -> str:
    run_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    if not run_id:
        raise ValueError("run_id is required")
    return run_id


def _config(ctx: dict[str, Any]) -> VLATrainConfig:
    run_id = _run_id(ctx.get("run_id") or "pi05-lora")
    raw_output = str(ctx.get("output_dir") or "").strip()
    output = (
        Path(raw_output).expanduser().resolve()
        if raw_output
        else (
            Path("/job/cache/training/vla") / run_id
            if os.environ.get("BLACKNODE_CLOUD_JOB_ID")
            else Path.cwd() / "training" / "vla" / run_id
        ).resolve()
    )
    return VLATrainConfig(
        run_id=run_id,
        dataset=dict(ctx.get("dataset") or {}),
        output_dir=str(output),
        steps=max(1, int(ctx.get("steps") or 5000)),
        batch_size=max(1, int(ctx.get("batch_size") or 8)),
        action_horizon=max(1, int(ctx.get("action_horizon") or 10)),
        action_mode=str(ctx.get("action_mode") or "absolute_joint"),
        learning_rate=float(ctx.get("learning_rate") or 5e-5),
        save_interval=max(1, int(ctx.get("save_interval") or 1000)),
        seed=int(ctx.get("seed") or 42),
        resume=bool(ctx.get("resume", True)),
        overwrite=bool(ctx.get("overwrite", False)),
    )


def _outputs(status: dict[str, Any]) -> dict[str, Any]:
    phase = str(status.get("phase") or "idle")
    error = str(status.get("error") or "")
    return {
        "ok": phase not in {"failed"},
        "running": bool(status.get("running")),
        "phase": phase,
        "step": int(status.get("step") or 0),
        "progress": float(status.get("progress") or 0.0),
        "status": status,
        "metrics": dict(status.get("metrics") or {}),
        "model": dict(status.get("model") or {}),
        "model_path": str(status.get("model_path") or ""),
        "report": (
            f"OpenPI π0.5 LoRA {phase}: step {int(status.get('step') or 0)}/"
            f"{int(status.get('steps') or 0)}; physical motion disarmed"
            + (f"; {error}" if error else "")
        ),
    }


@node(
    name="OpenPIFineTune",
    component="vla-openpi",
    live=True,
    category=_CATEGORY,
    description=(
        "Fine-tune OpenPI π0.5 with JAX LoRA on a BlacknodeDataset source and export a "
        "verified, inference-only Blacknode VLA model. Physical motion remains disarmed."
    ),
    inputs={
        "trigger": AnyPort,
        "action": Enum(["run", "start", "status", "stop", "check"], default="run"),
        "dataset": Dict(default={}),
        "run_id": Text(default="pi05-lora"),
        "output_dir": Text(default=""),
        "steps": Int(default=5000),
        "batch_size": Int(default=8),
        "action_horizon": Int(default=10),
        "action_mode": Enum(["absolute_joint", "delta"], default="absolute_joint"),
        "learning_rate": Float(default=0.00005),
        "save_interval": Int(default=1000),
        "seed": Int(default=42),
        "resume": Bool(default=True),
        "overwrite": Bool(default=False),
    },
    outputs={
        "ok": Bool,
        "running": Bool,
        "phase": Text,
        "step": Int,
        "progress": Float,
        "status": Dict,
        "metrics": Dict,
        "model": Dict,
        "model_path": Text,
        "report": Text,
    },
    primary_inputs=["action", "dataset", "steps", "batch_size", "output_dir"],
    primary_outputs=["model", "progress", "metrics", "report"],
)
def openpi_fine_tune(ctx: dict[str, Any]) -> dict[str, Any]:
    action = str(ctx.get("action") or "run").lower()
    run_id = _run_id(ctx.get("run_id") or "pi05-lora")
    try:
        if action == "status":
            status = vla_runtime.job_status(run_id)
        elif action == "stop":
            status = vla_runtime.stop_job(run_id)
        else:
            config = _config(ctx)
            OpenPIProvider().validate(config)
            if action == "check":
                status = {
                    **vla_runtime.job_status(run_id),
                    "phase": "ready",
                    "steps": config.steps,
                    "config": {**config.__dict__, "runner_path": ""},
                }
            elif action in {"start", "run"}:
                status = vla_runtime.start_job(config)
                if action == "run":
                    while status.get("running"):
                        time.sleep(0.25)
                        status = vla_runtime.job_status(run_id)
                    if status.get("phase") != "completed":
                        raise RuntimeError(str(status.get("error") or "OpenPI training failed"))
            else:
                raise ValueError("action must be run, start, status, stop, or check")
        return _outputs(status)
    except Exception as exc:
        if action == "run":
            raise
        return _outputs({
            **vla_runtime.job_status(run_id),
            "phase": "failed",
            "running": False,
            "error": str(exc),
        })
