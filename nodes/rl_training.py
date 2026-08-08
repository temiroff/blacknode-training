"""Blacknode nodes for managed SO-ARM101 reinforcement learning."""
from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Dict, Enum, Float, Image, Int, Text, node

from . import ppo_runtime


_CATEGORY = "Training"


def _emit_cloud_event(payload: dict[str, Any]) -> None:
    print(f"BLACKNODE_CLOUD_EVENT {json.dumps(payload, separators=(',', ':'))}", flush=True)


def _wait_for_training(run_id: str) -> dict[str, Any]:
    last_update = -1
    while True:
        status = ppo_runtime.job_status(run_id)
        update = int(status.get("update") or 0)
        if update != last_update:
            progress = min(99, max(0, round(100 * float(status.get("progress") or 0.0))))
            _emit_cloud_event({"type": "progress", "progress": progress})
            for name in (
                "mean_reward",
                "mean_distance_m",
                "success_rate",
                "policy_loss",
                "value_loss",
                "frames_per_second",
            ):
                value = status.get(name)
                if isinstance(value, int | float):
                    _emit_cloud_event(
                        {"type": "metric", "name": name, "value": value, "step": update}
                    )
            last_update = update
        if not bool(status.get("running")):
            if status.get("phase") == "failed" or status.get("error"):
                raise RuntimeError(str(status.get("error") or "PPO training failed"))
            _emit_cloud_event({"type": "progress", "progress": 100})
            return status
        time.sleep(0.25)


def _run_id(value: Any) -> str:
    run_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    if not run_id:
        raise ValueError("run_id is required")
    return run_id


def _config(ctx: dict[str, Any]) -> ppo_runtime.PPOTrainingConfig:
    run_id = _run_id(ctx.get("run_id") or "so101-reach-ppo")
    environment = ppo_runtime._validate_environment(dict(ctx.get("environment") or {}))
    raw_output = str(ctx.get("output_dir") or "").strip()
    output = (
        Path(raw_output).expanduser().resolve()
        if raw_output else (Path.cwd() / "training" / "ppo" / run_id).resolve()
    )
    config = ppo_runtime.PPOTrainingConfig(
        run_id=run_id,
        environment=environment,
        output_dir=str(output),
        device=str(ctx.get("device") or "auto"),
        updates=max(1, int(ctx.get("updates") or 500)),
        rollout_steps=max(4, min(1024, int(ctx.get("rollout_steps") or 32))),
        learning_rate=float(ctx.get("learning_rate") or 3e-4),
        hidden_dim=max(32, int(ctx.get("hidden_dim") or 128)),
        epochs=max(1, min(32, int(ctx.get("epochs") or 4))),
        minibatch_size=max(32, int(ctx.get("minibatch_size") or 4096)),
        gamma=float(ctx.get("gamma") or 0.99),
        gae_lambda=float(ctx.get("gae_lambda") or 0.95),
        clip_ratio=float(ctx.get("clip_ratio") or 0.2),
        entropy_coefficient=max(0.0, float(ctx.get("entropy_coefficient") or 0.01)),
        value_coefficient=max(0.0, float(ctx.get("value_coefficient") or 0.5)),
        max_gradient_norm=max(0.01, float(ctx.get("max_gradient_norm") or 0.5)),
        checkpoint_every=max(1, int(ctx.get("checkpoint_every") or 25)),
        seed=int(ctx.get("seed") or 42),
        resume=bool(ctx.get("resume", True)),
        viewer_enabled=bool(ctx.get("viewer_enabled", True)),
        viewer_provider=str(ctx.get("viewer_provider") or "viser"),
        viewer_port=max(1024, min(65535, int(ctx.get("viewer_port") or 8091))),
        viewer_fps=max(1, min(30, int(ctx.get("viewer_fps") or 15))),
        viewer_environment_index=max(0, int(ctx.get("viewer_environment_index") or 0)),
    )
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 < config.gamma <= 1.0 or not 0.0 < config.gae_lambda <= 1.0:
        raise ValueError("gamma and gae_lambda must be in (0, 1]")
    if not 0.01 <= config.clip_ratio <= 1.0:
        raise ValueError("clip_ratio must be between 0.01 and 1.0")
    return config


@node(
    name="PPOTraining", component="reinforcement-learning", live=True, category=_CATEGORY,
    description=(
        "Train a PPO policy on replicated SO-ARM101 Newton/Warp simulations. "
        "The physical robot remains disarmed and is never connected by this node."
    ),
    inputs={
        "trigger": AnyPort,
        "action": Enum(["start", "run", "status", "check", "stop"], default="start"),
        "environment": Dict(default={}),
        "run_id": Text(default="so101-reach-ppo"),
        "output_dir": Text(default=""),
        "device": Enum(["auto", "cuda", "cpu"], default="auto"),
        "updates": Int(default=500),
        "rollout_steps": Int(default=32),
        "learning_rate": Float(default=0.0003),
        "hidden_dim": Int(default=128),
        "epochs": Int(default=4),
        "minibatch_size": Int(default=4096),
        "gamma": Float(default=0.99),
        "gae_lambda": Float(default=0.95),
        "clip_ratio": Float(default=0.2),
        "entropy_coefficient": Float(default=0.01),
        "value_coefficient": Float(default=0.5),
        "max_gradient_norm": Float(default=0.5),
        "checkpoint_every": Int(default=25),
        "seed": Int(default=42),
        "resume": Bool(default=True),
        "viewer_enabled": Bool(default=True),
        "viewer_provider": Enum(["viser"], default="viser"),
        "viewer_port": Int(default=8091),
        "viewer_fps": Int(default=15),
        "viewer_environment_index": Int(default=0),
        "overwrite": Bool(default=False),
    },
    outputs={
        "ok": Bool, "running": Bool, "phase": Text, "update": Int,
        "status": Dict, "dashboard": Image, "viewer": Dict, "viewer_url": Text,
        "checkpoint": Text, "report": Text,
    },
    primary_inputs=["trigger", "action", "environment", "output_dir", "overwrite"],
    primary_outputs=["dashboard", "viewer_url", "checkpoint", "report"],
)
def ppo_training(ctx: dict[str, Any]) -> dict[str, Any]:
    action = str(ctx.get("action") or "start").lower()
    try:
        run_id = _run_id(ctx.get("run_id") or "so101-reach-ppo")
        if action == "status":
            status = ppo_runtime.job_status(run_id)
        elif action == "stop":
            status = ppo_runtime.stop_job(run_id)
        else:
            config = _config(ctx)
            output = Path(config.output_dir)
            if action == "check":
                status = {
                    **ppo_runtime.job_status(run_id), "phase": "ready",
                    "updates": config.updates, "output_dir": config.output_dir,
                    "device": config.device,
                    "environment_count": int(config.environment["environment_count"]),
                }
            elif action in {"start", "run"}:
                current = ppo_runtime.job_status(run_id)
                if bool(current.get("running")):
                    status = current
                else:
                    checkpoints = sorted(output.glob("checkpoint-*.pt")) if output.exists() else []
                    if output.exists() and bool(ctx.get("overwrite", False)):
                        shutil.rmtree(output)
                        checkpoints = []
                        config = replace(config, resume=False)
                        status = ppo_runtime.start_job(config)
                    elif checkpoints:
                        latest = ppo_runtime.checkpoint_info(checkpoints[-1])
                        if int(latest["update"]) >= config.updates:
                            status = {
                                **current, "phase": "completed", "running": False,
                                "update": int(latest["update"]), "updates": config.updates,
                                "progress": 1.0, "checkpoint": str(checkpoints[-1]),
                                "output_dir": str(output), "error": "",
                            }
                        else:
                            status = ppo_runtime.start_job(replace(config, resume=True))
                    else:
                        if output.exists() and any(output.iterdir()):
                            run_path = output / "run.json"
                            try:
                                prior = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
                            except Exception:  # noqa: BLE001
                                prior = {}
                            if prior.get("kind") != "blacknode.ppo-training-run":
                                raise FileExistsError(
                                    f"output_dir contains unrelated data: {output}; enable overwrite to restart"
                                )
                        status = ppo_runtime.start_job(replace(config, resume=False))
                if action == "run":
                    status = _wait_for_training(run_id)
            else:
                raise ValueError("action must be status, check, start, run, or stop")
        return ppo_runtime.node_outputs(status)
    except Exception as exc:  # noqa: BLE001
        if action == "run":
            raise
        status = {
            **ppo_runtime.job_status(str(ctx.get("run_id") or "so101-reach-ppo")),
            "phase": "failed", "error": str(exc),
        }
        return ppo_runtime.node_outputs(status)


@node(
    name="PPOCheckpointInspect", component="reinforcement-learning", category=_CATEGORY,
    description="Inspect a Blacknode PPO checkpoint and its simulation/safety contract.",
    inputs={"trigger": AnyPort, "checkpoint_path": Text(default="")},
    outputs={"ok": Bool, "checkpoint": Dict, "update": Int, "report": Text},
    primary_inputs=["trigger", "checkpoint_path"], primary_outputs=["checkpoint", "report"],
)
def ppo_checkpoint_inspect(ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        info = ppo_runtime.checkpoint_info(str(ctx.get("checkpoint_path") or ""))
        return {"ok": True, "checkpoint": info, "update": int(info["update"]),
                "report": f"PPO checkpoint valid at update {info['update']}: {info['path']}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "checkpoint": {}, "update": 0,
                "report": f"PPO checkpoint inspection FAILED: {exc}"}


@node(
    name="PPOPolicyEvaluate", component="reinforcement-learning", category=_CATEGORY,
    description="Evaluate a PPO checkpoint deterministically on simulated SO-ARM101 arms; never commands hardware.",
    inputs={
        "trigger": AnyPort, "action": Enum(["evaluate", "check"], default="evaluate"),
        "checkpoint_path": Text(default=""), "device": Enum(["auto", "cuda", "cpu"], default="auto"),
        "environment_count": Int(default=64),
    },
    outputs={"ok": Bool, "evaluated": Bool, "metrics": Dict, "success_rate": Float, "report": Text},
    primary_inputs=["trigger", "action", "checkpoint_path"], primary_outputs=["metrics", "report"],
)
def ppo_policy_evaluate(ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        info = ppo_runtime.checkpoint_info(str(ctx.get("checkpoint_path") or ""))
        if str(ctx.get("action") or "evaluate").lower() == "check":
            return {"ok": True, "evaluated": False, "metrics": {}, "success_rate": 0.0,
                    "report": f"PPO evaluation ready at update {info['update']}; choose action=evaluate"}
        metrics = ppo_runtime.evaluate_checkpoint(
            info["path"], str(ctx.get("device") or "auto"),
            max(1, int(ctx.get("environment_count") or 64)),
        )
        evaluation_path = Path(info["path"]).parent / "evaluation.json"
        ppo_runtime._atomic_json(evaluation_path, metrics)
        metrics = {**metrics, "path": str(evaluation_path)}
        return {"ok": True, "evaluated": True, "metrics": metrics,
                "success_rate": float(metrics["success_rate"]),
                "report": (
                    f"simulation evaluation: {100.0 * float(metrics['success_rate']):.1f}% success, "
                    f"mean distance {float(metrics['mean_distance_m']):.4f} m; hardware disarmed"
                )}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "evaluated": False, "metrics": {}, "success_rate": 0.0,
                "report": f"PPO evaluation FAILED: {exc}"}


@node(
    name="PPOPolicyExport", component="reinforcement-learning", category=_CATEGORY,
    description="Export a PPO checkpoint as a simulation-only SO-ARM101 policy artifact.",
    inputs={
        "trigger": AnyPort, "action": Enum(["export", "check"], default="export"),
        "checkpoint_path": Text(default=""), "output_dir": Text(default=""),
        "overwrite": Bool(default=False),
    },
    outputs={"ok": Bool, "exported": Bool, "artifact": Dict, "artifact_path": Text, "report": Text},
    primary_inputs=["trigger", "action", "checkpoint_path", "output_dir", "overwrite"],
    primary_outputs=["exported", "artifact_path", "report"],
)
def ppo_policy_export(ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        info = ppo_runtime.checkpoint_info(str(ctx.get("checkpoint_path") or ""))
        raw_output = str(ctx.get("output_dir") or "").strip()
        output = (
            Path(raw_output).expanduser().resolve()
            if raw_output else Path(info["path"]).parent / f"policy-{int(info['update']):08d}"
        )
        if str(ctx.get("action") or "export").lower() == "check":
            return {"ok": True, "exported": False, "artifact": {}, "artifact_path": str(output),
                    "report": f"PPO export ready at update {info['update']}; choose action=export"}
        artifact = ppo_runtime.export_policy_artifact(
            info["path"], output, overwrite=bool(ctx.get("overwrite", False))
        )
        return {"ok": True, "exported": True, "artifact": artifact,
                "artifact_path": str(artifact["path"]),
                "report": f"simulation-only PPO policy exported: {artifact['path']}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "exported": False, "artifact": {}, "artifact_path": "",
                "report": f"PPO policy export FAILED: {exc}"}
