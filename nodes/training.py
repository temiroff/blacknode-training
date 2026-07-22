"""Blacknode nodes for native offline action-chunking policy training."""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Dict, Enum, Float, Image, Int, List, Text, node

from . import data, runtime

_CATEGORY = "Training"


def _run_id(value: Any) -> str:
    run_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    if not run_id:
        raise ValueError("run_id is required")
    return run_id


def _config(ctx: dict[str, Any]) -> runtime.TrainingConfig:
    run_id = _run_id(ctx.get("run_id") or "act-training")
    dataset_path = data.resolve_dataset_path(str(ctx.get("dataset_path") or ""))
    raw_output = str(ctx.get("output_dir") or "").strip()
    output = Path(raw_output).expanduser().resolve() if raw_output else dataset_path.parent / "training" / run_id
    config = runtime.TrainingConfig(
        run_id=run_id,
        dataset_path=str(dataset_path),
        output_dir=str(output),
        device=str(ctx.get("device") or "auto"),
        steps=max(1, int(ctx.get("steps") or 5000)),
        batch_size=max(1, int(ctx.get("batch_size") or 8)),
        learning_rate=float(ctx.get("learning_rate") or 1e-4),
        weight_decay=max(0.0, float(ctx.get("weight_decay") or 1e-4)),
        chunk_size=max(1, int(ctx.get("chunk_size") or 32)),
        hidden_dim=max(32, int(ctx.get("hidden_dim") or 256)),
        attention_heads=max(1, int(ctx.get("attention_heads") or 8)),
        encoder_layers=max(1, int(ctx.get("encoder_layers") or 4)),
        decoder_layers=max(1, int(ctx.get("decoder_layers") or 2)),
        validation_fraction=min(0.5, max(0.0, float(ctx.get("validation_fraction") or 0.1))),
        eval_every=max(1, int(ctx.get("eval_every") or 250)),
        checkpoint_every=max(1, int(ctx.get("checkpoint_every") or 1000)),
        seed=int(ctx.get("seed") or 42),
        resume=bool(ctx.get("resume", True)),
    )
    if config.hidden_dim % config.attention_heads:
        raise ValueError("hidden_dim must be divisible by attention_heads")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    return config


@node(
    name="TrainingDatasetCheck", component="dataset-check", category=_CATEGORY,
    description="Validate Blacknode ACT-style HDF5 episodes without starting training.",
    inputs={"trigger": AnyPort, "dataset_path": Text(default="")},
    outputs={"ok": Bool, "dataset": Dict, "episode_count": Int, "frame_count": Int, "report": Text},
    primary_inputs=["trigger", "dataset_path"], primary_outputs=["dataset", "report"],
)
def training_dataset_check(ctx: dict) -> dict:
    try:
        summary = data.inspect_dataset(str(ctx.get("dataset_path") or ""))
        return {
            "ok": True, "dataset": summary,
            "episode_count": int(summary["episode_count"]), "frame_count": int(summary["total_frames"]),
            "report": (
                f"training dataset valid: {summary['episode_count']} episode(s), "
                f"{summary['total_frames']} frames, {len(summary['cameras'])} camera(s)"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "dataset": {}, "episode_count": 0, "frame_count": 0,
                "report": f"training dataset check FAILED: {exc}"}


@node(
    name="ACTTraining", component="training-jobs", live=True, category=_CATEGORY,
    description="Start or continue a managed Blacknode-native action-chunking training job. Active, completed, and checkpointed runs are reused automatically.",
    inputs={
        "trigger": AnyPort,
        "action": Enum(["start", "status", "check", "stop"], default="start"),
        "run_id": Text(default="act-training"), "dataset_path": Text(default=""), "output_dir": Text(default=""),
        "device": Enum(["auto", "cuda", "cpu"], default="auto"),
        "steps": Int(default=5000), "batch_size": Int(default=8),
        "learning_rate": Float(default=0.0001), "weight_decay": Float(default=0.0001),
        "chunk_size": Int(default=32), "hidden_dim": Int(default=256),
        "attention_heads": Int(default=8), "encoder_layers": Int(default=4), "decoder_layers": Int(default=2),
        "validation_fraction": Float(default=0.1), "eval_every": Int(default=250),
        "checkpoint_every": Int(default=1000), "seed": Int(default=42), "resume": Bool(default=True),
        "overwrite": Bool(default=False),
    },
    outputs={
        "ok": Bool, "running": Bool, "phase": Text, "step": Int,
        "status": Dict, "dashboard": Image, "checkpoint": Text, "report": Text,
    },
    primary_inputs=["trigger", "action", "dataset_path", "output_dir", "overwrite"],
    primary_outputs=["dashboard", "checkpoint", "report"],
)
def act_training(ctx: dict) -> dict:
    action = str(ctx.get("action") or "start").lower()
    try:
        run_id = _run_id(ctx.get("run_id") or "act-training")
        if action == "status":
            status = runtime.job_status(run_id)
        elif action == "stop":
            status = runtime.stop_job(run_id)
        else:
            config = _config(ctx)
            summary = data.inspect_dataset(config.dataset_path)
            output = Path(config.output_dir)
            if action == "check":
                status = {
                    **runtime.job_status(run_id),
                    "phase": "ready", "dataset": summary, "output_dir": config.output_dir,
                    "steps": config.steps, "device": config.device,
                }
            elif action == "start":
                current = runtime.job_status(run_id)
                if bool(current.get("running")):
                    status = current
                else:
                    checkpoints = sorted(output.glob("checkpoint-*.pt")) if output.exists() else []
                    if output.exists() and bool(ctx.get("overwrite", False)):
                        shutil.rmtree(output)
                        checkpoints = []
                        config = replace(config, resume=False)
                    elif checkpoints:
                        latest = runtime.checkpoint_info(checkpoints[-1])
                        latest_step = int(latest.get("step") or 0)
                        if latest_step >= config.steps:
                            status = {
                                **current,
                                "phase": "completed", "running": False,
                                "step": latest_step, "steps": config.steps, "progress": 1.0,
                                "checkpoint": str(checkpoints[-1]), "output_dir": str(output),
                                "error": "",
                            }
                        else:
                            config = replace(config, resume=True)
                            status = runtime.start_job(config)
                    else:
                        if output.exists() and any(output.iterdir()):
                            run_path = output / "run.json"
                            try:
                                prior_run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
                            except Exception:  # noqa: BLE001
                                prior_run = {}
                            if prior_run.get("kind") != "blacknode.training-run":
                                raise FileExistsError(
                                    f"output_dir contains unrelated data: {output}; enable overwrite to restart"
                                )
                        config = replace(config, resume=False)
                        status = runtime.start_job(config)
            else:
                raise ValueError("action must be status, check, start, or stop")
        phase = str(status.get("phase") or "unknown")
        ok = phase != "failed"
        report = (
            f"training {phase}: step {int(status.get('step') or 0)}/{int(status.get('steps') or 0)}"
            + (f"; {status['error']}" if status.get("error") else "")
        )
        return {
            "ok": ok, "running": bool(status.get("running")), "phase": phase,
            "step": int(status.get("step") or 0), "status": status,
            "dashboard": runtime.dashboard(status), "checkpoint": str(status.get("checkpoint") or ""),
            "report": report,
        }
    except Exception as exc:  # noqa: BLE001
        status = {**runtime.job_status(str(ctx.get("run_id") or "act-training")), "phase": "failed", "error": str(exc)}
        return {
            "ok": False, "running": False, "phase": "failed", "step": int(status.get("step") or 0),
            "status": status, "dashboard": runtime.dashboard(status), "checkpoint": "",
            "report": f"training action FAILED: {exc}",
        }


@node(
    name="ACTCheckpointInspect", component="checkpoints", category=_CATEGORY,
    description="Inspect a local Blacknode action-chunking checkpoint and its fixed dataset contract.",
    inputs={"trigger": AnyPort, "checkpoint_path": Text(default="")},
    outputs={"ok": Bool, "checkpoint": Dict, "step": Int, "report": Text},
    primary_inputs=["trigger", "checkpoint_path"], primary_outputs=["checkpoint", "report"],
)
def act_checkpoint_inspect(ctx: dict) -> dict:
    try:
        info = runtime.checkpoint_info(str(ctx.get("checkpoint_path") or ""))
        return {"ok": True, "checkpoint": info, "step": int(info["step"]),
                "report": f"checkpoint valid at step {info['step']}: {info['path']}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "checkpoint": {}, "step": 0, "report": f"checkpoint inspection FAILED: {exc}"}


@node(
    name="ACTPolicyPreview", component="policy-preview", category=_CATEGORY,
    description="Predict an action chunk for one recorded frame without connecting to or commanding robot hardware.",
    inputs={
        "trigger": AnyPort, "checkpoint_path": Text(default=""), "dataset_path": Text(default=""),
        "episode_index": Int(default=0), "frame_index": Int(default=0),
        "device": Enum(["auto", "cuda", "cpu"], default="auto"),
    },
    outputs={
        "ok": Bool, "prediction": Dict, "action": List, "target_action": List,
        "absolute_error": List, "action_chunk": List, "report": Text,
    },
    primary_inputs=["trigger", "checkpoint_path", "dataset_path"],
    primary_outputs=["prediction", "action", "report"],
)
def act_policy_preview(ctx: dict) -> dict:
    try:
        result = runtime.preview(
            str(ctx.get("checkpoint_path") or ""), str(ctx.get("dataset_path") or ""),
            int(ctx.get("episode_index") or 0), int(ctx.get("frame_index") or 0),
            str(ctx.get("device") or "auto"),
        )
        return {
            "ok": True, "prediction": result, "action": result["action"],
            "target_action": result["target_action"], "absolute_error": result["absolute_error"],
            "action_chunk": result["action_chunk"], "report": "prediction ready; robot motion was not commanded",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "prediction": {}, "action": [], "target_action": [], "absolute_error": [], "action_chunk": [],
                "report": f"policy preview FAILED: {exc}"}


@node(
    name="ACTPolicyExport", component="policy-artifacts", category=_CATEGORY,
    description="Export a trusted ACT checkpoint as an inference-only Blacknode policy artifact. Existing valid artifacts are reused unless overwrite is enabled.",
    inputs={
        "trigger": AnyPort,
        "action": Enum(["export", "check"], default="export"),
        "checkpoint_path": Text(default=""),
        "output_dir": Text(default=""),
        "overwrite": Bool(default=False),
    },
    outputs={"ok": Bool, "exported": Bool, "status": Text, "artifact": Dict, "artifact_path": Text, "report": Text},
    primary_inputs=["trigger", "action", "checkpoint_path", "output_dir", "overwrite"],
    primary_outputs=["exported", "status", "artifact_path", "report"],
)
def act_policy_export(ctx: dict) -> dict:
    try:
        checkpoint = str(ctx.get("checkpoint_path") or "")
        info = runtime.checkpoint_info(checkpoint)
        raw_output = str(ctx.get("output_dir") or "").strip()
        output = (
            Path(raw_output).expanduser().resolve()
            if raw_output
            else Path(info["path"]).parent / f"policy-{int(info['step']):08d}"
        )
        action = str(ctx.get("action") or "export").lower()
        if action == "check":
            return {
                "ok": True, "exported": False, "status": "checked_not_exported",
                "artifact": {}, "artifact_path": str(output),
                "report": f"ACT policy export ready at step {info['step']}; choose action=export",
            }
        if action != "export":
            raise ValueError(f"unsupported action: {action}; choose export or check")
        if output.exists() and not bool(ctx.get("overwrite", False)):
            try:
                artifact = runtime.policy_artifact_info(output)
            except Exception as exc:  # noqa: BLE001
                raise FileExistsError(
                    f"output directory exists but is not a valid policy artifact: {output}; enable overwrite to replace it"
                ) from exc
            return {
                "ok": True, "exported": True, "status": "exists", "artifact": artifact,
                "artifact_path": str(output),
                "report": f"EXISTS — valid policy artifact left unchanged. Enable overwrite to rebuild: {output}",
            }
        artifact = runtime.export_policy_artifact(
            checkpoint, output, overwrite=bool(ctx.get("overwrite", False)),
        )
        return {
            "ok": True, "exported": True, "status": "exported",
            "artifact": artifact, "artifact_path": str(artifact["path"]),
            "report": f"policy artifact exported: {artifact['path']}",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "exported": False, "status": "failed", "artifact": {}, "artifact_path": "",
                "report": f"policy export FAILED: {exc}"}


@node(
    name="PolicyArtifactLoad", component="policy-artifacts", category=_CATEGORY,
    description="Load and validate an exported Blacknode policy artifact manifest without starting inference.",
    inputs={"trigger": AnyPort, "artifact_path": Text(default="")},
    outputs={"ok": Bool, "artifact": Dict, "policy_type": Text, "report": Text},
    primary_inputs=["trigger", "artifact_path"], primary_outputs=["artifact", "report"],
)
def policy_artifact_load(ctx: dict) -> dict:
    try:
        artifact = runtime.policy_artifact_info(str(ctx.get("artifact_path") or ""))
        return {
            "ok": True, "artifact": artifact, "policy_type": str(artifact["policy_type"]),
            "report": (
                f"policy artifact ready: {artifact['policy_type']} · "
                f"{len(artifact['joint_names'])} joint(s) · {len(artifact['camera_names'])} camera(s)"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "artifact": {}, "policy_type": "", "report": f"policy artifact load FAILED: {exc}"}


@node(
    name="ACTPolicyReplay", component="policy-preview", category=_CATEGORY,
    description="Evaluate a loaded ACT policy across one recorded episode and emit a Dataset Browser-synchronized replay stream with prediction errors. Never commands hardware.",
    inputs={
        "trigger": AnyPort,
        "action": Enum(["evaluate", "check"], default="evaluate"),
        "artifact": Dict(default={}),
        "artifact_path": Text(default=""),
        "dataset_path": Text(default=""),
        "episode": Dict(default={}),
        "episode_index": Int(default=0),
        "device": Enum(["auto", "cuda", "cpu"], default="auto"),
        "sync_stream": Dict(default={}),
    },
    outputs={
        "ok": Bool, "evaluated": Bool, "replay": Dict, "stream": Dict,
        "metrics": Dict, "frame_count": Int, "report": Text,
    },
    primary_inputs=["trigger", "action", "artifact", "dataset_path"],
    primary_outputs=["stream", "metrics", "report"],
)
def act_policy_replay(ctx: dict) -> dict:
    try:
        artifact = dict(ctx.get("artifact") or {}) or str(ctx.get("artifact_path") or "")
        dataset_path = str(ctx.get("dataset_path") or "")
        selected_episode = dict(ctx.get("episode") or {})
        episode_index = int(selected_episode.get("episode_index", ctx.get("episode_index") or 0))
        checked = runtime.check_policy_replay(artifact, dataset_path, episode_index)
        if str(ctx.get("action") or "evaluate").lower() == "check":
            return {
                "ok": True, "evaluated": False, "replay": {}, "stream": {}, "metrics": {},
                "frame_count": int(checked["frames"]),
                "report": (
                    f"policy replay ready for episode {episode_index} · {checked['frames']} frame(s); "
                    "choose action=evaluate"
                ),
            }
        result = runtime.replay_policy(
            artifact, dataset_path, episode_index, str(ctx.get("device") or "auto"),
            dict(ctx.get("sync_stream") or {}),
        )
        metrics = dict(result["metrics"])
        return {
            "ok": True, "evaluated": True, "replay": result, "stream": result["stream"],
            "metrics": metrics, "frame_count": int(metrics["frames"]),
            "report": (
                f"policy replay ready: episode {episode_index} · {metrics['frames']} frame(s) · "
                f"MAE {metrics['mean_absolute_error']:.6f}; robot motion was not commanded"
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False, "evaluated": False, "replay": {}, "stream": {}, "metrics": {},
            "frame_count": 0, "report": f"policy replay FAILED: {exc}",
        }
