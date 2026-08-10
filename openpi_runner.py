"""Isolated OpenPI/JAX π0.5 LoRA runner used by the dedicated Cloud image."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any


def _event(payload: dict[str, Any]) -> None:
    print("BLACKNODE_VLA_EVENT " + json.dumps(payload, separators=(",", ":")), flush=True)


def _latest_checkpoint(path: Path) -> tuple[Path, int]:
    candidates = [(item, int(item.name)) for item in path.iterdir() if item.is_dir() and item.name.isdigit()]
    if not candidates:
        raise FileNotFoundError(f"OpenPI produced no checkpoint in {path}")
    return max(candidates, key=lambda item: item[1])


def _materialize_local_dataset(source: Path, cache_root: Path, dataset_id: str) -> tuple[str, Path]:
    import blacknode  # noqa: F401 - loads configured package components
    import cv2
    import numpy as np
    from blacknode.pkg.blacknode_dataset import adapters
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    view = adapters.LeRobotDatasetAdapter(source).open()
    repo_id = f"blacknode/{dataset_id}"
    target = cache_root / repo_id
    if (target / "meta" / "info.json").is_file():
        return repo_id, target
    if target.exists():
        raise FileExistsError(f"incomplete OpenPI dataset cache exists: {target}")
    first_episode = view.open_episode(view.episode_ids()[0])
    first_sample = next(first_episode.samples())
    first_images: dict[str, Any] = {}
    for camera, reference in first_sample.cameras.items():
        capture = cv2.VideoCapture(reference.video_path)
        ok, image = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"could not decode first {camera} frame")
        first_images[camera] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    state_dim = len(first_sample.observation)
    action_dim = len(first_sample.action)
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": list(view.robot_spec.state_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": list(view.robot_spec.action_names),
        },
    }
    for camera, image in first_images.items():
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": tuple(image.shape),
            "names": ["height", "width", "channel"],
        }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(view.metadata.fps),
        root=target,
        robot_type=view.robot_spec.robot_type or "blacknode-robot",
        features=features,
        use_videos=True,
    )
    for episode_id in view.episode_ids():
        captures: dict[str, Any] = {}
        try:
            for sample in view.open_episode(episode_id).samples():
                frame: dict[str, Any] = {
                    "observation.state": np.asarray(sample.observation, dtype=np.float32),
                    "action": np.asarray(sample.action, dtype=np.float32),
                    "task": sample.task or view.metadata.task,
                }
                for camera, reference in sample.cameras.items():
                    capture = captures.get(camera)
                    if capture is None:
                        capture = cv2.VideoCapture(reference.video_path)
                        captures[camera] = capture
                    ok, image = capture.read()
                    if not ok:
                        raise RuntimeError(
                            f"camera {camera} ended before frame {sample.frame_index}"
                        )
                    frame[f"observation.images.{camera}"] = cv2.cvtColor(
                        image, cv2.COLOR_BGR2RGB
                    )
                dataset.add_frame(frame)
            dataset.save_episode()
        finally:
            for capture in captures.values():
                capture.release()
    return repo_id, target


def _resolve_dataset(request: dict[str, Any], cache_root: Path) -> tuple[str, Path]:
    dataset = dict(request.get("dataset") or {})
    uri = str(
        dataset.get("local_uri")
        or dataset.get("uri")
        or dataset.get("metadata", {}).get("source_uri")
        or ""
    )
    revision = str(dataset.get("revision") or dataset.get("metadata", {}).get("source_revision") or "")
    if uri.startswith("hf://"):
        from huggingface_hub import snapshot_download

        repo_id = uri.removeprefix("hf://")
        target = cache_root / repo_id
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=target,
        )
        info_path = target / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        if str(info.get("codebase_version") or "") == "v3.0":
            dataset_id = repo_id.split("/", 1)[-1]
            return _materialize_local_dataset(target, cache_root, dataset_id)
        return repo_id, target
    source = Path(uri).expanduser().resolve()
    dataset_id = str(dataset.get("metadata", {}).get("dataset_id") or source.name)
    return _materialize_local_dataset(source, cache_root, dataset_id)


def run(request_path: Path) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    output = request_path.parent.resolve()
    cache_root = output / "cache" / "lerobot"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_LEROBOT_HOME"] = str(cache_root)
    repo_id, dataset_root = _resolve_dataset(request, cache_root)

    import numpy as np
    import openpi.models.model as model_api
    import openpi.transforms as transforms
    from openpi.models import pi0_config
    from openpi.policies import policy_config
    from openpi.training import config as config_api
    from openpi.training import optimizer, weight_loaders
    from scripts import compute_norm_stats, train

    camera_names = tuple(
        str(name)
        for name in request.get("dataset", {}).get("metadata", {}).get("camera_names", [])
    )
    if not camera_names:
        info = json.loads((cache_root / repo_id / "meta" / "info.json").read_text(encoding="utf-8"))
        camera_names = tuple(
            key.removeprefix("observation.images.")
            for key, value in info.get("features", {}).items()
            if isinstance(value, dict) and value.get("dtype") in {"image", "video"}
        )
        action_dim = int(info["features"]["action"]["shape"][0])
    else:
        info = json.loads((cache_root / repo_id / "meta" / "info.json").read_text(encoding="utf-8"))
        action_dim = int(info["features"]["action"]["shape"][0])
    if not camera_names:
        raise ValueError("OpenPI π0.5 training requires at least one camera stream")

    @dataclasses.dataclass(frozen=True)
    class BlacknodeInputs(transforms.DataTransformFn):
        model_type: model_api.ModelType

        def __call__(self, data: dict) -> dict:
            images = dict(data["images"])
            available = list(camera_names)
            base = np.asarray(images[available[0]])
            left = np.asarray(images[available[1]]) if len(available) > 1 else np.zeros_like(base)
            right = np.asarray(images[available[2]]) if len(available) > 2 else np.zeros_like(base)
            value = {
                "state": data["state"],
                "image": {
                    "base_0_rgb": base,
                    "left_wrist_0_rgb": left,
                    "right_wrist_0_rgb": right,
                },
                "image_mask": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_ if len(available) > 1 else np.False_,
                    "right_wrist_0_rgb": np.True_ if len(available) > 2 else np.False_,
                },
                "prompt": data["prompt"],
            }
            if "actions" in data:
                value["actions"] = data["actions"]
            return value

    @dataclasses.dataclass(frozen=True)
    class BlacknodeOutputs(transforms.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            return {"actions": np.asarray(data["actions"][..., :action_dim])}

    @dataclasses.dataclass(frozen=True)
    class BlacknodeDataConfig(config_api.DataConfigFactory):
        def create(self, assets_dirs: Path, model_config: model_api.BaseModelConfig):
            repack = transforms.Group(inputs=[transforms.RepackTransform({
                "images": {
                    camera: f"observation.images.{camera}" for camera in camera_names
                },
                "state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            })])
            data_transforms = transforms.Group(
                inputs=[BlacknodeInputs(model_config.model_type)],
                outputs=[BlacknodeOutputs()],
            )
            if request.get("action_mode") == "absolute_joint":
                mask = transforms.make_bool_mask(max(0, action_dim - 1), -1)
                data_transforms = data_transforms.push(
                    inputs=[transforms.DeltaActions(mask)],
                    outputs=[transforms.AbsoluteActions(mask)],
                )
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack,
                data_transforms=data_transforms,
                model_transforms=config_api.ModelTransformFactory()(model_config),
                action_sequence_keys=("action",),
                prompt_from_task=True,
            )

    model = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=int(request["action_horizon"]),
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    checkpoint_root = output / "checkpoints" / "blacknode_pi05_lora" / str(request["run_id"])
    has_checkpoint = checkpoint_root.is_dir() and any(
        item.is_dir() and item.name.isdigit() for item in checkpoint_root.iterdir()
    )
    config = config_api.TrainConfig(
        name="blacknode_pi05_lora",
        exp_name=str(request["run_id"]),
        model=model,
        data=BlacknodeDataConfig(
            repo_id=repo_id,
            base_config=config_api.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(request["base_model"])),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=min(1000, max(1, int(request["steps"]) // 10)),
            peak_lr=float(request["learning_rate"]),
            decay_steps=int(request["steps"]),
            decay_lr=float(request["learning_rate"]) / 10.0,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        assets_base_dir=str(output / "assets"),
        checkpoint_base_dir=str(output / "checkpoints"),
        seed=int(request["seed"]),
        batch_size=int(request["batch_size"]),
        num_workers=2,
        num_train_steps=int(request["steps"]),
        log_interval=max(1, min(100, int(request["steps"]) // 20)),
        save_interval=min(int(request["save_interval"]), int(request["steps"])),
        keep_period=None,
        overwrite=bool(request.get("overwrite")) and not has_checkpoint,
        resume=bool(request.get("resume")) and has_checkpoint,
        wandb_enabled=False,
    )
    config_api._CONFIGS.append(config)
    started_at = time.monotonic()
    _event({"type": "progress", "progress": 1, "step": 0})
    _event({
        "type": "metric", "name": "learning_rate",
        "value": float(request["learning_rate"]), "step": 0,
    })
    compute_norm_stats.main(config.name)
    _event({"type": "progress", "progress": 3, "step": 0})
    train.main(config)
    checkpoint_dir, final_step = _latest_checkpoint(config.checkpoint_dir)
    norm_stats_path = config.assets_dirs / repo_id / "norm_stats.json"
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    representative_sample = LeRobotDataset(repo_id=repo_id, root=dataset_root)[0]
    observation: dict[str, Any] = {
        "observation.state": np.asarray(representative_sample["observation.state"]),
        "prompt": str(representative_sample.get("task") or ""),
    }
    for camera in camera_names:
        image = np.asarray(representative_sample[f"observation.images.{camera}"])
        if image.ndim == 3 and image.shape[0] in {1, 3, 4}:
            image = np.moveaxis(image, 0, -1)
        observation[f"observation.images.{camera}"] = image
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    prediction = policy.infer(observation)
    actions = np.asarray(prediction["actions"])
    expected_shape = (int(request["action_horizon"]), action_dim)
    if actions.shape != expected_shape or not np.isfinite(actions).all():
        raise RuntimeError(
            f"OpenPI inference produced {actions.shape}; expected finite {expected_shape}"
        )
    runtime_seconds = time.monotonic() - started_at
    _event({
        "type": "metric", "name": "runtime_seconds",
        "value": runtime_seconds, "step": final_step,
    })
    memory = next((device.memory_stats() for device in __import__("jax").devices()), None)
    if memory and memory.get("peak_bytes_in_use") is not None:
        _event({
            "type": "metric", "name": "gpu_memory_gb",
            "value": float(memory["peak_bytes_in_use"]) / (1024 ** 3), "step": final_step,
        })
    result = {
        "kind": "blacknode.openpi-result",
        "schema_version": 1,
        "checkpoint_dir": str(checkpoint_dir),
        "norm_stats_path": str(norm_stats_path),
        "final_step": final_step,
        "metrics": {},
        "inference": {
            "verified": True,
            "action_shape": list(actions.shape),
            "action_dtype": str(actions.dtype),
            "finite": True,
            "camera_names": list(camera_names),
            "state_dim": int(info["features"]["observation.state"]["shape"][0]),
            "action_dim": action_dim,
        },
    }
    (output / "openpi-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _event({"type": "progress", "progress": 100, "step": final_step})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    arguments = parser.parse_args()
    run(arguments.request.resolve())
