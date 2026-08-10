"""Managed PPO training and artifacts for Blacknode simulation environments."""
from __future__ import annotations

import atexit
import base64
import html
import json
import shutil
import textwrap
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import torch
except Exception:  # pragma: no cover - package diagnostics report optional dependency
    torch = None

from .ppo_model import PPOActorCritic, PPOModelConfig


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("torch is required for PPO training")
    return torch


@dataclass(frozen=True)
class PPOTrainingConfig:
    run_id: str
    environment: dict[str, Any]
    output_dir: str
    device: str = "auto"
    updates: int = 500
    rollout_steps: int = 32
    learning_rate: float = 3e-4
    hidden_dim: int = 128
    epochs: int = 4
    minibatch_size: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    checkpoint_every: int = 25
    seed: int = 42
    resume: bool = True
    viewer_enabled: bool = True
    viewer_provider: str = "viser"
    viewer_port: int = 8091
    viewer_fps: int = 15
    viewer_environment_index: int = 0


@dataclass(frozen=True)
class PPOReplayConfig:
    run_id: str
    checkpoint_path: str
    device: str = "auto"
    episodes: int = 3
    viewer_provider: str = "viser"
    viewer_port: int = 8091
    viewer_fps: int = 15


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    _require_torch()
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # pragma: no cover - older supported Torch
        return torch.load(path, map_location=device)


def _environment_class() -> Any:
    try:
        from blacknode.pkg.blacknode_newton.rl import SO101ReachEnvironment
    except Exception as exc:  # pragma: no cover - package loader supplies namespace
        raise RuntimeError(
            "blacknode-newton/runtime must be installed and enabled for PPO training"
        ) from exc
    return SO101ReachEnvironment


def _validate_environment(value: dict[str, Any]) -> dict[str, Any]:
    environment = dict(value or {})
    if environment.get("kind") != "blacknode.rl-environment":
        raise ValueError("connect the environment output from SO101ReachTask")
    provider = environment.get("provider") if isinstance(environment.get("provider"), dict) else {}
    if provider.get("environment_type") != "so101-reach-v1":
        raise ValueError("PPO currently supports the SO-ARM101 reach environment")
    safety = environment.get("safety") if isinstance(environment.get("safety"), dict) else {}
    if safety.get("simulation_only") is not True or safety.get("physical_motion_authorized") is not False:
        raise ValueError("PPO training requires an explicitly simulation-only, motion-disarmed environment")
    return environment


def ppo_observation_contract(environment: dict[str, Any]) -> dict[str, Any]:
    """Return the simulator-neutral observation/action contract for SO-101 reach."""
    spec = _validate_environment(environment)
    joint_names = [str(name) for name in spec.get("joint_names") or []]
    joint_count = len(joint_names)
    if joint_count <= 0:
        raise ValueError("PPO environment must declare ordered joint_names")
    observation = dict(spec.get("observation") or {})
    action = dict(spec.get("action") or {})
    expected_observation = joint_count * 3 + 3
    if int(observation.get("dimension") or 0) != expected_observation:
        raise ValueError(
            f"SO-ARM101 PPO observation dimension must be {expected_observation}"
        )
    if int(action.get("dimension") or 0) != joint_count:
        raise ValueError("SO-ARM101 PPO action dimension must match joint_names")
    return {
        "kind": "blacknode.ppo-compatibility-contract",
        "schema_version": 1,
        "environment_type": "so101-reach-v1",
        "robot_profile": str(spec.get("robot_profile") or "so_arm101"),
        "joint_names": joint_names,
        "observation": {
            "dimension": expected_observation,
            "fields": [
                {
                    "name": "normalized_joint_positions", "size": joint_count,
                    "source": "joint_positions_rad", "normalization": "joint_limits",
                },
                {
                    "name": "scaled_joint_velocities", "size": joint_count,
                    "source": "joint_velocities_rad_s", "scale": 0.05,
                },
                {
                    "name": "scaled_target_minus_end_effector", "size": 3,
                    "source": "target_minus_end_effector_m", "divisor": 0.5,
                },
                {
                    "name": "previous_normalized_action", "size": joint_count,
                    "source": "previous_action", "initial": 0.0,
                },
            ],
        },
        "action": {
            "dimension": joint_count,
            "type": "bounded_joint_position_delta",
            "minimum": float(action.get("minimum", -1.0)),
            "maximum": float(action.get("maximum", 1.0)),
            "scale_rad": float(action.get("scale_rad") or 0.0),
            "output_application": "current_position_plus_scaled_delta",
        },
        "safety": {"simulation_only": True, "physical_motion_authorized": False},
    }


def _policy_device(device_name: str) -> Any:
    _require_torch()
    requested = str(device_name or "auto").strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    return torch.device(requested)


class PPOPolicy:
    """Prediction-only PPO artifact shared by Newton and simulator adapters."""

    def __init__(self, artifact: str | Path | dict[str, Any], device_name: str = "auto") -> None:
        from .runtime import policy_artifact_info

        self.info = policy_artifact_info(artifact)
        if self.info.get("policy_type") != "ppo-so101-reach":
            raise ValueError("PPOPolicy requires a ppo-so101-reach policy artifact")
        self.contract = dict(
            self.info.get("compatibility_contract")
            or ppo_observation_contract(dict(self.info.get("environment") or {}))
        )
        self.joint_names = [str(name) for name in self.contract.get("joint_names") or []]
        self.device = _policy_device(device_name)
        self.model_format = str(
            self.info.get("model_format") or "blacknode-ppo-state-dict"
        )
        model_path = Path(str(self.info["model_path"]))
        if self.model_format == "torchscript":
            self.model = torch.jit.load(str(model_path), map_location=self.device)
        elif self.model_format == "blacknode-ppo-state-dict":
            payload = _torch_load(model_path, self.device)
            if payload.get("kind") != "blacknode.ppo-policy-model":
                raise ValueError("artifact model is not a Blacknode PPO policy model")
            config = PPOModelConfig.from_dict(dict(payload["model_config"]))
            self.model = PPOActorCritic(config).to(self.device)
            self.model.load_state_dict(payload["model_state"])
        else:
            raise ValueError(f"unsupported PPO model_format: {self.model_format}")
        self.model.eval()
        self.previous_action = torch.zeros(
            len(self.joint_names), device=self.device, dtype=torch.float32
        )

    def reset(self) -> None:
        self.previous_action.zero_()

    def actions_for_observation(self, observation: Any) -> Any:
        with torch.no_grad():
            if self.model_format == "torchscript":
                actions = self.model(observation)
                if isinstance(actions, (tuple, list)):
                    actions = actions[0]
            else:
                actions = self.model.deterministic(observation)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        return torch.clamp(actions.to(self.device, dtype=torch.float32), -1.0, 1.0)

    def predict(
        self,
        qpos: list[float],
        images: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del images
        values = dict(context or {})
        if len(qpos) != len(self.joint_names):
            raise ValueError(
                f"expected {len(self.joint_names)} joint positions, got {len(qpos)}"
            )
        raw_limits = values.get("joint_limits")
        limits = dict(raw_limits) if isinstance(raw_limits, dict) else {}
        missing_limits = [name for name in self.joint_names if name not in limits]
        if missing_limits:
            raise ValueError("PPO observation is missing joint limits: " + ", ".join(missing_limits))
        lower = torch.tensor(
            [float(limits[name][0]) for name in self.joint_names],
            device=self.device, dtype=torch.float32,
        )
        upper = torch.tensor(
            [float(limits[name][1]) for name in self.joint_names],
            device=self.device, dtype=torch.float32,
        )
        if not bool(torch.isfinite(lower).all() and torch.isfinite(upper).all()) or bool(
            (lower >= upper).any()
        ):
            raise ValueError("PPO observation contains invalid joint limits")
        q = torch.tensor(qpos, device=self.device, dtype=torch.float32)
        raw_velocities = values.get("joint_velocities")
        velocities = (
            [float(raw_velocities[name]) for name in self.joint_names]
            if isinstance(raw_velocities, dict)
            else [float(value) for value in list(raw_velocities or [])]
        )
        if len(velocities) != len(self.joint_names):
            raise ValueError("PPO observation must include ordered joint velocities")
        target = [float(value) for value in list(values.get("target_m") or [])]
        end_effector = [float(value) for value in list(values.get("end_effector_m") or [])]
        if len(target) != 3 or len(end_effector) != 3:
            raise ValueError("PPO observation requires target_m and end_effector_m xyz values")
        normalized_q = torch.clamp((q - (lower + upper) * 0.5) / ((upper - lower) * 0.5), -1.0, 1.0)
        qd = torch.tensor(velocities, device=self.device, dtype=torch.float32) * 0.05
        delta = (
            torch.tensor(target, device=self.device, dtype=torch.float32)
            - torch.tensor(end_effector, device=self.device, dtype=torch.float32)
        ) / 0.5
        observation = torch.cat((normalized_q, qd, delta, self.previous_action)).unsqueeze(0)
        normalized_action = self.actions_for_observation(observation)[0]
        scale_rad = float(dict(self.contract.get("action") or {}).get("scale_rad") or 0.0)
        desired = torch.clamp(q + normalized_action * scale_rad, lower, upper)
        self.previous_action.copy_(normalized_action)
        return {
            "kind": "blacknode.policy-prediction", "schema_version": 1,
            "joint_names": list(self.joint_names),
            "action": desired.detach().cpu().tolist(),
            "normalized_action": normalized_action.detach().cpu().tolist(),
            "action_mode": "absolute_joint_position", "units": "radians",
            "physical_motion_authorized": False,
        }


class PPOTrainingJob:
    def __init__(self, config: PPOTrainingConfig) -> None:
        self.config = config
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name=f"blacknode-ppo-{config.run_id}"
        )
        self.phase = "starting"
        self.update = 0
        self.simulation_steps = 0
        self.mean_reward: float | None = None
        self.mean_distance_m: float | None = None
        self.success_rate: float | None = None
        self.policy_loss: float | None = None
        self.value_loss: float | None = None
        self.entropy: float | None = None
        self.frames_per_second: float | None = None
        self.checkpoint = ""
        self.error = ""
        self.started_at = _now()
        self.started_ns = time.time_ns()
        self.ended_at = ""
        self.ended_ns = 0
        self.actual_device = ""
        self.preview: dict[str, Any] = {
            "running": False, "viewer_url": "", "environment_index": 0,
            "error": "",
        }
        self.environment: Any | None = None
        self.logs: deque[str] = deque(maxlen=40)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def close_viewer(self) -> None:
        environment = self.environment
        if environment is not None:
            close_preview = getattr(environment, "close_preview", None)
            if callable(close_preview):
                close_preview()
            else:
                environment.close()
            self.environment = None
        with self.lock:
            self.preview = {**self.preview, "running": False, "viewer_url": ""}

    def _log(self, message: str) -> None:
        with self.lock:
            self.logs.append(f"{time.strftime('%H:%M:%S')} {message}")

    def status(self) -> dict[str, Any]:
        with self.lock:
            elapsed = max(0.0, ((self.ended_ns or time.time_ns()) - self.started_ns) / 1e9)
            running = self.thread.is_alive()
            viewer_running = bool(self.preview.get("running"))
            return {
                "kind": "blacknode.ppo-training-job",
                "schema_version": 1,
                "run_id": self.config.run_id,
                "phase": "stopping" if running and self.stop_event.is_set() else self.phase,
                "running": running,
                "stop_requested": self.stop_event.is_set(),
                "update": self.update,
                "updates": self.config.updates,
                "progress": min(1.0, self.update / max(1, self.config.updates)),
                "simulation_steps": self.simulation_steps,
                "mean_reward": self.mean_reward,
                "mean_distance_m": self.mean_distance_m,
                "success_rate": self.success_rate,
                "policy_loss": self.policy_loss,
                "value_loss": self.value_loss,
                "entropy": self.entropy,
                "frames_per_second": self.frames_per_second,
                "checkpoint": self.checkpoint,
                "output_dir": self.config.output_dir,
                "device": self.actual_device or self.config.device,
                "environment_count": int(self.config.environment.get("environment_count") or 0),
                "viewer_url": str(self.preview.get("viewer_url") or ""),
                "viewer": dict(self.preview),
                "viewer_running": viewer_running,
                "service_running": running or viewer_running,
                "mode": "training",
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "elapsed_seconds": elapsed,
                "error": self.error,
                "logs": list(self.logs),
                "physical_motion_authorized": False,
            }

    def _checkpoint_payload(self, model: Any, optimizer: Any, model_config: PPOModelConfig) -> dict[str, Any]:
        return {
            "kind": "blacknode.ppo-checkpoint",
            "schema_version": 1,
            "created_at": _now(),
            "update": self.update,
            "simulation_steps": self.simulation_steps,
            "model_config": model_config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "environment": dict(self.config.environment),
            "training_config": asdict(self.config),
            "metrics": {
                "mean_reward": self.mean_reward,
                "mean_distance_m": self.mean_distance_m,
                "success_rate": self.success_rate,
                "policy_loss": self.policy_loss,
                "value_loss": self.value_loss,
                "entropy": self.entropy,
                "frames_per_second": self.frames_per_second,
            },
            "safety": {"simulation_only": True, "physical_motion_authorized": False},
        }

    def _save_checkpoint(self, output: Path, model: Any, optimizer: Any, model_config: PPOModelConfig) -> Path:
        checkpoint = output / f"checkpoint-{self.update:08d}.pt"
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(self._checkpoint_payload(model, optimizer, model_config), temporary)
        temporary.replace(checkpoint)
        _atomic_json(output / "latest.json", {
            "checkpoint": str(checkpoint), "update": self.update,
            "simulation_steps": self.simulation_steps, "metrics": self.status(), "updated_at": _now(),
        })
        with self.lock:
            self.checkpoint = str(checkpoint)
        return checkpoint

    def _run(self) -> None:
        environment = None
        try:
            _require_torch()
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)
            spec = _validate_environment(self.config.environment)
            environment = _environment_class()(spec, device=self.config.device)
            device = environment.torch_device
            self.actual_device = str(device)
            observation_dim = int(spec["observation"]["dimension"])
            action_dim = int(spec["action"]["dimension"])
            model_config = PPOModelConfig(observation_dim, action_dim, self.config.hidden_dim)
            model = PPOActorCritic(model_config).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            if self.config.viewer_enabled:
                try:
                    self.preview = environment.start_preview({
                        "provider": self.config.viewer_provider,
                        "port": self.config.viewer_port,
                        "environment_index": self.config.viewer_environment_index,
                        "label": f"SO-ARM101 PPO · {self.config.run_id}",
                    })
                    self._log(
                        f"training preview started at {self.preview.get('viewer_url') or 'unknown URL'}"
                    )
                except Exception as exc:  # noqa: BLE001
                    self.preview = {
                        "running": False, "viewer_url": "", "environment_index": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                        "simulation_only": True, "physical_motion_authorized": False,
                    }
                    self._log(f"training preview unavailable: {self.preview['error']}")
            output = Path(self.config.output_dir).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            checkpoints = sorted(output.glob("checkpoint-*.pt"))
            if self.config.resume and checkpoints:
                payload = _torch_load(checkpoints[-1], device)
                if payload.get("kind") != "blacknode.ppo-checkpoint":
                    raise ValueError(f"unsupported checkpoint: {checkpoints[-1]}")
                if dict(payload.get("environment") or {}).get("provider") != spec.get("provider"):
                    raise ValueError("checkpoint environment does not match this training run")
                model.load_state_dict(payload["model_state"])
                optimizer.load_state_dict(payload["optimizer_state"])
                self.update = int(payload.get("update") or 0)
                self.simulation_steps = int(payload.get("simulation_steps") or 0)
                self.checkpoint = str(checkpoints[-1])
                self._log(f"resumed update {self.update} from {checkpoints[-1].name}")
            _atomic_json(output / "run.json", {
                "kind": "blacknode.ppo-training-run", "schema_version": 1,
                "created_at": self.started_at, "config": asdict(self.config),
                "safety": {"simulation_only": True, "physical_motion_authorized": False},
            })
            observation = environment.observe()
            environment_count = int(environment.environment_count)
            transition_count = self.config.rollout_steps * environment_count
            minibatch_size = min(max(1, self.config.minibatch_size), transition_count)
            with self.lock:
                self.phase = "running"
            self._log(
                f"started {environment_count} simulated SO-ARM101 arms on {device}; hardware disarmed"
            )
            started = time.perf_counter()
            starting_steps = self.simulation_steps
            next_preview_at = 0.0
            while self.update < self.config.updates and not self.stop_event.is_set():
                observations: list[Any] = []
                latents: list[Any] = []
                log_probabilities: list[Any] = []
                values: list[Any] = []
                rewards: list[Any] = []
                dones: list[Any] = []
                distances: list[Any] = []
                successes: list[Any] = []
                for _ in range(self.config.rollout_steps):
                    if self.stop_event.is_set():
                        break
                    with torch.no_grad():
                        action, latent, log_probability, value = model.sample(observation)
                    next_observation, reward, done, info = environment.step(action)
                    now = time.perf_counter()
                    if self.config.viewer_enabled and now >= next_preview_at:
                        preview_index = int(environment.preview_environment_index)
                        environment.render_preview({
                            "update": self.update,
                            "updates": self.config.updates,
                            "reward": float(reward[preview_index].detach().cpu()),
                            "success": bool(info["success"][preview_index].item()),
                        })
                        self.preview = environment.preview_status()
                        next_preview_at = now + 1.0 / max(1, self.config.viewer_fps)
                    observations.append(observation)
                    latents.append(latent)
                    log_probabilities.append(log_probability)
                    values.append(value)
                    rewards.append(reward)
                    dones.append(done)
                    distances.append(info["distance_m"])
                    successes.append(info["success"])
                    observation = next_observation
                if self.stop_event.is_set() or len(observations) != self.config.rollout_steps:
                    break
                with torch.no_grad():
                    next_value = model.critic(observation).squeeze(-1)
                reward_tensor = torch.stack(rewards)
                done_tensor = torch.stack(dones).to(torch.float32)
                value_tensor = torch.stack(values)
                advantages = torch.zeros_like(reward_tensor)
                gae = torch.zeros(environment_count, device=device)
                for index in range(self.config.rollout_steps - 1, -1, -1):
                    following_value = next_value if index == self.config.rollout_steps - 1 else value_tensor[index + 1]
                    alive = 1.0 - done_tensor[index]
                    delta = reward_tensor[index] + self.config.gamma * following_value * alive - value_tensor[index]
                    gae = delta + self.config.gamma * self.config.gae_lambda * alive * gae
                    advantages[index] = gae
                returns = advantages + value_tensor
                flat_observations = torch.stack(observations).reshape(transition_count, observation_dim)
                flat_latents = torch.stack(latents).reshape(transition_count, action_dim)
                old_log_probabilities = torch.stack(log_probabilities).reshape(transition_count)
                flat_advantages = advantages.reshape(transition_count)
                flat_returns = returns.reshape(transition_count)
                flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
                policy_losses: list[float] = []
                value_losses: list[float] = []
                entropies: list[float] = []
                for _ in range(self.config.epochs):
                    order = torch.randperm(transition_count, device=device)
                    for offset in range(0, transition_count, minibatch_size):
                        indexes = order[offset:offset + minibatch_size]
                        log_probability, entropy, value = model.evaluate(
                            flat_observations[indexes], flat_latents[indexes]
                        )
                        ratio = (log_probability - old_log_probabilities[indexes]).exp()
                        surrogate = ratio * flat_advantages[indexes]
                        clipped = ratio.clamp(1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * flat_advantages[indexes]
                        policy_loss = -torch.minimum(surrogate, clipped).mean()
                        value_loss = torch.nn.functional.mse_loss(value, flat_returns[indexes])
                        entropy_mean = entropy.mean()
                        loss = (
                            policy_loss + self.config.value_coefficient * value_loss
                            - self.config.entropy_coefficient * entropy_mean
                        )
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_gradient_norm)
                        optimizer.step()
                        policy_losses.append(float(policy_loss.detach()))
                        value_losses.append(float(value_loss.detach()))
                        entropies.append(float(entropy_mean.detach()))
                self.update += 1
                self.simulation_steps += transition_count
                elapsed = max(1e-6, time.perf_counter() - started)
                with self.lock:
                    self.mean_reward = float(reward_tensor.mean())
                    self.mean_distance_m = float(torch.stack(distances).mean())
                    self.success_rate = float(torch.stack(successes).to(torch.float32).mean())
                    self.policy_loss = sum(policy_losses) / max(1, len(policy_losses))
                    self.value_loss = sum(value_losses) / max(1, len(value_losses))
                    self.entropy = sum(entropies) / max(1, len(entropies))
                    self.frames_per_second = (self.simulation_steps - starting_steps) / elapsed
                if self.update == 1 or self.update % self.config.checkpoint_every == 0 or self.update == self.config.updates:
                    self._save_checkpoint(output, model, optimizer, model_config)
                    self._log(
                        f"update {self.update}/{self.config.updates}: distance={self.mean_distance_m:.4f}m "
                        f"success={100.0 * self.success_rate:.1f}%"
                    )
            if self.checkpoint == "" or self.update % self.config.checkpoint_every:
                self._save_checkpoint(output, model, optimizer, model_config)
            with self.lock:
                self.phase = "stopped" if self.stop_event.is_set() else "completed"
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.phase = "failed"
                self.error = f"{type(exc).__name__}: {exc}"
            self._log(self.error)
        finally:
            if environment is not None:
                if self.phase == "completed" and self.config.viewer_enabled:
                    try:
                        environment.render_preview({
                            "update": self.update,
                            "updates": self.config.updates,
                            "reward": self.mean_reward or 0.0,
                            "success": bool(self.success_rate and self.success_rate > 0.0),
                        })
                    except Exception:  # noqa: BLE001
                        pass
                    last_preview = environment.preview_status()
                    if bool(last_preview.get("running")):
                        release_batch = getattr(environment, "release_training_batch", None)
                        if callable(release_batch):
                            last_preview = dict(release_batch())
                        self.environment = environment
                        self.preview = last_preview
                        self._log("training complete; final preview remains open")
                    else:
                        environment.close()
                        self.preview = {**last_preview, "running": False}
                else:
                    last_preview = environment.preview_status()
                    environment.close()
                    self.preview = {**last_preview, "running": False, "viewer_url": ""}
            with self.lock:
                self.ended_at = _now()
                self.ended_ns = time.time_ns()


class PPOReplayJob:
    """Managed deterministic checkpoint replay in a one-arm Newton environment."""

    def __init__(self, config: PPOReplayConfig) -> None:
        self.config = config
        self.stop_event = threading.Event()
        self.lock = threading.RLock()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name=f"blacknode-ppo-replay-{config.run_id}"
        )
        self.phase = "starting_replay"
        self.update = 0
        self.replay_episode = 0
        self.mean_reward: float | None = None
        self.mean_distance_m: float | None = None
        self.success_rate: float | None = None
        self.error = ""
        self.actual_device = ""
        self.started_at = _now()
        self.started_ns = time.time_ns()
        self.ended_at = ""
        self.ended_ns = 0
        self.environment: Any | None = None
        self.preview: dict[str, Any] = {
            "running": False, "viewer_url": "", "environment_index": 0, "error": "",
        }
        self.logs: deque[str] = deque(maxlen=40)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def close_viewer(self) -> None:
        environment = self.environment
        if environment is not None:
            close_preview = getattr(environment, "close_preview", None)
            if callable(close_preview):
                close_preview()
            else:
                environment.close()
            self.environment = None
        with self.lock:
            self.preview = {**self.preview, "running": False, "viewer_url": ""}

    def _log(self, message: str) -> None:
        with self.lock:
            self.logs.append(f"{time.strftime('%H:%M:%S')} {message}")

    def status(self) -> dict[str, Any]:
        with self.lock:
            running = self.thread.is_alive()
            viewer_running = bool(self.preview.get("running"))
            elapsed = max(0.0, ((self.ended_ns or time.time_ns()) - self.started_ns) / 1e9)
            return {
                "kind": "blacknode.ppo-replay-job", "schema_version": 1,
                "run_id": self.config.run_id, "mode": "replay",
                "phase": "stopping" if running and self.stop_event.is_set() else self.phase,
                "running": running, "service_running": running or viewer_running,
                "viewer_running": viewer_running, "stop_requested": self.stop_event.is_set(),
                "update": self.update, "updates": self.update, "progress": 1.0,
                "simulation_steps": 0, "mean_reward": self.mean_reward,
                "mean_distance_m": self.mean_distance_m, "success_rate": self.success_rate,
                "policy_loss": None, "value_loss": None, "entropy": None,
                "frames_per_second": None, "checkpoint": self.config.checkpoint_path,
                "output_dir": str(Path(self.config.checkpoint_path).parent),
                "device": self.actual_device or self.config.device,
                "environment_count": 1, "replay_episode": self.replay_episode,
                "replay_episodes": self.config.episodes,
                "viewer_url": str(self.preview.get("viewer_url") or ""),
                "viewer": dict(self.preview), "started_at": self.started_at,
                "ended_at": self.ended_at, "elapsed_seconds": elapsed,
                "error": self.error, "logs": list(self.logs),
                "physical_motion_authorized": False,
            }

    def _run(self) -> None:
        environment = None
        try:
            _require_torch()
            checkpoint = Path(self.config.checkpoint_path).expanduser().resolve()
            payload = _torch_load(checkpoint, torch.device("cpu"))
            if payload.get("kind") != "blacknode.ppo-checkpoint":
                raise ValueError("checkpoint is not a Blacknode PPO checkpoint")
            spec = _validate_environment(dict(payload["environment"]))
            spec["environment_count"] = 1
            spec["seed"] = int(spec.get("seed") or 42) + 200_000
            environment = _environment_class()(spec, device=self.config.device)
            self.actual_device = str(environment.torch_device)
            model_config = PPOModelConfig.from_dict(dict(payload["model_config"]))
            model = PPOActorCritic(model_config).to(environment.torch_device)
            model.load_state_dict(payload["model_state"])
            model.eval()
            self.update = int(payload.get("update") or 0)
            self.preview = environment.start_preview({
                "provider": self.config.viewer_provider,
                "port": self.config.viewer_port,
                "environment_index": 0,
                "label": f"SO-ARM101 PPO Replay · update {self.update}",
            })
            self._log(f"checkpoint replay opened at {self.preview.get('viewer_url') or 'unknown URL'}")
            with self.lock:
                self.phase = "replaying"
            rewards: list[float] = []
            distances: list[float] = []
            successes = 0
            completed = 0
            render_interval = 1.0 / max(1, self.config.viewer_fps)
            next_render_at = 0.0
            control_interval = 1.0 / max(1, int(spec.get("control_hz") or 30))
            with torch.no_grad():
                for episode in range(1, self.config.episodes + 1):
                    if self.stop_event.is_set():
                        break
                    self.replay_episode = episode
                    observation = environment.reset()
                    for _ in range(int(spec.get("episode_steps") or 128)):
                        if self.stop_event.is_set():
                            break
                        started = time.perf_counter()
                        observation, reward, done, info = environment.step(
                            model.deterministic(observation)
                        )
                        reward_value = float(reward[0].detach().cpu())
                        distance_value = float(info["distance_m"][0].detach().cpu())
                        success = bool(info["success"][0].item())
                        rewards.append(reward_value)
                        distances.append(distance_value)
                        now = time.perf_counter()
                        if now >= next_render_at:
                            environment.render_preview({
                                "update": self.update, "updates": self.update,
                                "reward": reward_value, "success": success,
                                "mode": "replay", "replay_episode": episode,
                                "replay_episodes": self.config.episodes,
                            })
                            self.preview = environment.preview_status()
                            next_render_at = now + render_interval
                        if bool(done[0].item()):
                            completed += 1
                            successes += int(success)
                            break
                        remaining = control_interval - (time.perf_counter() - started)
                        if remaining > 0:
                            time.sleep(remaining)
            with self.lock:
                self.mean_reward = sum(rewards) / max(1, len(rewards))
                self.mean_distance_m = sum(distances) / max(1, len(distances))
                self.success_rate = successes / max(1, completed)
                self.phase = "stopped" if self.stop_event.is_set() else "replay_completed"
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.phase = "failed"
                self.error = f"{type(exc).__name__}: {exc}"
            self._log(self.error)
        finally:
            if environment is not None:
                last_preview = environment.preview_status()
                if self.phase == "replay_completed" and bool(last_preview.get("running")):
                    release_batch = getattr(environment, "release_training_batch", None)
                    if callable(release_batch):
                        last_preview = dict(release_batch())
                    self.environment = environment
                    self.preview = last_preview
                    self._log("replay complete; final frame remains open")
                else:
                    environment.close()
                    self.preview = {**last_preview, "running": False, "viewer_url": ""}
            with self.lock:
                self.ended_at = _now()
                self.ended_ns = time.time_ns()


_jobs: dict[str, PPOTrainingJob | PPOReplayJob] = {}
_jobs_lock = threading.RLock()


def start_job(config: PPOTrainingConfig) -> dict[str, Any]:
    _validate_environment(config.environment)
    with _jobs_lock:
        current = _jobs.get(config.run_id)
        if current and current.thread.is_alive():
            return current.status()
        if current:
            current.close_viewer()
        job = PPOTrainingJob(config)
        _jobs[config.run_id] = job
        job._log(f"initializing simulated environments (requested device: {config.device})")
        job.start()
        return job.status()


def start_replay_job(config: PPOReplayConfig) -> dict[str, Any]:
    checkpoint = Path(config.checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
    with _jobs_lock:
        current = _jobs.get(config.run_id)
        if current and current.thread.is_alive():
            return current.status()
        if current:
            current.close_viewer()
        job = PPOReplayJob(config)
        _jobs[config.run_id] = job
        job._log(f"loading checkpoint replay from {checkpoint.name}")
        job.start()
        return job.status()


def stop_job(run_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(run_id)
    if job is None:
        raise ValueError(f"PPO run {run_id!r} was not found")
    job.stop()
    return job.status()


def close_job_viewer(run_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(run_id)
    if job is None:
        raise ValueError(f"PPO run {run_id!r} was not found")
    if job.thread.is_alive():
        raise ValueError("stop the active PPO job before closing its viewer")
    job.close_viewer()
    return job.status()


def job_status(run_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(run_id)
    if job is not None:
        return job.status()
    return {
        "kind": "blacknode.ppo-training-job", "schema_version": 1, "run_id": run_id,
        "phase": "not_started", "running": False, "update": 0, "updates": 0,
        "progress": 0.0, "simulation_steps": 0, "mean_reward": None,
        "mean_distance_m": None, "success_rate": None, "policy_loss": None,
        "value_loss": None, "entropy": None, "frames_per_second": None,
        "checkpoint": "", "output_dir": "", "device": "", "error": "", "logs": [],
        "viewer_url": "", "viewer": {"running": False, "viewer_url": "", "error": ""},
        "viewer_running": False, "service_running": False, "mode": "training",
        "physical_motion_authorized": False,
    }


def control_training_job(run_id: str, action: str) -> dict[str, Any]:
    normalized = str(action or "status").lower()
    if normalized == "status":
        status = job_status(run_id)
    elif normalized == "stop":
        status = stop_job(run_id)
    elif normalized == "close-viewer":
        status = close_job_viewer(run_id)
    else:
        raise ValueError("PPOTraining direct control supports status, stop, or close-viewer")
    return node_outputs(status)


def runtime_status() -> dict[str, Any]:
    with _jobs_lock:
        statuses = [
            status for job in _jobs.values()
            if (status := job.status()).get("service_running")
        ]
    return {"ok": True, "active": bool(statuses), "managed_runs": statuses,
            "report": f"{len(statuses)} active PPO training or replay service(s)"}


def stop_runtime_services() -> dict[str, Any]:
    with _jobs_lock:
        jobs = [
            job for job in _jobs.values()
            if job.thread.is_alive() or bool(job.status().get("viewer_running"))
        ]
    for job in jobs:
        if job.thread.is_alive():
            job.stop()
        else:
            job.close_viewer()
    return {"ok": True, "stopped": {"ppo_runs": len(jobs)},
            "report": f"requested stop for {len(jobs)} PPO training or replay service(s)"}


def checkpoint_info(checkpoint_path: str | Path) -> dict[str, Any]:
    _require_torch()
    path = Path(str(checkpoint_path or "").strip()).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"checkpoint does not exist: {path}")
    payload = _torch_load(path, torch.device("cpu"))
    if payload.get("kind") != "blacknode.ppo-checkpoint":
        raise ValueError(f"unsupported PPO checkpoint: {path}")
    return {
        "kind": payload["kind"], "schema_version": int(payload.get("schema_version") or 0),
        "path": str(path), "update": int(payload.get("update") or 0),
        "simulation_steps": int(payload.get("simulation_steps") or 0),
        "model_config": dict(payload["model_config"]),
        "environment": dict(payload["environment"]), "metrics": dict(payload.get("metrics") or {}),
        "safety": dict(payload.get("safety") or {}),
    }


def export_policy_artifact(checkpoint_path: str | Path, output_dir: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    _require_torch()
    checkpoint = Path(str(checkpoint_path or "").strip()).expanduser().resolve()
    payload = _torch_load(checkpoint, torch.device("cpu"))
    if payload.get("kind") != "blacknode.ppo-checkpoint":
        raise ValueError("checkpoint is not a Blacknode PPO checkpoint")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"policy artifact directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "policy.pt"
    temporary = model_path.with_suffix(".pt.tmp")
    torch.save({
        "kind": "blacknode.ppo-policy-model", "schema_version": 1,
        "model_config": dict(payload["model_config"]), "model_state": payload["model_state"],
    }, temporary)
    temporary.replace(model_path)
    environment = dict(payload["environment"])
    compatibility_contract = ppo_observation_contract(environment)
    manifest = {
        "kind": "blacknode.policy-artifact", "schema_version": 1,
        "policy_type": "ppo-so101-reach", "backend": "blacknode-native",
        "model_format": "blacknode-ppo-state-dict",
        "created_at": _now(), "path": str(output), "model_file": model_path.name,
        "source_checkpoint": str(checkpoint), "step": int(payload.get("simulation_steps") or 0),
        "update": int(payload.get("update") or 0), "task": "reach",
        "robot_profile": "so_arm101", "action_mode": "bounded_joint_position_delta",
        "units": "normalized", "joint_names": list(environment["joint_names"]),
        "camera_names": [], "state_dim": int(environment["observation"]["dimension"]),
        "action_dim": int(environment["action"]["dimension"]),
        "model_config": dict(payload["model_config"]), "environment": environment,
        "compatibility_contract": compatibility_contract,
        "compatible_simulators": ["newton", "isaac-sim"],
        "metrics": dict(payload.get("metrics") or {}),
        "safety": {"simulation_only": True, "physical_motion_authorized": False},
    }
    _atomic_json(output / "manifest.json", manifest)
    return {**manifest, "model_path": str(model_path)}


def import_torchscript_policy(
    model_path: str | Path,
    environment: dict[str, Any],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    source: str = "isaac-sim",
) -> dict[str, Any]:
    """Package a compatible Isaac-trained TorchScript actor for Blacknode simulators."""
    _require_torch()
    source_model = Path(str(model_path or "").strip()).expanduser().resolve()
    if not source_model.is_file():
        raise ValueError(f"TorchScript policy does not exist: {source_model}")
    spec = _validate_environment(environment)
    contract = ppo_observation_contract(spec)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"policy artifact directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model = torch.jit.load(str(source_model), map_location="cpu")
    model.eval()
    observation_dim = int(dict(contract["observation"])["dimension"])
    action_dim = int(dict(contract["action"])["dimension"])
    with torch.no_grad():
        result = model(torch.zeros((1, observation_dim), dtype=torch.float32))
        if isinstance(result, (tuple, list)):
            result = result[0]
    result = torch.as_tensor(result)
    if tuple(result.shape) != (1, action_dim):
        raise ValueError(
            f"TorchScript actor must map [1,{observation_dim}] to [1,{action_dim}], "
            f"got {tuple(result.shape)}"
        )
    destination = output / "policy.pt"
    temporary = destination.with_suffix(".pt.tmp")
    shutil.copy2(source_model, temporary)
    temporary.replace(destination)
    manifest = {
        "kind": "blacknode.policy-artifact", "schema_version": 1,
        "policy_type": "ppo-so101-reach", "backend": "blacknode-native",
        "model_format": "torchscript", "created_at": _now(),
        "path": str(output), "model_file": destination.name,
        "source": str(source or "isaac-sim"), "source_model": str(source_model),
        "task": "reach", "robot_profile": str(spec.get("robot_profile") or "so_arm101"),
        "action_mode": "bounded_joint_position_delta", "units": "normalized",
        "joint_names": list(spec["joint_names"]), "camera_names": [],
        "state_dim": observation_dim, "action_dim": action_dim,
        "environment": dict(spec), "compatibility_contract": contract,
        "compatible_simulators": ["newton", "isaac-sim"],
        "safety": {"simulation_only": True, "physical_motion_authorized": False},
    }
    _atomic_json(output / "manifest.json", manifest)
    return {**manifest, "model_path": str(destination)}


def evaluate_policy_artifact(
    artifact: str | Path | dict[str, Any],
    device_name: str = "auto",
    environment_count: int = 64,
) -> dict[str, Any]:
    """Evaluate a native or imported PPO artifact in Newton."""
    policy = PPOPolicy(artifact, device_name)
    spec = dict(policy.info["environment"])
    spec["environment_count"] = max(1, min(1024, int(environment_count)))
    spec["seed"] = int(spec.get("seed") or 42) + 100_000
    environment = _environment_class()(spec, device=str(policy.device))
    try:
        observation = environment.observe()
        completed = 0
        successful = 0
        distances: list[float] = []
        rewards: list[float] = []
        for _ in range(int(spec.get("episode_steps") or 128)):
            actions = policy.actions_for_observation(observation)
            observation, reward, done, info = environment.step(actions)
            completed += int(done.sum().item())
            successful += int((done & info["success"]).sum().item())
            distances.append(float(info["distance_m"].mean().item()))
            rewards.append(float(reward.mean().item()))
        return {
            "kind": "blacknode.ppo-evaluation", "schema_version": 1,
            "artifact": str(policy.info["path"]),
            "source": str(policy.info.get("source") or "blacknode"),
            "environment_count": environment.environment_count,
            "completed_episodes": completed, "successful_episodes": successful,
            "success_rate": successful / max(1, completed),
            "mean_distance_m": sum(distances) / max(1, len(distances)),
            "mean_reward": sum(rewards) / max(1, len(rewards)),
            "device": str(environment.torch_device), "simulation_only": True,
            "physical_motion_authorized": False,
        }
    finally:
        environment.close()


def evaluate_checkpoint(
    checkpoint_path: str | Path, device_name: str = "auto", environment_count: int = 64,
) -> dict[str, Any]:
    """Run a deterministic policy in simulation and return episode metrics."""
    _require_torch()
    checkpoint = Path(str(checkpoint_path or "").strip()).expanduser().resolve()
    payload = _torch_load(checkpoint, torch.device("cpu"))
    if payload.get("kind") != "blacknode.ppo-checkpoint":
        raise ValueError("checkpoint is not a Blacknode PPO checkpoint")
    spec = dict(payload["environment"])
    spec["environment_count"] = max(1, min(1024, int(environment_count)))
    spec["seed"] = int(spec.get("seed") or 42) + 100_000
    environment = _environment_class()(spec, device=device_name)
    try:
        model_config = PPOModelConfig.from_dict(dict(payload["model_config"]))
        model = PPOActorCritic(model_config).to(environment.torch_device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        observation = environment.observe()
        completed = 0
        successful = 0
        distances: list[float] = []
        rewards: list[float] = []
        with torch.no_grad():
            for _ in range(int(spec.get("episode_steps") or 128)):
                observation, reward, done, info = environment.step(model.deterministic(observation))
                completed += int(done.sum().item())
                successful += int((done & info["success"]).sum().item())
                distances.append(float(info["distance_m"].mean().item()))
                rewards.append(float(reward.mean().item()))
        return {
            "kind": "blacknode.ppo-evaluation", "schema_version": 1,
            "checkpoint": str(checkpoint), "environment_count": environment.environment_count,
            "completed_episodes": completed, "successful_episodes": successful,
            "success_rate": successful / max(1, completed),
            "mean_distance_m": sum(distances) / max(1, len(distances)),
            "mean_reward": sum(rewards) / max(1, len(rewards)),
            "device": str(environment.torch_device), "simulation_only": True,
            "physical_motion_authorized": False,
        }
    finally:
        environment.close()


def dashboard(status: dict[str, Any]) -> str:
    phase = str(status.get("phase") or "unknown").upper()
    update = int(status.get("update") or 0)
    updates = int(status.get("updates") or 0)
    progress = max(0.0, min(1.0, float(status.get("progress") or 0.0)))
    distance = status.get("mean_distance_m")
    success = status.get("success_rate")
    error_lines = textwrap.wrap(str(status.get("error") or ""), width=72, break_long_words=True) or [""]
    color = "#22c55e" if phase in {"COMPLETED", "REPLAY_COMPLETED"} else "#ef4444" if phase == "FAILED" else "#8b5cf6"
    height = 210 + max(0, len(error_lines) - 1) * 18
    fill = int(472 * progress)
    distance_text = "—" if distance is None else f"{float(distance):.4f} m"
    success_text = "—" if success is None else f"{100.0 * float(success):.1f}%"
    error_svg = "".join(
        f'<text x="24" y="{194 + index * 18}" fill="#fca5a5" font-family="sans-serif" font-size="12">{html.escape(line)}</text>'
        for index, line in enumerate(error_lines)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="520" height="{height}" viewBox="0 0 520 {height}">
<rect width="100%" height="100%" rx="18" fill="#111827"/>
<circle cx="30" cy="34" r="7" fill="{color}"/><text x="48" y="40" fill="#f9fafb" font-family="sans-serif" font-size="19" font-weight="700">SO-ARM101 PPO · {phase}</text>
<text x="24" y="76" fill="#9ca3af" font-family="sans-serif" font-size="13">UPDATE</text><text x="24" y="99" fill="#f9fafb" font-family="monospace" font-size="20">{update} / {updates}</text>
<text x="250" y="76" fill="#9ca3af" font-family="sans-serif" font-size="13">DISTANCE</text><text x="250" y="99" fill="#f9fafb" font-family="monospace" font-size="20">{distance_text}</text>
<text x="410" y="76" fill="#9ca3af" font-family="sans-serif" font-size="13">SUCCESS</text><text x="410" y="99" fill="#f9fafb" font-family="monospace" font-size="20">{success_text}</text>
<rect x="24" y="122" width="472" height="14" rx="7" fill="#374151"/><rect x="24" y="122" width="{fill}" height="14" rx="7" fill="{color}"/>
<text x="24" y="166" fill="#d1d5db" font-family="sans-serif" font-size="13">Newton/Warp simulation · physical SO-ARM101 remains disarmed</text>
{error_svg}</svg>'''
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def node_outputs(status: dict[str, Any]) -> dict[str, Any]:
    phase = str(status.get("phase") or "unknown")
    return {
        "ok": phase != "failed", "running": bool(status.get("running")), "phase": phase,
        "update": int(status.get("update") or 0), "status": status,
        "dashboard": dashboard(status), "viewer": dict(status.get("viewer") or {}),
        "viewer_url": str(status.get("viewer_url") or ""),
        "viewer_running": bool(status.get("viewer_running")),
        "mode": str(status.get("mode") or "training"),
        "replay_episode": int(status.get("replay_episode") or 0),
        "replay_episodes": int(status.get("replay_episodes") or 0),
        "checkpoint": str(status.get("checkpoint") or ""),
        "report": (
            f"SO-ARM101 PPO {phase}: "
            + (
                f"episode {int(status.get('replay_episode') or 0)}/"
                f"{int(status.get('replay_episodes') or 0)}, checkpoint update "
                f"{int(status.get('update') or 0)}"
                if str(status.get("mode") or "training") == "replay"
                else f"update {int(status.get('update') or 0)}/{int(status.get('updates') or 0)}"
            )
            + "; physical motion disarmed"
            + (f"; {status['error']}" if status.get("error") else "")
        ),
    }


def _shutdown() -> None:
    stop_runtime_services()


atexit.register(_shutdown)
