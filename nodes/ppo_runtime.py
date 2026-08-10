"""Managed PPO training and artifacts for Blacknode simulation environments."""
from __future__ import annotations

import atexit
import base64
import html
import json
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
        self.logs: deque[str] = deque(maxlen=40)

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
            running = self.thread.is_alive()
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
                last_preview = environment.preview_status()
                environment.close()
                self.preview = {**last_preview, "running": False}
            with self.lock:
                self.ended_at = _now()
                self.ended_ns = time.time_ns()


_jobs: dict[str, PPOTrainingJob] = {}
_jobs_lock = threading.RLock()


def start_job(config: PPOTrainingConfig) -> dict[str, Any]:
    _validate_environment(config.environment)
    with _jobs_lock:
        current = _jobs.get(config.run_id)
        if current and current.thread.is_alive():
            return current.status()
        job = PPOTrainingJob(config)
        _jobs[config.run_id] = job
        job._log(f"initializing simulated environments (requested device: {config.device})")
        job.start()
        return job.status()


def stop_job(run_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(run_id)
    if job is None:
        raise ValueError(f"PPO run {run_id!r} was not found")
    job.stop()
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
        "physical_motion_authorized": False,
    }


def control_training_job(run_id: str, action: str) -> dict[str, Any]:
    normalized = str(action or "status").lower()
    if normalized == "status":
        status = job_status(run_id)
    elif normalized == "stop":
        status = stop_job(run_id)
    else:
        raise ValueError("PPOTraining direct control supports status or stop")
    return node_outputs(status)


def runtime_status() -> dict[str, Any]:
    with _jobs_lock:
        statuses = [job.status() for job in _jobs.values() if job.thread.is_alive()]
    return {"ok": True, "active": bool(statuses), "managed_runs": statuses,
            "report": f"{len(statuses)} active PPO training job(s)"}


def stop_runtime_services() -> dict[str, Any]:
    with _jobs_lock:
        jobs = [job for job in _jobs.values() if job.thread.is_alive()]
    for job in jobs:
        job.stop()
    return {"ok": True, "stopped": {"ppo_runs": len(jobs)},
            "report": f"requested stop for {len(jobs)} PPO training job(s)"}


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
    manifest = {
        "kind": "blacknode.policy-artifact", "schema_version": 1,
        "policy_type": "ppo-so101-reach", "backend": "blacknode-native",
        "created_at": _now(), "path": str(output), "model_file": model_path.name,
        "source_checkpoint": str(checkpoint), "step": int(payload.get("simulation_steps") or 0),
        "update": int(payload.get("update") or 0), "task": "reach",
        "robot_profile": "so_arm101", "action_mode": "bounded_joint_position_delta",
        "units": "normalized", "joint_names": list(environment["joint_names"]),
        "camera_names": [], "state_dim": int(environment["observation"]["dimension"]),
        "action_dim": int(environment["action"]["dimension"]),
        "model_config": dict(payload["model_config"]), "environment": environment,
        "metrics": dict(payload.get("metrics") or {}),
        "safety": {"simulation_only": True, "physical_motion_authorized": False},
    }
    _atomic_json(output / "manifest.json", manifest)
    return {**manifest, "model_path": str(model_path)}


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
    color = "#22c55e" if phase == "COMPLETED" else "#ef4444" if phase == "FAILED" else "#8b5cf6"
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
        "checkpoint": str(status.get("checkpoint") or ""),
        "report": (
            f"SO-ARM101 PPO {phase}: update {int(status.get('update') or 0)}/"
            f"{int(status.get('updates') or 0)}; physical motion disarmed"
            + (f"; {status['error']}" if status.get("error") else "")
        ),
    }


def _shutdown() -> None:
    stop_runtime_services()


atexit.register(_shutdown)
