"""blacknode-training node, model, and managed-job contracts."""
from __future__ import annotations

import base64
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import blacknode  # noqa: F401 - triggers package discovery
from blacknode.node import _NODE_REGISTRY
from blacknode.packages import _PACKAGE_REGISTRY, load_package, packages_root

_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_DATASET_DIR = packages_root() / "blacknode-dataset"
_NEWTON_DIR = packages_root() / "blacknode-newton"
with patch(
    "blacknode.packages._read_component_overrides",
    return_value=({"viewer-ovrtx": False, "rosbridge": False, "replay": False}, ""),
):
    load_package(_NEWTON_DIR)
with patch(
    "blacknode.packages._read_component_overrides",
    return_value=({
        "recording": True,
        "replay": True,
        "validation": True,
        "evaluation": False,
        "export": True,
        "publishing": True,
        "adapters": True,
    }, ""),
):
    load_package(_DATASET_DIR)
with patch(
    "blacknode.packages._read_component_overrides",
    return_value=({
        "dataset-check": True,
        "training-jobs": True,
        "checkpoints": True,
        "policy-preview": True,
        "policy-artifacts": True,
        "reinforcement-learning": True,
        "vla-openpi": True,
    }, ""),
):
    load_package(_PACKAGE_DIR)

from blacknode.pkg.blacknode_training import data, runtime
from blacknode.pkg.blacknode_training.model import ActionChunkingConfig, ActionChunkingTransformer, masked_l1_loss
from blacknode.pkg.blacknode_training.ppo_model import PPOActorCritic, PPOModelConfig
from blacknode.pkg.blacknode_training import ppo_runtime, rl_training
from blacknode.pkg.blacknode_training.vla_openpi import OpenPIProvider, VLATrainConfig
from blacknode.workflow import validate_workflow

try:
    import h5py
except ImportError:
    h5py = None

try:
    import torch
except ImportError:
    torch = None


EXPECTED = {
    "TrainingDatasetCheck", "ACTTraining", "ACTCheckpointInspect", "ACTPolicyPreview",
    "ACTPolicyExport", "PolicyArtifactLoad", "ACTPolicyReplay",
    "PPOTraining", "PPOCheckpointInspect", "PPOPolicyEvaluate", "PPOPolicyExport",
    "PPOPolicyImport",
    "OpenPIFineTune",
}


def test_training_components_are_optional_by_default():
    info = _PACKAGE_REGISTRY["blacknode-training"]
    assert not any(component["default"] for component in info.components.values())


def _write_episode(path: Path, index: int, frames: int = 5) -> None:
    assert h5py is not None
    with h5py.File(path / f"episode_{index}.hdf5", "w") as handle:
        handle.attrs["episode_index"] = index
        handle.attrs["fps"] = 10
        handle.attrs["task"] = "Move cube"
        observations = handle.create_group("observations")
        base = np.arange(frames * 2, dtype=np.float32).reshape(frames, 2) / 10
        observations.create_dataset("qpos", data=base)
        observations.create_dataset("leader", data=base + 0.1)
        images = observations.create_group("images")
        images.create_dataset("front", data=np.full((frames, 16, 20, 3), 20 + index, dtype=np.uint8))
        handle.create_dataset("action", data=base + 0.1)
        metadata = handle.create_group("metadata")
        metadata.create_dataset("joint_names", data=["shoulder", "gripper"], dtype=h5py.string_dtype("utf-8"))


def test_nodes_registered_and_motion_free():
    for name in EXPECTED:
        assert name in _NODE_REGISTRY
        definition = _NODE_REGISTRY[name]
        assert definition._bn_package == "blacknode-training"
        assert definition._bn_category == "Training"
    assert not any("robot" in output.lower() or "command" in output.lower() for output in _NODE_REGISTRY["ACTPolicyPreview"]._bn_outputs)
    assert not any("robot" in output.lower() or "command" in output.lower() for output in _NODE_REGISTRY["ACTPolicyReplay"]._bn_outputs)
    assert _NODE_REGISTRY["ACTTraining"]._bn_input_defaults["action"] == "start"
    assert _NODE_REGISTRY["ACTTraining"]._bn_input_defaults["resume"] is True
    assert _NODE_REGISTRY["ACTPolicyExport"]._bn_input_defaults["action"] == "export"
    assert _NODE_REGISTRY["ACTPolicyReplay"]._bn_input_defaults["action"] == "evaluate"
    assert _NODE_REGISTRY["PPOTraining"]._bn_input_defaults["viewer_enabled"] is True
    assert _NODE_REGISTRY["PPOTraining"]._bn_input_defaults["viewer_fps"] == 15
    assert _NODE_REGISTRY["PPOTraining"]._bn_input_defaults["replay_episodes"] == 3
    assert _NODE_REGISTRY["PPOTraining"]._bn_input_choices["viewer_provider"] == [
        "viser", "ovrtx",
    ]
    assert "viewer_provider" in _NODE_REGISTRY["PPOTraining"]._bn_primary_inputs
    assert "viewer_url" in _NODE_REGISTRY["PPOTraining"]._bn_outputs
    assert "viewer_running" in _NODE_REGISTRY["PPOTraining"]._bn_outputs
    assert _NODE_REGISTRY["OpenPIFineTune"]._bn_input_defaults["action"] == "run"
    assert _NODE_REGISTRY["OpenPIFineTune"]._bn_input_defaults["resume"] is True


def test_status_is_non_mutating_and_dashboard_is_svg():
    result = _NODE_REGISTRY["ACTTraining"]({"action": "status", "run_id": "never-started"})
    assert result["ok"]
    assert not result["running"]
    assert result["phase"] == "not_started"
    prefix = "data:image/svg+xml;base64,"
    assert result["dashboard"].startswith(prefix)
    svg = base64.b64decode(result["dashboard"][len(prefix):]).decode("utf-8")
    assert "hardware motion remains disarmed" in svg
    controlled = runtime.control_training_job("never-started", "status")
    assert controlled["phase"] == "not_started"
    assert controlled["dashboard"].startswith(prefix)


@pytest.mark.skipif(torch is None, reason="torch is installed by package setup")
def test_ppo_policy_builds_shared_semantic_observation_and_absolute_sim_target(tmp_path: Path):
    environment = {
        "kind": "blacknode.rl-environment", "schema_version": 1,
        "provider": {"environment_type": "so101-reach-v1"},
        "robot_profile": "so_arm101", "joint_names": ["shoulder", "gripper"],
        "observation": {"dimension": 9},
        "action": {"dimension": 2, "minimum": -1.0, "maximum": 1.0, "scale_rad": 0.1},
        "safety": {"simulation_only": True, "physical_motion_authorized": False},
    }
    config = PPOModelConfig(observation_dim=9, action_dim=2, hidden_dim=32)
    model = PPOActorCritic(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save({
        "kind": "blacknode.ppo-policy-model", "schema_version": 1,
        "model_config": config.to_dict(), "model_state": model.state_dict(),
    }, tmp_path / "policy.pt")
    manifest = {
        "kind": "blacknode.policy-artifact", "schema_version": 1,
        "policy_type": "ppo-so101-reach", "backend": "blacknode-native",
        "model_format": "blacknode-ppo-state-dict", "path": str(tmp_path),
        "model_file": "policy.pt", "action_mode": "bounded_joint_position_delta",
        "units": "normalized", "joint_names": ["shoulder", "gripper"],
        "camera_names": [], "state_dim": 9, "action_dim": 2,
        "environment": environment,
        "compatibility_contract": ppo_runtime.ppo_observation_contract(environment),
        "safety": {"simulation_only": True, "physical_motion_authorized": False},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    policy = ppo_runtime.PPOPolicy(tmp_path, "cpu")
    prediction = policy.predict(
        [0.2, 0.3], {}, context={
            "joint_velocities": {"shoulder": 0.0, "gripper": 0.0},
            "joint_limits": {"shoulder": [-1.0, 1.0], "gripper": [0.0, 0.8]},
            "target_m": [0.2, 0.0, 0.3], "end_effector_m": [0.1, 0.0, 0.2],
        },
    )

    assert prediction["action"] == pytest.approx([0.2, 0.3])
    assert prediction["normalized_action"] == pytest.approx([0.0, 0.0])
    assert prediction["physical_motion_authorized"] is False


@pytest.mark.skipif(torch is None, reason="torch is installed by package setup")
def test_imported_isaac_torchscript_actor_uses_shared_ppo_artifact_contract(tmp_path: Path):
    class Actor(torch.nn.Module):
        def forward(self, observation):
            return observation[:, :2] * 0.0

    source = tmp_path / "isaac-actor.pt"
    torch.jit.trace(Actor(), torch.zeros((1, 9), dtype=torch.float32)).save(str(source))
    environment = {
        "kind": "blacknode.rl-environment", "schema_version": 1,
        "provider": {"environment_type": "so101-reach-v1"},
        "robot_profile": "so_arm101", "joint_names": ["shoulder", "gripper"],
        "observation": {"dimension": 9},
        "action": {"dimension": 2, "minimum": -1.0, "maximum": 1.0, "scale_rad": 0.1},
        "safety": {"simulation_only": True, "physical_motion_authorized": False},
    }

    artifact = ppo_runtime.import_torchscript_policy(
        source, environment, tmp_path / "artifact", source="isaac-lab"
    )

    assert artifact["model_format"] == "torchscript"
    assert artifact["source"] == "isaac-lab"
    assert artifact["compatible_simulators"] == ["newton", "isaac-sim"]
    assert artifact["compatibility_contract"]["observation"]["dimension"] == 9
    assert ppo_runtime.PPOPolicy(artifact, "cpu").model_format == "torchscript"


def test_dashboard_wraps_long_errors_without_truncating_text():
    message = "A detailed training failure explains the output directory and recovery action. " * 4 + "TAIL_MARKER"
    encoded = runtime.dashboard({"phase": "failed", "step": 0, "steps": 100, "error": message})
    svg = base64.b64decode(encoded.split(",", 1)[1]).decode("utf-8")
    assert "TAIL_MARKER" in svg
    assert 'height="210"' not in svg


def test_missing_dataset_is_structured_error(tmp_path: Path):
    result = _NODE_REGISTRY["TrainingDatasetCheck"]({"dataset_path": str(tmp_path / "missing")})
    assert not result["ok"]
    assert "does not exist" in result["report"]


def test_model_shape_and_masked_loss():
    if torch is None:
        pytest.skip("torch is installed by Blacknode package setup")
    config = ActionChunkingConfig(
        state_dim=2, action_dim=2, camera_count=2, chunk_size=4,
        hidden_dim=32, attention_heads=4, encoder_layers=1, decoder_layers=1,
    )
    model = ActionChunkingTransformer(config)
    prediction = model(torch.randn(3, 2), torch.randn(3, 2, 3, 16, 20))
    assert prediction.shape == (3, 4, 2)
    target = torch.zeros_like(prediction)
    is_pad = torch.tensor([[False, False, True, True]] * 3)
    loss = masked_l1_loss(prediction, target, is_pad)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_ppo_model_shape_and_disarmed_environment_check():
    if torch is None:
        pytest.skip("torch is installed by Blacknode package setup")
    model = PPOActorCritic(PPOModelConfig(observation_dim=21, action_dim=6, hidden_dim=32))
    observation = torch.randn(5, 21)
    action, latent, log_probability, value = model.sample(observation)
    assert action.shape == (5, 6)
    assert latent.shape == (5, 6)
    assert log_probability.shape == (5,)
    assert value.shape == (5,)
    assert torch.all(action.abs() <= 1.0)
    unsafe = {
        "kind": "blacknode.rl-environment",
        "provider": {"environment_type": "so101-reach-v1"},
        "safety": {"simulation_only": False, "physical_motion_authorized": True},
    }
    checked = _NODE_REGISTRY["PPOTraining"]({"action": "check", "environment": unsafe})
    assert not checked["ok"]
    assert "simulation-only" in checked["report"]


def test_ppo_run_waits_for_completion_and_emits_cloud_telemetry(
    tmp_path: Path, capsys, monkeypatch,
):
    statuses = iter([
        {"phase": "not_started", "running": False, "update": 0, "updates": 2},
        {"phase": "running", "running": True, "update": 0, "updates": 2, "progress": 0.0},
        {
            "phase": "running", "running": True, "update": 1, "updates": 2,
            "progress": 0.5, "mean_reward": 4.25,
        },
        {
            "phase": "completed", "running": False, "update": 2, "updates": 2,
            "progress": 1.0, "checkpoint": str(tmp_path / "checkpoint.pt"),
        },
    ])
    monkeypatch.setattr(rl_training, "_config", lambda _ctx: ppo_runtime.PPOTrainingConfig(
        run_id="cloud-demo", environment={}, output_dir=str(tmp_path / "run"), updates=2,
    ))
    monkeypatch.setattr(ppo_runtime, "job_status", lambda _run_id: next(statuses))
    monkeypatch.setattr(ppo_runtime, "start_job", lambda _config: {"running": True})
    monkeypatch.setattr(rl_training.time, "sleep", lambda _seconds: None)

    result = _NODE_REGISTRY["PPOTraining"]({"action": "run", "run_id": "cloud-demo"})

    output = capsys.readouterr().out
    assert result["phase"] == "completed"
    assert '"type":"progress","progress":100' in output
    assert '"name":"mean_reward","value":4.25,"step":1' in output


def test_ppo_replay_uses_latest_checkpoint_and_managed_one_arm_job(
    tmp_path: Path, monkeypatch,
):
    output = tmp_path / "run"
    output.mkdir()
    checkpoint = output / "checkpoint-00000500.pt"
    checkpoint.write_bytes(b"test")
    environment = {
        "kind": "blacknode.rl-environment",
        "provider": {"environment_type": "so101-reach-v1"},
        "safety": {"simulation_only": True, "physical_motion_authorized": False},
    }
    monkeypatch.setattr(rl_training, "_config", lambda _ctx: ppo_runtime.PPOTrainingConfig(
        run_id="replay-test", environment=environment, output_dir=str(output),
        viewer_enabled=True, viewer_port=8091,
    ))
    captured: list[ppo_runtime.PPOReplayConfig] = []

    def start_replay(config):
        captured.append(config)
        return {
            "phase": "replaying", "running": True, "viewer_running": True,
            "mode": "replay", "update": 500, "updates": 500,
            "replay_episode": 1, "replay_episodes": config.episodes,
            "checkpoint": config.checkpoint_path,
            "viewer_url": "http://127.0.0.1:8091",
            "viewer": {"running": True, "viewer_url": "http://127.0.0.1:8091"},
        }

    monkeypatch.setattr(ppo_runtime, "start_replay_job", start_replay)
    result = _NODE_REGISTRY["PPOTraining"]({
        "action": "replay", "environment": environment,
        "run_id": "replay-test", "replay_episodes": 4,
    })

    assert result["running"] and result["viewer_running"]
    assert result["mode"] == "replay"
    assert result["checkpoint"] == str(checkpoint)
    assert captured[0].episodes == 4
    assert captured[0].checkpoint_path == str(checkpoint)


def test_completed_ppo_viewer_remains_a_managed_service(monkeypatch):
    class Thread:
        @staticmethod
        def is_alive():
            return False

    class Job:
        thread = Thread()

        def __init__(self):
            self.closed = False

        def status(self):
            return {
                "run_id": "finished", "phase": "completed", "running": False,
                "viewer_running": not self.closed, "service_running": not self.closed,
                "viewer_url": "" if self.closed else "http://127.0.0.1:8091",
                "viewer": {"running": not self.closed},
            }

        def close_viewer(self):
            self.closed = True

    job = Job()
    monkeypatch.setattr(ppo_runtime, "_jobs", {"finished": job})

    status = ppo_runtime.runtime_status()
    assert status["active"]
    assert status["managed_runs"][0]["phase"] == "completed"
    closed = ppo_runtime.close_job_viewer("finished")
    assert not closed["viewer_running"]
    assert not ppo_runtime.runtime_status()["active"]


def test_ppo_evaluation_writes_json_next_to_checkpoint(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "checkpoint-00000010.pt"
    checkpoint.write_bytes(b"test")
    monkeypatch.setattr(
        ppo_runtime,
        "checkpoint_info",
        lambda _path: {"path": str(checkpoint), "update": 10},
    )
    monkeypatch.setattr(
        ppo_runtime,
        "evaluate_checkpoint",
        lambda *_args: {
            "kind": "blacknode.ppo-evaluation",
            "success_rate": 0.75,
            "mean_distance_m": 0.02,
        },
    )

    result = _NODE_REGISTRY["PPOPolicyEvaluate"]({"checkpoint_path": str(checkpoint)})

    assert result["ok"] and result["success_rate"] == 0.75
    saved = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert saved["success_rate"] == 0.75


def test_template_validates():
    path = Path(__file__).resolve().parents[1] / "templates" / "act-training.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert validate_workflow(workflow).ok
    assert workflow["entrypoint"] == {"node_id": "training", "port": "dashboard"}
    assert workflow["node_meta"]["training"]["params"]["action"] == "start"
    assert workflow["node_meta"]["training"]["params"]["resume"] is True
    assert workflow["node_meta"]["hdf5_export"]["params"]["action"] == "export"
    assert workflow["node_meta"]["dataset_browser"]["type"] == "DatasetBrowser"
    assert workflow["node_meta"]["policy_replay"]["params"]["action"] == "evaluate"
    assert workflow["node_meta"]["policy_stream"]["params"]["action"] == "start"
    assert {"blacknode-training", "blacknode-dataset"} <= set(workflow["metadata"]["required_packages"])
    assert {
        (edge["from"], edge["from_port"], edge["to"], edge["to_port"])
        for edge in workflow["edges"]
    } >= {
        ("dataset_browser", "dataset", "dataset_validate", "dataset"),
        ("dataset_browser", "dataset", "hdf5_export", "dataset"),
        ("hdf5_export", "path", "training", "dataset_path"),
        ("policy_load", "artifact", "policy_replay", "artifact"),
        ("dataset_browser", "stream", "policy_replay", "sync_stream"),
        ("policy_replay", "stream", "policy_stream", "stream"),
    }


def test_openpi_template_is_a_direct_outcome_workflow():
    path = Path(__file__).resolve().parents[1] / "templates" / "openpi-pi05-finetune.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    result = validate_workflow(workflow)
    assert result.ok, result.errors
    assert workflow["entrypoint"] == {"node_id": "train", "port": "model"}
    assert [node["type"] for node in workflow["node_meta"].values()] == [
        "LeRobotDataset", "OpenPIFineTune",
    ]
    assert workflow["node_meta"]["train"]["params"]["action"] == "run"
    assert workflow["metadata"]["cloud"]["workload"] == "vla_train"


def test_openpi_provider_requires_an_immutable_remote_dataset(tmp_path: Path):
    config = VLATrainConfig(
        run_id="test",
        dataset={"kind": "blacknode.dataset-source", "uri": "hf://owner/dataset"},
        output_dir=str(tmp_path / "run"),
    )
    with pytest.raises(ValueError, match="immutable revision"):
        OpenPIProvider().validate(config)


def test_openpi_provider_runs_and_exports_a_vla_model(tmp_path: Path):
    runner = tmp_path / "fake_openpi.py"
    runner.write_text(
        "import json, pathlib, sys\n"
        "request = pathlib.Path(sys.argv[sys.argv.index('--request') + 1])\n"
        "output = request.parent\n"
        "checkpoint = output / 'checkpoints' / '1'\n"
        "checkpoint.mkdir(parents=True)\n"
        "(checkpoint / 'params').write_bytes(b'openpi-adapter')\n"
        "norm = output / 'norm_stats.json'\n"
        "norm.write_text('{\\\"action\\\": {}}')\n"
        "print('Step 1: loss=0.25, grad_norm=1.5', flush=True)\n"
        "(output / 'openpi-result.json').write_text(json.dumps({"
        "'checkpoint_dir': str(checkpoint), 'norm_stats_path': str(norm), "
        "'final_step': 1, 'metrics': {'loss': 0.25}, "
        "'inference': {'verified': True}}))\n",
        encoding="utf-8",
    )
    config = VLATrainConfig(
        run_id="real-output",
        dataset={
            "kind": "blacknode.dataset-source",
            "uri": "hf://owner/dataset",
            "revision": "a" * 40,
        },
        output_dir=str(tmp_path / "run"),
        steps=1,
        runner_path=str(runner),
    )
    provider = OpenPIProvider()
    prepared = provider.prepare(config)
    events = []
    result = provider.train(prepared, events.append, threading.Event())
    model = provider.export(prepared, result)

    assert model["kind"] == "blacknode.vla-model"
    assert model["architecture"] == "pi05"
    assert model["backend"] == "jax"
    assert model["training_method"] == "lora"
    assert model["dataset"]["revision"] == "a" * 40
    assert model["inference"]["verified"] is True
    assert (Path(model["path"]) / model["checkpoint"]).is_file()
    assert any(event.get("name") == "loss" for event in events)


def test_so101_ppo_template_validates_and_stays_simulation_only():
    path = Path(__file__).resolve().parents[1] / "templates" / "so101-ppo-training.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    external_task = _NODE_REGISTRY.get("SO101ReachTask", lambda _ctx: {})
    with patch.dict(_NODE_REGISTRY, {"SO101ReachTask": external_task}):
        result = validate_workflow(workflow)
    assert result.ok, result.errors
    assert workflow["entrypoint"] == {"node_id": "training", "port": "dashboard"}
    assert workflow["node_meta"]["training"]["params"]["action"] == "start"
    assert workflow["node_meta"]["training"]["params"]["resume"] is True
    assert workflow["node_meta"]["training"]["params"]["viewer_enabled"] is True
    assert workflow["node_meta"]["training"]["params"]["viewer_fps"] == 15
    assert workflow["node_meta"]["training"]["params"]["replay_episodes"] == 3
    assert "replay" in workflow["node_meta"]["training"]["input_choices"]["action"]
    assert workflow["node_meta"]["training"]["input_choices"]["viewer_provider"] == [
        "viser", "ovrtx",
    ]
    assert "viewer_provider" in workflow["node_meta"]["training"]["promoted_inputs"]
    assert "viewer_running" in workflow["node_meta"]["training"]["outputs"]
    assert {"blacknode-newton", "blacknode-training"} <= set(workflow["metadata"]["required_packages"])
    assert "blacknode-newton/viewer-viser" in workflow["metadata"]["required_components"]
    assert (
        "task", "environment", "training", "environment"
    ) in {
        (edge["from"], edge["from_port"], edge["to"], edge["to_port"])
        for edge in workflow["edges"]
    }

    cloud_path = path.with_name("so101-ppo-cloud-demo.json")
    cloud_workflow = json.loads(cloud_path.read_text(encoding="utf-8"))
    with patch.dict(_NODE_REGISTRY, {"SO101ReachTask": external_task}):
        cloud_result = validate_workflow(cloud_workflow)
    assert cloud_result.ok, cloud_result.errors
    assert cloud_workflow["entrypoint"] == {"node_id": "out", "port": "value"}
    assert cloud_workflow["node_meta"]["training"]["params"]["action"] == "run"
    assert cloud_workflow["node_meta"]["training"]["params"]["viewer_enabled"] is False
    assert "blacknode-newton/viewer-viser" not in cloud_workflow["metadata"]["required_components"]

    import_path = path.with_name("so101-ppo-isaac-import.json")
    import_workflow = json.loads(import_path.read_text(encoding="utf-8"))
    with patch.dict(_NODE_REGISTRY, {"SO101ReachTask": external_task}):
        import_result = validate_workflow(import_workflow)
    assert import_result.ok, import_result.errors
    assert import_workflow["entrypoint"] == {"node_id": "evaluate", "port": "metrics"}
    assert import_workflow["node_meta"]["import"]["params"]["action"] == "check"
    assert import_workflow["node_meta"]["evaluate"]["params"]["action"] == "check"
    assert (
        "import", "artifact", "evaluate", "artifact"
    ) in {
        (edge["from"], edge["from_port"], edge["to"], edge["to_port"])
        for edge in import_workflow["edges"]
    }


@pytest.mark.skipif(h5py is None, reason="h5py is installed by package setup")
def test_policy_replay_contract_without_loading_real_model(tmp_path: Path, monkeypatch):
    _write_episode(tmp_path, 0)
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    (policy_dir / "model.pt").write_bytes(b"synthetic")
    artifact = {
        "kind": "blacknode.policy-artifact", "schema_version": 1,
        "policy_type": "act", "backend": "blacknode-native",
        "path": str(policy_dir), "model_file": "model.pt", "step": 7,
        "units": "radians", "state_dim": 2, "action_dim": 2,
        "joint_names": ["shoulder", "gripper"], "camera_names": ["front"],
    }

    class FakePolicy:
        def __init__(self, _artifact, _device):
            pass

        def predict(self, qpos, _images):
            return {"action": list(qpos), "action_chunk": [list(qpos)]}

    monkeypatch.setattr(data, "torch", object())
    monkeypatch.setattr(runtime, "ACTPolicy", FakePolicy)
    replayed = _NODE_REGISTRY["ACTPolicyReplay"]({
        "action": "evaluate", "artifact": artifact, "dataset_path": str(tmp_path),
        "episode": {"episode_index": 0}, "device": "cpu",
        "sync_stream": {"kind": "blacknode.replay-stream", "token": "browser-token"},
    })
    assert replayed["ok"] and replayed["evaluated"]
    assert replayed["frame_count"] == 5
    assert replayed["stream"]["source_token"] == "browser-token"
    assert replayed["stream"]["frames_data"][0]["motion_commanded"] is False
    assert replayed["metrics"]["mean_absolute_error"] == pytest.approx(0.1)


@pytest.mark.skipif(h5py is None or torch is None, reason="h5py and torch are installed by package setup")
def test_dataset_training_checkpoint_and_preview(tmp_path: Path):
    _write_episode(tmp_path, 0)
    _write_episode(tmp_path, 1)
    summary = data.inspect_dataset(tmp_path)
    assert summary["episode_count"] == 2
    assert summary["joint_names"] == ["shoulder", "gripper"]
    output = tmp_path / "training"
    status = runtime.start_job(runtime.TrainingConfig(
        run_id="synthetic", dataset_path=str(tmp_path), output_dir=str(output),
        device="cpu", steps=2, batch_size=2, chunk_size=3,
        hidden_dim=32, attention_heads=4, encoder_layers=1, decoder_layers=1,
        validation_fraction=0.5, eval_every=1, checkpoint_every=1,
    ))
    assert status["running"]
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = runtime.job_status("synthetic")
        if not status["running"]:
            break
        time.sleep(0.05)
    assert status["phase"] == "completed", status
    checkpoint = Path(status["checkpoint"])
    assert checkpoint.exists()
    rerun = _NODE_REGISTRY["ACTTraining"]({
        "run_id": "synthetic", "dataset_path": str(tmp_path), "output_dir": str(output),
        "device": "cpu", "steps": 2, "batch_size": 2, "chunk_size": 3,
        "hidden_dim": 32, "attention_heads": 4, "encoder_layers": 1, "decoder_layers": 1,
    })
    assert rerun["ok"] and not rerun["running"]
    assert rerun["phase"] == "completed"
    assert rerun["step"] == 2
    info = runtime.checkpoint_info(checkpoint)
    assert info["step"] == 2
    prediction = runtime.preview(checkpoint, tmp_path, 0, 0, "cpu")
    assert len(prediction["action"]) == 2
    assert len(prediction["target_action"]) == 2
    assert len(prediction["absolute_error"]) == 2
    assert len(prediction["action_chunk"]) == 3
    assert prediction["motion_commanded"] is False
    artifact = runtime.export_policy_artifact(checkpoint, tmp_path / "policy")
    assert artifact["kind"] == "blacknode.policy-artifact"
    assert artifact["joint_names"] == ["shoulder", "gripper"]
    policy = runtime.ACTPolicy(artifact, "cpu")
    live_prediction = policy.predict(
        [0.0, 0.1], {"front": np.full((16, 20, 3), 20, dtype=np.uint8)},
    )
    assert len(live_prediction["action"]) == 2
    exported = _NODE_REGISTRY["ACTPolicyExport"]({
        "action": "check", "checkpoint_path": str(checkpoint),
    })
    assert exported["ok"] and not exported["exported"]
    loaded = _NODE_REGISTRY["PolicyArtifactLoad"]({"artifact_path": artifact["path"]})
    assert loaded["ok"] and loaded["artifact"]["model_path"].endswith("model.pt")
    checked_replay = _NODE_REGISTRY["ACTPolicyReplay"]({
        "action": "check", "artifact": loaded["artifact"], "dataset_path": str(tmp_path),
        "episode_index": 0,
    })
    assert checked_replay["ok"] and not checked_replay["evaluated"]
    assert checked_replay["frame_count"] == 5
    replayed = _NODE_REGISTRY["ACTPolicyReplay"]({
        "action": "evaluate", "artifact": loaded["artifact"], "dataset_path": str(tmp_path),
        "episode_index": 0, "device": "cpu",
        "sync_stream": {"kind": "blacknode.replay-stream", "token": "recorded-episode"},
    })
    assert replayed["ok"] and replayed["evaluated"]
    assert replayed["stream"]["kind"] == "blacknode.replay-stream"
    assert replayed["stream"]["source_token"] == "recorded-episode"
    assert len(replayed["stream"]["frames_data"]) == 5
    assert replayed["metrics"]["mean_absolute_error"] >= 0
    assert replayed["replay"]["motion_commanded"] is False
