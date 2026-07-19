"""blacknode-training node, model, and managed-job contracts."""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import numpy as np
import pytest

import blacknode  # noqa: F401 - triggers package discovery
from blacknode.node import _NODE_REGISTRY
from blacknode.pkg.blacknode_training import data, runtime
from blacknode.pkg.blacknode_training.model import ActionChunkingConfig, ActionChunkingTransformer, masked_l1_loss
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
}


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


def test_status_is_non_mutating_and_dashboard_is_svg():
    result = _NODE_REGISTRY["ACTTraining"]({"action": "status", "run_id": "never-started"})
    assert result["ok"]
    assert not result["running"]
    assert result["phase"] == "not_started"
    prefix = "data:image/svg+xml;base64,"
    assert result["dashboard"].startswith(prefix)
    svg = base64.b64decode(result["dashboard"][len(prefix):]).decode("utf-8")
    assert "never commands robot motion" in svg


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


def test_template_validates():
    path = Path(__file__).resolve().parents[1] / "templates" / "act-training.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert validate_workflow(workflow).ok
    assert workflow["entrypoint"] == {"node_id": "training", "port": "dashboard"}
    assert workflow["node_meta"]["training"]["params"]["action"] == "status"
    assert workflow["node_meta"]["hdf5_export"]["params"]["action"] == "check"
    assert workflow["node_meta"]["dataset_browser"]["type"] == "DatasetBrowser"
    assert workflow["node_meta"]["policy_replay"]["params"]["action"] == "check"
    assert workflow["node_meta"]["policy_stream"]["params"]["action"] == "status"
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
