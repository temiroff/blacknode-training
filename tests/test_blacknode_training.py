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


EXPECTED = {"TrainingDatasetCheck", "ACTTraining", "ACTCheckpointInspect", "ACTPolicyPreview"}


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
