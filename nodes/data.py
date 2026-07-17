"""HDF5 episode inspection and action-chunk dataset utilities."""
from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

try:
    import h5py
except Exception:  # pragma: no cover - surfaced by package health and nodes
    h5py = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def require_dependencies(*, needs_torch: bool = False) -> None:
    missing: list[str] = []
    if h5py is None:
        missing.append("h5py")
    if np is None:
        missing.append("numpy")
    if needs_torch and torch is None:
        missing.append("torch")
    if missing:
        raise RuntimeError(
            "missing training dependencies: " + ", ".join(missing)
            + "; restart Blacknode for automatic package setup"
        )


def resolve_dataset_path(value: str | Path) -> Path:
    path = Path(str(value or "").strip()).expanduser().resolve()
    if not str(value or "").strip():
        raise ValueError("dataset_path is required")
    if not path.is_dir():
        raise ValueError(f"HDF5 dataset directory does not exist: {path}")
    return path


def _episode_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else 2**31 - 1, path.name)


def episode_files(path: Path) -> list[Path]:
    files = sorted(path.glob("episode_*.hdf5"), key=_episode_key)
    if not files:
        raise ValueError(f"no episode_*.hdf5 files found in {path}")
    return files


def _strings(dataset: Any) -> list[str]:
    values = dataset.asstr()[:] if hasattr(dataset, "asstr") else dataset[:]
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def inspect_dataset(value: str | Path) -> dict[str, Any]:
    root = resolve_dataset_path(value)
    require_dependencies()
    assert h5py is not None and np is not None
    files = episode_files(root)
    expected: dict[str, Any] | None = None
    episodes: list[dict[str, Any]] = []
    total_frames = 0
    for position, file_path in enumerate(files):
        with h5py.File(file_path, "r") as handle:
            for required in ("observations/qpos", "action", "metadata/joint_names"):
                if required not in handle:
                    raise ValueError(f"{file_path.name} is missing /{required}")
            qpos = handle["observations/qpos"]
            action = handle["action"]
            if len(qpos.shape) != 2 or len(action.shape) != 2:
                raise ValueError(f"{file_path.name} qpos and action must both be rank-2")
            if qpos.shape[0] != action.shape[0] or not qpos.shape[0]:
                raise ValueError(f"{file_path.name} qpos/action frame counts differ or are empty")
            if qpos.shape[1] != action.shape[1]:
                raise ValueError(f"{file_path.name} action dimension must match qpos/joint dimension")
            joint_names = _strings(handle["metadata/joint_names"])
            if len(joint_names) != qpos.shape[1]:
                raise ValueError(f"{file_path.name} joint_names length does not match qpos")
            cameras: dict[str, list[int]] = {}
            if "observations/images" in handle:
                for name, image_data in sorted(handle["observations/images"].items()):
                    if len(image_data.shape) != 4 or image_data.shape[0] != qpos.shape[0] or image_data.shape[-1] != 3:
                        raise ValueError(f"{file_path.name} camera {name} must have shape [T,H,W,3]")
                    if image_data.dtype != np.dtype("uint8"):
                        raise ValueError(f"{file_path.name} camera {name} must contain uint8 RGB images")
                    cameras[str(name)] = [int(size) for size in image_data.shape[1:]]
            if not cameras:
                raise ValueError(f"{file_path.name} has no /observations/images cameras")
            schema = {
                "state_dim": int(qpos.shape[1]),
                "action_dim": int(action.shape[1]),
                "joint_names": joint_names,
                "cameras": cameras,
                "fps": int(handle.attrs.get("fps") or 0),
                "task": str(handle.attrs.get("task") or ""),
                "image_color_space": str(handle.attrs.get("image_color_space") or "RGB").upper(),
            }
            if schema["fps"] <= 0:
                raise ValueError(f"{file_path.name} must declare a positive fps attribute")
            if schema["image_color_space"] != "RGB":
                raise ValueError(f"{file_path.name} image_color_space must be RGB")
            if expected is None:
                expected = schema
            elif schema != expected:
                raise ValueError(f"{file_path.name} schema differs from {files[0].name}")
            if not np.isfinite(qpos[:]).all() or not np.isfinite(action[:]).all():
                raise ValueError(f"{file_path.name} contains non-finite qpos or action values")
            frames = int(qpos.shape[0])
            total_frames += frames
            episodes.append({
                "episode_index": int(handle.attrs.get("episode_index", position)),
                "file": str(file_path),
                "frames": frames,
                "task": str(handle.attrs.get("task") or ""),
            })
    assert expected is not None
    return {
        "kind": "blacknode.training-dataset",
        "schema_version": 1,
        "path": str(root),
        "episode_count": len(episodes),
        "total_frames": total_frames,
        **expected,
        "episodes": episodes,
    }


def split_episodes(count: int, validation_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if count < 1:
        raise ValueError("dataset has no episodes")
    indexes = list(range(count))
    random.Random(int(seed)).shuffle(indexes)
    if count == 1 or validation_fraction <= 0:
        return indexes, []
    validation_count = max(1, min(count - 1, round(count * float(validation_fraction))))
    return sorted(indexes[validation_count:]), sorted(indexes[:validation_count])


def compute_statistics(files: list[Path], indexes: list[int]) -> dict[str, list[float]]:
    require_dependencies()
    assert h5py is not None and np is not None
    if not indexes:
        raise ValueError("training split is empty")
    sums: dict[str, Any] = {}
    square_sums: dict[str, Any] = {}
    count = 0
    for index in indexes:
        with h5py.File(files[index], "r") as handle:
            qpos = np.asarray(handle["observations/qpos"][:], dtype=np.float64)
            action = np.asarray(handle["action"][:], dtype=np.float64)
            if not sums:
                sums = {"qpos": qpos.sum(axis=0), "action": action.sum(axis=0)}
                square_sums = {"qpos": np.square(qpos).sum(axis=0), "action": np.square(action).sum(axis=0)}
            else:
                sums["qpos"] += qpos.sum(axis=0)
                sums["action"] += action.sum(axis=0)
                square_sums["qpos"] += np.square(qpos).sum(axis=0)
                square_sums["action"] += np.square(action).sum(axis=0)
            count += qpos.shape[0]
    result: dict[str, list[float]] = {}
    for name in ("qpos", "action"):
        mean = sums[name] / count
        variance = np.maximum(square_sums[name] / count - np.square(mean), 1e-12)
        result[f"{name}_mean"] = mean.astype(np.float32).tolist()
        result[f"{name}_std"] = np.sqrt(variance).astype(np.float32).tolist()
    return result


class HDF5ActionChunkDataset(torch.utils.data.Dataset if torch is not None else object):
    def __init__(
        self,
        files: list[Path],
        indexes: list[int],
        *,
        camera_names: list[str],
        chunk_size: int,
        statistics: dict[str, list[float]],
    ) -> None:
        require_dependencies(needs_torch=True)
        assert h5py is not None and np is not None and torch is not None
        self.files = files
        self.indexes = list(indexes)
        self.camera_names = list(camera_names)
        self.chunk_size = max(1, int(chunk_size))
        self.statistics = statistics
        self.samples: list[tuple[int, int]] = []
        self._handles: dict[int, Any] = {}
        for file_index in self.indexes:
            with h5py.File(files[file_index], "r") as handle:
                self.samples.extend((file_index, frame) for frame in range(int(handle["action"].shape[0])))
        self.qpos_mean = np.asarray(statistics["qpos_mean"], dtype=np.float32)
        self.qpos_std = np.asarray(statistics["qpos_std"], dtype=np.float32)
        self.action_mean = np.asarray(statistics["action_mean"], dtype=np.float32)
        self.action_std = np.asarray(statistics["action_std"], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def _handle(self, file_index: int) -> Any:
        handle = self._handles.get(file_index)
        if handle is None:
            assert h5py is not None
            handle = h5py.File(self.files[file_index], "r")
            self._handles[file_index] = handle
        return handle

    def __getitem__(self, sample_index: int) -> dict[str, Any]:
        assert np is not None and torch is not None
        file_index, frame_index = self.samples[sample_index]
        handle = self._handle(file_index)
        qpos = (np.asarray(handle["observations/qpos"][frame_index], dtype=np.float32) - self.qpos_mean) / self.qpos_std
        images = np.stack([
            np.asarray(handle[f"observations/images/{name}"][frame_index], dtype=np.float32).transpose(2, 0, 1) / 255.0
            for name in self.camera_names
        ])
        action_data = handle["action"]
        end = min(frame_index + self.chunk_size, int(action_data.shape[0]))
        valid = end - frame_index
        actions = np.zeros((self.chunk_size, int(action_data.shape[1])), dtype=np.float32)
        actions[:valid] = (
            np.asarray(action_data[frame_index:end], dtype=np.float32) - self.action_mean
        ) / self.action_std
        is_pad = np.ones(self.chunk_size, dtype=np.bool_)
        is_pad[:valid] = False
        return {
            "qpos": torch.from_numpy(qpos),
            "images": torch.from_numpy(images),
            "actions": torch.from_numpy(actions),
            "is_pad": torch.from_numpy(is_pad),
        }

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass
