# blacknode-training

Blacknode-native offline robot policy training with no LeRobot, Hugging Face,
GR00T, ROS, or robot-runtime dependency. The package consumes the ACT-style
HDF5 episodes produced by `blacknode-dataset`, trains a compact vision-and-state
action-chunking transformer in PyTorch, saves resumable checkpoints, and can
preview predictions against recorded frames without commanding hardware.

## Install

Place this repository under `packages/blacknode-training` and restart
Blacknode. Startup automatically installs declared dependencies:

```text
torch>=2.4
h5py>=3.11
numpy>=1.24
```

PyTorch is intentionally isolated in this optional package; recording remains
lightweight when training is not installed.

## Workflow

1. Record successful episodes with `blacknode-dataset`.
2. Run `EpisodeDatasetValidate`.
3. Use `HDF5EpisodeExport` with `action=export` and `include_images=true`.
4. Open **Blacknode Native ACT Training**.
5. Set the HDF5 directory in the `Text` node.
6. Set `ACTTraining.action=check` and cook the dashboard.
7. Choose `action=start` only after the dataset and output settings are valid.
8. Return the action to `status` to monitor without starting another job.
9. Use `action=stop` for a cooperative stop; the latest completed step is
   checkpointed and can be resumed with `resume=true`.

The template defaults to `status`. Cooking a graph never implicitly starts
training.

## Nodes

| Node | Purpose |
| --- | --- |
| `TrainingDatasetCheck` | Validate episode files, state/action dimensions, joint order, cameras, FPS, frame counts, and finite numeric data. |
| `ACTTraining` | Check, start, monitor, stop, or resume one managed background training run. Its dashboard shows phase, progress, train loss, validation loss, and failures. |
| `ACTCheckpointInspect` | Read the fixed schema, normalization statistics, split, model configuration, step, and metrics from a checkpoint. |
| `ACTPolicyPreview` | Predict a denormalized future action chunk for one recorded frame. It has no robot connection or motion output. |

## Training contract

The loader requires one or more `episode_<index>.hdf5` files with:

```text
/observations/qpos             float [T, state_dim]
/observations/images/<camera>  uint8 [T, height, width, 3]
/action                        float [T, action_dim]
/metadata/joint_names          UTF-8 [state_dim]
```

Every episode must have the same state/action dimensions, ordered joint names,
camera names, camera resolutions, and FPS. The package refuses mismatches and
non-finite state/action values. It does not resize vectors or reorder joints.

Training and validation are split by whole episode. Normalization statistics
are computed only from training episodes. At timestep `t`, the model receives
the normalized follower joint state plus all RGB cameras and predicts
`chunk_size` normalized future actions starting at `t`. Padded actions at the
end of an episode are excluded from the L1 loss.

The model is an ACT-style action-chunking transformer baseline: a shared CNN
encodes each camera into spatial tokens, a transformer encoder fuses those
tokens with robot state, and learned action queries decode the future action
chunk. It is not a byte-for-byte reproduction of the original ACT CVAE. The
clear checkpoint contract lets later package versions add a canonical CVAE or
diffusion model without changing the recorder.

## Run outputs

```text
<output-dir>/
  run.json
  latest.json
  checkpoint-00001000.pt
  checkpoint-00002000.pt
  ...
```

Each checkpoint includes model and optimizer state, model configuration,
normalization statistics, dataset joint/camera schema, episode split, training
configuration, and metrics. Writes use a temporary file and atomic rename.
Only load checkpoints produced locally or by a trusted source.

## Safety and limitations

- This package performs offline training and prediction only.
- It never imports a robot or ROS package and cannot command motion.
- Validation loss measures action imitation, not real-world task success.
- Always inspect predictions and add a separately reviewed, disarmed safety
  controller before using a trained policy on hardware.
- Background jobs live in the Blacknode server process. Stop training before
  restarting the server; completed checkpoints remain resumable.

## Test

```powershell
$env:PYTHONPATH="python"
python -m pytest packages/blacknode-training/tests
blacknode validate packages/blacknode-training/templates/act-training.json
```
