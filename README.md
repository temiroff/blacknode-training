# blacknode-training

Blacknode-native robot policy training from the HDF5 episodes produced by
`blacknode-dataset`. The package trains a compact vision-and-state
action-chunking transformer in PyTorch, saves resumable checkpoints, reports
live metrics, and previews predictions against recorded frames.

## Install

Place this repository under `packages/blacknode-training` and restart
Blacknode. Startup automatically installs declared dependencies:

```text
torch>=2.4
h5py>=3.11
numpy>=1.24
```

PyTorch is contained in this optional package, keeping recording installations
focused on acquisition.

## Workflow

1. Record successful episodes with `blacknode-dataset`.
2. Open **Blacknode Native ACT Training**.
3. Use `DatasetBrowser` to choose the dataset root and dataset ID.
4. Inspect the selected episode and camera, then run `EpisodeDatasetValidate`.
5. Set `HDF5EpisodeExport.action=export` with `include_images=true` and cook it
   once. Return the action to `check` after the export is ready.
6. Confirm `TrainingDatasetCheck` reports the expected episodes, joints, and
   cameras.
7. Set `ACTTraining.action=check` and cook the dashboard.
8. Choose `action=start` only after the dataset and output settings are valid.
9. Return the action to `status` to monitor the current job.
10. Use `action=stop` for a cooperative stop; the latest completed step is
   checkpointed and can be resumed with `resume=true`.
11. Export the checkpoint, load it with `PolicyArtifactLoad`, and set
    `ACTPolicyReplay.action=evaluate` for the selected Dataset Browser episode.
12. Start the connected `StreamPublisher`. Playing or seeking the Dataset
    Browser sends policy-predicted actions to the external evaluation apps.

The browser selects the Blacknode-native dataset. The HDF5 export node produces
the ACT training view and passes its path directly into dataset checking,
training, and policy preview. The export defaults to `check`, and training
defaults to `status`, so cooking the graph never implicitly exports data or
starts training.

## Nodes

| Node | Purpose |
| --- | --- |
| `TrainingDatasetCheck` | Validate episode files, state/action dimensions, joint order, cameras, FPS, frame counts, and finite numeric data. |
| `ACTTraining` | Check, start, monitor, stop, or resume one managed background training run. Its dashboard shows phase, progress, train loss, validation loss, and failures. |
| `ACTCheckpointInspect` | Read the fixed schema, normalization statistics, split, model configuration, step, and metrics from a checkpoint. |
| `ACTPolicyPreview` | Predict and display a denormalized future action chunk for one recorded frame. |
| `ACTPolicyExport` | Export model weights, schema, normalization, camera order, joint order, and metrics as an inference-only policy artifact. |
| `PolicyArtifactLoad` | Validate and load an exported policy manifest for a deployment workflow. |
| `ACTPolicyReplay` | Evaluate a loaded artifact across every frame in one recorded episode, report prediction error, and emit a browser-synchronized replay stream. |

## Training contract

The loader requires one or more `episode_<index>.hdf5` files with:

```text
/observations/qpos             float [T, state_dim]
/observations/images/<camera>  uint8 [T, height, width, 3]
/action                        float [T, action_dim]
/metadata/joint_names          UTF-8 [state_dim]
```

Every episode must have the same state/action dimensions, ordered joint names,
camera names, camera resolutions, and FPS. Validation rejects mismatches and
non-finite state/action values; vectors and joint order are preserved exactly.

Training and validation are split by whole episode. Normalization statistics
are computed only from training episodes. At timestep `t`, the model receives
the normalized follower joint state plus all RGB cameras and predicts
`chunk_size` normalized future actions starting at `t`. Padded actions at the
end of an episode are excluded from the L1 loss.

The model is a Blacknode action-chunking transformer: a shared CNN
encodes each camera into spatial tokens, a transformer encoder fuses those
tokens with robot state, and learned action queries decode the future action
chunk. Its fixed checkpoint contract keeps the dataset schema, normalization
statistics, model configuration, and training state together.

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

`ACTPolicyExport` writes `manifest.json` plus `model.pt`. The manifest is the
stable handoff to Blacknode policy runtimes and declares absolute joint-position
actions in radians, ordered joints and cameras, normalization statistics, and
the source training step. The exported model omits optimizer state.

`ACTPolicyReplay` checks the artifact and HDF5 dataset contracts before loading
the model. Evaluation produces predicted actions, recorded targets, per-joint
absolute error, episode MAE/RMSE/max error, and a `blacknode.replay-stream`
handle. Connect that handle to `StreamPublisher`. When `sync_stream` receives
`DatasetBrowser.stream`, the recorded video timeline controls policy replay, so
Maya, ROS 2, Isaac Sim, and other subscribers see the prediction for the frame
currently under review. This inference-only path never publishes robot commands.

## Safety and limitations

- This package performs offline training and recorded-frame prediction.
- Hardware control belongs to an explicitly armed Blacknode policy controller.
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

## License

Apache-2.0, same as Blacknode.
