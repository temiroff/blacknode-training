# blacknode-training

`blacknode-training` provides offline robot-policy dataset checks, managed PyTorch and PPO training, resumable checkpoints, simulation or recorded-frame evaluation, and inference artifact export.

## Components

| Component | Purpose |
|---|---|
| `dataset-check` | Validate episode schema and readiness |
| `training-jobs` | Start, monitor, stop, and resume ACT training |
| `checkpoints` | Inspect checkpoint schema, metrics, and configuration |
| `policy-preview` | Preview and replay predictions on recorded episodes |
| `policy-artifacts` | Export and load inference artifacts |
| `reinforcement-learning` | Train, stop, resume, evaluate, and export PPO policies for vectorized Newton environments |

All components are optional so normal robot runtimes do not install the training stack.

### SO-ARM101 reinforcement learning

Open `so101-ppo-training.json`, then press **Run** on `PPOTraining`. The connected
`SO101ReachTask` creates replicated SO-ARM101 articulations from the bundled USD
asset, with joint-state and target observations and bounded joint-position-delta
actions. Newton advances the articulations through Warp while PPO updates the
policy on PyTorch.

The dashboard reports update count, target distance, success rate, simulation
throughput, and the latest atomic checkpoint. The node's **Stop** button requests
a cooperative stop; running the same output directory with `resume=true`
continues from the newest checkpoint. Use `PPOPolicyEvaluate` before exporting a
simulation-only artifact with `PPOPolicyExport`.

For a one-shot local or Cloud container run, open `SO-ARM101 PPO Cloud Demo`.
Its `PPOTraining` node uses `action=run`, waits for the managed training thread,
emits progress and reward metrics, evaluates the final checkpoint, and exports
`evaluation.json` plus `policy.pt`. The ordinary PPO template keeps
`action=start` for interactive editor control.

With `viewer_enabled=true`, training starts a read-only Viser preview and
Blacknode opens it in the simulation pane. The preview copies one selected arm
into a separate one-articulation render model at up to `viewer_fps` (15 by
default), so the complete environment batch remains dedicated to learning. Use
the **Environment index** control inside the viewer to inspect another arm. The
green sphere is the reach target and the orange line is the end-effector trail.
The preview contains no joint-command or hardware-arm controls.

## Workflow

1. Record episodes with `blacknode-dataset`.
2. Open `act-training.json` and select the dataset.
3. Start or resume `ACTTraining`; the upstream HDF5 exporter prepares the training view.
4. Monitor train/validation loss and stop cooperatively when needed.
5. Export a checkpoint with `ACTPolicyExport`.
6. Evaluate it with `ACTPolicyReplay` before using a separately armed controller.

Training splits by episode and computes normalization from training episodes only. Checkpoints include model and optimizer state, ordered joints/cameras, dataset schema, normalization, configuration, and metrics. Exported policy artifacts omit optimizer state and retain the deployment contract.

## Safety

This package performs offline training, simulated reinforcement learning, and
prediction evaluation. It never commands hardware. PPO checkpoints and
artifacts explicitly retain `physical_motion_authorized=false`; a future
hardware evaluation path must add a separately armed safety controller. Treat
checkpoints as trusted executable data and review predictions before connecting
an artifact to a motion controller.

## Install and verify

```powershell
blacknode packages install https://github.com/temiroff/blacknode-training.git
$env:PYTHONPATH="python"
python -m pytest packages/blacknode-training/tests
blacknode validate packages/blacknode-training/templates/act-training.json
blacknode validate packages/blacknode-training/templates/so101-ppo-training.json
```

See [AGENTS.md](AGENTS.md) for dataset, checkpoint, and managed-job rules.
