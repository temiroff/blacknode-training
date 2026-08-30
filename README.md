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
| `reinforcement-learning` | Train, stop, resume, evaluate, export, and qualify PPO policies for extension-owned vectorized environments |

All components are optional so normal robot runtimes do not install the training stack.

### Provider-neutral reinforcement learning

`PPOTraining` accepts a `blacknode.rl-environment` from any loaded extension
package. The environment declares its provider factory, tensor dimensions,
semantic observation fields, action mapping, timing, domain randomization, and
simulation-only safety state. Training no longer depends on a particular
simulator package.

`PPOPolicyExport` writes a portable TorchScript actor by default and fingerprints
the model plus compatibility contract. Connect the exported artifact and one or
more `PPOPolicyEvaluate` results to `PPOPolicyQualify`. The qualification binds
explicit success, episode-count, distance, and scenario thresholds to the exact
artifact while leaving physical motion unauthorized. Physical approval belongs
to `blacknode-motion` and remains bound to a calibrated robot and safety gate.

See the core [sim-to-real lifecycle guide](../../docs/sim-to-real-policy-lifecycle.md)
for the environment contract and deployment flow.

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

With `viewer_enabled=true`, training starts a read-only preview and Blacknode
opens it in the simulation pane. Select `viser` for the lightweight interactive
training view, including the reach target, trail, metrics, and environment
selector. Select `ovrtx` for the RTX-rendered USD view after enabling
`blacknode-newton/viewer-ovrtx` in **Packages** and installing its prerequisites.
Both providers copy one selected arm into a separate one-articulation render
model at up to `viewer_fps` (15 by default), so the complete environment batch
remains dedicated to learning. Neither preview contains joint-command or
hardware-arm controls.

When training completes, Blacknode releases the replicated training batch and
keeps the final one-arm preview open. Press **Replay checkpoint** on
`PPOTraining` to run the latest checkpoint deterministically for
`replay_episodes` at visible control speed. Replay uses one simulated arm,
leaves its final frame open, and closes only through **Close viewer**, **Stop
all**, or server shutdown.

`PPOPolicyExport` now records a simulator-neutral
`blacknode.ppo-compatibility-contract`. The exported SO-ARM101 reach policy can
be evaluated in Newton or through Blacknode's Isaac Sim bridge. The contract
preserves ordered joint names, the 21 semantic observation fields, normalized
six-joint delta actions, task scale, and its simulation-only safety state.

For the reverse path, open **SO-ARM101 Isaac PPO Import**. Export the
deterministic Isaac actor as TorchScript with shape `[batch,21] -> [batch,6]`,
connect the matching `SO101ReachTask` environment, and run `PPOPolicyImport`.
The importer validates the tensor interface and creates a normal Blacknode PPO
artifact. Connect that artifact to `PPOPolicyEvaluate` to evaluate it in
Newton. Arbitrary Isaac training checkpoints require an actor export matching
this observation and action contract.

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
artifacts explicitly retain `physical_motion_authorized=false`; deployment uses
a separate qualification-bound authorization and armed safety controller. Treat
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
