# blacknode-training

`blacknode-training` provides offline robot-policy dataset checks, managed PyTorch training, resumable checkpoints, recorded-frame evaluation, and inference artifact export.

## Components

| Component | Purpose |
|---|---|
| `dataset-check` | Validate episode schema and readiness |
| `training-jobs` | Start, monitor, stop, and resume ACT training |
| `checkpoints` | Inspect checkpoint schema, metrics, and configuration |
| `policy-preview` | Preview and replay predictions on recorded episodes |
| `policy-artifacts` | Export and load inference artifacts |

All components are optional so normal robot runtimes do not install the training stack.

## Workflow

1. Record episodes with `blacknode-dataset`.
2. Open `act-training.json` and select the dataset.
3. Start or resume `ACTTraining`; the upstream HDF5 exporter prepares the training view.
4. Monitor train/validation loss and stop cooperatively when needed.
5. Export a checkpoint with `ACTPolicyExport`.
6. Evaluate it with `ACTPolicyReplay` before using a separately armed controller.

Training splits by episode and computes normalization from training episodes only. Checkpoints include model and optimizer state, ordered joints/cameras, dataset schema, normalization, configuration, and metrics. Exported policy artifacts omit optimizer state and retain the deployment contract.

## Safety

This package performs offline training and recorded-frame prediction only. It never commands hardware. Treat checkpoints as trusted executable data and review predictions before connecting an artifact to a motion controller.

## Install and verify

```powershell
blacknode packages install https://github.com/temiroff/blacknode-training.git
$env:PYTHONPATH="python"
python -m pytest packages/blacknode-training/tests
blacknode validate packages/blacknode-training/templates/act-training.json
```

See [AGENTS.md](AGENTS.md) for dataset, checkpoint, and managed-job rules.
