# blacknode-training Agent Instructions

This is an independent Blacknode extension-package repository. Check and
commit its Git state separately from the containing Blacknode checkout.

## Scope

Own offline robot-policy dataset inspection, PyTorch training, checkpoints,
metrics, prediction-only evaluation, and inference artifact export. Do not own recording, hardware
discovery, ROS transport, or robot motion.

## Rules

- Keep PyTorch and HDF5 imports guarded so package discovery works before
  optional dependencies are installed.
- Dedicated training workflows default to their named local operation. Opening
  a graph is passive; pressing Run starts or resumes the managed job and shows
  live progress with an explicit stop control.
- Training runs in a managed background job with explicit stop and resume.
- Write checkpoints atomically and include dataset schema, normalization
  statistics, model configuration, joint names, and camera names.
- Split training and validation by episode, never by individual frame.
- Compute normalization statistics from training episodes only.
- Prediction nodes never command hardware. A future controller belongs in a
  robot/ROS package and must retain explicit arming and safety limits.
- Tests use synthetic data only and require no robot, camera, ROS, network, or
  Hugging Face access.

## Verification

From the Blacknode root:

```powershell
$env:PYTHONPATH="python"
python -m pytest packages/blacknode-training/tests
blacknode validate packages/blacknode-training/templates/act-training.json
```

## Documentation voice

Describe Blacknode datasets, training jobs, model architecture, checkpoints,
metrics, and policy preview directly. Do not position the package against other
training products or describe speculative integrations in public docs.
