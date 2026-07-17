# blacknode-training Agent Instructions

This is an independent Blacknode extension-package repository. Check and
commit its Git state separately from the containing Blacknode checkout.

## Scope

Own offline robot-policy dataset inspection, PyTorch training, checkpoints,
metrics, and prediction-only evaluation. Do not own recording, hardware
discovery, ROS transport, or robot motion.

## Rules

- Keep PyTorch and HDF5 imports guarded so package discovery works before
  optional dependencies are installed.
- Training actions default to `check` or `status`; a graph cook must not start
  an expensive job implicitly.
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
