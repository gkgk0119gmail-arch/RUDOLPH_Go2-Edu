#!/bin/bash
# Fine-tune the HiPAN low teacher after the tracking-reward fix (squared error). CKPT = starting checkpoint.
cd "$(dirname "$0")/.."; unset PYTHONPATH; export OMNI_KIT_ACCEPT_EULA=YES
PY=${ISAACLAB:-../../IsaacLab}/.venv/bin/python; mkdir -p logs
$PY scripts/rl.py train --task Go2-HiPAN-Low-FT --num_envs 4096 --max_iterations ${N:-3000} --seed 2 --run_name teacher_ft \
  --checkpoint "$(realpath "$CKPT")" > logs/train_hipan_low_ft.log 2>&1 || { echo train failed; exit 1; }
echo "teacher ft done: $(ls -dt logs/rsl_rl/hipan_low/*_teacher_ft | head -1)"
