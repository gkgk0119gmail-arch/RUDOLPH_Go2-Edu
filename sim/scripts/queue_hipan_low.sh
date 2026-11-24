#!/bin/bash
# HiPAN low-level teacher (PPO, privileged heightmap + domain + motion state).
cd "$(dirname "$0")/.."; unset PYTHONPATH; export OMNI_KIT_ACCEPT_EULA=YES
PY=${ISAACLAB:-../../IsaacLab}/.venv/bin/python; mkdir -p logs
$PY scripts/rl.py train --task Go2-HiPAN-Low --num_envs 4096 --max_iterations ${N:-6000} --seed 1 --run_name teacher \
  > logs/train_hipan_low.log 2>&1 || { echo train failed; exit 1; }
echo "teacher done: $(ls -dt logs/rsl_rl/hipan_low/*_teacher | head -1)"
