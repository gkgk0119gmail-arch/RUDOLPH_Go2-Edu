#!/bin/bash
# Gait fine-tune of the HiPAN low teacher (feet air time + feet slide terms). CKPT = starting checkpoint.
cd "$(dirname "$0")/.."; unset PYTHONPATH; export OMNI_KIT_ACCEPT_EULA=YES
PY=${ISAACLAB:-../../IsaacLab}/.venv/bin/python; mkdir -p logs
$PY scripts/rl.py train --task Go2-HiPAN-Low-Gait --num_envs 4096 --max_iterations ${N:-1500} --seed 3 --run_name ${RUN:-teacher_gait} \
  --checkpoint "$(realpath "$CKPT")" > logs/train_hipan_low_gait.log 2>&1 || { echo train failed; exit 1; }
echo "gait teacher done: $(ls -dt logs/rsl_rl/hipan_low/*_${RUN:-teacher_gait} | head -1)"
