#!/bin/bash
# HiPAN high-level teacher (PPO, privileged map, frozen low student inside). N iterations, RUN name.
cd "$(dirname "$0")/.."; unset PYTHONPATH; export OMNI_KIT_ACCEPT_EULA=YES
PY=${ISAACLAB:-../../IsaacLab}/.venv/bin/python; mkdir -p logs
$PY scripts/train_hipan_nav.py --envs ${ENVS:-1024} --iters ${N:-4000} --run ${RUN:-teacher} ${RESUME:+--resume "$RESUME"} \
  > logs/train_hipan_nav.log 2>&1 || { echo train failed; exit 1; }
echo "nav teacher done: $(ls -dt logs/rsl_rl/hipan_nav/*_${RUN:-teacher} | head -1)"
