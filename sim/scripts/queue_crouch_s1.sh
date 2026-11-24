#!/bin/bash
# Stage 1 (height): warm start vs from scratch, same iterations / seed, then the same evaluation.
cd "$(dirname "$0")/.."; unset PYTHONPATH; export OMNI_KIT_ACCEPT_EULA=YES
PY=${ISAACLAB:-../../IsaacLab}/.venv/bin/python
INIT=${INIT:-logs/crouch_init.pt}; mkdir -p logs
[ -f "$INIT" ] || $PY scripts/extend_policy.py weights/go2_posture_walk_model_1999.pt "$INIT" --extra 1 --std 0.6; N=${N:-1500}
run() { # name, extra args
  name=$1; shift
  $PY scripts/rl.py train --task Go2-Crouch-Walk --num_envs 4096 --max_iterations $N --seed 1 --run_name $name "$@" > logs/train_$name.log 2>&1 || { echo "$name train failed"; return 1; }
  d=$(ls -dt logs/rsl_rl/go2_crouch_walk/*_$name | head -1); c=$(ls $d/model_*.pt | sort -V | tail -1)
  $PY scripts/eval_crouch.py --ckpt $c --out results/gap_search/eval_$name.json > logs/eval_$name.log 2>&1 || echo "$name eval failed"
  echo "$name done: $c"
}
run warm --checkpoint "$(realpath "$INIT")" --no-resume_optimizer_state
run scratch
