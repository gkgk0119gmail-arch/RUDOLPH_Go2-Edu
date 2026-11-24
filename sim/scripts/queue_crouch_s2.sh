#!/bin/bash
# Stage 2 (domain randomisation, lower height limit): continue both stage-1 policies, then flat + stress evaluation.
cd "$(dirname "$0")/.."; unset PYTHONPATH; export OMNI_KIT_ACCEPT_EULA=YES
PY=${ISAACLAB:-../../IsaacLab}/.venv/bin/python; N=${N:-1500}; mkdir -p logs
L=logs/rsl_rl/go2_crouch_walk; R=results/gap_search
for s in scratch warm; do
  i=$(ls -dt $L/*_$s | head -1)/model_1499.pt
  $PY scripts/rl.py train --task Go2-Crouch-Robust --num_envs 4096 --max_iterations $N --seed 1 --run_name s2_$s \
    --checkpoint "$(realpath $i)" > logs/train_s2_$s.log 2>&1 || { echo "s2_$s train failed"; continue; }
  c=$(ls $(ls -dt $L/*_s2_$s | head -1)/model_*.pt | sort -V | tail -1)
  for st in "" --stress; do
    $PY scripts/eval_crouch.py --ckpt $c $st --per 48 --out $R/eval_s2_$s${st:+_stress}.json > logs/eval_s2_$s${st:+_stress}.log 2>&1 || echo "s2_$s eval failed"
  done
  echo "s2_$s done: $c"
done
