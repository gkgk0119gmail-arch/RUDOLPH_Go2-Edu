#!/bin/bash
# Stage 4 (delay + lifted feet + random terrain): teacher -> student -> Isaac (flat / stress, 0 / 20 / 40 ms) + MuJoCo.
cd "$(dirname "$0")/.."; unset PYTHONPATH; export OMNI_KIT_ACCEPT_EULA=YES
PY=${ISAACLAB:-../../IsaacLab}/.venv/bin/python; mkdir -p logs; R=results/gap_search; L=logs/rsl_rl/go2_crouch_walk
$PY scripts/rl.py train --task Go2-Crouch-Step --num_envs 4096 --max_iterations 2000 --seed 1 --run_name s4 \
  --checkpoint "$PWD/weights/go2_crouch_teacher_s2.pt" > logs/train_s4.log 2>&1 || { echo train failed; exit 1; }
cp "$(ls $(ls -dt $L/*_s4 | head -1)/model_*.pt | sort -V | tail -1)" weights/go2_crouch_teacher_s4.pt
$PY scripts/distill_crouch.py --task Go2-Crouch-Step --teacher weights/go2_crouch_teacher_s4.pt --out weights/go2_crouch_student_s4.pt > logs/distill_s4.log 2>&1 || { echo distill failed; exit 1; }
for w in teacher student; do for st in "" --stress; do for d in 0 20 40; do
  n=eval_s4_$w${st:+_stress}_d$d
  $PY scripts/eval_crouch.py --ckpt weights/go2_crouch_${w}_s4.pt $st --head --delay_ms $d --per 48 --out $R/$n.json > logs/$n.log 2>&1 || echo "$n failed"
done; done; done
OMP_NUM_THREADS=1 $PY scripts/mujoco_check.py --ckpt weights/go2_crouch_student_s4.pt --out $R/mujoco_student_s4.json --seeds 5 > logs/mujoco_s4.log 2>&1 || echo mujoco failed
