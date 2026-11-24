#!/bin/bash
# Stage 3 (control delay): teacher -> student -> Isaac (flat / stress, 0 / 20 / 40 ms) + MuJoCo.
cd "$(dirname "$0")/.."; unset PYTHONPATH; export OMNI_KIT_ACCEPT_EULA=YES
PY=${ISAACLAB:-../../IsaacLab}/.venv/bin/python; mkdir -p logs; R=results/gap_search; L=logs/rsl_rl/go2_crouch_walk
$PY scripts/rl.py train --task Go2-Crouch-Delay --num_envs 4096 --max_iterations 1500 --seed 1 --run_name s3 \
  --checkpoint "$PWD/weights/go2_crouch_teacher_s2.pt" > logs/train_s3.log 2>&1 || { echo train failed; exit 1; }
cp "$(ls $(ls -dt $L/*_s3 | head -1)/model_*.pt | sort -V | tail -1)" weights/go2_crouch_teacher_s3.pt
$PY scripts/distill_crouch.py --task Go2-Crouch-Delay --teacher weights/go2_crouch_teacher_s3.pt --out weights/go2_crouch_student_s3.pt > logs/distill_s3.log 2>&1 || { echo distill failed; exit 1; }
for w in teacher student; do for st in "" --stress; do for d in 0 20 40; do
  n=eval_s3_$w${st:+_stress}_d$d
  $PY scripts/eval_crouch.py --ckpt weights/go2_crouch_${w}_s3.pt $st --head --delay_ms $d --per 48 --out $R/$n.json > logs/$n.log 2>&1 || echo "$n failed"
done; done; done
OMP_NUM_THREADS=1 $PY scripts/mujoco_check.py --ckpt weights/go2_crouch_student_s3.pt --out $R/mujoco_student_s3.json --seeds 5 > logs/mujoco_s3.log 2>&1 || echo mujoco failed
