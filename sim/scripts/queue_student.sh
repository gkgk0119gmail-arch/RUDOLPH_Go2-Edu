#!/bin/bash
cd "$(dirname "$0")/.."; unset PYTHONPATH; export OMNI_KIT_ACCEPT_EULA=YES
PY=${ISAACLAB:-../../IsaacLab}/.venv/bin/python; mkdir -p logs
$PY scripts/distill_crouch.py --teacher weights/go2_crouch_teacher_s2.pt --out weights/go2_crouch_student.pt > logs/distill.log 2>&1 || { echo distill failed; exit 1; }
for st in "" --stress; do
  $PY scripts/eval_crouch.py --ckpt weights/go2_crouch_student.pt $st --per 48 --out results/gap_search/eval_student${st:+_stress}.json > logs/eval_student${st:+_stress}.log 2>&1 || echo "eval failed $st"
done
