#!/bin/bash
# After teacher training: teacher eval -> depth student DAgger -> student eval (fixed goal / goal switching).
# TEACHER=<run dir>  (waits for model_3999.pt)
cd "$(dirname "$0")/.."; unset PYTHONPATH; export OMNI_KIT_ACCEPT_EULA=YES
PY=${ISAACLAB:-../../IsaacLab}/.venv/bin/python; mkdir -p logs results/hipan_nav weights
T=$TEACHER/model_${LAST:-3999}.pt
while [ ! -f "$T" ] || pgrep -f scripts/train_hipan_nav.py >/dev/null; do sleep 60; done
set -e
$PY scripts/eval_hipan_nav.py --teacher $T --out results/hipan_nav/eval_teacher.json > logs/eval_teacher.log 2>&1
$PY scripts/eval_hipan_nav.py --teacher $T --switch 0.003 --out results/hipan_nav/eval_teacher_switch.json > logs/eval_teacher_switch.log 2>&1
$PY scripts/distill_hipan_nav.py --teacher $T --out weights/go2_hipan_nav_student.pt > logs/distill_hipan_nav.log 2>&1
$PY scripts/eval_hipan_nav.py --teacher $T --student weights/go2_hipan_nav_student.pt --out results/hipan_nav/eval_student.json > logs/eval_student.log 2>&1
$PY scripts/eval_hipan_nav.py --teacher $T --student weights/go2_hipan_nav_student.pt --switch 0.003 --out results/hipan_nav/eval_student_switch.json > logs/eval_student_switch.log 2>&1
echo chain done
