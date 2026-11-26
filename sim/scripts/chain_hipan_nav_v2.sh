#!/bin/bash
# v2 (upright reward) teacher: eval + posture -> student DAgger -> student eval + posture. TEACHER=<run dir> LAST=<iter> TAG=v2
cd "$(dirname "$0")/.."; unset PYTHONPATH; export OMNI_KIT_ACCEPT_EULA=YES
PY=${ISAACLAB:-../../IsaacLab}/.venv/bin/python; TAG=${TAG:-v2}; R=results/hipan_nav
ST=weights/go2_hipan_nav_student_$TAG.pt
sleep 120; while pgrep -f scripts/train_hipan_nav.py >/dev/null || pgrep -f "distill_hipan_nav.py --teacher logs/rsl_rl/hipan_nav/2026-11-26_00-40-15" >/dev/null; do sleep 60; done
T=$(ls -v $TEACHER/model_*.pt | tail -1); echo teacher $T
set -e
$PY scripts/eval_hipan_nav.py --teacher $T --out $R/eval_teacher_$TAG.json > logs/eval_teacher_$TAG.log 2>&1
$PY scripts/posture_hipan_nav.py --teacher $T --out $R/posture_teacher_$TAG.json > logs/posture_teacher_$TAG.log 2>&1
$PY scripts/distill_hipan_nav.py --teacher $T --out $ST > logs/distill_hipan_nav_$TAG.log 2>&1
$PY scripts/eval_hipan_nav.py --teacher $T --student $ST --out $R/eval_student_$TAG.json > logs/eval_student_$TAG.log 2>&1
$PY scripts/eval_hipan_nav.py --teacher $T --student $ST --switch 0.003 --out $R/eval_student_switch_$TAG.json > logs/eval_student_switch_$TAG.log 2>&1
$PY scripts/posture_hipan_nav.py --teacher $T --student $ST --out $R/posture_student_$TAG.json > logs/posture_student_$TAG.log 2>&1
echo chain $TAG done
