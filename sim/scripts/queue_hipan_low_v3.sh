#!/bin/bash
# HiPAN low v3 (real-robot fixes): teacher fine-tune from teacher_ft2 -> student without foot contacts -> eval.
cd "$(dirname "$0")/.."; unset PYTHONPATH; export OMNI_KIT_ACCEPT_EULA=YES
PY=${ISAACLAB:-../../IsaacLab}/.venv/bin/python; mkdir -p logs results/hipan_low
CKPT=${CKPT:-logs/rsl_rl/hipan_low/2026-11-25_10-44-34_teacher_ft2/model_9998.pt}
$PY scripts/rl.py train --task Go2-HiPAN-Low-V3 --num_envs 4096 --max_iterations ${N:-1500} --seed 4 --run_name teacher_v3 \
  --checkpoint "$(realpath "$CKPT")" > logs/train_hipan_low_v3.log 2>&1 || { echo train failed; exit 1; }
RUN=$(ls -dt logs/rsl_rl/hipan_low/*_teacher_v3 | head -1); T=$(ls -v $RUN/model_*.pt | tail -1); echo "teacher v3: $T"
$PY scripts/eval_hipan_low.py --ckpt $T --out results/hipan_low/eval_v3_teacher_flat.json > logs/eval_v3_teacher_flat.log 2>&1
$PY scripts/eval_hipan_low.py --ckpt $T --rough --out results/hipan_low/eval_v3_teacher_rough.json > logs/eval_v3_teacher_rough.log 2>&1
$PY scripts/distill_hipan_low.py --task Go2-HiPAN-Low-V3 --teacher $T --no_contact --out weights/go2_hipan_low_student_v3.pt \
  > logs/distill_hipan_low_v3.log 2>&1 || { echo distill failed; exit 1; }
for R in "" "--rough"; do for D in 0 20 40; do
  $PY scripts/eval_hipan_low.py --ckpt $T --student weights/go2_hipan_low_student_v3.pt $R --delay_ms $D \
    --out results/hipan_low/eval_v3_student${R:+_rough}_d$D.json > logs/eval_v3_student${R:+_rough}_d$D.log 2>&1
done; done
echo "v3 done"
