#!/bin/bash
# After the v1 pilot parse finishes: 8B (prompt v2) on the condition-C sentences, then on the pilot set. CPU unless
# a GPU with >= 18 GB free shows up.
cd "$(dirname "$0")/.."; PY=~/miniconda3/envs/roadpush/bin/python
while pgrep -f "clue.parse --data results/clue/pilot_llm_v1.jsonl" > /dev/null; do sleep 60; done
dev() { f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$f" -gt 18000 ] && echo cuda:0 || echo cpu; }
D=$(dev); env -u PYTHONPATH $PY -m clue.parse --data results/clue/c_sentences.jsonl --device $D --batch 4 \
  --out results/clue/parse_8b_v2_c.jsonl > logs/clue_parse_8b_v2_c.log 2>&1
D=$(dev); env -u PYTHONPATH $PY -m clue.parse --data results/clue/pilot_llm_v1.jsonl --device $D --batch 4 \
  --out results/clue/parse_8b_v2_pilot.jsonl > logs/clue_parse_8b_v2_pilot.log 2>&1
echo done
