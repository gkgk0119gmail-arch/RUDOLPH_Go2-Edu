"""Merge a clue-parser LoRA adapter into the base Qwen3-VL weights (for GGUF conversion / llama.cpp on the NX).

    python -m clue.merge_lora --adapter OUT/lora_8b_v1 --out OUT/merged_8b_v1
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

m = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(m, a.adapter).merge_and_unload()
m.save_pretrained(a.out, safe_serialization=True)
AutoProcessor.from_pretrained(a.model).save_pretrained(a.out)
print("merged ->", a.out)
