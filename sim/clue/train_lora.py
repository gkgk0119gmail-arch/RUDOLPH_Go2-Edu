"""LoRA fine-tune of the clue parser (Qwen3-VL text side) on synthetic sentence -> compact JSON pairs.

    python -m clue.train_lora --data results/clue/train_v1.jsonl --out ~/3DDOG/rudolph_jobs/out/lora_8b_v1
Target = only the fields that were said (gen_train.compact), prompt = PROMPTS["ft1"]. Loss on the reply tokens only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

from .prompts import PROMPTS

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=2.0)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--val", type=int, default=200, help="held-out synthetic rows for a loss check")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

random.seed(a.seed)
torch.manual_seed(a.seed)
proc = AutoProcessor.from_pretrained(a.model)
tok = proc.tokenizer
model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda:0")
model.config.use_cache = False
cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                 target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()


def encode(row):
    user = [{"role": "user", "content": [{"type": "text", "text": PROMPTS["ft1"].format(u=row["clue"])}]}]
    reply = json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))
    p = proc.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
    ids_p = tok(p, add_special_tokens=False)["input_ids"]
    ids = tok(p + reply, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]
    return ids, [-100] * len(ids_p) + ids[len(ids_p):]


rows = [json.loads(l) for l in open(a.data)]
random.shuffle(rows)
val, train = rows[: a.val], rows[a.val:]
enc_tr, enc_va = [encode(r) for r in train], [encode(r) for r in val]
print(f"train {len(enc_tr)}  val {len(enc_va)}  mean tokens {sum(len(i) for i, _ in enc_tr) / len(enc_tr):.0f}", flush=True)


def batches(enc, bs, shuffle):
    idx = list(range(len(enc)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        ch = [enc[j] for j in idx[i:i + bs]]
        n = max(len(x) for x, _ in ch)
        pad = tok.pad_token_id or 0
        ids = torch.tensor([x + [pad] * (n - len(x)) for x, _ in ch])
        lab = torch.tensor([y + [-100] * (n - len(y)) for _, y in ch])
        att = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x, _ in ch])
        yield ids.cuda(), lab.cuda(), att.cuda()


@torch.no_grad()
def val_loss():
    model.eval()
    tot, n = 0.0, 0
    for ids, lab, att in batches(enc_va, a.bs, False):
        tot += model(input_ids=ids, attention_mask=att, labels=lab).loss.item()
        n += 1
    model.train()
    return tot / max(n, 1)


steps = math.ceil(len(enc_tr) / a.bs * a.epochs)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
warm = max(steps // 20, 10)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
print(f"steps {steps}  val loss before {val_loss():.4f}", flush=True)
model.train()
step, t0 = 0, time.time()
while step < steps:
    for ids, lab, att in batches(enc_tr, a.bs, True):
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 25 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / step:.2f}s/step mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        if step % 200 == 0 or step == steps:
            print(f"step {step} val loss {val_loss():.4f}", flush=True)
        if step >= steps:
            break
model.save_pretrained(a.out)
json.dump(vars(a), open(os.path.join(a.out, "train_args.json"), "w"), indent=1)
print("saved", a.out, flush=True)
