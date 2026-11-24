"""Extend a 50-input posture-walk checkpoint to more inputs: old columns copied, new ones zero (same behaviour at
first), exploration std reset, optimizer state dropped.

    python scripts/extend_policy.py weights/go2_posture_walk_model_1999.pt out.pt --extra 1 --std 0.6
"""
import argparse

import torch


def extend(ck: dict, extra: int, std: float) -> dict:
    out = dict(ck)
    for net in ("actor_state_dict", "critic_state_dict"):
        sd = {k: v.clone() for k, v in ck[net].items()}
        w = sd["mlp.0.weight"]
        sd["mlp.0.weight"] = torch.cat([w, w.new_zeros(w.shape[0], extra)], dim=1)
        if "distribution.std_param" in sd:
            sd["distribution.std_param"] = torch.full_like(sd["distribution.std_param"], std)
        out[net] = sd
    out["optimizer_state_dict"] = {"state": {}, "param_groups": ck["optimizer_state_dict"]["param_groups"]}
    out["iter"] = 0
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--extra", type=int, default=1)
    ap.add_argument("--std", type=float, default=0.6)
    a = ap.parse_args()
    torch.save(extend(torch.load(a.src, map_location="cpu", weights_only=True), a.extra, a.std), a.dst)
