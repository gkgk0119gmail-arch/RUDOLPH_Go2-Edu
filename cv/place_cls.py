"""Place classification from the Go2 head camera with pretrained models, no training (first check).

Two label levels:
  struct : corridor | room | lobby | stairs          (what the map region is; the 2D map geometry will be the main
                                                      source later, the camera is a check / fallback)
  func   : sim/clue/schema.py PLACES + lab            (what the place is used for: toilet, office, classroom ...)
Models:
  places365 : ResNet50 trained on Places365 (365 scene classes), summed into our labels with MAP365
  siglip2-b : google/siglip2-base-patch16-224, zero-shot with text prompts (small enough for the NX)
  siglip2-L : google/siglip2-so400m-patch14-384, zero-shot (desktop reference)
The head camera is 16:9 wide and 0.3 m above the floor: each frame is scored as 3 square crops (left, centre, right)
and the probabilities are averaged.

    python -m cv.place_cls --live                         # camera window with the top labels of each model
    python -m cv.place_cls --images a.jpg dir/ clip.mp4   # print labels per image (mp4: --fps frames per second)
    python -m cv.place_cls --clips                        # accuracy on cv/data/baseline clips that name a place:
                                                          #   empty_corridor, empty_lab, lie_3m_none_corridor ...
Output of --clips: cv/results/place/{frames.csv, summary.md}
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "sim"))
from clue.schema import PLACES  # noqa: E402  (entrance lobby stairs elevator toilet kitchen office classroom storage hall)

W365 = ROOT / "weights" / "places365"
STRUCT = ("corridor", "room", "lobby", "stairs")
FUNC = tuple(PLACES) + ("lab",)
# struct of each func label (corridor has no func label of its own)
FUNC_STRUCT = {"entrance": "lobby", "lobby": "lobby", "stairs": "stairs", "elevator": "lobby", "toilet": "room",
               "kitchen": "room", "office": "room", "classroom": "room", "storage": "room", "hall": "lobby",
               "lab": "room"}
# ground truth from a place token in a clip name (empty_<place>, or a memo token like lie_3m_none_corridor)
GT_PLACE = {"corridor": ("corridor", None), "hallway": ("corridor", None), "lab": ("room", "lab"),
            "room": ("room", None), "office": ("room", "office"), "classroom": ("room", "classroom"),
            "storage": ("room", "storage"), "toilet": ("room", "toilet"), "kitchen": ("room", "kitchen"),
            "lobby": ("lobby", "lobby"), "hall": ("lobby", "hall"), "entrance": ("lobby", "entrance"),
            "stairs": ("stairs", "stairs"), "elevator": ("lobby", "elevator")}

# Places365 category -> (struct, func). Unlisted categories count as "other" (not added to any label).
MAP365 = {
    "corridor": ("corridor", None),
    "lobby": ("lobby", "lobby"), "reception": ("lobby", "lobby"), "atrium/public": ("lobby", "lobby"),
    "waiting_room": ("lobby", "lobby"), "elevator_lobby": ("lobby", "elevator"), "elevator/door": ("lobby", "elevator"),
    "elevator_shaft": ("lobby", "elevator"), "entrance_hall": ("lobby", "entrance"), "doorway/outdoor": (None, "entrance"),
    "auditorium": ("lobby", "hall"), "conference_center": ("lobby", "hall"), "banquet_hall": ("lobby", "hall"),
    "ballroom": ("lobby", "hall"), "gymnasium/indoor": ("lobby", "hall"),
    "staircase": ("stairs", "stairs"), "escalator/indoor": ("stairs", "stairs"),
    "office": ("room", "office"), "office_cubicles": ("room", "office"), "home_office": ("room", "office"),
    "conference_room": ("room", "office"), "computer_room": ("room", "office"), "server_room": ("room", "lab"),
    "classroom": ("room", "classroom"), "lecture_room": ("room", "classroom"),
    "kindergarden_classroom": ("room", "classroom"),
    "storage_room": ("room", "storage"), "closet": ("room", "storage"), "utility_room": ("room", "storage"),
    "archive": ("room", "storage"), "pantry": ("room", "kitchen"), "kitchen": ("room", "kitchen"),
    "bathroom": ("room", "toilet"), "shower": ("room", "toilet"), "locker_room": ("room", "toilet"),
    "biology_laboratory": ("room", "lab"), "chemistry_lab": ("room", "lab"), "physics_laboratory": ("room", "lab"),
    "clean_room": ("room", "lab"), "engine_room": ("room", "lab"),
    "living_room": ("room", None), "bedroom": ("room", None), "dining_room": ("room", None),
    "television_room": ("room", None), "recreation_room": ("room", None), "dorm_room": ("room", None),
    "hospital_room": ("room", None), "art_studio": ("room", None), "music_studio": ("room", None),
    "library/indoor": ("room", None), "basement": ("room", "storage"), "garage/indoor": ("room", "storage"),
}

# zero-shot prompts: several phrasings per label, text embeddings averaged
P_STRUCT = {
    "corridor": ["a hallway", "a long corridor inside a building", "a narrow indoor passage with doors on the sides"],
    "room": ["a room", "a small room inside a building", "an enclosed room with furniture"],
    "lobby": ["a lobby", "a large open hall inside a building", "a wide open indoor space"],
    "stairs": ["a staircase", "stairs inside a building"],
}
P_FUNC = {
    "entrance": ["a building entrance", "an entrance door of a building"], "lobby": ["a lobby", "a reception area"],
    "stairs": ["a staircase"], "elevator": ["an elevator", "elevator doors"],
    "toilet": ["a restroom", "a toilet", "a bathroom"], "kitchen": ["a kitchen", "a pantry"],
    "office": ["an office", "an office with desks and chairs"], "classroom": ["a classroom", "a lecture room"],
    "storage": ["a storage room", "a room full of boxes and shelves"], "hall": ["an auditorium", "a large hall"],
    "lab": ["a laboratory", "a research lab with computers and equipment"],
}
TEMPLATES = ["a photo of {}.", "a photo of {} taken from near the floor."]


def crops(img: np.ndarray) -> list[np.ndarray]:
    """3 square crops across the width (left, centre, right), RGB."""
    h, w = img.shape[:2]
    s = min(h, w)
    xs = [0, (w - s) // 2, w - s] if w > s else [0]
    return [cv2.cvtColor(img[:s, x:x + s], cv2.COLOR_BGR2RGB) for x in xs]


class Places365:
    name = "places365"

    def __init__(self, device):
        import torchvision
        self.dev = device
        m = torchvision.models.resnet50(num_classes=365)
        ck = torch.load(W365 / "resnet50_places365.pth.tar", map_location="cpu", weights_only=False)
        m.load_state_dict({k.replace("module.", ""): v for k, v in ck["state_dict"].items()})
        self.m = m.eval().to(device)
        cats = [ln.split()[0][3:] for ln in (W365 / "categories_places365.txt").read_text().splitlines()]
        missing = [k for k in MAP365 if k not in cats]
        if missing:
            print("places365: categories not found:", missing)
        self.cats = cats
        self.A_s = np.zeros((365, len(STRUCT)), np.float32)
        self.A_f = np.zeros((365, len(FUNC)), np.float32)
        for i, c in enumerate(cats):
            s, f = MAP365.get(c, (None, None))
            if s:
                self.A_s[i, STRUCT.index(s)] = 1
            if f:
                self.A_f[i, FUNC.index(f)] = 1
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, img):
        x = np.stack([cv2.resize(c, (224, 224), interpolation=cv2.INTER_AREA) for c in crops(img)])
        x = torch.from_numpy(x).to(self.dev).permute(0, 3, 1, 2).float() / 255
        p = torch.softmax(self.m((x - self.mean) / self.std), 1).mean(0).cpu().numpy()
        top = [(self.cats[i], float(p[i])) for i in np.argsort(-p)[:3]]
        ps, pf = p @ self.A_s, p @ self.A_f
        # renormalise over our labels (the rest of the mass is "other")
        return {"struct": ps / max(ps.sum(), 1e-9), "func": pf / max(pf.sum(), 1e-9), "other": 1 - ps.sum(),
                "raw": top}


class SigLIP:
    def __init__(self, repo, name, device):
        from transformers import AutoModel, AutoProcessor
        self.name, self.dev = name, device
        self.proc = AutoProcessor.from_pretrained(repo)
        self.m = AutoModel.from_pretrained(repo, dtype=torch.float16).eval().to(device)
        self.T_s = self.text(P_STRUCT, STRUCT)
        self.T_f = self.text(P_FUNC, FUNC)

    @torch.no_grad()
    def text(self, prompts, labels):
        out = []
        for lab in labels:
            txt = [t.format(p) for p in prompts[lab] for t in TEMPLATES]
            tok = self.proc(text=txt, padding="max_length", max_length=64, return_tensors="pt").to(self.dev)
            e = self.m.get_text_features(**tok)
            e = getattr(e, "pooler_output", e)
            e = torch.nn.functional.normalize(e.float(), dim=-1).mean(0)
            out.append(torch.nn.functional.normalize(e, dim=0))
        return torch.stack(out)

    @torch.no_grad()
    def __call__(self, img):
        x = self.proc(images=crops(img), return_tensors="pt").to(self.dev)
        e = self.m.get_image_features(pixel_values=x["pixel_values"].half(),
                                      **{k: v for k, v in x.items() if k != "pixel_values"})
        e = getattr(e, "pooler_output", e)
        e = torch.nn.functional.normalize(e.float(), dim=-1)
        scale = self.m.logit_scale.exp().float()
        ps = torch.softmax(scale * e @ self.T_s.T, 1).mean(0).cpu().numpy()
        pf = torch.softmax(scale * e @ self.T_f.T, 1).mean(0).cpu().numpy()
        return {"struct": ps, "func": pf, "other": 0.0, "raw": []}


def load_models(names, device):
    ms = []
    for n in names:
        if n == "places365":
            ms.append(Places365(device))
        elif n == "siglip2-b":
            ms.append(SigLIP("google/siglip2-base-patch16-224", n, device))
        elif n == "siglip2-L":
            ms.append(SigLIP("google/siglip2-so400m-patch14-384", n, device))
        else:
            raise SystemExit(f"unknown model {n}")
    return ms


def top(p, labels, k=2):
    return [(labels[i], float(p[i])) for i in np.argsort(-p)[:k]]


def fmt(r):
    s = " ".join(f"{l} {v:.2f}" for l, v in top(r["struct"], STRUCT))
    f = " ".join(f"{l} {v:.2f}" for l, v in top(r["func"], FUNC))
    return f"[{s}]  [{f}]" + (f"  other {r['other']:.2f}" if r["other"] else "")


def frames_of(path: Path, fps: float):
    if path.is_dir():
        for f in sorted(path.glob("*.jpg")):
            yield f.name, cv2.imread(str(f))
    elif path.suffix == ".mp4":
        cap = cv2.VideoCapture(str(path))
        step, k = max(1, round((cap.get(cv2.CAP_PROP_FPS) or 14.5) / fps)), 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if k % step == 0:
                yield f"{k:05d}", img
            k += 1
    else:
        yield path.name, cv2.imread(str(path))


def gt_of(name: str):
    for tok in re.sub(r"_\d+$", "", name).split("_"):
        if tok in GT_PLACE:
            return GT_PLACE[tok]
    return None


def run_live(models):
    from .go2_cam import Go2Camera
    cam = Go2Camera()
    win = "place  [q] quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)
    last = -1
    try:
        while True:
            last, _, img = cam.read_new(last)
            vis = img.copy()
            for i, m in enumerate(models):
                r = m(img)
                txt = f"{m.name}: " + fmt(r)
                cv2.putText(vis, txt, (15, 40 + 38 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
                cv2.putText(vis, txt, (15, 40 + 38 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow(win, vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


def run_images(models, paths, fps):
    for p in paths:
        for fid, img in frames_of(Path(p), fps):
            print(f"{p} {fid}")
            for m in models:
                r = m(img)
                print(f"   {m.name:10s} {fmt(r)}" + (f"   places365 top: {r['raw']}" if r["raw"] else ""))


def run_clips(models, fps):
    data = ROOT / "data" / "baseline"
    clips = [c for c in sorted(data.iterdir()) if (c.suffix == ".mp4" or c.is_dir()) and not c.name.startswith(".")
             and gt_of(c.stem)] if data.exists() else []
    if not clips:
        raise SystemExit(f"no clip in {data} names a place (empty_corridor, empty_lab, ..._corridor)")
    out = ROOT / "results" / "place"
    out.mkdir(parents=True, exist_ok=True)
    rows, dt = [], defaultdict(list)
    for c in clips:
        gs, gf = gt_of(c.stem)
        for fid, img in frames_of(c, fps):
            for m in models:
                t0 = time.perf_counter()
                r = m(img)
                dt[m.name].append(time.perf_counter() - t0)
                rows.append({"model": m.name, "clip": c.stem, "frame": fid, "gt_struct": gs, "gt_func": gf or "",
                             "struct": STRUCT[int(np.argmax(r["struct"]))], "func": FUNC[int(np.argmax(r["func"]))],
                             **{f"p_{s}": round(float(v), 3) for s, v in zip(STRUCT, r["struct"])}})
        print(f"{c.stem}: done")
    with open(out / "frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    names = [m.name for m in models]
    L = ["# Place classification, pretrained (Go2 head camera)", "",
         "Per-frame accuracy (3 crops averaged). Clip vote = majority of the clip's frames.", "",
         "| model | struct acc | func acc (clips with a func label) | clip vote struct | ms/frame (desktop) |",
         "|---|---|---|---|---|"]
    for n in names:
        R = [r for r in rows if r["model"] == n]
        sa = np.mean([r["struct"] == r["gt_struct"] for r in R])
        F = [r for r in R if r["gt_func"]]
        fa = np.mean([r["func"] == r["gt_func"] for r in F]) if F else float("nan")
        votes = [Counter(r["struct"] for r in R if r["clip"] == c.stem).most_common(1)[0][0] == gt_of(c.stem)[0]
                 for c in clips]
        L.append(f"| {n} | {sa:.2f} | {fa:.2f} | {sum(votes)}/{len(votes)} | {1000 * np.median(dt[n][3:] or dt[n]):.0f} |")
    L += ["", "## confusion (struct, rows = truth)", ""]
    for n in names:
        R = [r for r in rows if r["model"] == n]
        L += [f"**{n}**", "", "| truth \\ pred | " + " | ".join(STRUCT) + " |", "|---|" + "---|" * len(STRUCT)]
        for g in STRUCT:
            G = [r for r in R if r["gt_struct"] == g]
            if G:
                L.append(f"| {g} | " + " | ".join(str(sum(r["struct"] == s for r in G)) for s in STRUCT) + " |")
        L.append("")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[4:4 + 2 + len(names)]))
    print(f"-> {out / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--live", action="store_true")
    g.add_argument("--images", nargs="+")
    g.add_argument("--clips", action="store_true")
    ap.add_argument("--models", nargs="+", default=["places365", "siglip2-b", "siglip2-L"])
    ap.add_argument("--fps", type=float, default=1.0, help="frames per second taken from videos")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    models = load_models(a.models, a.device)
    if a.live:
        run_live(models)
    elif a.images:
        run_images(models, a.images, a.fps)
    else:
        run_clips(models, a.fps)


if __name__ == "__main__":
    main()
