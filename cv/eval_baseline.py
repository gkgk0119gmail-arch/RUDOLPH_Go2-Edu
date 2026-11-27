"""Person-detection baseline of COCO-pretrained YOLO on the recorded clips (cv/data/baseline/*).

    python -m cv.eval_baseline                                   # default models / sizes
    python -m cv.eval_baseline --models yolo11n.pt yolo11x.pt --imgsz 640 1280

Clip-level labels only (from the scenario name), so the metrics are per frame:
    person clips : recall = share of frames with >= 1 person box at conf >= thr
    empty clips  : false-alarm rate = share of frames with >= 1 person box at conf >= thr
A hit in a person clip is not checked against the person's position (a box on a poster would count), so
look at the saved overlays (cv/results/baseline/overlays/<model>/<clip>.jpg) before trusting a number.
Output: cv/results/baseline/{frames.csv, summary.md}.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "baseline"
OUT = ROOT / "results" / "baseline"
THRS = (0.25, 0.5)


def parse(scn: str) -> dict:
    p = scn.split("_")
    if p[0] == "empty":
        return {"pose": "empty", "dist": "-", "occl": "-"}
    return {"pose": p[0], "dist": p[1], "occl": p[2]}


def video_frames(path: Path, fps: float) -> list[np.ndarray]:
    """Frames of an .mp4 at about `fps` (the head camera records ~14.5 fps)."""
    cap = cv2.VideoCapture(str(path))
    src = cap.get(cv2.CAP_PROP_FPS) or 14.5
    step = max(1, round(src / fps))
    out, k = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if k % step == 0:
            out.append(img)
        k += 1
    cap.release()
    return out


def sheet(imgs: list[np.ndarray], n: int = 6, w: int = 480) -> np.ndarray:
    idx = np.linspace(0, len(imgs) - 1, min(n, len(imgs))).astype(int)
    tiles = [cv2.resize(imgs[i], (w, int(imgs[i].shape[0] * w / imgs[i].shape[1]))) for i in idx]
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11x.pt"])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640, 1280])
    ap.add_argument("--device", default="0")
    ap.add_argument("--fps", type=float, default=5.0, help="frames per second taken from .mp4 clips")
    a = ap.parse_args()
    from ultralytics import YOLO

    # clips: frame folders from cv.record (<name>/NNN.jpg + meta.json) or videos from the panel (<name>.mp4, rec.sh)
    clips = sorted(d for d in DATA.iterdir() if (d / "meta.json").exists() or d.suffix == ".mp4") \
        if DATA.exists() else []
    if not clips:
        raise SystemExit(f"no clips in {DATA}; record with the panel (영상 녹화) or python -m cv.record")
    frames = {c.name: (video_frames(c, a.fps) if c.suffix == ".mp4" else sorted(c.glob("*.jpg"))) for c in clips}
    scen = {c.name: (re.sub(r"_\d+$", "", c.stem) if c.suffix == ".mp4"      # lie_3m_none_2 -> lie_3m_none
                     else json.loads((c / "meta.json").read_text())["scenario"]) for c in clips}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in a.models:
        model = YOLO(m)
        for sz in a.imgsz:
            tag = f"{Path(m).stem}@{sz}"
            dt = []
            for c in clips:
                vis = []
                for i, f in enumerate(frames[c.name]):
                    img = cv2.imread(str(f)) if isinstance(f, Path) else f
                    t0 = time.perf_counter()
                    r = model.predict(img, imgsz=sz, classes=[0], conf=0.1, device=a.device, verbose=False)[0]
                    dt.append(time.perf_counter() - t0)
                    confs = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
                    rows.append({"model": tag, "clip": c.name, "scenario": scen[c.name], **parse(scen[c.name]),
                                 "frame": f.name if isinstance(f, Path) else i, "max_conf": float(confs.max()) if confs.size else 0.0,
                                 "n_025": int((confs >= 0.25).sum()), "n_05": int((confs >= 0.5).sum())})
                    vis.append(r.plot(conf=True, line_width=2))
                od = OUT / "overlays" / tag
                od.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(od / f"{c.stem}.jpg"), sheet(vis), [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"{tag}: {len(dt)} frames, desktop GPU {1000 * np.median(dt[5:] or dt):.1f} ms/frame")

    with open(OUT / "frames.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # summary: per model, per scenario group
    tags = list(dict.fromkeys(r["model"] for r in rows))
    lines = ["# Person detection baseline (COCO-pretrained YOLO, Go2 head camera)", "",
             f"clips {len(clips)}, frames {sum(len(v) for v in frames.values())}. "
             "Cells: recall at conf 0.25 / 0.5 (person) or false-alarm rate at conf 0.25 / 0.5 (empty).", ""]

    def rate(sel, key):
        return np.mean([r[key] > 0 for r in sel]) if sel else float("nan")

    for group in ("pose", "dist", "occl", "scenario"):
        vals = sorted({r[group] for r in rows}, key=lambda v: (v == "empty", v))
        lines += [f"## by {group}", "", "| " + group + " | " + " | ".join(tags) + " |",
                  "|---|" + "---|" * len(tags)]
        for v in vals:
            cells = []
            for t in tags:
                sel = [r for r in rows if r["model"] == t and r[group] == v]
                cells.append(f"{rate(sel, 'n_025'):.2f} / {rate(sel, 'n_05'):.2f}")
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
        lines.append("")
    agg = defaultdict(dict)
    for t in tags:
        pos = [r for r in rows if r["model"] == t and r["pose"] != "empty"]
        neg = [r for r in rows if r["model"] == t and r["pose"] == "empty"]
        agg[t] = (rate(pos, "n_025"), rate(pos, "n_05"), rate(neg, "n_025"), rate(neg, "n_05"))
    lines += ["## overall", "", "| model | recall 0.25 | recall 0.5 | false alarm 0.25 | false alarm 0.5 |",
              "|---|---|---|---|---|"]
    lines += [f"| {t} | " + " | ".join(f"{x:.2f}" for x in agg[t]) + " |" for t in tags]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-(len(tags) + 2):]))
    print(f"-> {OUT / 'summary.md'}")


if __name__ == "__main__":
    main()
