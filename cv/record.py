"""Record labelled clips from the Go2 head camera for the person-detection baseline.

    python -m cv.record                       # live window with YOLO boxes; keys below
    python -m cv.record --no-det              # live window only

Keys in the window:
    r   record a clip (asks the scenario name in the terminal, 3 s countdown, then --sec seconds at --fps)
    q   quit

Scenario name = what is in view for the WHOLE clip (the eval counts every frame as "person present" / "empty"):
    <pose>_<dist>_<occl>[_<note>]
        pose : stand | sit | crouch | lie (on back) | prone (face down) | side (lying on side) | slump (collapsed, curled)
        dist : 1m | 2m | 3m | 5m | 8m          (camera to person)
        occl : none | part (about half hidden by desk / chair / box) | heavy (only a limb or the head visible)
    empty_<place>[_<note>]                     (no person anywhere in the image, for false alarms)
    e.g.  lie_3m_none, prone_5m_part_dark, empty_corridor, empty_lab_clutter
Frames go to cv/data/baseline/<scenario>[_k]/NNN.jpg (+ meta.json), raw camera images without overlays.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2

from .go2_cam import Go2Camera

ROOT = Path(__file__).resolve().parent
NAME_RE = re.compile(r"^((stand|sit|crouch|lie|prone|side|slump)_\d+m_(none|part|heavy)|empty_[a-z0-9]+)(_[a-z0-9]+)*$")


def out_dir(name: str) -> Path:
    base = ROOT / "data" / "baseline"
    d, k = base / name, 1
    while d.exists():
        k += 1
        d = base / f"{name}_{k}"
    d.mkdir(parents=True)
    return d


def draw(img, res):
    vis = img.copy()
    if res is not None:
        for b in res.boxes:
            if int(b.cls) != 0:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"person {float(b.conf):.2f}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=float, default=10.0)
    ap.add_argument("--fps", type=float, default=5.0, help="saved frames per second (camera is ~14.5)")
    ap.add_argument("--model", default=str(ROOT / "weights" / "yolo11s.pt"), help="live overlay only, not used for the saved frames")
    ap.add_argument("--no-det", action="store_true")
    ap.add_argument("--iface", default=None)
    a = ap.parse_args()

    model = None
    if not a.no_det:
        from ultralytics import YOLO
        model = YOLO(a.model)
    cam = Go2Camera(a.iface)
    win = "go2 head camera  [r] record  [q] quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)
    last = -1
    rec, rec_dir, rec_t0, rec_next, rec_k, countdown_end = False, None, 0.0, 0.0, 0, 0.0
    try:
        while True:
            last, t, img = cam.read_new(last)
            now = time.time()
            if rec and now >= countdown_end:
                if now >= rec_next:
                    cv2.imwrite(str(rec_dir / f"{rec_k:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    rec_k += 1
                    rec_next += 1.0 / a.fps
                if now - countdown_end >= a.sec:
                    rec = False
                    meta = json.loads((rec_dir / "meta.json").read_text())
                    meta.update(frames=rec_k, sec=a.sec, fps=a.fps)
                    (rec_dir / "meta.json").write_text(json.dumps(meta, indent=1))
                    print(f"  saved {rec_k} frames -> {rec_dir.relative_to(ROOT)}")
            res = model.predict(img, classes=[0], conf=0.25, verbose=False)[0] if model else None
            vis = draw(img, res)
            if rec:
                left = countdown_end - now
                txt = f"get ready {left:.0f}" if left > 0 else f"REC {rec_dir.name} {now - countdown_end:.0f}/{a.sec:.0f}s"
                cv2.putText(vis, txt, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and not rec:
                name = input("scenario (e.g. lie_3m_part, empty_corridor; blank = cancel): ").strip().lower()
                if not name:
                    continue
                if not NAME_RE.match(name):
                    print("  name does not follow <pose>_<dist>m_<occl>[_note] or empty_<place>; see --help")
                    continue
                rec_dir = out_dir(name)
                (rec_dir / "meta.json").write_text(json.dumps({"scenario": name, "start": time.time(),
                                                               "iface": cam.iface}, indent=1))
                rec, rec_k = True, 0
                countdown_end = time.time() + 3.0
                rec_next = countdown_end
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
