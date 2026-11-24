"""Korean testimony templates in the speaker's own frame (8 directions, near / far) and their meaning.
Written by us, not collected from people: good for a pipeline check, not evidence about real speech."""

import math

import numpy as np

# bearing relative to the speaker's facing, clockwise on the map image (y down): 0 front, 90 right, 180 back, 270 left
DIRS = {"front": 0, "front-right": 45, "right": 90, "back-right": 135, "back": 180, "back-left": 225, "left": 270,
        "front-left": 315}
_W = {"front": ["제 앞쪽", "제가 지금 보고 있는 방향", "정면"],
      "front-right": ["오른쪽 앞", "앞에서 약간 오른편"], "right": ["오른쪽", "제 오른편", "오른손 쪽"],
      "back-right": ["오른쪽 뒤", "뒤에서 오른편"], "back": ["제 뒤쪽", "등 뒤", "제가 등지고 있는 쪽"],
      "back-left": ["왼쪽 뒤", "뒤에서 왼편"], "left": ["왼쪽", "제 왼편", "왼손 쪽"],
      "front-left": ["왼쪽 앞", "앞에서 약간 왼편"]}
_D = {"near": ["가까운 방", "바로 옆 방", "얼마 안 떨어진 방"], "far": ["저 멀리 있는 방", "한참 들어간 안쪽 방", "꽤 떨어진 방"]}
_T = ["{d}으로 {r}에 한 명 더 있어요.", "사람이 더 있어요! {d}, {r}이요.", "콜록… {r}인데, {d}이에요. 거기서 소리가 났어요.",
      "친구가 {d} {r}에 갇혀 있어요. 빨리요!", "{d} {r} 쪽에서 살려달라는 소리를 들었어요."]


def make(seed: int, clue_dir: float, far: bool):
    """-> (sentence, facing [rad, map, y down], meaning dict). clue_dir as atan2(dy, dx) with y down."""
    rng = np.random.default_rng([seed, 5])
    facing = float(rng.uniform(-math.pi, math.pi))
    rel = math.degrees(clue_dir - facing) % 360
    name = min(DIRS, key=lambda k: abs((rel - DIRS[k] + 180) % 360 - 180))
    dist = "far" if far else "near"
    pick = lambda a: a[int(rng.integers(len(a)))]  # noqa: E731
    return pick(_T).format(d=pick(_W[name]), r=pick(_D[dist])), facing, {"direction": name, "distance": dist}


def to_map(meaning, facing):
    return facing + math.radians(DIRS[meaning["direction"]]), meaning["distance"] == "far"
