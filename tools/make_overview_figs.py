"""RUDOLPH 연구 개요도 + 노드별 설명 그림 + 흐름 움짤.

python3 tools/make_overview_figs.py   ->  docs/images/overview/*.png, overview_flow.gif
내용(수치)은 2026-11-27 기준 README·sim/README·docs/hipan_nav_design.md·docs/rudolph_search.md 에서 가져왔다.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/images/overview"
OUT.mkdir(parents=True, exist_ok=True)
IMG = ROOT / "docs/images"
CLUE = ROOT / "sim/results/clue"

from matplotlib import font_manager as _fm
for _f in ("Regular", "Bold"):
    _p = f"/usr/share/fonts/opentype/noto/NotoSansCJK-{_f}.ttc"
    if Path(_p).exists():
        _fm.fontManager.addfont(_p)
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

W, H = 160, 90          # drawing units (16:9)
DPI = 120               # 16x9 in -> 1920x1080

INK = "#1f2328"
MUTED = "#5b636e"
BG = "#ffffff"
C = {  # node colors
    "parse": "#7b4fc9",
    "percep": "#2f7f8f",
    "search": "#d9730d",
    "nav": "#2563c9",
    "low": "#2e8b47",
    "robot": "#444b55",
}
TINT = {k: v + "1c" for k, v in C.items()}
ST = {"ok": "#2e8b47", "run": "#c98a0d", "todo": "#8a929c"}


def canvas(title=None, color=None, sub=None):
    fig = plt.figure(figsize=(16, 9), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    if title:
        ax.add_patch(FancyBboxPatch((0, 82), W, 8, boxstyle="square,pad=0", fc=color or INK, ec="none"))
        ax.text(4, 86.6, title, color="white", fontsize=24, weight="bold", va="center")
        if sub:
            ax.text(W - 4, 86.6, sub, color="white", fontsize=13, va="center", ha="right", alpha=0.95)
    return fig, ax


def box(ax, x, y, w, h, color, fill=None, lw=2.2, r=1.6, alpha=1.0, z=2):
    from matplotlib.colors import to_rgba
    fc = to_rgba(fill if fill is not None else "white")
    ec = to_rgba(color)
    if alpha < 1:  # dimmed: fade both colors toward white, keep the patch opaque
        fc = tuple(1 - (1 - c) * alpha for c in fc[:3]) + (1,)
        ec = tuple(1 - (1 - c) * alpha for c in ec[:3]) + (1,)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z))


def pill(ax, x, y, text, color, fs=10.5, ha="left", z=5):
    t = ax.text(x, y, text, fontsize=fs, color="white", weight="bold", va="center", ha=ha, zorder=z,
                bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.8", fc=color, ec="none"))
    return t


def arrow(ax, p, q, color=INK, lw=2.0, label=None, lpos=0.5, loff=(0, 1.4), fs=10.5, rad=0.0, ls="-", z=4,
          lcolor=None, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=color, lw=lw, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=0, shrinkB=0))
    if label:
        mx = p[0] + (q[0] - p[0]) * lpos + loff[0]
        my = p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(mx, my, label, fontsize=fs, color=lcolor or color, ha=ha, va="center", zorder=z + 1,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))


def lines(ax, x, y, items, fs=12, dy=None, color=INK, bullet="•", wrap_indent=1.6):
    """items: str | (str, dict(style)) ; '  ' prefix = continuation line (no bullet)."""
    dy = dy or fs * 0.205
    for it in items:
        style = {}
        if isinstance(it, tuple):
            it, style = it
        if it == "":
            y -= dy * 0.5
            continue
        if it.startswith("  "):
            ax.text(x + wrap_indent, y, it.strip(), fontsize=style.get("fs", fs), color=style.get("color", color),
                    va="top", weight=style.get("weight", "normal"))
        elif bullet and not style.get("nobullet"):
            ax.text(x, y, bullet, fontsize=style.get("fs", fs), color=style.get("bcolor", MUTED), va="top")
            ax.text(x + wrap_indent, y, it, fontsize=style.get("fs", fs), color=style.get("color", color), va="top",
                    weight=style.get("weight", "normal"))
        else:
            ax.text(x, y, it, fontsize=style.get("fs", fs), color=style.get("color", color), va="top",
                    weight=style.get("weight", "normal"))
        y -= dy * style.get("dyk", 1.0)
    return y


def heading(ax, x, y, text, color, fs=15):
    ax.text(x, y, text, fontsize=fs, weight="bold", color=color, va="center")
    ax.plot([x, x + 2.2], [y - 2.0, y - 2.0], color=color, lw=3, solid_capstyle="round")


def add_img(fig, path, x, y, w, h, frame=None, caption=None, ax=None, crop=None):
    """Place an image inside the unit box (x, y, w, h), keeping its aspect ratio (centered)."""
    im = Image.open(path)
    if frame is not None:
        n = getattr(im, "n_frames", 1)
        im.seek(max(0, min(frame if frame >= 0 else n + frame, n - 1)))
    im = im.convert("RGB")
    if crop:
        W0, H0 = im.size
        im = im.crop((int(crop[0] * W0), int(crop[1] * H0), int(crop[2] * W0), int(crop[3] * H0)))
    iw, ih = im.size
    # unit box in pixels: 1 unit = 12 px both ways (16 in * 120 / 160)
    bw, bh = w * 12, h * 12
    s = min(bw / iw, bh / ih)
    pw, ph = iw * s / 12, ih * s / 12
    x0, y0 = x + (w - pw) / 2, y + (h - ph) / 2
    iax = fig.add_axes([x0 / W, y0 / H, pw / W, ph / H])
    iax.imshow(im)
    iax.set_xticks([])
    iax.set_yticks([])
    for s_ in iax.spines.values():
        s_.set_color("#c9ced6")
    if caption and ax is not None:
        ax.text(x + w / 2, y0 - 1.0, caption, fontsize=9.5, color=MUTED, ha="center", va="top")
    return (x0, y0, pw, ph)


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("wrote", p)
    return p


# ---------------------------------------------------------------------------------------------------------------
# 0. 전체 개요도
# ---------------------------------------------------------------------------------------------------------------
NODES = {  # key: (x, y, w, h)
    "parse": (34, 60, 36, 18),
    "percep": (34, 33, 36, 22),
    "search": (79, 39, 38, 39),
    "nav": (79, 5, 31, 26),
    "low": (120, 5, 30, 26),
}


def node_box(ax, key, num, name, role, model, learn, run, status, stext, dim=False):
    x, y, w, h = NODES[key]
    a = 0.18 if dim else 1.0
    box(ax, x, y, w, h, C[key], fill=TINT[key], alpha=a)
    tx = x + 1.6
    ty = y + h - 2.6
    ax.text(tx, ty, f"{num} {name}", fontsize=15, weight="bold", color=C[key], va="center", alpha=a, zorder=6)
    ty -= 3.6
    for t, fs, col, wt in [(role, 11.2, INK, "normal")]:
        for ln in t.split("\n"):
            ax.text(tx, ty, ln, fontsize=fs, color=col, va="center", alpha=a, zorder=6, weight=wt)
            ty -= 2.5
    ty -= 0.6
    for lab, val in [("모델", model), ("학습", learn), ("실행", run)]:
        if not val:
            continue
        vs = val.split("\n")
        ax.text(tx, ty, lab, fontsize=10, color=C[key], weight="bold", va="center", alpha=a, zorder=6)
        for i, ln in enumerate(vs):
            ax.text(tx + 4.6, ty, ln, fontsize=10.3, color=INK, va="center", alpha=a, zorder=6)
            ty -= 2.3
    t = pill(ax, x + w - 1.2, y + 2.4, stext, ST[status], fs=9.8, ha="right", z=7)
    t.set_alpha(a)
    t.get_bbox_patch().set_alpha(a)


def inp(ax, x, y, w, h, title, sub, dim=False):
    a = 0.18 if dim else 1.0
    box(ax, x, y, w, h, "#9aa3ad", fill="#f4f6f8", lw=1.4, alpha=a)
    ax.text(x + 1.3, y + h - 2.3, title, fontsize=11.5, weight="bold", color=INK, va="center", alpha=a, zorder=6)
    ax.text(x + 1.3, y + h - 5.0, sub, fontsize=9.6, color=MUTED, va="center", alpha=a, zorder=6)


OVER_STEPS = [
    # (active keys, caption)
    (None, "RUDOLPH 전체 구조: 사람의 제보로 수색 순서를 바꾸고, 낮은 틈은 몸을 낮춰 통과하며 요구조자를 찾는다"),
    ({"in_text", "parse"}, "① 현장에서 만난 사람이 말한다 → VLM이 말을 단서 JSON으로 번역 (번역만, 점수는 계산하지 않음)"),
    ({"in_lidar", "in_cam", "percep"}, "② LiDAR로 3D 지도를 만들고 2D 격자로 요약, 머리 카메라로 사람을 찾는다"),
    ({"parse", "percep", "search"}, "③ 단서 + 지도 → 요구조자 확률 지도 → '어디로 가면 가장 빨리 찾을까' 목표점 선택"),
    ({"in_depth", "search", "nav"}, "④ 목표점만 받고 지도 없이 depth로 장애물을 보며 속도·몸 높이·roll 명령 (10 Hz)"),
    ({"nav", "low", "robot"}, "⑤ 명령대로 몸을 낮추거나 기울이며 걷는 관절 목표를 낸다 (50 Hz)"),
    ({"percep", "search"}, "⑥ 사람을 찾거나 새 제보가 오면 확률 지도를 고치고 목표점을 다시 고른다 (반복)"),
]


def overview(active=None, caption=None, show_side=True):
    fig, ax = canvas()
    on = (lambda k: True) if active is None else (lambda k: k in active)

    ax.text(4, 86.0, "RUDOLPH", fontsize=27, weight="bold", color=INK, va="center")
    ax.text(29.5, 86.0, "언어와 확률적 단서를 따라, 장애물 아래로 몸을 낮춰 요구조자를 찾는 4족 로봇",
            fontsize=16, color=INK, va="center")
    ax.text(4, 81.0, "Unitree Go2 EDU Plus · Hesai XT16 · RealSense D435i · Jetson Orin NX   |   2026-11-27 기준",
            fontsize=11, color=MUTED, va="center")

    # inputs
    ax.text(3, 77.5, "입력", fontsize=12, weight="bold", color=MUTED)
    inp(ax, 3, 63, 25, 11, "사람의 제보", "지금 텍스트, 나중에 음성", dim=not on("in_text"))
    inp(ax, 3, 46, 25, 9, "Hesai XT16 3D LiDAR", "점군 10 Hz, 몸 위 앞쪽", dim=not on("in_lidar"))
    inp(ax, 3, 34, 25, 9, "Go2 머리 RGB 카메라", "1280×720, 14.5 fps", dim=not on("in_cam"))
    inp(ax, 3, 13, 25, 9, "RealSense D435i depth", "848×480, 머리 위, 11.6° 아래", dim=not on("in_depth"))

    node_box(ax, "parse", "①", "제보 해석", "사람 말 → 구조화된 단서 (JSON)",
             "Qwen3-VL-8B", "가중치 학습 없음 (프롬프트 설계)", "연구실 5090 서버, 단서당 1.5 s",
             "ok", "텍스트로 동작", dim=not on("parse"))
    node_box(ax, "percep", "②", "지도 · 사람 탐지", "3D 지도 → 2D 격자 (바닥·장애물·본 곳)\n머리 카메라로 요구조자 발견",
             "FAST-LIO2 + LiveGrid\n사람 검출기 (사전학습)", "없음 (기존 알고리즘·검출기)", "Orin NX",
             "run", "지도 완료 · 탐지 진행 중", dim=not on("percep"))
    node_box(ax, "search", "③", "수색: 목표점 선택",
             "단서 → 요구조자 확률 지도\n시간에 따라 번짐, 봤는데 없으면 깎음\n경계 후보마다 점수 → 다음 목표점",
             "점수 = 기대 요구조자 수\n          / (이동 시간 + 10 s)",
             "2D 수색 시뮬에서 신뢰도 값 학습\n(차분진화, 건물 60개)", "Orin NX, 계획 0.2 s",
             "run", "실로봇 1차 (녹화 재생)", dim=not on("search"))
    node_box(ax, "nav", "④", "상위 항법", "목표점 + depth만 보고 (지도 없음)\n속도·몸 높이·roll 결정",
             "CNN + GRU, 10 Hz", "teacher PPO (특권 지도)\n→ student DAgger (depth)", None,
             "ok", "실로봇 7회 중 4회 도착", dim=not on("nav"))
    node_box(ax, "low", "⑤", "하위 보행", "명령한 속도·높이·roll로\n걷는 관절 목표 12개",
             "MLP, 50 Hz (C++)", "teacher PPO (높이맵)\n→ student 증류 (몸 감각)", None,
             "ok", "실로봇 동작 · v4 학습 중", dim=not on("low"))

    if on("search"):
        add_img(fig, CLUE / "belief_demo.png", 80.5, 42.2, 35, 12.3, crop=(0.0, 0.0, 0.5, 1.0))
    # robot
    ra = 1.0 if on("robot") else 0.18
    box(ax, 154, 5, 4.8, 26, C["robot"], fill="#eceef1", lw=1.6, alpha=ra)
    ax.text(156.4, 18, "Go2 관절 모터", fontsize=10.5, rotation=90, ha="center", va="center", color=INK,
            weight="bold", alpha=ra)

    def A(k):
        return 1.0 if (active is None or k <= active) else 0.15

    def arr(keys, *a, **kw):
        al = A(set(keys))
        col = kw.pop("color", INK)
        if al < 1:
            col = "#d5d9de"
            kw["lcolor"] = "#c2c7cd"
        arrow(ax, *a, color=col, **kw)

    arr({"in_text", "parse"}, (28, 68.5), (34, 68.5))
    arr({"in_lidar", "percep"}, (28, 50.5), (34, 50.5))
    arr({"in_cam", "percep"}, (28, 38.5), (34, 38.5))
    arr({"parse", "search"}, (70, 69), (79, 69), label="단서 JSON", loff=(0, 2.0), color=C["parse"])
    arr({"percep", "search"}, (70, 49), (79, 49), label="2D 격자", loff=(0, 2.0), color=C["percep"])
    arr({"percep", "search"}, (70, 41.5), (79, 41.5), label="발견 위치", loff=(0, -2.0), color=C["percep"])
    arr({"search", "nav"}, (96, 39), (96, 31), label="목표점 (계속 바뀜)", loff=(10.5, 0), color=C["search"])
    arr({"in_depth", "nav"}, (28, 17.5), (79, 17.5), label="depth 64×112 (지도 없이 이것만)", lpos=0.5,
        loff=(0, 2.0), color=C["nav"])
    arr({"nav", "low"}, (110, 22), (120, 22), label="명령 c", loff=(0, 2.2), fs=10, color=C["nav"])
    arr({"low", "nav"}, (120, 13), (110, 13), label="운동 상태", loff=(0, -2.1), fs=9.5, color=C["low"])
    arr({"low", "robot"}, (150, 22), (154, 22), color=C["low"])
    arr({"low", "robot"}, (154, 11), (150, 11), color=C["robot"])

    # side panel: where things run / time scales
    if show_side:
        sx, sy = 121, 39
        box(ax, sx, sy, 37, 39, "#c9ced6", fill="#fafbfc", lw=1.2)
        ax.text(sx + 1.6, sy + 36.4, "한눈에 보기", fontsize=13, weight="bold", color=INK, va="center")
        y = sy + 32.6
        rows = [
            ("무엇이 새로운가", None),
            ("탐색 중 만난 사람의 말로 수색 순서를 계속 바꿈", "-"),
            ("틈은 치우지 않고 몸을 낮춰 통과 (HiPAN식)", "-"),
            ("", None),
            ("주기", None),
            ("③ 수색: 몇 초마다 · 새 단서·발견 때", C["search"]),
            ("④ 상위 항법: 10 Hz", C["nav"]),
            ("⑤ 하위 보행: 50 Hz", C["low"]),
            ("", None),
            ("학습 위치", None),
            ("④⑤ RTX 3090 · Isaac Lab (시뮬)", INK),
            ("①  연구실 5090 (추론만)", INK),
            ("시뮬에서 학습 → 실로봇에 그대로 (zero-shot)", "-"),
        ]
        for t, col in rows:
            if t == "":
                y -= 1.2
                continue
            if col is None:
                ax.text(sx + 1.6, y, t, fontsize=10.5, color=MUTED, weight="bold", va="center")
            else:
                ax.text(sx + 1.6, y, "•", fontsize=10.5, color=MUTED, va="center")
                ax.text(sx + 3.2, y, t, fontsize=10.3, color=INK if col == "-" else col, va="center")
            y -= 2.55

    # loop arrow: detection/new clue -> search (feedback)
    if caption:
        ax.add_patch(FancyBboxPatch((0, 0), W, 0.01, boxstyle="square,pad=0", fc=BG, ec="none"))
    if caption:
        ax.text(W / 2, 1.6, caption, fontsize=14, ha="center", va="center", color=INK, weight="bold",
                bbox=dict(boxstyle="round,pad=0.45", fc="#fff6cc", ec="#e0c25a"))
    return fig


# ---------------------------------------------------------------------------------------------------------------
# 노드별 설명 그림 공통 레이아웃: 왼쪽 = 담당·입출력, 가운데 = 학습 방법, 오른쪽 = 결과·상태
# ---------------------------------------------------------------------------------------------------------------
def detail_frame(key, title, sub):
    fig, ax = canvas(title, C[key], sub)
    for x0, w0 in [(3, 46), (52, 54), (109, 48)]:
        box(ax, x0, 3, w0, 76, "#dfe3e8", fill="#fbfcfd", lw=1.1, r=1.2, z=1)
    heading(ax, 5, 75.5, "담당하는 일", C[key])
    heading(ax, 54, 75.5, "어떻게 학습하나", C[key])
    heading(ax, 111, 75.5, "지금 상태 (2026-11-27)", C[key])
    return fig, ax


def flow_box(ax, x, y, w, h, title, body, color, fill=None, fs=10.5, tfs=11.5):
    box(ax, x, y, w, h, color, fill=fill if fill is not None else "white", lw=1.8, r=1.2, z=3)
    ax.text(x + 1.2, y + h - 2.0, title, fontsize=tfs, weight="bold", color=color, va="center", zorder=6)
    yy = y + h - 4.6
    for ln in body:
        ax.text(x + 1.2, yy, ln, fontsize=fs, color=INK, va="center", zorder=6)
        yy -= fs * 0.2


def io_rows(ax, x, y, rows, key):
    for lab, val in rows:
        ax.text(x, y, lab, fontsize=11, weight="bold", color=C[key], va="top")
        for i, ln in enumerate(val.split("\n")):
            ax.text(x + 7, y, ln, fontsize=11, color=INK, va="top")
            y -= 2.45
        y -= 0.5
    return y


def node1_parser():
    k = "parse"
    fig, ax = detail_frame(k, "① 제보 해석 (VLM)", "사람 말 → 단서 JSON  ·  Qwen3-VL-8B  ·  학습 없음")
    y = lines(ax, 5, 71, [
        "현장에서 만난 사람의 말을 수색 노드가 쓸 수 있는",
        "  구조화된 단서로 번역한다",
        "번역만 한다. 확률 지도·점수 계산은 코드가 맡는다",
        ("  VLM이 직접 점수를 매기면 더 느렸다 (5×7 건물 204 s", {"color": MUTED}),
        ("  vs 말만 해석 113 s, 단서 안 씀 178 s)", {"color": MUTED}),
        "방향은 항상 로봇 기준 (말을 들은 순간의 정면 = 0°)",
    ], fs=11.5)
    y -= 1.5
    y = io_rows(ax, 5, y, [
        ("입력", "제보 문장 (지금 텍스트, 나중에 음성)"),
        ("출력", "utterance (단서/구조요청/무관), anchor,\n방향 8칸, 거리, 경로 토큰 (복도·n번째 방),\n"
                 "인원, 인상착의, 이동 여부, 확신도"),
    ], k)
    # example
    y -= 0.5
    ax.text(5, y, "예시", fontsize=11, weight="bold", color=C[k], va="top")
    box(ax, 5, y - 27.5, 42, 25, "#d8cdee", fill="#f7f3fd", lw=1.2, r=1.0)
    ax.text(6.4, y - 4.0, "“어… 복도 따라가다가 왼쪽 세 번째 방에", fontsize=11, color=INK, va="center")
    ax.text(6.4, y - 6.6, "   두 명 갇혀 있는 것 같아요”", fontsize=11, color=INK, va="center")
    ax.text(26, y - 9.2, "▼", fontsize=12, color=C[k], ha="center", va="center")
    js = ['{ "utterance": "clue",', '  "semantic": ["side:left", "room:3"],', '  "count": 2,',
          '  "mobility": "static",', '  "certainty": "mid" }']
    for i, s in enumerate(js):
        ax.text(6.4, y - 12.2 - i * 2.4, s, fontsize=10.5, family="monospace", color=INK, va="center")
    ax.text(6.4, y - 25.3, "(형식을 보여 주는 예시 문장)", fontsize=9, color=MUTED, va="center")

    # learning
    ax.text(79, 69, "가중치 학습 없음: 프롬프트와 평가 데이터로 개선", fontsize=13, weight="bold", color=INK,
            ha="center", va="center")
    steps = [
        ("1  단서 스키마 설계 (v2)", ["말에서 뽑을 필드와 경로 토큰 문법을 정함", "모르면 null (추측 금지)"]),
        ("2  평가 문장 만들기", ["라벨을 먼저 뽑고 다른 LLM이 구어로 작성: 150문장",
                              "말 더듬기·자기 정정·사투리·구조 요청 섞임"]),
        ("3  프롬프트 반복 개선 → v3f 채택", ["필드별 정확도 + 혼동표로 틀리는 유형 확인",
                                       "8B / 2B, GPU / NX(Q4) 비교"]),
        ("4  수색 성능으로 최종 평가", ["파서가 낸 단서로 2D 수색을 돌려 정답 단서",
                                 "대비 이득을 얼마나 유지하는지 측정"]),
    ]
    yy = 64
    for i, (t, b) in enumerate(steps):
        flow_box(ax, 55, yy - 11, 48, 11, t, b, C[k])
        if i < len(steps) - 1:
            arrow(ax, (79, yy - 11), (79, yy - 13.2), color=C[k], lw=1.6)
        yy -= 13.6
    ax.text(79, 7.5, "이 노드만 학습하지 않는 이유: 말 → 의미 번역은 사전학습 VLM이 이미 잘함,", fontsize=10,
            color=MUTED, ha="center", va="center")
    ax.text(79, 5.2, "연구 기여는 해석 이후의 확률 지도와 수색 판단 쪽 (③)", fontsize=10, color=MUTED, ha="center",
            va="center")

    # status
    y = lines(ax, 111, 71, [
        ("수색 이득 유지율 (의미 단서 “왼쪽 n번째 방”)", {"weight": "bold"}),
        "  Qwen3-VL-8B: 정답 단서 이득의 81 %",
        "  Qwen3-VL-2B: 54 % (절반 가까이 잃음)",
        "",
        ("속도", {"weight": "bold"}),
        "  5090 서버 (NX에서 Wi-Fi): 단서당 1.5–1.7 s",
        "  NX 단독 (llama.cpp Q4): 8.8 s, 정확도 손실 없음",
        "  해석에 20 s가 걸려도 수색 시간 손해는 거의 없음",
        ("  → 빠른 해석보다 정확한 해석이 중요", {"color": C[k]}),
        "",
        ("알려진 약점", {"weight": "bold"}),
        "  머뭇거리는 말은 확신도를 낮게 읽어 이득 감소",
        "  “등지고 있는 쪽”, “오른쪽 + 멀리”에서 자주 틀림",
        "",
        ("다음", {"weight": "bold"}),
        "  실제 사람이 한 말로 시험 (지금은 LLM이 쓴 문장)",
        "  음성 입력, 애매하면 되묻기 (Hi Robot식)",
    ], fs=11.2, bullet=None, wrap_indent=1.2)
    pill(ax, 111.5, 6.5, "실로봇: 텍스트 단서 → 5090 파서 → NX 수색 노드 연결 완료", ST["ok"], fs=10)
    return save(fig, "node1_parser.png")


def node2_percep():
    k = "percep"
    fig, ax = detail_frame(k, "② 지도 · 사람 탐지", "FAST-LIO2 · LiveGrid · 머리 카메라 검출  ·  학습 없음")
    y = lines(ax, 5, 71, [
        "XT16 + Go2 IMU로 3D 지도를 실시간으로 만든다",
        "3D 지도를 0.2 m 2D 격자로 요약해 ③에 넘긴다",
        "  칸마다: 바닥 높이, 장애물, 빈 곳, 본 곳",
        "머리 RGB 카메라로 사람을 찾는다",
        ("  지도에 찍힌 곳 ≠ 수색한 곳: 문 너머로 라이다가", {"color": MUTED}),
        ("  본 방도 카메라로 봐야 ‘찾아본 곳’이 된다", {"color": MUTED}),
    ], fs=11.5)
    y -= 1.5
    y = io_rows(ax, 5, y, [
        ("입력", "XT16 점군 10 Hz, Go2 IMU,\n머리 카메라 1280×720 14.5 fps"),
        ("출력", "/cloud_registered, /Odometry,\n2D 격자, 카메라로 본 칸, 발견 위치"),
    ], k)
    add_img(fig, IMG / "map_20261122_171243.png", 5, 4.5, 42, y - 7, ax=ax,
            caption="직접 매핑한 실내 지도 (FAST-LIO2, 약 23만 점)")

    ax.text(79, 69, "학습하는 노드가 아님: 기존 알고리즘 + 사전학습 검출기", fontsize=13, weight="bold", color=INK,
            ha="center", va="center")
    flow_box(ax, 55, 44, 48, 21, "3D → 2D 격자 (LiveGrid) 규칙", [
        "바닥: 로봇 3 m 안, 몸보다 0.15–1.2 m 아래 점의 10번째 백분위",
        "  (FAST-LIO z가 한 번 걷는 동안 −0.1 → +0.6 m 흐름)",
        "장애물: 바닥 + 0.12–0.5 m 점 3번 이상, 지나간 광선의 25 % 이상",
        "  → 잠깐 지나간 사람은 나중 광선이 지움",
        "빈 칸: 광선이 2번 이상 지나갔거나 바닥이 보인 칸",
        "계산 범위: 본 영역 + 2 m (NX에서 3 ms/스캔)",
    ], C[k], fs=10.2)
    flow_box(ax, 55, 20, 48, 21, "사람 탐지 (진행 중)", [
        "머리 카메라 멀티캐스트 H264 → NX 하드웨어 디코딩 확인",
        "사람 검출 = 요구조자 발견 → 확률 지도에서 그 칸 처리",
        "장소 분류 (방·복도·로비) → 지도 영역에 라벨",
        "  → “왼쪽 세 번째 방” 같은 의미 단서를 실제 지도에 연결",
        "지금은 사람이 토픽/rviz 클릭으로 ‘발견’을 알려 줌",
    ], C[k], fs=10.2)
    flow_box(ax, 55, 6, 48, 11, "카메라 시야 = ‘본 곳’", [
        "지금은 자리표시 값: 시야 90°, 4 m, 검출 확률 0.8",
        "→ 실측 값으로 교체 예정",
    ], C[k], fs=10.2)

    lines(ax, 111, 71, [
        ("완료", {"weight": "bold"}),
        "  XT16·D435i ROS2 드라이버, 센서 TF 보정",
        "  CycloneDDS 대역폭 문제 해결",
        "  FAST-LIO2 실시간 매핑, 녹화 재생으로 격자 확인",
        "",
        ("진행 중", {"weight": "bold"}),
        "  머리 카메라 사람 검출·장소 분류",
        "",
        ("다음", {"weight": "bold"}),
        "  루프 클로저: 출발점 복귀 오차 0.4–0.7 m → 0.3 m 이하",
        "  z 드리프트 보정",
        "  낮은 틈 높이 (천장 여유) 격자 추가",
        "",
        ("센서 역할 분담", {"weight": "bold"}),
        "  XT16: 지도만 · D435i: ④ 항법 입력만",
        "  머리 카메라: 사람·장소 인식만",
        "  (내장 L1 LiDAR는 좌표계가 맞지 않아 사용 안 함)",
    ], fs=11.2, bullet=None, wrap_indent=1.2)
    pill(ax, 111.5, 6.5, "지도 완료 · 사람 탐지 진행 중", ST["run"], fs=10)
    return save(fig, "node2_perception.png")


def node3_search():
    k = "search"
    fig, ax = detail_frame(k, "③ 수색: 목표점 선택", "단서 → 요구조자 확률 지도 → 다음 목표점  ·  신뢰도 값 학습")
    y = lines(ax, 5, 71, [
        "“지금 어디로 가면 가장 빨리 사람을 찾나”를 정한다",
        "단서를 요구조자 확률 지도로 바꾸고 관측으로 갱신",
        "지도 경계(프론티어) 후보마다 점수 → 최고점이 목표",
        "새 단서·발견·도착 때마다 다시 계획",
    ], fs=11.5)
    y -= 1.2
    y = io_rows(ax, 5, y, [
        ("입력", "단서 JSON (①), 2D 격자·본 곳·발견 (②),\n로봇 위치"),
        ("출력", "목표점 /goal_pose → ④ 상위 항법"),
    ], k)
    y -= 0.3
    ax.text(5, y, "확률 지도 만들기", fontsize=11, weight="bold", color=C[k], va="top")
    lines(ax, 5, y - 2.8, [
        "방향 부채꼴(±30°) × 거리 고리(±20 %)",
        "확신도로 균일 분포와 섞음 (high 0.9 / mid 0.6 / low 0.3)",
        "시간에 따라 이웃 칸으로 번짐 (갇힘 0, 이동 중 빠름)",
        "카메라로 봤는데 없음 → 그 칸 × (1 − 검출확률)",
        "의미 단서: 문을 본 방부터 순서 확정 → 그 방에 모음",
    ], fs=10.4)
    add_img(fig, CLUE / "belief_demo.png", 4, 10, 44, 20, ax=ax, crop=(0.0, 0.0, 0.69, 1.0),
            caption="단서 확률 지도: 처음 → 10분 뒤 번짐 (정지/모름/이동) → 복도를 봤는데 없음")

    # learning (center)
    ax.text(79, 69.5, "무엇을 학습하나: 점수식은 고정, 단서 신뢰도 값을 학습", fontsize=12.5, weight="bold",
            color=INK, ha="center", va="center")
    box(ax, 55, 57.5, 48, 9, C[k], fill=TINT[k], lw=1.6, r=1.0)
    ax.text(79, 63.6, "점수 = 기대 요구조자 수 / (이동 시간 + 10 s)", fontsize=12.5, weight="bold", color=C[k],
            ha="center", va="center")
    ax.text(79, 60.0, "기대 수 = 단서 확률 질량 + 단서 없는 남은 인원 × 그쪽 미탐색 면적 비율", fontsize=9.8,
            color=INK, ha="center", va="center")
    flow_box(ax, 55, 40, 48, 15, "1  점수식 선택 (2D 수색 시뮬, 건물 120개)", [
        "계획 문서의 선형식 a·커버리지 + b·확률 − c·거리는",
        "  단서 없을 때도 있을 때도 더 느림 (537 s vs 458 s)",
        "→ ‘시간당 찾을 사람 수’ 비율식으로 교체, 가중치가 필요 없음",
    ], C[k], fs=10.2)
    arrow(ax, (79, 40), (79, 37.8), color=C[k], lw=1.6)
    flow_box(ax, 55, 21.5, 48, 16, "2  제보자별 신뢰도 갱신 (학습 대상)", [
        "단서가 가리킨 곳에 사람이 없으면 같은 제보자의",
        "  나머지 단서도 덜 믿음",
        "값 3개 (처음 신뢰도 등)를 차분진화로 탐색",
        "  학습: 건물 60개 × 6조건 → 시험: 처음 보는 건물 120개",
        "학습값 0.89 / 0.20 / 0.41 ≈ 손으로 정한 값과 같은 성능",
    ], C[k], fs=10.2)
    arrow(ax, (79, 21.5), (79, 19.3), color=C[k], lw=1.6)
    flow_box(ax, 55, 6, 48, 13.3, "3  같은 판단 코드를 실로봇에 (LiveSearch)", [
        "시뮬레이터 Search와 같은 코드에 실제 FAST-LIO 입력",
        "시뮬 검증: 맞는 단서 393 s, 1/3 틀림 393 s (기존 410 / 443)",
        "녹화 재생: 계획 0.22 s, 목표가 복도 끝·갈림길에 찍힘",
    ], C[k], fs=10.2)

    lines(ax, 111, 71, [
        ("2D 수색 시뮬 결과 (6명 중 3명 단서, 건물 120개)", {"weight": "bold"}),
        "  지목된 3명 발견: 단서 없음 367 s",
        "  → 방향·거리 단서 250 s, 의미 단서 144 s (−61 %)",
        "  정확한 위치를 알 때(상한) 138 s와 거의 같음",
        "  단서 틀림 1/3: 제보자별 신뢰도로 이득 유지",
        "",
        ("선행 연구와의 차이", {"weight": "bold"}),
        "  “단서로 빨라진다” 자체는 기존 결과 (COMRES, SLOOP)",
        "  우리: 모르는 건물에서 문을 본 뒤 단서 확률이 바뀜,",
        "  여러 명 + 단서 없는 사람, 확률 번짐, 틀린 단서 대처",
    ], fs=10.8, bullet=None, wrap_indent=1.2)
    add_img(fig, IMG / "rudolph_search_replay.png", 111, 9, 44, 30, ax=ax, crop=(0.33, 0.25, 0.86, 0.68),
            caption="실로봇 녹화 재생: ★ 고른 목표, ◆ 단서 들은 곳, 색 = 단서 확률")
    pill(ax, 111.5, 4.6, "2D 검증 완료 · 실로봇 녹화 재생 확인 · --live 실행 예정", ST["run"], fs=9.8)
    return save(fig, "node3_search.png")


def teacher_student(ax, x, y, key, t_title, t_in, t_how, s_title, s_in, s_how, link):
    """Two stacked boxes teacher -> student."""
    w = 48
    flow_box(ax, x, y + 22, w, 18, t_title, t_in + [""] + t_how, C[key], fill="#ffffff", fs=10.1)
    arrow(ax, (x + w / 2, y + 22), (x + w / 2, y + 18.8), color=C[key], lw=2.0)
    ax.text(x + w / 2 + 1.5, y + 20.4, link, fontsize=10, color=C[key], va="center", weight="bold")
    flow_box(ax, x, y, w, 18.5, s_title, s_in + [""] + s_how, C[key], fill=TINT[key], fs=10.1)


def node4_nav():
    k = "nav"
    fig, ax = detail_frame(k, "④ 상위 항법 (HiPAN식)", "목표점 + depth → 속도·몸 높이·roll  ·  10 Hz  ·  teacher → student")
    y = lines(ax, 5, 71, [
        "③이 준 목표점까지 간다. 지도 없이 depth만 본다",
        "머리 위 장애물을 보면 미리 몸을 낮추고,",
        "  지나가면 다시 일어선다",
        "목표가 도중에 바뀌어도 따라간다 (③이 계속 바꾸므로)",
    ], fs=11.5)
    y -= 1.2
    y = io_rows(ax, 5, y, [
        ("입력", "D435i depth 64×112 (4 m에서 자름),\n몸 기준 목표 방향·거리, ⑤의 운동 상태 추정"),
        ("출력", "c = [vx, vy, ωz, 몸 높이 0.12–0.36 m,\nroll ±0.5 rad] → ⑤ 하위 보행"),
    ], k)
    add_img(fig, IMG / "sim_nav_student_v2_creep.gif", 4, 4, 44, y - 6.5, frame=-30, ax=ax,
            caption="시뮬: 기울어진 판 밑에서 0.19 m로 숙여 통과 (오른쪽 아래 = 입력 depth)")

    teacher_student(
        ax, 55, 22, k,
        "1  teacher: PPO, 특권 정보 사용 (시뮬에서만)",
        ["입력: 로봇 주변 지도 2×41×31 (바닥 장애물 높이,",
         "  머리 위 여유 높이) + 실제 운동 상태"],
        ["Isaac Lab, 12×12 m 방 48개 (문·탁자·낮은 보·기운 판)",
         "PGCL: 기준 경로 위 하위목표 1 m 간격 → 성공하면 늘림",
         "보상: 도착 5, 진행 3, 충돌 −2.5, 트인 곳에선 일어서기"],
        "2  student: DAgger 증류 (실로봇용)",
        ["입력: depth 64×112 + 목표 + 운동 상태 추정",
         "구조: CNN + GRU(256) → c, 그리고 teacher 지도 잠재 추정"],
        ["손실 ‖c − ĉ‖² + 0.1‖z − ẑ‖², 32스텝 BPTT",
         "도중 목표 변경 (스텝마다 0.3 %)으로 ③에 대비",
         "GRU 기억: 바로 밑에선 보가 안 보여도 숙인 채 통과"],
        "teacher 행동을 따라 하도록")
    ax.text(79, 69.5, "보이는 정보가 많은 teacher를 먼저 강화학습으로 만들고,", fontsize=11.5, color=INK,
            ha="center", va="center")
    ax.text(79, 66.9, "로봇이 실제로 가진 센서만 쓰는 student가 따라 하게 한다", fontsize=11.5, color=INK,
            ha="center", va="center")
    flow_box(ax, 55, 5, 48, 14.5, "3  다음 재학습 (실로봇에서 찾은 차이 반영)", [
        "depth: 기울기 11.6°, ±4° 자세, 왼쪽 빈 띠·테두리 빈 선,",
        "  픽셀 누락 5 %, 거리² 노이즈",
        "depth → 관절까지 지연 (예상 100–150 ms, 시뮬은 0)",
        "새 하위 보행 (⑤ v4/v5) 위에서 이어 학습",
    ], C[k], fs=10.1, fill="#ffffff")
    ax.add_patch(FancyBboxPatch((55, 5), 48, 14.5, boxstyle="round,pad=0,rounding_size=1.2", fc="none",
                                ec=C[k], lw=1.8, ls="--", zorder=4))

    lines(ax, 111, 71, [
        ("시뮬 (학습에 안 쓴 지도 480개)", {"weight": "bold"}),
        "  teacher v2: 도착 83 %, 넘어짐 6.5 %",
        "  student v2 (지도 없음): 도착 83 %, 넘어짐 8 %",
        "  도중 목표 변경 시 84 %, 평균 0.84 m/s",
        "  트인 곳에서 숙인 채 걷기 25 % → 9 % (v2 보상)",
        "  실패는 주로 머리 위 장애물 근처에서 넘어짐",
        "",
        ("실로봇 (2026-11-26, 목표 2–3 m)", {"weight": "bold"}),
        "  7회 중 4회 도착, 사용자 정지 2, 넘어짐 1",
        "  머리 위 장애물 앞에서 0.14 m로 숙였다가 2.5 s 뒤 0.30 m",
        "  남은 문제: 명령 속도 0.9–1.0 m/s, 불필요한 roll, 헤맴",
    ], fs=10.8, bullet=None, wrap_indent=1.2)
    add_img(fig, IMG / "real_nav_paths.png", 111, 9, 44, 26, ax=ax,
            caption="실로봇 7회 경로 (색 = student가 낸 몸 높이)")
    pill(ax, 111.5, 4.6, "v2 실로봇 1차 실행 · 다음: 실측 depth·지연 넣어 재학습", ST["ok"], fs=9.8)
    return save(fig, "node4_nav.png")


def node5_low():
    k = "low"
    fig, ax = detail_frame(k, "⑤ 하위 보행 (HiPAN식)", "속도·높이·roll 명령 → 관절 12개  ·  50 Hz C++  ·  teacher → student")
    y = lines(ax, 5, 71, [
        "④의 명령대로 걷는다: 몸 높이 0.12–0.36 m,",
        "  roll, 앞뒤·옆·회전 속도",
        "낮은 틈 아래를 기어가듯 걷는 능력은 여기서 나온다",
        "시뮬에서만 학습, 실로봇에 그대로 올림 (zero-shot)",
    ], fs=11.5)
    y -= 1.2
    y = io_rows(ax, 5, y, [
        ("입력", "몸 감각 50스텝 (관절 각·속도, IMU, 직전 행동)\n+ 명령 c"),
        ("출력", "관절 목표 12개 (PD 제어), 운동 상태 추정\n(속도·높이·roll → ④가 사용)"),
        ("실행", "NX C++ rclcpp 노드 go2_hipan_low"),
    ], k)
    add_img(fig, IMG / "real_nav_creep_depth.gif", 4, 4, 44, y - 6.5, frame=17, ax=ax,
            caption="실로봇: 숙이라는 명령(0.14 m)을 따라 낮춰 걷기")

    ax.text(79, 69.5, "④와 같은 teacher → student 구조, 다만 입력이 다르다", fontsize=12.5, weight="bold",
            color=INK, ha="center", va="center")
    teacher_student(
        ax, 55, 26, k,
        "1  teacher: PPO, 4096대 병렬 (Isaac Lab)",
        ["입력: 몸 감각 + 발밑 높이맵 + 실제 속도 (시뮬에서만)"],
        ["거친 지형 커리큘럼 (계단·경사·구멍), 밀기·질량·게인 무작위",
         "높이 명령 0.30 m부터 오차 < 2.5 cm면 1 cm씩 낮춤",
         "높이 보상 exp(−e²/s), 가중치 0.6 (논문과 다름)"],
        "2  student: 증류 (실로봇용)",
        ["입력: 몸 감각 50스텝만 (높이맵·속도 없음)",
         "MLP → 행동 + 운동 상태 추정"],
        ["제어 지연 0–40 ms 무작위로 학습",
         "v3부터 발 접촉 입력 제거 (실로봇 센서 영점이 흘러감)",
         "teacher와 평지 성능 같음"],
        "행동을 따라 하도록")
    # fine-tune chain
    ax.text(55, 23.5, "3  실로봇에서 본 문제를 하나씩 고치는 미세조정", fontsize=11.5, weight="bold", color=C[k],
            va="center")
    chain = [("ft2", "첫 실로봇\n보행", "ok"), ("v3", "발 끌림·흔들림\n실로봇 시험 (v3c)", "ok"),
             ("v4", "낮은 자세에서\n공중시간 보상 끔", "run"), ("v5", "한 발 오래\n딛기 벌점", "todo")]
    cx = 55
    for i, (n, d, s) in enumerate(chain):
        box(ax, cx, 6, 10.8, 14.5, ST[s], fill="white", lw=1.8, r=1.0, z=3)
        ax.text(cx + 5.4, 17.6, n, fontsize=13, weight="bold", color=ST[s], ha="center", va="center", zorder=6)
        for j, ln in enumerate(d.split("\n")):
            ax.text(cx + 5.4, 13.6 - j * 2.3, ln, fontsize=8.9, color=INK, ha="center", va="center", zorder=6)
        lab = {"ok": "완료", "run": "학습 중", "todo": "다음"}[s]
        ax.text(cx + 5.4, 7.6, lab, fontsize=9, color=ST[s], ha="center", va="center", weight="bold", zorder=6)
        if i < len(chain) - 1:
            arrow(ax, (cx + 10.8, 13.2), (cx + 12.4, 13.2), color=MUTED, lw=1.4)
        cx += 12.4

    lines(ax, 111, 71, [
        ("시뮬 (teacher_ft2, 평지 0.4 m/s)", {"weight": "bold"}),
        "  높이 0.12–0.36 m 추종 오차 ≤ 1 cm",
        "  0.16 m 이하에서 느려짐 (실측상 거의 엎드린 높이)",
        "  student: 평지는 teacher와 같음,",
        "  거친 지형 넘어짐 15–16 % (teacher 2–3 %)",
        "",
        ("실로봇 (2026-11-26, zero-shot)", {"weight": "bold"}),
        "  서기·걷기·옆걸음·회전 성공",
        "  실제 속도 = 명령의 0.70배 (시뮬 0.87배)",
        "  정책의 속도·높이·roll 추정은 정확",
        "  회전·옆걸음 때 발을 끎 → v3",
        "  제자리 회전 때 안쪽 앞발이 80–88 % 딛고 있음 → v5",
        "",
        ("v4 미세조정 이유", {"weight": "bold"}),
        "  v3가 0.12 m 높이 계단에서만 더 넘어짐 (13 vs 4 %)",
        "  → 몸을 낮추면 발 들 공간이 없음: 0.20 m 아래에서",
        "  공중 시간 보상을 서서히 끔 (지금 3090에서 학습 중)",
        "",
        ("다음", {"weight": "bold"}),
        "  v4/v5 → student 재증류 → 같은 코스에서 실로봇 비교",
    ], fs=10.6, bullet=None, wrap_indent=1.2)
    pill(ax, 111.5, 4.6, "실로봇 동작 (ft2 → v3c) · v4 학습 중", ST["ok"], fs=10)
    return save(fig, "node5_low.png")


# ---------------------------------------------------------------------------------------------------------------
# 학습 방식 한 장 요약
# ---------------------------------------------------------------------------------------------------------------
def training_summary():
    fig, ax = canvas("노드별 학습 방식 한눈에", INK, "무엇을 · 어디서 · 무엇으로 배우나  ·  2026-11-27")
    cols = [("노드", 4, 26), ("하는 일", 30, 32), ("학습 방법", 62, 36), ("학습 데이터·환경", 98, 30),
            ("상태", 128, 29)]
    ty = 77
    for t, x, w in cols:
        ax.text(x + 1, ty, t, fontsize=13, weight="bold", color=MUTED, va="center")
    ax.plot([4, 157], [74.6, 74.6], color="#c9ced6", lw=1.2)
    rows = [
        ("parse", "① 제보 해석", ["사람 말 → 단서 JSON"], ["가중치 학습 없음", "프롬프트 설계·개선 (v3f)"],
         ["LLM이 쓴 구어 150문장", "수색 이득 유지율로 평가"], ("ok", "동작 (텍스트)")),
        ("percep", "② 지도·사람 탐지", ["3D 지도 → 2D 격자", "사람·장소 인식"], ["학습 없음", "FAST-LIO2 + 사전학습 검출기"],
         ["실측 매핑·녹화 재생"], ("run", "지도 완료 · 탐지 진행")),
        ("search", "③ 수색: 목표점", ["확률 지도 → 다음 목표점"], ["점수식 고정 (비율식)", "신뢰도 값 3개: 차분진화"],
         ["2D 수색 시뮬", "학습 건물 60 / 시험 120"], ("run", "실로봇 녹화 재생")),
        ("nav", "④ 상위 항법", ["목표 + depth →", "속도·높이·roll"], ["teacher: PPO + PGCL (특권 지도)",
                                                             "student: DAgger (depth + GRU)"],
         ["Isaac Lab 장애물 방 48개", "RTX 3090"], ("ok", "실로봇 4/7 도착")),
        ("low", "⑤ 하위 보행", ["명령 → 관절 12개"], ["teacher: PPO (높이맵, 커리큘럼)",
                                                   "student: 증류 (몸 감각 50스텝)"],
         ["Isaac Lab 거친 지형 4096대", "RTX 3090"], ("ok", "실로봇 동작 · v4 학습 중")),
    ]
    y = 70
    rh = 13.2
    for key, name, does, how, data, (st, stx) in rows:
        box(ax, 4, y - rh + 1.6, 153, rh - 1.2, C[key], fill=TINT[key], lw=1.4, r=1.2)
        ax.add_patch(FancyBboxPatch((4, y - rh + 1.6), 1.4, rh - 1.2, boxstyle="square,pad=0", fc=C[key], ec="none",
                                    zorder=3))
        cy = y - rh / 2 + 1.0
        ax.text(7, cy, name, fontsize=14.5, weight="bold", color=C[key], va="center")
        for x, items in [(31, does), (63, how), (99, data)]:
            n = len(items)
            for i, t in enumerate(items):
                ax.text(x, cy + (n - 1) * 1.45 - i * 2.9, t, fontsize=11.5, color=INK, va="center")
        pill(ax, 129, cy, stx, ST[st], fs=10.5)
        y -= rh
    ax.text(80, 2.8, "④⑤ 공통: 시뮬에서만 볼 수 있는 정보를 쓰는 teacher를 강화학습으로 만들고, "
                     "실로봇 센서만 쓰는 student가 따라 하게 한 뒤 실로봇에 그대로 올린다",
            fontsize=11.5, color=MUTED, ha="center", va="center")
    return save(fig, "training_summary.png")


def flow_gif():
    frames = []
    for act, cap in OVER_STEPS:
        fig = overview(active=act, caption=cap, show_side=act is None)
        fig.canvas.draw()
        im = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
        plt.close(fig)
        frames.append(im.resize((1280, 720), Image.LANCZOS))
    durs = [3500] + [2600] * (len(frames) - 2) + [3000]
    pal = [f.quantize(colors=128, method=Image.MEDIANCUT) for f in frames]
    p = OUT / "overview_flow.gif"
    pal[0].save(p, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True)
    print("wrote", p)


if __name__ == "__main__":
    save(overview(), "overview.png")
    node1_parser()
    node2_percep()
    node3_search()
    node4_nav()
    node5_low()
    training_summary()
    flow_gif()
