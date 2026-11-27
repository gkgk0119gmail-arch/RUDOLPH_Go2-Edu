"""제어판 '영상 녹화' 칸: Go2 내장 머리 카메라 영상을 NX 에서 녹화하고 이름을 붙여 저장한다.

녹화는 NX 의 tools/panel/rec.sh 가 한다 (remote.sh rec ...). 노트북은 버튼과 이름 입력만.
저장 위치: NX ~/Desktop/3DOG/cv/data/baseline/<이름>.mp4 → 데스크톱에서 cv/eval_baseline.py 로 평가.

이름 = 클립 내내 화면에 보이는 상황:
    <자세>_<거리>_<가림>[_메모]    예: lie_3m_none, prone_5m_part_dark
    empty_<장소>[_메모]           사람이 화면에 전혀 없는 장면 (헛검출 측정용)
"""
import re
import time

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QMessageBox, QPushButton)

POSES = [("stand", "서 있음"), ("sit", "앉음"), ("crouch", "쪼그림"), ("lie", "바로 누움"), ("prone", "엎드림"),
         ("side", "옆으로 누움"), ("slump", "웅크리고 쓰러짐")]
DISTS = ["1m", "2m", "3m", "5m", "8m"]
OCCLS = [("none", "안 가려짐"), ("part", "절반쯤 가려짐"), ("heavy", "팔다리·머리만 보임")]
PLACES = ["lab", "corridor", "lobby", "room", "office", "stairs", "toilet", "storage", "clutter"]
RULE = re.compile(r"^((stand|sit|crouch|lie|prone|side|slump)_\d+m_(none|part|heavy)|empty_[a-z0-9]+)(_[a-z0-9]+)*$")
GREEN, RED, GREY, ORANGE = "#1f9d55", "#d64545", "#8a8f98", "#d97a00"


def clean(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]", "", s.strip().lower().replace(" ", "_"))).strip("_")


class RecordBox(QGroupBox):
    def __init__(self, ssh, say):
        super().__init__("영상 녹화 (내장 머리 카메라 → NX cv/data/baseline)")
        self.ssh, self.say = ssh, say
        self.state, self.t0, self.pending_sec = "unknown", 0.0, ""
        g = QGridLayout(self)
        g.setColumnStretch(1, 1)

        # 1줄: 상태 · 시작 · 정지
        h = QHBoxLayout()
        self.lb_state = QLabel()
        h.addWidget(self.lb_state)
        h.addStretch(1)
        self.b_start = QPushButton("● 녹화 시작")
        self.b_start.setStyleSheet("color:%s; font-weight:bold" % RED)
        self.b_start.clicked.connect(self.start)
        self.b_stop = QPushButton("■ 정지")
        self.b_stop.clicked.connect(self.stop)
        h.addWidget(self.b_start)
        h.addWidget(self.b_stop)
        self.lb_saved = QLabel("")
        self.lb_saved.setStyleSheet("color:#555")
        h.addWidget(self.lb_saved)
        g.addLayout(h, 0, 0, 1, 3)

        # 2줄: 제목 직접 입력 · 저장 · 버리기
        g.addWidget(QLabel("제목"), 1, 0)
        self.title = QLineEdit()
        self.title.setPlaceholderText("직접 입력하거나 아래에서 고르세요 — 예: lie_3m_none, empty_corridor")
        self.title.returnPressed.connect(self.save)
        g.addWidget(self.title, 1, 1)
        h = QHBoxLayout()
        self.b_save = QPushButton("저장")
        self.b_save.setStyleSheet("font-weight:bold")
        self.b_save.clicked.connect(self.save)
        self.b_discard = QPushButton("버리기")
        self.b_discard.clicked.connect(self.discard)
        h.addWidget(self.b_save)
        h.addWidget(self.b_discard)
        g.addLayout(h, 1, 2)

        # 3줄: 자세 · 거리 · 가림 · 메모
        g.addWidget(QLabel("선택"), 2, 0)
        h = QHBoxLayout()
        self.pose, self.dist, self.occl = QComboBox(), QComboBox(), QComboBox()
        for k, ko in POSES:
            self.pose.addItem("%s  %s" % (k, ko), k)
        for d in DISTS:
            self.dist.addItem(d, d)
        for k, ko in OCCLS:
            self.occl.addItem("%s  %s" % (k, ko), k)
        self.pose.setCurrentIndex(3)
        self.dist.setCurrentIndex(2)
        self.memo = QLineEdit()
        self.memo.setPlaceholderText("메모 (선택: dark, desk …)")
        self.memo.setMinimumWidth(120)
        for lb, w in [("자세", self.pose), ("거리", self.dist), ("가림", self.occl), ("메모", self.memo)]:
            h.addWidget(QLabel(lb))
            h.addWidget(w)
        h.addStretch(1)
        g.addLayout(h, 2, 1, 1, 2)

        # 4줄: 사람 없는 장면
        h = QHBoxLayout()
        self.empty = QCheckBox("사람 없는 장면 (empty)")
        self.empty.setToolTip("화면 어디에도 사람이 없는 장면 — 헛검출(오경보) 측정용")
        self.place = QComboBox()
        self.place.setEditable(True)
        self.place.addItems(PLACES)
        self.place.setEnabled(False)
        h.addWidget(self.empty)
        h.addWidget(QLabel("장소"))
        h.addWidget(self.place)
        h.addStretch(1)
        g.addLayout(h, 3, 1, 1, 2)

        # 5줄: 최근 저장
        self.lb_recent = QLabel("")
        self.lb_recent.setStyleSheet("color:#555")
        self.lb_recent.setWordWrap(True)
        g.addWidget(self.lb_recent, 4, 0, 1, 3)

        for w in (self.pose, self.dist, self.occl, self.place):
            w.currentIndexChanged.connect(self.compose)
        self.place.editTextChanged.connect(self.compose)
        self.memo.textChanged.connect(self.compose)
        self.empty.toggled.connect(self.on_empty)
        self.compose()

        self.tick = QTimer(self, timeout=self.show_state)
        self.tick.start(500)
        self.set_state("unknown")
        QTimer.singleShot(1500, self.refresh)

    # ------------------------------------------------------------ 제목
    def on_empty(self, on):
        for w in (self.pose, self.dist, self.occl):
            w.setEnabled(not on)
        self.place.setEnabled(on)
        self.compose()

    def compose(self, *_):
        memo = clean(self.memo.text())
        if self.empty.isChecked():
            parts = ["empty", clean(self.place.currentText()) or "place"]
        else:
            parts = [self.pose.currentData(), self.dist.currentData(), self.occl.currentData()]
        self.title.setText("_".join(parts + ([memo] if memo else [])))

    # ------------------------------------------------------------ 상태
    def set_state(self, st, sec=None):
        self.state = st
        if st == "rec":
            self.t0 = time.time() - float(sec or 0)
        self.b_start.setEnabled(st in ("idle", "unknown"))
        self.b_stop.setEnabled(st in ("rec", "unknown"))
        self.b_save.setEnabled(st in ("pending", "unknown"))
        self.b_discard.setEnabled(st in ("pending", "unknown"))
        self.show_state()

    def show_state(self):
        st = self.state
        if st == "rec":
            s = int(time.time() - self.t0)
            txt, col = "녹화 중  %02d:%02d" % (s // 60, s % 60), RED
        elif st == "pending":
            txt, col = "정지됨 (%s 초) — 제목 확인 후 저장" % self.pending_sec, ORANGE
        elif st == "idle":
            txt, col = "대기", GREY
        else:
            txt, col = "상태 확인 중 …", GREY
        self.lb_state.setText('<span style="color:%s; font-size:16px">●</span> <b>%s</b>' % (col, txt))

    def refresh(self):
        def done(code, out):
            st = dict(ln.split("=", 1) for ln in out.splitlines() if "=" in ln)
            if "rec" not in st:
                self.set_state("unknown")
                return
            self.pending_sec = st.get("pending_sec", "")
            if st["rec"] == "1":
                self.set_state("rec", st.get("rec_sec", 0))
            elif st.get("pending") == "1":
                self.set_state("pending")
            else:
                self.set_state("idle")
            self.lb_saved.setText("저장 %s개" % st.get("saved", "?"))

        self.ssh.run(["rec", "status"], done, timeout_ms=8000)
        self.ssh.run(["rec", "list"], lambda code, out: self.lb_recent.setText(
            "최근: " + " · ".join("%s %s s" % tuple((ln.split() + ["?"])[:2]) for ln in out.split("\n")[:6] if ln.strip())
            if out.strip() and code == 0 else ""), timeout_ms=8000)

    # ------------------------------------------------------------ 버튼
    def run(self, args, then=None):
        self.say("→ rec " + " ".join(args))
        for b in (self.b_start, self.b_stop, self.b_save, self.b_discard):
            b.setEnabled(False)

        def done(code, out):
            self.say(out if out.strip() else ("✓ 끝" if code == 0 else "✗ 실패 (코드 %d)" % code))
            if then:
                then(code, out)
            self.refresh()

        self.ssh.run(["rec"] + args, done, timeout_ms=20000)

    def start(self):
        self.run(["start"])

    def stop(self):
        self.run(["stop"])

    def save(self):
        if self.state != "pending":
            if self.state == "rec":
                QMessageBox.information(self, "저장", "녹화 중입니다. 먼저 ■ 정지를 누르세요.")
            return
        name = clean(self.title.text())
        if not name:
            QMessageBox.warning(self, "저장", "제목이 비었습니다.")
            return
        if name != self.title.text().strip():
            self.title.setText(name)
        if not RULE.match(name) and QMessageBox.question(
                self, "제목 규칙", "'%s' 는 이름 규칙과 다릅니다 (자세_거리_가림 또는 empty_장소).\n"
                "평가 표에서 자세·거리·가림으로 묶이지 않습니다. 그대로 저장할까요?" % name,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.run(["save", name])

    def discard(self):
        if QMessageBox.question(self, "버리기", "방금 녹화한 영상을 지울까요?", QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) == QMessageBox.Yes:
            self.run(["discard"])

    def shutdown(self):
        self.tick.stop()
