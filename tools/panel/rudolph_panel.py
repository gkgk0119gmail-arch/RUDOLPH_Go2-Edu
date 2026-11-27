#!/usr/bin/env python3
"""RUDOLPH 제어판 (노트북). NX 의 작업을 버튼으로 띄우고 내리고, 단서를 채팅하듯 보낸다.

    python3 tools/panel/rudolph_panel.py                      # NX = unitree@192.168.123.18 (유선)
    RUDOLPH_NX=unitree@<NX wifi IP> python3 tools/panel/rudolph_panel.py

필요한 것: PyQt5 (sudo apt install python3-pyqt5), NX 에 비밀번호 없는 SSH (ssh-copy-id unitree@<NX>).
ROS 는 필요 없다. 모든 것이 SSH 하나(ControlMaster) 위에서 NX 의 tools/panel/remote.sh 를 부른다.

원칙: 제어판은 로봇을 실시간으로 움직이지 않는다. 비상 정지는 언제나 리모컨 L2+B.
"""
import base64
import html
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PyQt5.QtCore import QProcess, Qt, QTimer
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout,
                             QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QSplitter,
                             QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

HOST = os.environ.get("RUDOLPH_NX", "unitree@192.168.123.18")
NX_ROOT = os.environ.get("RUDOLPH_NX_ROOT", "Desktop/3DOG")          # NX 홈 기준
REMOTE = NX_ROOT + "/tools/panel/remote.sh"
REPO = Path(__file__).resolve().parents[2]
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2", "-o", "ControlMaster=auto",
            "-o", "ControlPath=" + os.path.expanduser("~/.ssh/cm-rudolph-%r@%h:%p"), "-o", "ControlPersist=120"]
JOBS = [("sensors", "센서만"), ("slam", "SLAM"), ("low", "하위 보행"), ("nav", "항법"), ("search", "수색")]
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\r")

GREEN, RED, GREY, ORANGE = "#1f9d55", "#d64545", "#8a8f98", "#d97a00"


def dot(color):
    return '<span style="color:%s; font-size:16px">●</span>' % color


class Ssh:
    """One ControlMaster connection; every call is a short ssh over it. Never blocks the UI."""

    def __init__(self, parent):
        self.parent, self.procs, self.closed = parent, set(), False

    def run(self, args, done=None, timeout_ms=60000):
        if self.closed:
            return None
        p = QProcess(self.parent)
        p.setProcessChannelMode(QProcess.MergedChannels)
        self.procs.add(p)

        def fin(*_):
            self.procs.discard(p)
            try:
                out = bytes(p.readAll()).decode("utf-8", "replace")
                code = p.exitCode()
            except RuntimeError:      # QProcess already deleted while closing
                return
            if done and not self.closed:
                done(code, ANSI.sub("", out))
            p.deleteLater()

        p.finished.connect(fin)
        p.start("ssh", SSH_OPTS + [HOST, " ".join(shlex.quote(a) for a in [REMOTE] + list(args))])
        def kill_late():
            try:
                if p.state() != QProcess.NotRunning:
                    p.kill()
            except RuntimeError:
                pass

        QTimer.singleShot(timeout_ms, kill_late)
        return p

    def stream(self, args, on_line, on_end):
        if self.closed:
            return None
        p = QProcess(self.parent)
        p.setProcessChannelMode(QProcess.MergedChannels)
        self.procs.add(p)
        buf = [b""]

        def read():
            try:
                buf[0] += bytes(p.readAll())
            except RuntimeError:
                return
            *lines, buf[0] = buf[0].split(b"\n")
            for ln in lines:
                on_line(ANSI.sub("", ln.decode("utf-8", "replace")))

        def fin(*_):
            self.procs.discard(p)
            if not self.closed:
                on_end()

        p.readyRead.connect(read)
        p.finished.connect(fin)
        p.start("ssh", SSH_OPTS + [HOST, " ".join(shlex.quote(a) for a in [REMOTE] + list(args))])
        return p

    def close(self):
        self.closed = True
        for p in list(self.procs):
            try:
                p.finished.disconnect()
                p.kill()
                p.waitForFinished(500)
            except (RuntimeError, TypeError):
                pass
        self.procs.clear()


class Panel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RUDOLPH 제어판 — " + HOST)
        self.resize(1280, 860)
        self.ssh = Ssh(self)
        self.st = {}
        self.seen_chat = set()
        self.log_job = None

        root = QWidget()
        v = QVBoxLayout(root)
        warn = QLabel("비상 정지 = 리모컨 <b>L2+B</b> · 제어판은 로봇을 직접 움직이지 않습니다 · 서 있는 로봇의 보행은 "
                      "<b>앉히고 종료</b>로만 내립니다")
        warn.setStyleSheet("background:#fff1dc; color:#5a3a00; padding:6px 10px; border-radius:6px")
        v.addWidget(warn)

        chips = QHBoxLayout()
        self.chips = {}
        for k in ["ssh", "go2", "load", "temp", "wifi"]:
            lb = QLabel()
            lb.setStyleSheet("padding:3px 10px; border:1px solid #d0d4da; border-radius:10px")
            chips.addWidget(lb)
            self.chips[k] = lb
        chips.addStretch(1)
        self.note = QLabel("")
        self.note.setStyleSheet("color:#444")
        chips.addWidget(self.note)
        v.addLayout(chips)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.build_jobs())
        split.addWidget(self.build_chat())
        split.setSizes([700, 560])
        vs = QSplitter(Qt.Vertical)
        vs.addWidget(split)
        vs.addWidget(self.build_tabs())
        vs.setSizes([520, 320])
        v.addWidget(vs, 1)
        self.setCentralWidget(root)

        self.set_chip("ssh", GREY, "SSH …")
        self.t_status = QTimer(self, timeout=self.poll_status)
        self.t_status.start(5000)
        self.t_screen = QTimer(self, timeout=self.poll_screen)
        self.t_screen.start(2000)
        QTimer.singleShot(100, self.poll_status)
        QTimer.singleShot(300, self.start_logs)

    # ---------------------------------------------------------------- layout
    def build_jobs(self):
        box = QGroupBox("실행 (NX)")
        g = QGridLayout(box)
        g.setColumnStretch(2, 1)
        self.dots, self.state_lb = {}, {}

        def row(r, key, name):
            d = QLabel(dot(GREY))
            g.addWidget(d, r, 0)
            g.addWidget(QLabel("<b>%s</b>" % name), r, 1)
            s = QLabel("")
            s.setStyleSheet("color:#555")
            g.addWidget(s, r, 2)
            self.dots[key], self.state_lb[key] = d, s
            h = QHBoxLayout()
            g.addLayout(h, r, 3)
            return h

        def btn(h, text, fn, tip=""):
            b = QPushButton(text)
            b.clicked.connect(fn)
            b.setToolTip(tip)
            h.addWidget(b)
            return b

        h = row(0, "sensors", "센서만")
        btn(h, "시작", lambda: self.cmd(["start", "sensors"]), "Hesai + RealSense 848x480 (항법 시야)")
        btn(h, "정지", lambda: self.cmd(["stop", "sensors"]))

        h = row(1, "slam", "SLAM")
        btn(h, "시작", lambda: self.cmd(["start", "slam"]), "FAST-LIO2 + Hesai + RealSense (센서 포함)")
        btn(h, "정지", lambda: self.cmd(["stop", "slam"]))
        btn(h, "맵 저장", lambda: self.cmd(["savemap"]), "maps/map_YYYYmmdd_HHMMSS.pcd")

        h = row(2, "mode", "내장 보행")
        btn(h, "확인", lambda: self.cmd(["mode", "check"]))
        btn(h, "끄기", self.mode_release, "스포츠 모드 해제 — 힘이 풀립니다. 하위 보행 전에")
        btn(h, "다시 켜기", self.mode_restore, "스포츠 모드 복구 (로봇이 누운 상태)")

        h = row(3, "low", "하위 보행")
        self.low_model = QComboBox()
        self.low_model.setEditable(True)
        self.low_model.addItems(["v3c", "old", "v3", "v4c"])
        self.low_model.setToolTip("deploy/weights/hipan_low_student_<이름>.bin (old = 11-26 기존)")
        self.low_src = QComboBox()
        self.low_src.addItem("리모컨", "remote")
        self.low_src.addItem("항법 명령", "topic")
        h.addWidget(self.low_model)
        h.addWidget(self.low_src)
        btn(h, "시작", self.low_start)
        b = btn(h, "앉히고 종료", lambda: self.cmd(["stop", "low"]), "걸어서 서기 → 천천히 앉기 → 종료")
        b.setStyleSheet("font-weight:bold")

        h = row(4, "nav", "항법")
        self.nav_follow = QCheckBox("수색 목표")
        self.nav_follow.setToolTip("켜면 /goal_pose (수색 LIVE·rviz 2D Goal) 를 따라감, 끄면 아래 x y")
        self.nav_follow.setChecked(True)
        self.nav_x, self.nav_y, self.nav_v = QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()
        for w, lo, hi, val, pre in [(self.nav_x, -10, 10, 2, "x "), (self.nav_y, -10, 10, 0, "y "),
                                    (self.nav_v, 0.1, 0.8, 0.3, "v ")]:
            w.setRange(lo, hi)
            w.setSingleStep(0.1 if w is self.nav_v else 0.5)
            w.setValue(val)
            w.setPrefix(pre)
            w.setDecimals(1)
        self.nav_follow.toggled.connect(lambda on: (self.nav_x.setEnabled(not on), self.nav_y.setEnabled(not on)))
        self.nav_x.setEnabled(False)
        self.nav_y.setEnabled(False)
        for w in [self.nav_follow, self.nav_x, self.nav_y, self.nav_v]:
            h.addWidget(w)
        btn(h, "시작", self.nav_start)
        btn(h, "정지", lambda: self.cmd(["stop", "nav"]), "항법만 내림 — 하위 보행은 0.5 초 뒤 제자리 서기")

        h = row(5, "search", "수색")
        self.search_mode = QComboBox()
        self.search_mode.addItem("SHADOW (rviz 만)", "shadow")
        self.search_mode.addItem("LIVE (항법이 걸어감)", "live")
        h.addWidget(self.search_mode)
        btn(h, "시작", self.search_start)
        btn(h, "정지", lambda: self.cmd(["stop", "search"]))

        h = QHBoxLayout()
        g.addLayout(h, 6, 0, 1, 4)
        for text, fn in [("NX 터미널", self.open_term), ("RViz (노트북)", self.open_rviz),
                         ("명령어 다시 읽기", self.load_usage)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            h.addWidget(b)
        h.addStretch(1)
        g.setRowStretch(7, 1)
        return box

    def build_chat(self):
        box = QGroupBox("단서 (로봇 기준 방향: '로봇 오른쪽 5미터')")
        v = QVBoxLayout(box)
        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(False)
        v.addWidget(self.chat, 1)
        h = QHBoxLayout()
        self.chat_in = QLineEdit()
        self.chat_in.setPlaceholderText("들은 단서를 그대로 입력하고 Enter — 예: 복도 끝 왼쪽 방에 두 명 있어요")
        self.chat_in.returnPressed.connect(self.send_clue)
        h.addWidget(self.chat_in, 1)
        b = QPushButton("보내기")
        b.clicked.connect(self.send_clue)
        h.addWidget(b)
        v.addLayout(h)
        b = QPushButton("요구조자 발견 (지금 로봇 위치)")
        b.clicked.connect(self.send_victim)
        v.addWidget(b)
        self.chat_sys("수색(search)이 돌고 있을 때 보낸 단서가 해석되어 목표가 바뀝니다. 해석은 5090 서버(약 2 초).")
        return box

    def build_tabs(self):
        self.tabs = QTabWidget()
        mono = QFont("monospace")
        mono.setStyleHint(QFont.Monospace)
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        h = QHBoxLayout()
        h.addWidget(QLabel("작업 화면 (tmux):"))
        self.screen_job = QComboBox()
        for k, name in JOBS:
            self.screen_job.addItem(name, k)
        self.screen_job.currentIndexChanged.connect(self.poll_screen)
        h.addWidget(self.screen_job)
        h.addStretch(1)
        v.addLayout(h)
        self.screen = QPlainTextEdit(readOnly=True)
        self.screen.setFont(mono)
        v.addWidget(self.screen)
        self.tabs.addTab(w, "화면")
        self.logs = {}
        for k, name in JOBS + [("job", "작업·알림")]:
            t = QPlainTextEdit(readOnly=True)
            t.setFont(mono)
            t.setMaximumBlockCount(3000)
            self.logs[k] = t
            self.tabs.addTab(t, name)
        self.usage = QPlainTextEdit(readOnly=True)
        self.usage.setFont(mono)
        self.tabs.addTab(self.usage, "명령어")
        self.tabs.currentChanged.connect(lambda i: self.tabs.widget(i) is self.usage and not self.usage.toPlainText()
                                         and self.load_usage())
        return self.tabs

    # ---------------------------------------------------------------- helpers
    def set_chip(self, k, color, text):
        self.chips[k].setText("%s %s" % (dot(color), html.escape(text)))

    def say(self, text):
        self.note.setText(time.strftime("%H:%M:%S ") + text.strip().splitlines()[-1] if text.strip() else "")
        self.logs["job"].appendPlainText(time.strftime("[%H:%M:%S] ") + text.rstrip())

    def cmd(self, args, then=None):
        self.say("→ " + " ".join(args))

        def done(code, out):
            self.say(out if out.strip() else ("✓ 끝" if code == 0 else "✗ 실패 (코드 %d)" % code))
            if then:
                then(code, out)
            QTimer.singleShot(800, self.poll_status)

        self.ssh.run(args, done)

    def ask(self, title, text):
        return QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.No) \
            == QMessageBox.Yes

    # ---------------------------------------------------------------- actions
    def mode_release(self):
        if self.ask("내장 보행 끄기", "끄는 순간 다리 힘이 풀립니다.\n\n로봇이 바닥에 누워 있거나 매달려 있나요?"):
            self.cmd(["mode", "release"])

    def mode_restore(self):
        if self.ask("내장 보행 다시 켜기", "하위 보행 노드가 꺼져 있고, 로봇이 누워 있나요?"):
            self.cmd(["mode", "restore"])

    def low_start(self):
        m, s = self.low_model.currentText().strip(), self.low_src.currentData()
        if self.ask("하위 보행 시작 (LIVE)",
                    "모델 %s, 명령: %s\n\n· 내장 보행을 껐나요? (끄기)\n· 로봇이 누워 있고 주변이 비었나요?\n\n"
                    "시작하면 3 초 동안 일어섭니다. 리모컨 start = 보행, L2+A = 서기/앉기, L2+B = 비상 정지."
                    % (m, self.low_src.currentText())):
            self.cmd(["start", "low", m, s])

    def nav_start(self):
        v = "%.1f" % self.nav_v.value()
        if self.st.get("low_src") and self.st.get("low_src") != "topic":
            QMessageBox.information(self, "항법", "하위 보행이 '리모컨' 명령으로 돌고 있습니다.\n"
                                    "항법 명령을 받으려면 하위 보행을 '항법 명령'으로 다시 시작하세요.")
            return
        if self.nav_follow.isChecked():
            args = ["start", "nav", "", "", v]
            what = "수색/rviz 목표(/goal_pose)"
        else:
            x, y = "%.1f" % self.nav_x.value(), "%.1f" % self.nav_y.value()
            args = ["start", "nav", x, y, v]
            what = "앞 %s m, 왼쪽 %s m" % (x, y)
        if self.ask("항법 시작 (LIVE)", "목표: %s, 속도 %s\n\n로봇이 걸어갑니다. 리모컨 스틱을 움직이면 바로 사람이 잡습니다."
                    % (what, v)):
            self.cmd(args)

    def search_start(self):
        md = self.search_mode.currentData()
        if md == "live" and not self.ask("수색 LIVE", "수색이 /goal_pose 를 냅니다. 항법이 돌고 있으면 로봇이 그쪽으로 걸어갑니다."):
            return
        self.cmd(["start", "search", md])

    def send_clue(self):
        text = self.chat_in.text().strip()
        if not text:
            return
        self.chat_in.clear()
        self.chat_me(text)
        b64 = base64.b64encode(text.encode()).decode()
        self.ssh.run(["clue", b64], lambda code, out: self.chat_sys(out.strip() or "✗ 응답 없음", ok=code == 0))

    def send_victim(self):
        self.chat_me("요구조자 발견! (지금 로봇 위치)")
        self.ssh.run(["victim"], lambda code, out: self.chat_sys(out.strip() or "✗ 응답 없음", ok=code == 0))

    def open_term(self):
        term = shutil.which("gnome-terminal")
        cmd = ["ssh", "-t"] + SSH_OPTS[:2] + [HOST, "cd %s; exec bash -l" % NX_ROOT]
        if term:
            subprocess.Popen([term, "--title", "NX " + HOST, "--"] + cmd)
        else:
            QMessageBox.information(self, "NX 터미널", " ".join(cmd))

    def open_rviz(self):
        cfg = REPO / "rviz" / "slam_lio.rviz"
        if not shutil.which("rviz2"):
            QMessageBox.information(self, "RViz", "이 노트북에 rviz2 가 없습니다 (ROS2 source 후 제어판을 다시 여세요).")
            return
        subprocess.Popen(["rviz2", "-d", str(cfg)] if cfg.exists() else ["rviz2"])

    def load_usage(self):
        self.ssh.run(["usage"], lambda code, out: self.usage.setPlainText(out.replace("@@@ ", "\n━━━━ ")))

    # ---------------------------------------------------------------- chat
    def chat_add(self, who, text, align, bg, fg="#111"):
        self.chat.moveCursor(QTextCursor.End)
        self.chat.insertHtml(
            '<table width="100%%" cellspacing="0" cellpadding="0"><tr><td align="%s">'
            '<div style="color:#888; font-size:11px">%s %s</div>'
            '<table cellpadding="8" style="background:%s"><tr><td style="color:%s">%s</td></tr></table>'
            '</td></tr></table><br>' % (align, html.escape(who), time.strftime("%H:%M:%S"), bg, fg,
                                         html.escape(text).replace("\n", "<br>")))
        self.chat.moveCursor(QTextCursor.End)

    def chat_me(self, text):
        self.chat_add("나", text, "right", "#dbe9ff")

    def chat_robot(self, text):
        self.chat_add("RUDOLPH", text, "left", "#f1f2f4")

    def chat_sys(self, text, ok=True):
        self.chat_add("제어판", text, "left", "#ffffff", "#1f6f3f" if ok else "#a12020")

    @staticmethod
    def korean(msg):
        msg = re.sub(r"^heard: (.*) \(parsing\.\.\.\)$", r"들음: \1 (해석 중…)", msg)
        msg = re.sub(r"^clue \(([\d.]+)s\): (.*?) -> ", r"해석 완료 (\1초): ", msg)
        msg = msg.replace("no map yet: clue ignored", "아직 지도가 없어 단서를 버렸습니다")
        msg = msg.replace("parser failed", "해석 서버 실패").replace("clue ignored", "단서 버림")
        return msg

    # ---------------------------------------------------------------- polling
    def poll_status(self):
        def done(code, out):
            if code == 255 or "=" not in out:
                self.set_chip("ssh", RED, "SSH ✗")
                self.note.setText(out.strip().splitlines()[-1] if out.strip() else "NX 연결 안 됨: " + HOST)
                return
            st = dict(ln.split("=", 1) for ln in out.splitlines() if "=" in ln)
            self.st = st
            self.set_chip("ssh", GREEN, "SSH ✓")
            self.set_chip("go2", GREEN if st.get("go2") == "1" else RED, "Go2 " + ("✓" if st.get("go2") == "1" else "✗"))
            self.set_chip("load", GREY, "부하 " + st.get("load", "?"))
            t = int(st.get("temp", "0") or 0)
            self.set_chip("temp", ORANGE if t >= 75 else GREY, "온도 %d°C" % t)
            q = int(st.get("wifi", "0") or 0)
            self.set_chip("wifi", RED if q < 30 else GREY, "Wi-Fi %d/70" % q)
            for k, _ in JOBS:
                on, tm = st.get(k) == "1", st.get(k + "_tmux") == "1"
                self.dots[k].setText(dot(GREEN if on else GREY))
                extra = ""
                if k == "low" and on:
                    extra = "모델 %s · 명령 %s" % (st.get("low_model") or "?",
                                                {"remote": "리모컨", "topic": "항법"}.get(st.get("low_src"), "?"))
                if k == "slam" and on:
                    extra = "FAST-LIO " + ("✓" if st.get("fastlio") == "1" else "…")
                if k == "sensors" and on:
                    extra = "Hesai %s · RealSense %s" % ("✓" if st.get("hesai") == "1" else "…",
                                                        "✓" if st.get("realsense") == "1" else "…")
                if k == "search" and on:
                    extra = "LIVE" if st.get("search_live") == "1" else "SHADOW"
                if on and not tm:
                    extra += " (제어판 밖에서 띄움)"
                self.state_lb[k].setText(("실행 중 · " + extra if extra else "실행 중") if on else "꺼짐")

        self.ssh.run(["status"], done, timeout_ms=8000)

    def poll_screen(self):
        if self.tabs.currentIndex() != 0 or not self.isVisible():
            return
        job = self.screen_job.currentData()

        def done(code, out):
            sb = self.screen.verticalScrollBar()
            at_end = sb.value() >= sb.maximum() - 2
            self.screen.setPlainText(out)
            if at_end:
                sb.setValue(sb.maximum())

        self.ssh.run(["screen", job], done, timeout_ms=5000)

    def start_logs(self):
        def line(ln):
            m = re.match(r"^==> .*/(\w+?)(?:_last)?\.log <==$", ln)
            if m:
                self.log_job = m.group(1)
                return
            if not ln.strip() or self.log_job not in self.logs:
                return
            self.logs[self.log_job].appendPlainText(ln)
            if self.log_job == "search" and "[rudolph_search]:" in ln and ln not in self.seen_chat:
                self.seen_chat.add(ln)
                self.chat_robot(self.korean(ln.split("[rudolph_search]:", 1)[1].strip()))

        def end():
            QTimer.singleShot(3000, self.start_logs)       # NX 재부팅·wifi 끊김: 다시 붙는다

        self.ssh.stream(["logs"], line, end)

    def shutdown(self):
        for t in (self.t_status, self.t_screen):
            t.stop()
        self.ssh.close()

    def closeEvent(self, e):
        self.shutdown()
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Panel()
    app.aboutToQuit.connect(w.shutdown)
    w.show()
    sys.exit(app.exec_())
