#!/usr/bin/env bash
# ============================================================================
#  rec.sh — Go2 내장 머리 카메라 영상 녹화 (제어판 '영상 녹화' 칸이 remote.sh rec ... 로 부른다)
#
#     rec.sh start            녹화 시작 (임시 파일 cv/data/baseline/.rec/current.mp4)
#     rec.sh stop             녹화 정지 (mp4 를 정상 마무리) → 이름을 정해 save 하거나 discard
#     rec.sh save <이름>      임시 영상을 cv/data/baseline/<이름>.mp4 로 (같은 이름이 있으면 _2, _3 …)
#     rec.sh discard          임시 영상 버리기
#     rec.sh status           rec=0|1 rec_sec= pending=0|1 pending_sec= saved=<개수>
#     rec.sh list             최근 저장한 영상 15개 (이름 · 길이)
#
#  카메라: 메인보드가 230.1.1.1:1720 으로 보내는 RTP/H264 (1280x720, ~14.5 fps).
#  디코딩 없이 그대로 mp4 에 담는다 → NX CPU 약 1 %, 코어 0-3·nice 10 (보행 노드와 안 겹치게).
#  이름 규칙 (person-detection 기준 성능, cv/eval_baseline.py):
#     <자세>_<거리>_<가림>[_메모]   자세 stand|sit|crouch|lie|prone|side|slump, 거리 1m|3m|5m|8m, 가림 none|part|heavy
#     empty_<장소>[_메모]          사람이 화면에 전혀 없는 장면
# ============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$ROOT/cv/data/baseline"
TMP="$DIR/.rec"
PIDF="$TMP/pid"; CUR="$TMP/current.mp4"; T0F="$TMP/t0"; LOG="$TMP/gst.log"
mkdir -p "$TMP"

alive () { [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
secs () {   # mp4 길이 (초)
    gst-discoverer-1.0 "$1" 2>/dev/null | awk '/Duration/ {split($2, a, ":"); printf "%.1f", a[1]*3600 + a[2]*60 + a[3]; exit}'
}
iface () { ip -4 -o addr | awk '/ 192\.168\.123\./ {print $2; exit}'; }

case "${1:-}" in
start)
    alive && { echo "✗ 이미 녹화 중입니다"; exit 1; }
    [ -f "$CUR" ] && { echo "✗ 저장 안 한 영상이 있습니다 — 저장하거나 버린 뒤 다시 시작하세요"; exit 1; }
    ifc="$(iface)"; ifc="${ifc:-eth0}"
    setsid nohup taskset -c 0-3 nice -n 10 gst-launch-1.0 -e -q \
        udpsrc address=230.1.1.1 port=1720 multicast-iface="$ifc" \
        ! application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,payload=96 \
        ! rtph264depay ! h264parse ! mp4mux ! filesink location="$CUR" > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    date +%s.%N > "$T0F"
    # mp4 는 첫 키프레임(IDR)부터 써진다. 키프레임 간격만큼 비어 있을 수 있어 최대 6 초 기다린다
    for i in $(seq 1 30); do
        { [ -s "$CUR" ] || ! alive; } && break
        sleep 0.2
    done
    if ! alive; then
        echo "✗ 녹화 프로그램이 바로 꺼졌습니다:"; tail -3 "$LOG"; rm -f "$CUR"; exit 1
    fi
    if [ ! -s "$CUR" ]; then
        kill -INT "$(cat "$PIDF")" 2>/dev/null; sleep 1; kill -9 "$(cat "$PIDF")" 2>/dev/null; rm -f "$CUR" "$PIDF"
        echo "✗ 카메라 영상이 안 들어옵니다 (Go2 본체가 켜져 있나요? 유선 $ifc)"; exit 1
    fi
    echo "✓ 녹화 시작 ($ifc)"
    ;;
stop)
    if ! alive; then
        [ -f "$CUR" ] && { echo "✓ 이미 정지됨 — 저장하거나 버리세요 ($(secs "$CUR") 초)"; exit 0; }
        echo "✗ 녹화 중이 아닙니다"; exit 1
    fi
    pid="$(cat "$PIDF")"
    kill -INT "$pid"                      # gst-launch -e: Ctrl+C → EOS → mp4 정상 마무리
    for i in $(seq 1 25); do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid"; echo "⚠ 5 초 안에 안 끝나 강제 종료 — 영상이 깨졌을 수 있습니다"
    fi
    rm -f "$PIDF"
    s="$(secs "$CUR")"
    [ -n "$s" ] || { echo "✗ 영상이 깨졌습니다 (버리고 다시 찍으세요)"; exit 1; }
    echo "✓ 정지: $s 초 — 이름을 정해 저장하세요"
    ;;
save)
    name="${2:-}"
    [[ "$name" =~ ^[a-z0-9]+(_[a-z0-9]+)*$ ]] || { echo "✗ 이름은 영문 소문자·숫자·_ 만: '$name'"; exit 1; }
    alive && { echo "✗ 녹화 중입니다 — 먼저 정지하세요"; exit 1; }
    [ -f "$CUR" ] || { echo "✗ 저장할 영상이 없습니다"; exit 1; }
    out="$DIR/$name.mp4"; k=2
    while [ -e "$out" ]; do out="$DIR/${name}_$k.mp4"; k=$((k + 1)); done
    mv "$CUR" "$out"
    echo "✓ 저장: cv/data/baseline/$(basename "$out") ($(secs "$out") 초)"
    ;;
discard)
    alive && { echo "✗ 녹화 중입니다 — 먼저 정지하세요"; exit 1; }
    [ -f "$CUR" ] || { echo "✓ 버릴 영상이 없습니다"; exit 0; }
    rm -f "$CUR"; echo "✓ 버림"
    ;;
status)
    if alive; then
        echo "rec=1"; echo "rec_sec=$(awk -v t0="$(cat "$T0F")" -v t="$(date +%s.%N)" 'BEGIN {printf "%.0f", t - t0}')"
    else echo "rec=0"; fi
    if ! alive && [ -f "$CUR" ]; then echo "pending=1"; echo "pending_sec=$(secs "$CUR")"; else echo "pending=0"; fi
    echo "saved=$(ls "$DIR"/*.mp4 2>/dev/null | wc -l)"
    ;;
list)
    ls -t "$DIR"/*.mp4 2>/dev/null | head -15 | while read -r f; do echo "$(basename "$f" .mp4) $(secs "$f")"; done
    ;;
*)
    sed -n '3,18p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
