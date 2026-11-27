#!/usr/bin/env bash
# ============================================================================
#  link.sh — 노트북 브리지를 SSH 터널 안으로 NX 브리지에 붙인다 (laptop.sh 가 부른다)
#
#     tools/zenoh/link.sh unitree@<NX>     (앞에서 계속 돈다. Ctrl+C·TERM 이면 둘 다 내린다)
#
#     노트북 127.0.0.1:17447 ──ssh -L──▶ NX 127.0.0.1:7447 (NX 브리지)
#
#  · 와이파이가 끊기면 ssh 가 약 9 초 뒤 끝나고 2 초 뒤 다시 연다. 노트북 브리지는
#    exit_on_failure=false 라 터널이 돌아오면 알아서 다시 붙는다.
#  · NX 쪽 터널 sshd 는 remote.sh tunnel 이 코어 0-3·nice 19 로 묶는다 (보행 노드 보호).
# ============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${1:?unitree@<NX 주소>}"
NX_ROOT="${RUDOLPH_NX_ROOT:-Desktop/3DOG}"
ZB="$HERE/$(uname -m)/zenoh-bridge-ros2dds"
LPORT=17447
LOGD="${TMPDIR:-/tmp}"
O=(-o BatchMode=yes -o ConnectTimeout=4 -o ServerAliveInterval=3 -o ServerAliveCountMax=3)

[ -x "$ZB" ] || { echo "브리지 실행 파일이 없습니다: $ZB  (tools/zenoh/fetch.sh)" >&2; exit 1; }
export ROS_LOCALHOST_ONLY=0   # ★ 1 이면 아래 CYCLONEDDS_URI 와 부딪혀 브리지가 죽는다 (Precondition Not Met). lo 고정은 CYCLONEDDS_URI 가 한다
export CYCLONEDDS_URI="file://$HERE/laptop_cyclonedds.xml"   # 브리지만 lo 에 (중복 방지)

LOOP=""; BP=""
cleanup () {
    [ -n "$LOOP" ] && kill "$LOOP" 2>/dev/null
    [ -n "$BP" ] && kill "$BP" 2>/dev/null
    for p in $(pgrep -f "ssh .*-L 127.0.0.1:$LPORT:127.0.0.1:7447"); do kill "$p" 2>/dev/null; done
    wait 2>/dev/null
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

(
    trap 'kill $SP 2>/dev/null; exit 0' TERM
    while :; do
        ssh "${O[@]}" -n -o ExitOnForwardFailure=yes \
            -L "127.0.0.1:$LPORT:127.0.0.1:7447" "$HOST" "$NX_ROOT/tools/panel/remote.sh tunnel" >/dev/null &
        SP=$!
        sleep 2
        kill -0 "$SP" 2>/dev/null && echo "$(date +%T) 터널 열림 → $HOST"
        wait "$SP"
        echo "$(date +%T) 터널 끊김 (rc=$?) — 2 초 뒤 다시"
        sleep 2
    done
) >>"$LOGD/rudolph_tunnel.log" 2>&1 &
LOOP=$!

"$ZB" -c "$HERE/laptop.json5" -e "tcp/127.0.0.1:$LPORT" >"$LOGD/rudolph_laptop_bridge.log" 2>&1 &
BP=$!
wait "$BP"
