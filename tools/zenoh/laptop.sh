#!/usr/bin/env bash
# ============================================================================
#  laptop.sh — 노트북에서 NX 토픽을 RViz 로 보기 (Zenoh 브리지 + SSH 터널 + RViz)
#
#     RUDOLPH_NX=unitree@<NX 와이파이 주소> tools/zenoh/laptop.sh
#
#  제어판의 'RViz (노트북)' 버튼이 이것을 부른다. 하는 일:
#    1) NX 브리지 켜기 (remote.sh bridge start — 이미 떠 있으면 그대로)
#    2) 노트북 브리지 + SSH 터널 (link.sh)
#    3) rviz2 (tools/zenoh/laptop.rviz). RViz 를 닫으면 노트북 브리지·터널도 내린다 (NX 브리지는 남김)
#  노트북: Ubuntu 22.04 + ROS 2 Humble. NX(Foxy)와 메시지 정의가 같아서 그대로 보인다.
# ============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${1:-${RUDOLPH_NX:-unitree@192.168.123.18}}"
NX_ROOT="${RUDOLPH_NX_ROOT:-Desktop/3DOG}"
ZB="$HERE/$(uname -m)/zenoh-bridge-ros2dds"

[ -x "$ZB" ] || { echo "[laptop] 브리지 실행 파일 받기 (처음 한 번)"; "$HERE/fetch.sh" || exit 1; }
if [ -z "${ROS_DISTRO:-}" ]; then
    set +u; source /opt/ros/humble/setup.bash || { echo "[laptop] ✗ ROS 2 Humble 이 없습니다"; exit 1; }; set -u
fi
# 노트북 RViz 와 브리지는 노트북 안(lo)에서만 만난다. RViz 도 브리지와 같은 Cyclone + lo 설정으로:
#   ★ 2026-11-27 실측: 점군 1장 1.3 MB. FastDDS 로 받으면 10 초에 6 장, Cyclone(lo) 로 받으면 17 장 (2 Hz 로 솎은 것 중)
if [ -f "/opt/ros/${ROS_DISTRO}/lib/librmw_cyclonedds_cpp.so" ]; then
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=0 CYCLONEDDS_URI="file://$HERE/laptop_cyclonedds.xml"
else
    export ROS_LOCALHOST_ONLY=1
    echo "[laptop] ⚠ rmw_cyclonedds_cpp 가 없어 FastDDS 로 봅니다 — 점군이 자주 빠집니다. 설치: sudo apt install ros-humble-rmw-cyclonedds-cpp"
fi

echo "[laptop] NX 브리지: $(ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" "$NX_ROOT/tools/panel/remote.sh bridge start" 2>&1 | tail -1)"

LP=""
cleanup () { [ -n "$LP" ] && kill -TERM -- "-$LP" 2>/dev/null; }
trap cleanup EXIT
trap 'exit 130' INT TERM
if pgrep -f "$ZB" >/dev/null 2>&1; then
    echo "[laptop] 노트북 브리지가 이미 떠 있습니다 — 그대로 씁니다"
else
    setsid "$HERE/link.sh" "$HOST" >/dev/null 2>&1 &
    LP=$!
    sleep 3
    kill -0 "$LP" 2>/dev/null || { echo "[laptop] ✗ 브리지가 바로 죽었습니다"; tail -5 "${TMPDIR:-/tmp}/rudolph_laptop_bridge.log"; exit 1; }
    echo "[laptop] 브리지 → SSH 터널 → $HOST  (로그: ${TMPDIR:-/tmp}/rudolph_laptop_bridge.log, rudolph_tunnel.log)"
fi
echo "[laptop] RViz — Fixed Frame camera_init. '2D Goal Pose' = 항법 목표, 'Publish Point' = 요구조자 발견 위치"
rviz2 -d "$HERE/laptop.rviz"
