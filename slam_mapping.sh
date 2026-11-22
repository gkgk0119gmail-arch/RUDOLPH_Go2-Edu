#!/bin/bash
# 실시간 3D LIO 매핑: Hesai 드라이버 + IMU 브리지 + FAST-LIO2 + 맵 누적기 + rviz (Ctrl+C 로 전부 종료)
#   맵 저장 (다른 터미널, source ~/Desktop/3DOG/env.sh 후):
#     ros2 service call /map_accumulator/save_map std_srvs/srv/Trigger   → maps/map_YYYYmmdd_HHMMSS.pcd
#   RealSense 는 매핑에 안 쓰므로 기본 OFF.  같이 켜려면: ./slam_mapping.sh --with-realsense
#   rviz 없이: ./slam_mapping.sh rviz:=false
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/env.sh"
PIDS=()

SENSOR_ARGS="--no-realsense"; LAUNCH_ARGS=()
for a in "$@"; do
  if [ "$a" = "--with-realsense" ]; then SENSOR_ARGS=""; else LAUNCH_ARGS+=("$a"); fi
done

"$DIR/start_sensors.sh" $SENSOR_ARGS & PIDS+=($!)
sleep 4   # let the lidar driver come up
ros2 launch "$DIR/launch/slam_lio.launch.py" "${LAUNCH_ARGS[@]}" & PIDS+=($!)

trap 'kill ${PIDS[@]} 2>/dev/null; wait' INT TERM
wait
