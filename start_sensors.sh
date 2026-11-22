#!/bin/bash
# Hesai XT16 + RealSense D435i ROS2 드라이버 실행 (Ctrl+C로 전부 종료)
#   ./start_sensors.sh                 Hesai + RealSense
#   ./start_sensors.sh --no-realsense  Hesai 만 (매핑 시: RealSense 는 SLAM 에 안 쓰고 CPU/발열만 큼)
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/env.sh"
PIDS=()
USE_RS=1
for a in "$@"; do [ "$a" = "--no-realsense" ] && USE_RS=0; done

# Hesai: start.py 는 rviz 까지 같이 띄우므로 드라이버 노드만 직접 실행 → /lidar_points
(cd $HOME/hesai_ws && exec ros2 run hesai_ros_driver hesai_ros_driver_node) & PIDS+=($!)

if [ $USE_RS = 1 ]; then
  # RealSense: 드라이버 내장 pointcloud 는 4.51.1 에서 동작 안 함 → 끄고 depth_image_proc 사용
  ros2 launch realsense2_camera rs_launch.py \
      enable_gyro:=true enable_accel:=true unite_imu_method:=2 \
      depth_module.profile:=640x480x15 rgb_camera.profile:=640x480x15 \
      align_depth.enable:=true pointcloud.enable:=false & PIDS+=($!)
  # depth 영상 → /camera/depth/points
  ros2 launch "$DIR/launch/rs_depth_points.launch.py" & PIDS+=($!)
fi

trap 'kill ${PIDS[@]} 2>/dev/null; wait' INT TERM
wait
