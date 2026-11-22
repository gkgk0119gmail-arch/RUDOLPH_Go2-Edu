#!/bin/bash
# 센서 드라이버 + 정적 TF + rviz2 한 번에 실행 (Ctrl+C 로 전부 종료)
DIR="$(cd "$(dirname "$0")" && pwd)"
source "$DIR/env.sh"
PIDS=()

# 센서 드라이버 (Hesai + RealSense)
"$DIR/start_sensors.sh" & PIDS+=($!)

# ── 정적 TF (x y z yaw pitch roll  parent child) ──
# Go2 내장 L1 라이다: Unitree go2_description URDF radar_joint 값 (pitch 2.8782 rad)
ros2 run tf2_ros static_transform_publisher 0.28945 0 -0.046825 0 2.8782 0 base_link utlidar_lidar & PIDS+=($!)
# Hesai XT16 (Go2 EDU 표준 마운트):
#   GO2_Hesai / go2-3d-mapping 의 lidar-in-IMU = (0.171, 0, 0.0908) + URDF imu_joint (-0.02557, 0, 0.04232)
#   → base_link 기준 (0.1454, 0, 0.1331). z 는 바닥 높이로 확인.
#   yaw +90°: 이 로봇의 XT16 은 점군 +y 가 로봇 정면 → RealSense 점군과 대조해 측정 (0° 일 때 겹침 0%, +90° 에서 최대)
ros2 run tf2_ros static_transform_publisher 0.1454 0 0.1331 1.5708 0 0 base_link hesai_lidar & PIDS+=($!)
# D435i (머리 위 브라켓): x 는 사진/URDF Head_upper(0.285) 기준 추정,
#   z·pitch 는 깊이영상 바닥평면 피팅 (바닥에서 ~0.40 m, 아래로 ~13°) 으로 측정
#   yaw 는 Hesai 점군 대조 시 ±2° 안에서 흔들림 → 0 유지 (겹침 ~70-80%, 중앙 오차 ~5cm)
ros2 run tf2_ros static_transform_publisher 0.32 0 0.055 0 0.227 0 base_link camera_link & PIDS+=($!)

sleep 3
rviz2 -d "$DIR/rviz/all_sensors.rviz" & PIDS+=($!)

trap 'kill ${PIDS[@]} 2>/dev/null; pkill -f hesai_ros_driver_node; pkill -f realsense2_camera_node; wait' INT TERM EXIT
wait
