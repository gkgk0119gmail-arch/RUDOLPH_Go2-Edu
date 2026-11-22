# 설치 / 재현 절차 (Jetson Orin NX, Ubuntu 20.04, ROS2 Foxy)

이 저장소를 `~/Desktop/3DOG`에 clone 했다고 가정합니다. (`REPO=~/Desktop/3DOG`)

## 0. 기본 패키지 · ROS apt 저장소

중국 미러(tuna)의 ROS2 인덱스가 실제 파일과 맞지 않아 404가 나고, ROS 서명키도 만료되어 있었습니다.

```bash
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
     -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu focal main" \
  | sudo tee /etc/apt/sources.list.d/ros2-official.list
sudo apt update

sudo apt install -y ros-foxy-realsense2-camera ros-foxy-realsense2-description ros-foxy-librealsense2 \
                    ros-foxy-depth-image-proc ros-foxy-pcl-ros libboost-all-dev libyaml-cpp-dev wmctrl
```

(선택) 한글 입력: `tools/install_korean.sh`

## 1. DDS 커널 버퍼

```bash
sudo cp $REPO/config/60-dds-buffers.conf /etc/sysctl.d/
sudo sysctl -p /etc/sysctl.d/60-dds-buffers.conf
```

## 2. Unitree ROS2 메시지 (`unitree_go`, `unitree_api`) + CycloneDDS 0.10

[unitree_ros2](https://github.com/unitreerobotics/unitree_ros2) 공식 절차대로 `~/unitree_ros2/cyclonedds_ws`에서
cyclonedds(0.10.x)를 먼저 빌드한 뒤:

```bash
cd ~/unitree_ros2/cyclonedds_ws
source /opt/ros/foxy/setup.bash
colcon build --packages-skip cyclonedds      # rmw_cyclonedds_cpp, unitree_api/go/hg
```

`~/unitree_ros2/setup.sh`의 `NetworkInterface name="enp3s0"` → **`eth0`** 로 수정 (Orin NX 유선 포트 이름).
이 저장소의 `env.sh`는 자체 `cyclonedds.xml`을 쓰므로 `setup.sh`를 source 할 필요는 없습니다.

## 3. Hesai XT16 드라이버

```bash
mkdir -p ~/hesai_ws/src && cd ~/hesai_ws/src
git clone --recurse-submodules https://github.com/HesaiTechnology/HesaiLidar_ROS_2.0.git
cp $REPO/config/hesai_xt16_config.yaml HesaiLidar_ROS_2.0/config/config.yaml
cd ~/hesai_ws && source /opt/ros/foxy/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

설정 핵심: `device_ip_address: 192.168.123.20`, `use_timestamp_type: 1`(수신 시각), `send_imu_ros: false`.
드라이버 노드는 **소스 폴더의** `config/config.yaml`을 읽습니다.
`start.py` 런치는 rviz까지 띄우므로 `ros2 run hesai_ros_driver hesai_ros_driver_node`로 실행합니다 (`start_sensors.sh`).

## 4. SLAM 워크스페이스 (FAST-LIO2 + 자체 패키지)

```bash
mkdir -p ~/slam_ws/src && cd ~/slam_ws/src
git clone --recursive -b ROS2 https://github.com/hku-mars/FAST_LIO.git
cd FAST_LIO && git checkout cc3c9e8b1469b33b8072764d6db2348ed6dc597f
git apply $REPO/patches/fast_lio_hesai_foxy.patch
cd ~/slam_ws/src
for p in go2_imu_bridge go2_map_tools livox_ros_driver2; do ln -s $REPO/ros2_ws/src/$p .; done

cd ~/slam_ws
source /opt/ros/foxy/setup.bash && source ~/unitree_ros2/cyclonedds_ws/install/setup.bash
MAKEFLAGS=-j6 colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

패치 내용 (`patches/fast_lio_hesai_foxy.patch`):

- `preprocess.h/.cpp`: `lidar_type: 5` (HESAI) 추가 — 필드 `x y z intensity ring timestamp(double, 절대초)`,
  점별 시간을 헤더 기준 ms 오프셋으로 변환, NaN·blind 제거, 시간순 정렬
- `laserMapping.cpp`: Foxy 서비스 콜백 시그니처 수정, 빈 버퍼 PCD 저장 시 크래시 방지

## 5. 확인

```bash
source $REPO/env.sh
$REPO/view_all.sh        # rviz에 Hesai + RealSense가 겹쳐 보이면 OK
$REPO/slam_mapping.sh    # 정지 상태에서 /Odometry 10 Hz, 드리프트 ~mm 수준이면 OK
```
