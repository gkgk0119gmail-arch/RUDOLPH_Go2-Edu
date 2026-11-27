# 노트북에서 NX 토픽 보기 (Zenoh 브리지 + SSH 터널 + RViz)

제어판의 **`RViz (노트북)`** 버튼 하나로 됩니다. 직접 하려면:

```bash
RUDOLPH_NX=unitree@<NX 와이파이 주소> tools/zenoh/laptop.sh
```

```
NX 노드들 ──(DDS, eth0 그대로)── NX 브리지 127.0.0.1:7447 ══(SSH 터널)══ 노트북 브리지 ──(DDS, lo)── RViz
Go2 머리 카메라 (RTP/H264 멀티캐스트) ── front_cam_pub.py ── /go2/front/image ──┘
```

## 왜 이렇게 하나

- NX 의 DDS 는 `cyclonedds.xml` 로 **유선 eth0 에만** 묶여 있습니다 (Go2 유선 통신·카메라/점군 트래픽 보호).
  그래서 와이파이의 노트북 RViz 에는 아무것도 안 보였습니다 (2026-11-27). 이 설정은 그대로 두고, 와이파이로는 브리지 하나만 나갑니다.
- NX 브리지는 NX 안(127.0.0.1)에서만 듣고, 노트북은 SSH 터널로 들어옵니다 → SSH 키가 있는 노트북만 `/goal_pose` 를 보낼 수 있습니다.
- 학교 와이파이를 아끼려고 큰 토픽은 브리지가 솎아 보냅니다 ([nx.json5](nx.json5) `pub_max_frequencies`).

## 노트북에서 보이는 것 ([laptop.rviz](laptop.rviz), Fixed Frame `camera_init`)

| 표시 | 토픽 | 노트북 도착 |
|---|---|---|
| 누적 지도 | `/map_cloud` | 5 초마다 |
| 현재 스캔 | `/cloud_registered_body` (장당 1.3 MB) | 2 Hz |
| D435i 깊이 | `/camera/depth/image_rect_raw` (848x480) | 2 Hz |
| **Go2 머리 카메라** | `/go2/front/image` (480x270 컬러) | 약 1.4 Hz |
| 로봇 위치·지나온 길 | `/Odometry`, `/path`, `/tf` | 10 Hz, 1 Hz, 그대로 |
| 수색 | `/rudolph/map`, `/belief`, `/searched`, `/goal` | 수색 노드가 켜져 있을 때 |

노트북 → NX: RViz `2D Goal Pose` → `/goal_pose` (항법 목표), `Publish Point` → `/clicked_point` (요구조자 발견 위치). 그 밖의 제어는 없습니다.

## 머리 카메라 (front_cam_pub.py)

메인보드가 230.1.1.1:1720 으로 보내는 RTP/H264 (1280x720, 14.5 fps) 를 Jetson 하드웨어 디코더로 풀고 480x270 으로 줄여
`sensor_msgs/Image` 로 냅니다. `remote.sh bridge start/stop` 이 브리지와 같이 켜고 끕니다 (tmux `rd_frontcam`, 코어 0-3·nice 10).

- NX CPU: 코어 하나의 약 22 % (전체 프레임 디코딩). `--keyframes 1` 이면 키프레임만 → 약 5.5 %, 대신 0.5 Hz (2 초에 한 장).
- 같은 멀티캐스트를 쓰는 녹화(`tools/panel/rec.sh`)와 동시에 돌아도 됩니다.

## 노트북 준비 (한 번만)

```bash
sudo apt install -y ros-humble-rmw-cyclonedds-cpp     # 없으면 FastDDS 로 보는데 점군이 자주 빠진다
ssh-copy-id unitree@<NX 와이파이 주소>
```

브리지 실행 파일(약 17 MB)은 레포에 없습니다. `laptop.sh` 가 처음 실행할 때 [fetch.sh](fetch.sh) 로 받습니다.
NX 에는 `tools/zenoh/aarch64/` 에 이미 있습니다 (없으면 NX 에서 `tools/zenoh/fetch.sh`).

## 함정 (2026-11-27 실측)

- **버전 1.7.2 고정.** 1.10.2 x86_64 는 GLIBC 2.38 을 요구해 Ubuntu 22.04(2.35) 노트북에서 안 뜹니다.
  1.7.2 는 x86_64 GLIBC 2.34, aarch64 2.29 까지 (NX 20.04 = 2.31). NX(Foxy) ↔ 노트북(Humble) 그대로 통합니다.
- **노트북 브리지는 `lo` 에만** ([laptop_cyclonedds.xml](laptop_cyclonedds.xml)). `ros_localhost_only: true` 로 두면
  노트북 → NX 메시지가 정확히 5 배로 중복됐습니다 (6 개 보내 30 개 도착). 같은 설정에 `ROS_LOCALHOST_ONLY=1` 을 같이 주면
  브리지가 `Precondition Not Met` 으로 죽어서, 브리지는 `ROS_LOCALHOST_ONLY=0` + 이 xml 로 띄웁니다. 고친 뒤 6 → 6.
- **RViz 는 Reliable 로 구독.** 큰 메시지(점군 1.3 MB, 깊이 0.8 MB)는 노트북 UDP 수신 버퍼(0.2 MB)를 넘쳐 Best Effort 로는
  0 장이었습니다. Reliable 은 빠진 조각을 다시 받아 10 초에 17 장. RViz 도 Cyclone + 같은 xml 로 띄웁니다 (FastDDS 는 10 초에 6 장).

## 파일

| 파일 | 어디서 | 하는 일 |
|---|---|---|
| `laptop.sh` | 노트북 | NX 브리지 켜기 → `link.sh` → rviz2. RViz 를 닫으면 노트북 쪽만 내림 |
| `link.sh` | 노트북 | SSH 터널(17447→NX 7447, 끊기면 다시) + 노트북 브리지 |
| `laptop.json5`, `laptop_cyclonedds.xml`, `laptop.rviz` | 노트북 | 노트북 브리지·DDS·RViz 설정 |
| `nx.json5` | NX | 보낼 토픽·주파수 |
| `front_cam_pub.py` | NX | 머리 카메라 → `/go2/front/image` |
| `fetch.sh` | 둘 다 | 브리지 1.7.2 받기 |
