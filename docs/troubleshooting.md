# 트러블슈팅 기록

구축하면서 실제로 겪은 문제와 해결 방법입니다.

## 1. DDS 토픽 속도 붕괴

**증상:** 센서를 여러 개 띄우고 rviz를 켜자 RealSense 컬러 0 Hz, 깊이 4 Hz, Hesai 2 Hz로 떨어짐.

**원인:** CycloneDDS를 `eth0` + 기본 멀티캐스트로 쓰면 NX 내부 노드끼리 주고받는 대용량 데이터
(720p 컬러 ≈ 80 MB/s)도 유선 LAN으로 멀티캐스트되고, 수신 버퍼가 넘쳐 조각난 UDP 패킷이 버려짐.

**해결:** [`cyclonedds.xml`](../cyclonedds.xml) — `AllowMulticast=spdp` (탐색만 멀티캐스트, 데이터는 유니캐스트 →
같은 호스트 내부 트래픽은 루프백으로 처리), `SocketReceiveBufferSize min=16MB`, 커널 `rmem_max` 32 MB,
RealSense 해상도 640×480@15. → 깊이 13 Hz, 점군 11.5 Hz, Hesai 10 Hz, Go2 `/lowstate` 470 Hz 유지.

## 2. Hesai 타임스탬프가 2020년

**증상:** `/lidar_points` 헤더 시간이 1589933597 (2020-05-20). IMU(시스템 시간)와 6년 차이 → LIO 불가.

**원인:** XT16 내부 시계가 PTP/GPS 동기화 없이 Free Run.

**해결:** Hesai 드라이버 `use_timestamp_type: 1` (패킷 수신 시각). 점별 `timestamp`도 시스템 시간 기준으로 바뀌며,
한 프레임 내 점 시간 분포(0.1 s)는 유지됨.

## 3. Hesai TF yaw가 90° 틀어짐

**증상:** rviz에서 RealSense 점군이 Hesai 지도와 다른 방향을 가리킴.

**원인:** 이 로봇의 XT16 장착 방향에서 점군 +y가 로봇 정면. 참고한 공개 설정들은 yaw 0.

**해결:** 위치를 아는 RealSense 점군을 기준으로 Hesai yaw를 360° 탐색 → +88~90°에서 정합 최대.
`base_link → hesai_lidar` yaw = +1.5708, FAST-LIO `extrinsic_R = Rz(+90°)`.

## 4. RealSense 드라이버 내장 pointcloud가 안 나옴

**증상:** `pointcloud.enable:=true` 시 `/camera/depth/color/points` 무발행 + 깊이 스트림까지 ~4 Hz로 저하.
로그: `No stream match for pointcloud chosen texture Process - Color`.

**해결:** 내장 필터 끄고 `depth_image_proc::PointCloudXyzNode`로 변환 ([launch/rs_depth_points.launch.py](../launch/rs_depth_points.launch.py)).

## 5. FAST-LIO `/map_save`가 노드를 죽임

**증상:** `ros2 service call /map_save ...` → `pcl::IOException: Input point cloud has no data!` 후 FAST-LIO 종료, 매핑 데이터 소실.

**원인:** ROS2 브랜치에서 저장 버퍼를 채우는 코드가 주석 처리되어 있어 항상 빈 클라우드를 저장하려 함.

**해결:** 자체 [`go2_map_tools/map_accumulator`](../ros2_ws/src/go2_map_tools) 노드로 저장
(`/map_accumulator/save_map`). FAST-LIO 쪽은 `pcd_save_en: false` + 빈 버퍼 가드 패치.

## 6. rviz가 메모리 9 GB 사용

**증상:** 매핑 중 rviz 2 fps, RAM 60 % 점유.

**원인:** `/cloud_registered`를 `Decay Time: 100000`으로 표시 → 모든 스캔을 rviz가 그대로 누적.

**해결:** map_accumulator가 10 cm 복셀 전역 지도(`/map_cloud`, transient_local)를 2초마다 발행, rviz는 그것만 표시. → 0.16 GB, 31 fps.

## 7. FAST-LIO가 IMU를 못 받음

**증상:** `/Odometry` 0 Hz, 로그에 `incompatible QoS ... RELIABILITY`.

**해결:** `go2_imu_bridge` 퍼블리셔를 RELIABLE(KeepLast 200)로 (FAST-LIO 구독이 RELIABLE).

## 8. Go2 내장 L1 LiDAR 정합 실패

`/utlidar/cloud`(frame `utlidar_lidar`)를 URDF `radar_joint`(xyz 0.28945 0 −0.046825, pitch 2.8782)로 변환하면
바닥이 −0.49 m로 나오고 Hesai 점군과 어떤 오프셋/회전에서도 정합되지 않음 (최선 중앙 오차 0.42 m).
→ 실제 좌표계가 프레임 이름과 다른 것으로 추정. 프로젝트에서는 사용하지 않음.

## 9. 기타

| 문제 | 해결 |
|---|---|
| `~/unitree_ros2/setup.sh`가 `enp3s0` 사용 → Go2 통신 불가 | `eth0`으로 수정 |
| ROS apt: `EXPKEYSIG F42ED6FBAB17C654`, 미러 404 | 키 갱신 + `packages.ros.org` 추가 ([setup.md](setup.md)) |
| `ros2 topic hz`가 대용량 메시지에서 실제보다 훨씬 낮게 나옴 | 파이썬 역직렬화 병목. `raw=True` 구독으로 개수만 세서 측정 |
| Hesai `start.py`가 rviz를 하나 더 띄움 | `ros2 run hesai_ros_driver hesai_ros_driver_node` 직접 실행 |
| rviz2(Foxy)에 DepthCloud 플러그인 없음 | `/camera/depth/points`를 PointCloud2로 표시 |
| Hesai 로그 `firetime` / `channel fov` 에러 | 설정 예시 경로 때문, 동작 무관 |
| RealSense 로그 `auto_exposure_limit ... HW not ready` | 펌웨어 미지원 옵션, 동작 무관 |
