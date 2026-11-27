# RUDOLPH 제어판 (노트북)

NX 작업을 버튼으로 띄우고 내리고, 단서를 채팅하듯 보내는 노트북 창입니다. 터미널에 명령을 붙여넣을 필요가 없습니다.
mlcs_mpc 의 `car_panel` 방식을 따랐습니다. 노트북 → SSH 연결 하나(ControlMaster) → NX 의 `remote.sh` 창구.

```
┌ 비상 정지 = 리모컨 L2+B 안내 ──────────────────────────────────────────────┐
├ 상태 칩: SSH · Go2 · 부하 · 온도 · Wi-Fi                                     ┤
├ 실행 (NX)                                      ├ 단서 채팅                     ┤
│ ● 센서만     [시작][정지]                      │  나: 로봇 오른쪽 5미터에 한 명 │
│ ● SLAM       [시작][정지][맵 저장]             │  RUDOLPH: 들음 … (해석 중)     │
│ ● 내장 보행  [확인][끄기][다시 켜기]           │  RUDOLPH: 해석 완료 (1.7초) …  │
│ ● 하위 보행  [v3c▾][리모컨▾][시작][앉히고 종료]│  [입력칸            ][보내기]  │
│ ● 항법       [☑수색 목표][x][y][v][시작][정지] │  [요구조자 발견]               │
│ ● 수색       [SHADOW▾][시작][정지]             │                               │
├ 탭: 화면(tmux) · 센서만 · SLAM · 하위 보행 · 항법 · 수색 · 작업·알림 · 명령어 ┤
```

## 실행

```bash
sudo apt install -y python3-pyqt5            # 처음 한 번
ssh-copy-id unitree@<NX 주소>                 # 처음 한 번 (비밀번호 없는 SSH)
RUDOLPH_NX=unitree@<NX wifi 주소> python3 tools/panel/rudolph_panel.py
```

`RUDOLPH_NX` 를 안 주면 유선 `unitree@192.168.123.18`. 노트북에 ROS 는 필요 없습니다. `RViz (노트북)` 버튼만 노트북의 rviz2 를 씁니다.

## 원칙

- **제어판은 로봇을 실시간으로 움직이지 않습니다.** 비상 정지는 언제나 리모컨 L2+B.
- **서 있는 로봇의 보행 노드는 kill 하지 않습니다.** `앉히고 종료` = `deploy/stop_hipan_low.sh` (서기 → 천천히 앉기 → 종료).
- 위험한 버튼(내장 보행 끄기·다시 켜기, 하위 보행 시작, 항법 시작, 수색 LIVE)은 확인 창을 거칩니다.
- **NX 부하**: 조회(status 5초마다, 화면 2초마다는 그 탭을 볼 때만)는 코어 0-3·nice 19, bash·ps·tmux 만 씁니다. status 1회 약 0.07 s.
  실행 명령(start)은 nice 를 물려주지 않습니다 (보행 노드 rt 우선순위 보호).

## 구조

| 파일 | 어디서 | 하는 일 |
|---|---|---|
| `rudolph_panel.py` | 노트북 | PyQt5 창. 모든 일을 `ssh <NX> remote.sh <명령>` 으로 |
| `remote.sh` | NX | 창구: `status screen logs start stop savemap mode clue victim usage` (맨 위 주석에 전체) |
| `send_clue.py` | NX | 단서 → `/rudolph/clue_text`, 발견 → `/rudolph/victim_here` (수색 노드가 없으면 안 보내고 알려 줌) |
| `../zenoh/` | 둘 다 | `RViz (노트북)` 버튼: Zenoh 브리지 + SSH 터널로 지도·점군·깊이·**Go2 머리 카메라**를 노트북 RViz 에 ([README](../zenoh/README.md)) |
| `record_box.py` | 노트북 | '영상 녹화' 칸: 녹화 시작·정지 → 제목(직접 입력 또는 자세·거리·가림 / empty·장소 선택) → 저장·버리기 |
| `rec.sh` | NX | `remote.sh rec start\|stop\|save <이름>\|discard\|status\|list`. 머리 카메라 H264 를 디코딩 없이 mp4 로 (CPU ~1 %) → `cv/data/baseline/<이름>.mp4` |

- 녹화한 영상은 데스크톱에서 `cv/fetch.sh` 로 가져와 `python -m cv.eval_baseline` 으로 사람 검출 기준 성능을 잽니다.
  녹화는 제어판 창을 닫아도 NX 에서 계속되고, 다시 열면 상태(녹화 중·저장 대기)를 이어 받습니다.

- 제어판으로 띄운 작업은 NX tmux 세션 `rd_<작업>` 에서 돕니다. NX 에서 직접 보기: `tmux attach -t rd_low`.
  출력은 `deploy/logs/panel/<작업>_<시각>.log` 에도 남고 `<작업>_last.log` 가 마지막 실행을 가리킵니다.
- 제어판 밖(NX 바탕화면 터미널, `start_hipan_live.sh` 등)에서 띄운 것도 상태에는 `실행 중 (제어판 밖에서 띄움)` 으로 보이고 정지 버튼도 듣습니다.
  단 화면·로그 탭에는 안 나옵니다.
- 단서 문장은 base64 로 보내서 따옴표·한글이 깨지지 않습니다. 수색 노드의 상태 줄(`[rudolph_search]: …`)이 채팅에 RUDOLPH 말풍선으로 올라옵니다.
- 작업 명령은 NX 의 기존 스크립트와 같습니다: 센서만 = `start_sensors.sh --rs-sim`, SLAM = `slam_mapping.sh --with-realsense-sim rviz:=false pgo:=false`
  (맵 저장 = `/map_accumulator/save_map`), 하위 보행 = `start_hipan_live.sh` 와 같은 가중치 표, 항법 = `run_hipan_nav.py --live`, 수색 = `rudolph_search.py [--live]`.

## 보통 순서

1. SLAM 시작 → 2. 내장 보행 확인 → 끄기 (로봇 누운 상태) → 3. 하위 보행 시작 (항법 명령) →
4. 수색 시작 (LIVE) → 5. 항법 시작 (수색 목표) → 6. 들은 단서를 채팅에 입력 → 7. 사람을 찾으면 `요구조자 발견` →
8. 끝: 항법 정지 → 앉히고 종료 → 내장 보행 다시 켜기

## 시험 기록

- 2026-11-27 (데스크톱 유선, NX 에서 보행 시험 중): status·screen·usage 조회와 창 그리기만 시험. SSH ✓, Go2 ✓,
  제어판 밖에서 돌던 하위 보행을 `모델 v3c · 명령 리모컨 (제어판 밖에서 띄움)` 으로 정확히 읽음, 닫을 때 경고 없음.
  **start / stop / mode / clue / victim 은 로봇에서 아직 안 눌러 봤습니다** (보행 시험 중이라 건드리지 않음).
- 2026-11-27 (사용자, 노트북): 하위 보행 v3c 리모컨 시작·앉히고 종료 ✓, 센서만 ✓, SLAM ✓. 노트북 RViz 에 아무것도 안 보임
  → NX DDS 가 eth0 전용이라서. Zenoh 브리지로 해결 ([../zenoh](../zenoh/README.md)), 데스크톱(Humble)에서 RViz 로 지도·점군·깊이·머리 카메라 확인.
