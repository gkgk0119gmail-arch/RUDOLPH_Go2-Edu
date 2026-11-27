#!/usr/bin/env bash
# ============================================================================
#  remote.sh — 노트북 제어판(rudolph_panel.py)이 SSH 로 부르는 NX 쪽 창구
#
#  사람이 직접 쓸 일은 거의 없다. 제어판이 SSH 연결 하나 위에서
#      ssh unitree@<NX> ~/Desktop/3DOG/tools/panel/remote.sh <명령> [인자]
#  로 부른다. NX 터미널에서 직접 불러도 같다.
#
#  ★ 원칙
#    · 제어판은 로봇을 실시간으로 움직이지 않는다. 비상 정지는 언제나 리모컨 L2+B.
#    · 조회 명령(status/screen/logs/usage)은 코어 0-3 + nice 19, bash·ps·tmux 만 쓴다.
#      (보행 노드 hipan_low_node 는 rt_priority 80 으로 다른 코어에서 돈다. 조회가 그 루프를 늦추면 안 된다)
#    · ★ 실행 명령(start)에는 nice·taskset 을 물려주지 않는다. 보행 노드가 nice 19 로 뜨면 루프가 밀린다.
#    · 서 있는 로봇의 보행 노드는 절대 kill 하지 않는다. 보행 정지 = stop low = stop_hipan_low.sh (걸어서 서기 → 천천히 앉기 → 종료).
#
#  명령:
#     status                 상태 key=value 줄 (제어판이 5초마다)
#     screen <작업>          tmux 화면 마지막 80줄 (그 탭을 볼 때만)
#     logs                   작업 로그들 tail -F (연결 하나로 계속)
#     start sensors          Hesai + RealSense (848x480 항법 시야)
#     start slam             FAST-LIO2 + Hesai + RealSense 848x480 (센서 포함, rviz 없음)
#     start low <old|v3c|v3|..> <remote|topic>   HiPAN 하위 보행 LIVE (스포츠 모드 해제·로봇 누운 상태)
#     start nav [x y] [v]    HiPAN 항법 LIVE. x y 없으면 /goal_pose(수색·rviz)가 목표
#     start search <shadow|live>   RUDOLPH 수색 (live = /goal_pose 발행 → 항법이 걸어감)
#     stop <작업>            Ctrl+C 로 정리하며 내림. low 는 stop_hipan_low.sh (안전하게 앉힘)
#     savemap                SLAM 맵 저장 (map_accumulator → maps/*.pcd)
#     mode check|release|restore   Go2 내장 보행(스포츠 모드) 확인 / 끄기(힘 풀림!) / 다시 켜기
#     clue <base64>          단서 문장 → /rudolph/clue_text (base64 = 따옴표·한글 안전)
#     victim                 지금 로봇 위치에서 요구조자 발견 → /rudolph/victim_here
#     usage                  이 저장소 주요 스크립트 맨 위 사용법 주석
#     bridge start|stop      노트북 RViz 용 Zenoh 브리지 (tools/zenoh/nx.json5, 127.0.0.1:7447, 코어 0-3·nice 10)
#     tunnel                 노트북 SSH 터널(tools/zenoh/link.sh)의 NX 쪽. 그 sshd 를 코어 0-3·nice 19 로 묶고 기다림
# ============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NX: ~/Desktop/3DOG
PANEL="$ROOT/tools/panel"
LOGDIR="$ROOT/deploy/logs/panel"
JOB_LOG="$LOGDIR/job.log"
JOBS="sensors slam low nav search"
mkdir -p "$LOGDIR"

light () { taskset -cp 0-3 $$ >/dev/null 2>&1; renice -n 19 -p $$ >/dev/null 2>&1; }
envs () { source "$ROOT/env.sh" >/dev/null 2>&1; }

PROCS=""
# tmux 서버는 처음 받은 명령줄을 끝까지 달고 있어서 빼야 한다 (작업이 끝나도 '돌고 있음'으로 보인다)
procs () { ps -eo args= | grep -v -e '^tmux' -e 'remote.sh'; }
has () { case "$PROCS" in *"$1"*) return 0 ;; esac; return 1; }
flag () { if has "$2"; then echo "$1=1"; else echo "$1=0"; fi; }
sess () { tmux has-session -t "rd_$1" 2>/dev/null; }

# 작업별 '살아 있음' 판정에 쓰는 프로세스 이름 (패널 밖에서 띄운 것도 잡힌다)
pat_of () {
    case "$1" in
        sensors) echo 'start_sensors.sh' ;;
        slam)    echo 'slam_mapping.sh' ;;
        low)     echo 'lib/go2_hipan_low/hipan_low_node' ;;
        nav)     echo 'run_hipan_nav.py' ;;
        search)  echo 'rudolph_search.py' ;;
    esac
}

# 패턴에 맞는 PID (이 셸과 부모 sshd 는 뺀다 — pkill -f 가 자기 셸을 죽이는 문제)
pids_of () {
    local p="$1"
    ps -eo pid=,args= | while read -r pid args; do
        [ "$pid" = "$$" ] || [ "$pid" = "$PPID" ] && continue
        case "$args" in tmux*|*remote.sh*) continue ;; *"$p"*) echo "$pid" ;; esac
    done
}

run_in_tmux () {   # <작업> <명령>  — 화면은 tmux, 출력은 로그 파일에도 (script)
    local job="$1" cmd="$2" ts log
    ts=$(date +%Y%m%d_%H%M%S)
    log="$LOGDIR/${job}_$ts.log"
    ln -sfn "$log" "$LOGDIR/${job}_last.log"
    tmux kill-session -t "rd_$job" 2>/dev/null
    tmux new -d -s "rd_$job" "cd '$ROOT' && script -q -f -e -c \"$cmd\" '$log'; echo \"[종료 코드 \$?]\" | tee -a '$log'; echo; echo '[끝남 — 제어판에서 다시 시작할 수 있습니다]'; exec bash -l"
    echo "✓ 시작: $job (NX 에서 보기: tmux attach -t rd_$job)"
}

case "${1:-}" in
status)
    light
    PROCS="$(procs)"
    for j in $JOBS; do
        if has "$(pat_of $j)"; then echo "$j=1"; else echo "$j=0"; fi
        if sess "$j"; then echo "${j}_tmux=1"; else echo "${j}_tmux=0"; fi
    done
    flag hesai     'hesai_ros_driver_node'
    flag realsense 'realsense2_camera_node'
    flag fastlio   'fastlio_mapping'
    flag bridge    'zenoh-bridge-ros2dds'
    flag search_live 'rudolph_search.py --live'
    # 보행 노드 인자에서 모델·명령 출처
    w=""; src=""
    for a in $(ps -eo args= | grep -m1 '[l]ib/go2_hipan_low/hipan_low_node'); do
        case "$a" in weights:=*) w="${a##*/}"; w="${w%.bin}"; w="${w#hipan_low_}"; w="${w#student_}"; [ "$w" = student ] && w=old ;; cmd_source:=*) src="${a#cmd_source:=}" ;; esac
    done
    echo "low_model=$w"; echo "low_src=$src"
    if ping -c1 -W1 192.168.123.161 >/dev/null 2>&1; then echo go2=1; else echo go2=0; fi
    read -r l _ < /proc/loadavg; echo "load=$l"
    t=0; for z in /sys/class/thermal/thermal_zone*; do
        read -r ty < "$z/type"; [ "$ty" = tj-therm ] && { read -r t < "$z/temp"; break; }
    done; echo "temp=$((t / 1000))"
    q=$(awk '/wlan0/ {gsub(/\./,"",$3); print $3}' /proc/net/wireless 2>/dev/null); echo "wifi=${q:-0}"
    ;;
screen)
    light
    j="${2:-}"
    tmux capture-pane -p -J -S -80 -t "rd_$j" 2>/dev/null || {
        echo "(제어판으로 띄운 '$j' 화면이 없습니다)"
        [ -n "$j" ] && pids_of "$(pat_of "$j")" | head -1 | grep -q . && echo "(단, 제어판 밖에서 띄운 $j 가 돌고 있습니다 — NX 바탕화면 터미널)"
    }
    ;;
logs)
    light
    # tail -F: 새로 시작하면 *_last.log 링크가 바뀌어도 따라간다. '==> 파일 <==' 머리줄로 제어판이 탭을 나눈다
    exec tail -n 40 -F $(for j in $JOBS; do echo "$LOGDIR/${j}_last.log"; done) "$JOB_LOG" 2>/dev/null
    ;;
start)
    # ★ light 를 부르지 않는다 (위 원칙). 이 셸이 nice 상태면 띄우지 않는다
    if [ "$(nice)" != 0 ]; then echo "✗ 명령 연결이 nice $(nice) 상태입니다 — 띄우지 않습니다. 제어판을 다시 여세요"; exit 1; fi
    job="${2:-}"; PROCS="$(procs)"
    [ -n "$(pat_of "$job")" ] || { echo "✗ start sensors|slam|low|nav|search"; exit 1; }
    if has "$(pat_of "$job")"; then echo "✗ $job 이(가) 이미 돌고 있습니다 — 먼저 정지하세요"; exit 1; fi
    case "$job" in
    sensors)
        has slam_mapping.sh && { echo "✗ SLAM 이 센서를 이미 띄웠습니다 (SLAM 은 센서 포함)"; exit 1; }
        run_in_tmux sensors "./start_sensors.sh --rs-sim" ;;
    slam)
        has start_sensors.sh && { echo "✗ '센서만' 이 돌고 있습니다 — 먼저 정지하세요 (SLAM 이 센서를 같이 띄웁니다)"; exit 1; }
        run_in_tmux slam "./slam_mapping.sh --with-realsense-sim rviz:=false pgo:=false" ;;
    low)
        m="${3:-old}"; s="${4:-remote}"
        [[ "$m" =~ ^[A-Za-z0-9_]+$ ]] || { echo "✗ 모델 이름: $m"; exit 1; }
        [[ "$s" =~ ^(remote|topic)$ ]] || { echo "✗ 명령 출처: $s (remote|topic)"; exit 1; }
        # start_hipan_live.sh 와 같은 표
        case $m in old) W=hipan_low_student.bin; UC=true ;; v3) W=hipan_low_student_v3.bin; UC=false ;; *) W=hipan_low_student_$m.bin; UC=true ;; esac
        [ -f "$ROOT/deploy/weights/$W" ] || { echo "✗ 가중치 없음: deploy/weights/$W"; exit 1; }
        run_in_tmux low "source env.sh; source ~/slam_ws/install/setup.bash; cd deploy; mkdir -p logs/live; ros2 run go2_hipan_low hipan_low_node --ros-args -p weights:=$ROOT/deploy/weights/$W -p use_contact:=$UC -p mode:=live -p cmd_source:=$s -p log_dir:=$ROOT/deploy/logs/live" ;;
    nav)
        has hipan_low_node || { echo "✗ 하위 보행(low)이 먼저 돌아야 합니다"; exit 1; }
        g=""
        if [ -n "${3:-}" ]; then
            [[ "${3}${4:-}" =~ ^[-0-9.]+$ ]] || { echo "✗ 목표 x y: ${3:-} ${4:-}"; exit 1; }
            g="--goal $3 ${4:-0}"
        fi
        v="${5:-0.3}"; [[ "$v" =~ ^0?\.[0-9]+$|^1(\.0+)?$ ]] || { echo "✗ 속도 v (0~1): $v"; exit 1; }
        run_in_tmux nav "source env.sh; cd deploy; export OPENBLAS_NUM_THREADS=1; nice -n 5 python3 run_hipan_nav.py $g --v_des $v --live" ;;
    search)
        md="${3:-shadow}"; [[ "$md" =~ ^(shadow|live)$ ]] || { echo "✗ shadow|live: $md"; exit 1; }
        has fastlio_mapping || echo "⚠ SLAM(FAST-LIO)이 안 돌고 있습니다 — 수색은 지도가 생길 때까지 기다립니다"
        f=""; [ "$md" = live ] && f="--live"
        run_in_tmux search "source env.sh; cd deploy; nice -n 5 python3 rudolph_search.py $f" ;;
    esac
    ;;
stop)
    job="${2:-}"
    case "$job" in
    low)
        # ★ kill 금지: 노드가 스스로 서기 → 앉기 → 종료
        "$ROOT/deploy/stop_hipan_low.sh" ;;
    sensors|slam|nav|search)
        if sess "$job" && [ -n "$(pids_of "$(pat_of "$job")")" ]; then
            tmux send-keys -t "rd_$job" C-c
        else
            for p in $(pids_of "$(pat_of "$job")"); do kill -INT "$p" 2>/dev/null; done
        fi
        for i in $(seq 1 20); do
            [ -z "$(pids_of "$(pat_of "$job")")" ] && { echo "✓ 정지: $job"; exit 0; }
            sleep 0.5
        done
        echo "✗ $job 이(가) 10 초 안에 안 내려갔습니다 — 화면 탭을 보세요" ; exit 1 ;;
    *) echo "✗ stop sensors|slam|low|nav|search"; exit 1 ;;
    esac
    ;;
savemap)
    envs
    { echo; echo "=== $(date +%H:%M:%S) 맵 저장 ==="; } >> "$JOB_LOG"
    timeout 30 ros2 service call /map_accumulator/save_map std_srvs/srv/Trigger 2>&1 | tail -3 | tee -a "$JOB_LOG"
    ;;
mode)
    a="${2:-check}"
    envs
    { echo; echo "=== $(date +%H:%M:%S) 내장 보행 모드: $a ==="; } >> "$JOB_LOG"
    case "$a" in
        check)   python3 "$ROOT/deploy/mode_switch.py" check 2>&1 | tee -a "$JOB_LOG" ;;
        release)
            PROCS="$(procs)"
            python3 "$ROOT/deploy/mode_switch.py" release --yes 2>&1 | tee -a "$JOB_LOG" ;;
        restore)
            PROCS="$(procs)"
            has hipan_low_node && { echo "✗ 하위 보행 노드가 돌고 있습니다 — 먼저 '앉히고 종료'" | tee -a "$JOB_LOG"; exit 1; }
            python3 "$ROOT/deploy/mode_switch.py" select mcf 2>&1 | tee -a "$JOB_LOG" ;;
        *) echo "✗ mode check|release|restore"; exit 1 ;;
    esac
    ;;
clue)
    envs
    txt="$(printf '%s' "${2:-}" | base64 -d 2>/dev/null)" || { echo "✗ 단서 인코딩"; exit 1; }
    [ -n "$txt" ] || { echo "✗ 빈 단서"; exit 1; }
    printf '%s' "$txt" | python3 "$PANEL/send_clue.py" clue
    ;;
victim)
    envs
    python3 "$PANEL/send_clue.py" victim
    ;;
bridge)
    ZB="$ROOT/tools/zenoh/aarch64/zenoh-bridge-ros2dds"
    case "${2:-start}" in
    start)
        [ -n "$(pids_of zenoh-bridge-ros2dds)" ] && { echo "✓ 브리지 이미 실행 중"; exit 0; }
        [ -x "$ZB" ] || { echo "✗ 브리지 실행 파일 없음: $ZB (tools/zenoh/fetch.sh)"; exit 1; }
        # 보행과 무관한 중계라 코어 0-3·nice 10 (보행 노드는 rt 우선순위로 다른 코어)
        log="$LOGDIR/bridge_$(date +%Y%m%d_%H%M%S).log"
        ln -sfn "$log" "$LOGDIR/bridge_last.log"
        tmux kill-session -t rd_bridge 2>/dev/null
        tmux new -d -s rd_bridge "source '$ROOT/env.sh'; exec taskset -c 0-3 nice -n 10 '$ZB' -c '$ROOT/tools/zenoh/nx.json5' 2>&1 | tee '$log'"
        # 내장 머리 카메라 → /go2/front/image (하드웨어 디코딩, 480x270 2 Hz). 브리지와 같이 켜고 같이 끈다
        if [ -z "$(pids_of front_cam_pub.py)" ]; then
            tmux kill-session -t rd_frontcam 2>/dev/null
            tmux new -d -s rd_frontcam "source '$ROOT/env.sh'; exec taskset -c 0-3 nice -n 10 python3 '$ROOT/tools/zenoh/front_cam_pub.py' 2>&1 | tee '$LOGDIR/frontcam_last.log'"
        fi
        sleep 1
        [ -n "$(pids_of zenoh-bridge-ros2dds)" ] && echo "✓ 브리지 시작 (127.0.0.1:7447) + 머리 카메라" || { echo "✗ 브리지가 바로 끝남 — $log"; tail -3 "$log"; exit 1; }
        ;;
    stop)
        for p in $(pids_of zenoh-bridge-ros2dds) $(pids_of front_cam_pub.py); do kill -INT "$p" 2>/dev/null; done
        sleep 1
        for p in $(pids_of zenoh-bridge-ros2dds) $(pids_of front_cam_pub.py); do kill "$p" 2>/dev/null; done
        tmux kill-session -t rd_bridge 2>/dev/null; tmux kill-session -t rd_frontcam 2>/dev/null
        echo "✓ 브리지 정지" ;;
    *) echo "✗ bridge start|stop"; exit 1 ;;
    esac
    ;;
tunnel)
    # link.sh 의 'ssh -L ... remote.sh tunnel' 로만 불린다. $PPID = 이 터널 연결의 sshd
    taskset -acp 0-3 "$PPID" >/dev/null 2>&1
    renice -n 19 -p "$PPID" >/dev/null 2>&1
    light
    # 연결이 끊기면 echo 가 SIGPIPE 로 끝나 남지 않는다
    while sleep 5; do echo || exit 0; done
    ;;
usage)
    light
    for f in start_sensors.sh slam_mapping.sh deploy/start_hipan_live.sh deploy/stop_hipan_low.sh deploy/nav_live.sh deploy/mode_switch.py deploy/rudolph_search.py; do
        [ -f "$ROOT/$f" ] || continue
        echo "@@@ $f"
        case "$f" in
            *.py) awk 'NR==1 && /^"""/ {p=1; sub(/^"""/,""); print; next} p && /"""/ {sub(/""".*/,""); print; exit} p {print}' "$ROOT/$f" ;;
            *)    awk 'NR==1{next} /^#/{print; next} {exit}' "$ROOT/$f" ;;
        esac
    done
    ;;
*)
    sed -n '3,32p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
