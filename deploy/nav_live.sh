#!/bin/bash
# (Re)start the nav student in a terminal on the NX desktop.  ./nav_live.sh [goal_x goal_y] [extra args...]
#   default: goal 2 m straight ahead of the current pose, v_des 0.3, LIVE (publishes /hipan/cmd)
for p in $(pgrep -f "python3 run_hipan_nav.py"); do kill -INT "$p"; done
sleep 2
GX=${1:-2}; GY=${2:-0}; shift 2 2>/dev/null
TAG=$(date +%H%M%S)
export DISPLAY=:0 XAUTHORITY=$HOME/.Xauthority DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
gnome-terminal --title="HiPAN nav LIVE goal ($GX, $GY) $TAG" -- bash -c "source ~/Desktop/3DOG/env.sh; cd ~/Desktop/3DOG/deploy; export OPENBLAS_NUM_THREADS=1; nice -n 5 python3 run_hipan_nav.py --goal $GX $GY --v_des 0.3 --live $* 2>&1 | tee logs/nav_live_$TAG.out; exec bash" >/dev/null 2>&1
sleep 5
grep LIVE ~/Desktop/3DOG/deploy/logs/nav_live_$TAG.out | sed "s/.*hipan_nav\]: //"
