#!/bin/bash
# FAST-LIO + RealSense depth 848x480 (sim FOV) in tmux "slam", plus a viewer terminal on the NX desktop.
cd ~/Desktop/3DOG
tmux kill-session -t slam 2>/dev/null
tmux new-session -d -s slam "./slam_mapping.sh --with-realsense-sim rviz:=false pgo:=false   # pgo:= is used by the NX launch file; ignored otherwise 2>&1 | tee /tmp/slam.log"
export DISPLAY=:0 XAUTHORITY=$HOME/.Xauthority DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
gnome-terminal --title="SLAM (FAST-LIO + RealSense 848x480)" -- bash -c "tmux attach -t slam" >/dev/null 2>&1
