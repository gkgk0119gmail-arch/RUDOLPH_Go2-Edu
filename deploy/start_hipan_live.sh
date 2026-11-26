#!/bin/bash
# Start go2_hipan_low LIVE in a terminal on the NX desktop (sport mode must be released; robot lying).
export DISPLAY=:0 XAUTHORITY=$HOME/.Xauthority DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
gnome-terminal --title="HiPAN low (walking policy)" -- bash -c "source ~/Desktop/3DOG/env.sh; source ~/slam_ws/install/setup.bash; cd ~/Desktop/3DOG/deploy; ros2 run go2_hipan_low hipan_low_node --ros-args -p weights:=\$HOME/Desktop/3DOG/deploy/weights/hipan_low_student.bin -p mode:=live -p log_dir:=\$HOME/Desktop/3DOG/deploy/logs/live 2>&1 | tee logs/live/node.out; exec bash" >/dev/null 2>&1
