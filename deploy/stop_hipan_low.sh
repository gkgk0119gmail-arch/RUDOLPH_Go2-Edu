#!/bin/bash
# Safe stop of go2_hipan_low: the node walks back to the standing pose, sits down slowly, then exits.
# Never kill the node while the robot stands (it collapses). Usage: ./stop_hipan_low.sh
source ~/Desktop/3DOG/env.sh >/dev/null 2>&1
ros2 topic pub --once /hipan/sit_and_exit std_msgs/msg/Empty '{}' >/dev/null
for i in $(seq 1 40); do
  pgrep -f "lib/go2_hipan_low/hipan_low_node" >/dev/null || { echo "node exited (robot lying)"; exit 0; }
  sleep 0.5
done
echo "node still running after 20 s: check the robot; emergency only: L2+B on the remote"; exit 1
