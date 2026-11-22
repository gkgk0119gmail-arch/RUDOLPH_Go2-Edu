# Go2 + 센서 ROS2 환경 (사용법: source <repo>/env.sh)
GO2_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GO2_REPO
source /opt/ros/foxy/setup.bash
source $HOME/unitree_ros2/cyclonedds_ws/install/setup.bash
source $HOME/hesai_ws/install/setup.bash
[ -f $HOME/slam_ws/install/setup.bash ] && source $HOME/slam_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# eth0 + 데이터는 유니캐스트 (로컬 대용량 토픽이 Go2 랜선으로 안 나가게), 수신버퍼 16MB
export CYCLONEDDS_URI="file://$GO2_REPO/cyclonedds.xml"
export PYTHONUNBUFFERED=1
