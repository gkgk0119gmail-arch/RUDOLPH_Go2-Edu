"""Live 3D LIO mapping: Go2 IMU bridge + FAST-LIO2 (Hesai XT16) + static TFs + rviz.

TF tree:
  camera_init (FAST-LIO world) -> body (= go2_imu, from FAST-LIO)
  body -> base_link            (inverse of Go2 URDF imu_joint)
  base_link -> hesai_lidar     (top mount, yaw +90deg)
  base_link -> camera_link     (D435i on head)

Sensor drivers (Hesai / RealSense) are started separately by start_sensors.sh.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)


def static_tf(name, xyz, ypr, parent, child):
    return Node(package='tf2_ros', executable='static_transform_publisher', name=name,
                arguments=[*map(str, xyz), *map(str, ypr), parent, child], output='log')


def generate_launch_description():
    rviz = LaunchConfiguration('rviz')
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),

        Node(package='go2_imu_bridge', executable='go2_imu_bridge', name='go2_imu_bridge',
             parameters=[{'frame_id': 'body'}], output='screen'),

        Node(package='fast_lio', executable='fastlio_mapping', name='fastlio_mapping',
             parameters=[os.path.join(ROOT, 'config', 'fastlio_go2_xt16.yaml')], output='screen'),

        # voxel map for rviz (/map_cloud) + PCD saving:
        #   ros2 service call /map_accumulator/save_map std_srvs/srv/Trigger
        Node(package='go2_map_tools', executable='map_accumulator', name='map_accumulator',
             parameters=[{'save_dir': os.path.join(ROOT, 'maps')}], output='screen'),

        static_tf('tf_body_base', (0.02557, 0, -0.04232), (0, 0, 0), 'body', 'base_link'),
        static_tf('tf_base_hesai', (0.1454, 0, 0.1331), (1.5708, 0, 0), 'base_link', 'hesai_lidar'),
        static_tf('tf_base_cam', (0.32, 0, 0.055), (0, 0.227, 0), 'base_link', 'camera_link'),

        Node(package='rviz2', executable='rviz2', name='rviz2',
             arguments=['-d', os.path.join(ROOT, 'rviz', 'slam_lio.rviz')],
             condition=IfCondition(rviz), output='log'),
    ])
