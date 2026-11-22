"""D435i depth image -> /camera/depth/points (sensor_msgs/PointCloud2, frame: camera_depth_optical_frame).

realsense-ros 4.51.1's built-in pointcloud filter doesn't publish on this setup
(and it drags depth down to ~4 Hz), so we build the cloud with depth_image_proc.
"""
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    return LaunchDescription([
        ComposableNodeContainer(
            name='rs_points_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                ComposableNode(
                    package='depth_image_proc',
                    plugin='depth_image_proc::PointCloudXyzNode',
                    name='rs_point_cloud_xyz',
                    remappings=[
                        ('image_rect', '/camera/depth/image_rect_raw'),
                        ('camera_info', '/camera/depth/camera_info'),
                        ('points', '/camera/depth/points'),
                    ],
                ),
            ],
            output='screen',
        ),
    ])
