cmake_minimum_required(VERSION 3.8)
project(go2_imu_bridge)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(unitree_go REQUIRED)

add_executable(go2_imu_bridge src/go2_imu_bridge.cpp)
target_compile_features(go2_imu_bridge PUBLIC cxx_std_17)
ament_target_dependencies(go2_imu_bridge rclcpp sensor_msgs unitree_go)

install(TARGETS go2_imu_bridge DESTINATION lib/${PROJECT_NAME})
ament_package()
