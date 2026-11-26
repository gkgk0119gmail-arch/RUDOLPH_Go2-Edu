cmake_minimum_required(VERSION 3.8)
project(go2_hipan_low)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -Wall)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unitree_go REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(include ${EIGEN3_INCLUDE_DIR})

add_executable(hipan_low_node src/hipan_low_node.cpp)
target_compile_features(hipan_low_node PUBLIC cxx_std_17)
ament_target_dependencies(hipan_low_node rclcpp std_msgs unitree_go)
target_link_libraries(hipan_low_node pthread)

add_executable(hipan_low_selftest src/selftest.cpp)
target_compile_features(hipan_low_selftest PUBLIC cxx_std_17)

install(TARGETS hipan_low_node hipan_low_selftest DESTINATION lib/${PROJECT_NAME})
ament_package()
