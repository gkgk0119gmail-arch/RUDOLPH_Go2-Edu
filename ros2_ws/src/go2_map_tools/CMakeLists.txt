cmake_minimum_required(VERSION 3.8)
project(go2_map_tools)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

add_executable(map_accumulator src/map_accumulator.cpp)
target_compile_features(map_accumulator PUBLIC cxx_std_17)
target_include_directories(map_accumulator PUBLIC ${PCL_INCLUDE_DIRS})
target_link_libraries(map_accumulator ${PCL_LIBRARIES} stdc++fs)
ament_target_dependencies(map_accumulator rclcpp sensor_msgs std_srvs pcl_conversions)

install(TARGETS map_accumulator DESTINATION lib/${PROJECT_NAME})
ament_package()
