cmake_minimum_required(VERSION 2.4.6)
project(mvs_ros)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  sensor_msgs
  cv_bridge
  message_generation
  std_msgs
)

add_message_files(
  FILES
  DepthMsg.msg
  bgr_frameMSG.msg
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

catkin_package()
