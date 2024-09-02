cmake_minimum_required(VERSION 2.4.6)
project(dso_ros)

set(CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  sensor_msgs
  cv_bridge
  message_generation
  std_msgs
)



set(DSO_PATH $ENV{DSO_PATH})
message("\n\n---- LOOKING FOR DSO at\n\"${DSO_PATH}\"")


find_package(Pangolin 0.2 REQUIRED)
message("\n\n---- FOUND Pangolin Headers at\n\"${Pangolin_INCLUDE_DIRS}\"")
message("---- FOUND Pangolin Libs at\n\"${Pangolin_LIBRARIES}\"")

find_package(OpenCV REQUIRED)
message("\n\n---- FOUND OpenCV Headers at\n\"${OpenCV_INCLUDE_DIRS}\"")
message("---- FOUND OpenCV Libs at\n\"${OpenCV_LIBS}\"")

find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread) 
find_library(DSO_LIBRARY dso ${DSO_PATH}/build/lib)

find_package(PCL 1.10 REQUIRED)

# SSE flags

set(CMAKE_CXX_FLAGS
   "${SSE_FLAGS} -O2 -g -std=c++0x -march=native -fno-omit-frame-pointer"
)

add_message_files(
  FILES
  poseMSG.msg
  bgr_frameMSG.msg
  DepthMsg.msg
)

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${DSO_PATH}/src
  ${DSO_PATH}/thirdparty/Sophus
  ${Pangolin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

include_directories(
  ${PROJECT_SOURCE_DIR}/src
  ${DSO_PATH}/src
  ${DSO_PATH}/thirdparty/Sophus
  ${OpenCV_INCLUDE_DIRS}
  ${Pangolin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
)

catkin_package()

## Declare a C++ executable

add_library(mapping SHARED
  include/pointcloudmapping.cpp
)

target_link_libraries(mapping
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(dso_live src/main.cpp)

target_link_libraries(dso_live
    ${DSO_LIBRARY}
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES})

add_executable(mapping_node src/mapping_node.cpp)

target_link_libraries(mapping_node
  ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES}
  mapping
)

add_executable(filter src/filter.cpp)

target_link_libraries(filter
  ${PCL_LIBRARIES}
  ${catkin_LIBRARIES})

install(
  TARGETS dso_live 
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


