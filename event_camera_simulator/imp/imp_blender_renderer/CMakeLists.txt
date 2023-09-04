cmake_minimum_required(VERSION 2.8.3)
project(imp_blender_renderer)

find_package(catkin_simple REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cppzmq REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${cppzmq_INCLUDE_DIR})
catkin_simple()

set(HEADERS
    include/esim/imp_blender_renderer/blender_renderer.hpp
)

set(SOURCES
    src/blender_renderer.cpp
)

cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${ZeroMQ_LIBRARY})

cs_install()
cs_export()
