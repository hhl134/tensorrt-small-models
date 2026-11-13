cmake_minimum_required(VERSION 3.16)
project(aliked_lightglue)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

set(TENSORRT_ROOT "/home/hhl/workspace/3rdparty/TensorRT-10.8.0.43" CACHE PATH "TensorRT installation root")
set(SPSG_ROOT "/home/hhl/workspace/deploy/tensorrt/SuperPoint-SuperGlue-TensorRT" CACHE PATH "SPSG tree used for tensorrtbuffer")
if(NOT EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
    message(FATAL_ERROR "TensorRT headers not found under ${TENSORRT_ROOT}")
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${SPSG_ROOT}
    ${SPSG_ROOT}/3rdparty/tensorrtbuffer/include
    ${TENSORRT_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

link_directories(${TENSORRT_ROOT}/lib)
set(CMAKE_BUILD_RPATH "${TENSORRT_ROOT}/lib")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")

add_library(aliked_lightglue_lib SHARED src/pipeline.cpp)
target_link_libraries(aliked_lightglue_lib
    nvinfer
    nvonnxparser
    ${SPSG_ROOT}/3rdparty/tensorrtbuffer/lib/libtensorrtbuffer.so
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    yaml-cpp
)

add_executable(aliked_lightglue_image inference_image.cpp)
target_link_libraries(aliked_lightglue_image aliked_lightglue_lib)

add_executable(aliked_lightglue_hpatches inference_hpatches.cpp)
target_link_libraries(aliked_lightglue_hpatches aliked_lightglue_lib)
