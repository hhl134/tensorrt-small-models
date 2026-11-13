cmake_minimum_required(VERSION 3.5)
project(superpointglue)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "release")
add_definitions(-w)

# Local TensorRT 10.8. The upstream project targeted TensorRT 8.4.
set(TENSORRT_ROOT "/home/hhl/workspace/3rdparty/TensorRT-10.8.0.43" CACHE PATH "TensorRT installation root")
if(NOT EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
  message(FATAL_ERROR "TensorRT headers not found under ${TENSORRT_ROOT}")
endif()

include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)
# DT_RPATH (not RUNPATH) so libnvinfer can dlopen libnvinfer_builder_resource
# from this directory without a manual LD_LIBRARY_PATH.
set(CMAKE_BUILD_RPATH "${TENSORRT_ROOT}/lib")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")

add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/tensorrtbuffer)


find_package(OpenCV 4.2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)


include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${TENSORRT_ROOT}/include
)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-O3")
cuda_add_library(${PROJECT_NAME}_lib SHARED
  src/super_point.cpp
  src/super_glue.cpp
  src/grid_sample.cu
)

target_link_libraries(${PROJECT_NAME}_lib
  nvinfer
  nvonnxparser
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  yaml-cpp
  tensorrtbuffer
)

add_executable(${PROJECT_NAME}_image inference_image.cpp)
add_executable(${PROJECT_NAME}_sequence inference_sequence.cpp)
add_executable(${PROJECT_NAME}_hpatches inference_hpatches.cpp)

target_link_libraries(${PROJECT_NAME}_image ${PROJECT_NAME}_lib)
target_link_libraries(${PROJECT_NAME}_sequence  ${PROJECT_NAME}_lib)
target_link_libraries(${PROJECT_NAME}_hpatches ${PROJECT_NAME}_lib)
