cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
project(yolo_tensorrt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 120)

set(TENSORRT_ROOT "/home/hhl/workspace/3rdparty/TensorRT-10.8.0.43" CACHE PATH "TensorRT installation root")
set(SPSG_ROOT "/home/hhl/workspace/deploy/tensorrt/SuperPoint-SuperGlue-TensorRT" CACHE PATH "SPSG tree used for tensorrtbuffer")
set(WTS_ROOT "/home/hhl/workspace/deploy/tensorrt/tensorrtx/yolov8/yolov8_trt10" CACHE PATH "tensorrtx YOLOv8 wts sources")
if(NOT EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
    message(FATAL_ERROR "TensorRT headers not found under ${TENSORRT_ROOT}")
endif()

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(yaml-cpp REQUIRED)

link_directories(${TENSORRT_ROOT}/lib)
set(CMAKE_BUILD_RPATH "${TENSORRT_ROOT}/lib;${CMAKE_BINARY_DIR}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")

add_library(yolo_onnx_lib SHARED src/pipeline.cpp)
target_include_directories(yolo_onnx_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${SPSG_ROOT}
    ${SPSG_ROOT}/3rdparty/tensorrtbuffer/include
    ${TENSORRT_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(yolo_onnx_lib
    nvinfer
    nvonnxparser
    ${SPSG_ROOT}/3rdparty/tensorrtbuffer/lib/libtensorrtbuffer.so
    ${OpenCV_LIBS}
    CUDA::cudart
    yaml-cpp
)

add_executable(yolo_onnx inference_onnx.cpp src/yolo_gpu.cu)
target_link_libraries(yolo_onnx yolo_onnx_lib)

add_library(yolo_plugins SHARED ${WTS_ROOT}/plugin/yololayer.cu)
target_include_directories(yolo_plugins BEFORE PUBLIC
    ${WTS_ROOT}/include
    ${WTS_ROOT}/plugin
    ${TENSORRT_ROOT}/include
)
target_link_libraries(yolo_plugins nvinfer CUDA::cudart)

file(GLOB WTS_SOURCES ${WTS_ROOT}/src/*.cpp ${WTS_ROOT}/src/*.cu)
add_executable(yolo_wts ${WTS_ROOT}/yolov8_det.cpp ${WTS_SOURCES})
target_include_directories(yolo_wts BEFORE PRIVATE
    ${WTS_ROOT}/include
    ${WTS_ROOT}/plugin
    ${TENSORRT_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(yolo_wts PRIVATE USE_FP16)
target_link_libraries(yolo_wts nvinfer CUDA::cudart yolo_plugins ${OpenCV_LIBS})

# INT8 build: do not define USE_FP16 so model.cpp takes the USE_INT8 branch from config.h.
add_executable(yolo_wts_int8 ${WTS_ROOT}/yolov8_det.cpp ${WTS_SOURCES})
target_include_directories(yolo_wts_int8 BEFORE PRIVATE
    ${WTS_ROOT}/include
    ${WTS_ROOT}/plugin
    ${TENSORRT_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_compile_definitions(yolo_wts_int8 PRIVATE USE_INT8)
target_link_libraries(yolo_wts_int8 nvinfer CUDA::cudart yolo_plugins ${OpenCV_LIBS})
