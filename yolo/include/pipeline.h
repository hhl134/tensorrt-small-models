#ifndef YOLO_ONNX_PIPELINE_H_
#define YOLO_ONNX_PIPELINE_H_

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include "3rdparty/tensorrtbuffer/include/buffers.h"
#include "3rdparty/tensorrtbuffer/include/logger.h"

struct YoloConfig {
    int image_width{};
    int image_height{};
    int num_classes{};
    float conf_threshold{};
    float nms_threshold{};
    bool fp16{true};
    bool int8{false};
    std::string calib_dir;
    std::string calib_cache;
    std::vector<std::string> input_tensor_names;
    std::vector<std::string> output_tensor_names;
    std::string onnx_file;
    std::string engine_file;
};

struct PipelineConfig {
    YoloConfig yolo;

    PipelineConfig(const std::string &config_file, const std::string &model_dir);
};

class YoloOnnx {
public:
    explicit YoloOnnx(const YoloConfig &config);
    ~YoloOnnx();

    bool build();
    bool infer(const float *nchw, std::vector<float> &output, nvinfer1::Dims &output_dims);
    void *deviceBuffer(const std::string &name) const;
    nvinfer1::Dims tensorShape(const std::string &name) const;
    cudaStream_t stream() const { return stream_; }
    bool enqueue();

private:
    bool construct();
    bool deserialize();
    void save();
    bool bind();

    YoloConfig config_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    tensorrt_common::TensorRTUniquePtr<nvinfer1::IExecutionContext> context_;
    std::unique_ptr<tensorrt_buffer::BufferManager> buffers_;
    cudaStream_t stream_{nullptr};
};

#endif
