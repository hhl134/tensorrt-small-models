#ifndef MIXVPR_PIPELINE_H_
#define MIXVPR_PIPELINE_H_

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>

#include "3rdparty/tensorrtbuffer/include/buffers.h"
#include "3rdparty/tensorrtbuffer/include/logger.h"

struct MixVPRConfig {
    int image_width{};
    int image_height{};
    int desc_dim{};
    bool fp16{true};
    bool force_norm_fp32{true};
    float mean[3]{};
    float std[3]{};
    std::vector<std::string> input_tensor_names;
    std::vector<std::string> output_tensor_names;
    std::string onnx_file;
    std::string engine_file;
};

struct PipelineConfig {
    MixVPRConfig mixvpr;

    PipelineConfig(const std::string &config_file, const std::string &model_dir);
};

class MixVPR {
public:
    explicit MixVPR(const MixVPRConfig &config);
    ~MixVPR();

    bool build();
    bool infer(const cv::Mat &bgr, std::vector<float> &descriptor);

private:
    bool construct();
    bool deserialize();
    void save();
    bool bind();

    MixVPRConfig config_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    tensorrt_common::TensorRTUniquePtr<nvinfer1::IExecutionContext> context_;
    std::unique_ptr<tensorrt_buffer::BufferManager> buffers_;
    cudaStream_t stream_{nullptr};
};

#endif
