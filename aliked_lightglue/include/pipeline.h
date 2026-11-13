#ifndef ALIKED_LIGHTGLUE_PIPELINE_H_
#define ALIKED_LIGHTGLUE_PIPELINE_H_

#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>

#include "3rdparty/tensorrtbuffer/include/buffers.h"
#include "3rdparty/tensorrtbuffer/include/logger.h"

struct AlikedConfig {
    int image_width{};
    int image_height{};
    int max_keypoints{};
    int desc_dim{};
    float detection_threshold{};
    std::vector<std::string> input_tensor_names;
    std::vector<std::string> output_tensor_names;
    std::string onnx_file;
    std::string engine_file;
};

struct LightGlueConfig {
    float match_threshold{};
    int min_keypoints{};
    int opt_keypoints{};
    int max_keypoints{};
    int desc_dim{};
    bool fp16{true};
    std::vector<std::string> input_tensor_names;
    std::vector<std::string> output_tensor_names;
    std::string onnx_file;
    std::string engine_file;
};

struct PipelineConfig {
    AlikedConfig aliked;
    LightGlueConfig lightglue;

    PipelineConfig(const std::string &config_file, const std::string &model_dir);
};

struct Features {
    int count{0};
    std::vector<float> scores;
    std::vector<float> keypoints;
    std::vector<float> descriptors;
};

class TrtModule {
public:
    TrtModule(std::string onnx_file, std::string engine_file,
              std::vector<std::string> input_names, std::vector<std::string> output_names);
    ~TrtModule();

    bool build();
    bool infer();

protected:
    bool construct();
    bool deserialize();
    void save();
    bool bind();
    bool bind_max_shapes();

    std::string onnx_file_;
    std::string engine_file_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    int profile_min_k_{0};
    int profile_opt_k_{0};
    int profile_max_k_{0};
    int desc_dim_{128};
    bool allow_fp16_{true};
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    tensorrt_common::TensorRTUniquePtr<nvinfer1::IExecutionContext> context_;
    std::unique_ptr<tensorrt_buffer::BufferManager> buffers_;
    cudaStream_t stream_{nullptr};
};

class Aliked : public TrtModule {
public:
    explicit Aliked(const AlikedConfig &config);

    bool infer(const cv::Mat &bgr, Features &features);

private:
    AlikedConfig config_;
};

class LightGlueMatcher : public TrtModule {
public:
    explicit LightGlueMatcher(const LightGlueConfig &config);

    bool match(const Features &left, const Features &right, std::vector<cv::DMatch> &matches);

private:
    LightGlueConfig config_;
};

#endif
