#include "pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <utility>

#include <yaml-cpp/yaml.h>

using namespace tensorrt_common;
using namespace tensorrt_log;
using namespace tensorrt_buffer;
using namespace nvinfer1;

namespace {

std::string join_path(const std::string &folder, const std::string &file) {
    if (folder.empty()) {
        return file;
    }
    if (folder.back() == '/') {
        return folder + file;
    }
    return folder + "/" + file;
}

std::vector<std::string> read_names(const YAML::Node &node) {
    std::vector<std::string> names;
    for (size_t i = 0; i < node.size(); ++i) {
        names.push_back(node[i].as<std::string>());
    }
    return names;
}

}  // namespace

PipelineConfig::PipelineConfig(const std::string &config_file, const std::string &model_dir) {
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node aliked_node = root["aliked"];
    aliked.image_width = aliked_node["image_width"].as<int>();
    aliked.image_height = aliked_node["image_height"].as<int>();
    aliked.max_keypoints = aliked_node["max_keypoints"].as<int>();
    aliked.desc_dim = aliked_node["desc_dim"].as<int>();
    aliked.detection_threshold = aliked_node["detection_threshold"].as<float>();
    aliked.input_tensor_names = read_names(aliked_node["input_tensor_names"]);
    aliked.output_tensor_names = read_names(aliked_node["output_tensor_names"]);
    aliked.onnx_file = join_path(model_dir, aliked_node["onnx_file"].as<std::string>());
    aliked.engine_file = join_path(model_dir, aliked_node["engine_file"].as<std::string>());

    YAML::Node glue_node = root["lightglue"];
    lightglue.match_threshold = glue_node["match_threshold"].as<float>();
    lightglue.min_keypoints = glue_node["min_keypoints"].as<int>();
    lightglue.opt_keypoints = glue_node["opt_keypoints"].as<int>();
    lightglue.max_keypoints = glue_node["max_keypoints"].as<int>();
    lightglue.desc_dim = glue_node["desc_dim"].as<int>();
    lightglue.fp16 = !glue_node["fp16"] || glue_node["fp16"].as<bool>();
    lightglue.input_tensor_names = read_names(glue_node["input_tensor_names"]);
    lightglue.output_tensor_names = read_names(glue_node["output_tensor_names"]);
    lightglue.onnx_file = join_path(model_dir, glue_node["onnx_file"].as<std::string>());
    lightglue.engine_file = join_path(model_dir, glue_node["engine_file"].as<std::string>());
}

TrtModule::TrtModule(std::string onnx_file, std::string engine_file,
                     std::vector<std::string> input_names, std::vector<std::string> output_names)
        : onnx_file_(std::move(onnx_file)),
          engine_file_(std::move(engine_file)),
          input_names_(std::move(input_names)),
          output_names_(std::move(output_names)) {
    setReportableSeverity(Logger::Severity::kINFO);
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        stream_ = nullptr;
    }
}

TrtModule::~TrtModule() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

bool TrtModule::deserialize() {
    std::ifstream file(engine_file_, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);
    std::vector<char> stream(size);
    file.read(stream.data(), static_cast<std::streamsize>(size));
    TensorRTUniquePtr<IRuntime> runtime{createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime) {
        return false;
    }
    engine_.reset(runtime->deserializeCudaEngine(stream.data(), size));
    return engine_ != nullptr;
}

void TrtModule::save() {
    if (engine_file_.empty() || !engine_) {
        return;
    }
    TensorRTUniquePtr<IHostMemory> data{engine_->serialize()};
    if (!data) {
        return;
    }
    std::ofstream file(engine_file_, std::ios::binary);
    if (!file) {
        return;
    }
    file.write(reinterpret_cast<const char *>(data->data()), static_cast<std::streamsize>(data->size()));
}

bool TrtModule::construct() {
    auto builder = TensorRTUniquePtr<IBuilder>(createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }
    const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    auto config = TensorRTUniquePtr<IBuilderConfig>(builder->createBuilderConfig());
    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!network || !config || !parser) {
        return false;
    }
    if (!parser->parseFromFile(onnx_file_.c_str(), static_cast<int>(gLogger.getReportableSeverity()))) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 8192_MiB);
    if (profile_max_k_ > 0) {
        auto *profile = builder->createOptimizationProfile();
        const int desc_dim = desc_dim_;
        auto set_pair = [&](const std::string &kpt_name, const std::string &desc_name) {
            profile->setDimensions(kpt_name.c_str(), OptProfileSelector::kMIN, Dims3(1, profile_min_k_, 2));
            profile->setDimensions(kpt_name.c_str(), OptProfileSelector::kOPT, Dims3(1, profile_opt_k_, 2));
            profile->setDimensions(kpt_name.c_str(), OptProfileSelector::kMAX, Dims3(1, profile_max_k_, 2));
            profile->setDimensions(desc_name.c_str(), OptProfileSelector::kMIN, Dims3(1, profile_min_k_, desc_dim));
            profile->setDimensions(desc_name.c_str(), OptProfileSelector::kOPT, Dims3(1, profile_opt_k_, desc_dim));
            profile->setDimensions(desc_name.c_str(), OptProfileSelector::kMAX, Dims3(1, profile_max_k_, desc_dim));
        };
        set_pair(input_names_[0], input_names_[1]);
        set_pair(input_names_[2], input_names_[3]);
        config->addOptimizationProfile(profile);
    }
    if (allow_fp16_) {
        config->setFlag(BuilderFlag::kFP16);
        config->setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
    }
    int fp32_layers = 0;
    for (int i = 0; i < network->getNbLayers() && allow_fp16_; ++i) {
        auto *layer = network->getLayer(i);
        bool force_fp32 = false;
        if (layer->getType() == LayerType::kUNARY) {
            auto operation = static_cast<IUnaryLayer *>(layer)->getOperation();
            force_fp32 = operation == UnaryOperation::kEXP || operation == UnaryOperation::kLOG;
        } else if (layer->getType() == LayerType::kREDUCE) {
            auto operation = static_cast<IReduceLayer *>(layer)->getOperation();
            force_fp32 = operation == ReduceOperation::kSUM;
        }
        if (!force_fp32) {
            continue;
        }
        layer->setPrecision(DataType::kFLOAT);
        for (int output_index = 0; output_index < layer->getNbOutputs(); ++output_index) {
            layer->setOutputType(output_index, DataType::kFLOAT);
        }
        ++fp32_layers;
    }
    std::cout << "FP32 Exp/Log/ReduceSum layers: " << fp32_layers << std::endl;
    auto profile_stream = makeCudaStream();
    if (!profile_stream) {
        return false;
    }
    config->setProfileStream(*profile_stream);
    TensorRTUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }
    TensorRTUniquePtr<IRuntime> runtime{createInferRuntime(gLogger.getTRTLogger())};
    if (!runtime) {
        return false;
    }
    engine_.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_) {
        return false;
    }
    save();
    return true;
}

bool TrtModule::build() {
    if (!deserialize() && !construct()) {
        return false;
    }
    context_ = TensorRTUniquePtr<IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        return false;
    }
    if (!bind_max_shapes()) {
        return false;
    }
    buffers_ = std::make_unique<BufferManager>(engine_, 0, context_.get());
    return bind();
}

bool TrtModule::bind_max_shapes() {
    if (engine_->getNbOptimizationProfiles() > 0 && stream_) {
        context_->setOptimizationProfileAsync(0, stream_);
    }
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char *name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) != TensorIOMode::kINPUT) {
            continue;
        }
        const auto dims = engine_->getTensorShape(name);
        bool dynamic = false;
        for (int d = 0; d < dims.nbDims; ++d) {
            dynamic = dynamic || dims.d[d] < 0;
        }
        if (!dynamic) {
            continue;
        }
        const auto max_dims = engine_->getProfileShape(name, 0, OptProfileSelector::kMAX);
        if (!context_->setInputShape(name, max_dims)) {
            std::cerr << "Failed to set max shape for " << name << std::endl;
            return false;
        }
    }
    return true;
}

bool TrtModule::bind() {
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char *name = engine_->getIOTensorName(i);
        if (!context_->setTensorAddress(name, buffers_->getDeviceBuffer(name))) {
            std::cerr << "Failed to bind tensor " << name << std::endl;
            return false;
        }
    }
    return true;
}

bool TrtModule::infer() {
    if (!stream_) {
        return false;
    }
    buffers_->copyInputToDeviceAsync(stream_);
    if (!context_->enqueueV3(stream_)) {
        return false;
    }
    buffers_->copyOutputToHostAsync(stream_);
    return cudaStreamSynchronize(stream_) == cudaSuccess;
}

Aliked::Aliked(const AlikedConfig &config)
        : TrtModule(config.onnx_file, config.engine_file, config.input_tensor_names, config.output_tensor_names),
          config_(config) {}

bool Aliked::infer(const cv::Mat &bgr, Features &features) {
    cv::Mat rgb, resized, fp32;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(config_.image_width, config_.image_height), 0, 0, cv::INTER_LINEAR);
    resized.convertTo(fp32, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(fp32, channels);
    auto *input = static_cast<float *>(buffers_->getHostBuffer(config_.input_tensor_names[0]));
    const int plane = config_.image_height * config_.image_width;
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input + c * plane, channels[c].ptr<float>(), plane * sizeof(float));
    }
    if (!TrtModule::infer()) {
        return false;
    }
    const int capacity = config_.max_keypoints;
    const int desc_dim = config_.desc_dim;
    auto *keypoints = static_cast<float *>(buffers_->getHostBuffer(config_.output_tensor_names[0]));
    auto *scores = static_cast<float *>(buffers_->getHostBuffer(config_.output_tensor_names[1]));
    auto *descriptors = static_cast<float *>(buffers_->getHostBuffer(config_.output_tensor_names[2]));
    auto *peaks = static_cast<float *>(buffers_->getHostBuffer(config_.output_tensor_names[3]));
    features.scores.clear();
    features.keypoints.clear();
    features.descriptors.clear();
    features.scores.reserve(capacity);
    features.keypoints.reserve(capacity * 2);
    features.descriptors.reserve(capacity * desc_dim);
    for (int i = 0; i < capacity; ++i) {
        if (peaks[i] <= config_.detection_threshold) {
            continue;
        }
        features.scores.push_back(scores[i]);
        features.keypoints.push_back(keypoints[i * 2]);
        features.keypoints.push_back(keypoints[i * 2 + 1]);
        features.descriptors.insert(
                features.descriptors.end(),
                descriptors + i * desc_dim,
                descriptors + (i + 1) * desc_dim);
    }
    features.count = static_cast<int>(features.scores.size());
    if (const char *keep_env = std::getenv("ALG_KEEP")) {
        const int keep = std::atoi(keep_env);
        if (keep > 0 && features.count > keep) {
            features.scores.resize(static_cast<size_t>(keep));
            features.keypoints.resize(static_cast<size_t>(keep) * 2);
            features.descriptors.resize(static_cast<size_t>(keep) * static_cast<size_t>(desc_dim));
            features.count = keep;
        }
    }
    return true;
}

LightGlueMatcher::LightGlueMatcher(const LightGlueConfig &config)
        : TrtModule(config.onnx_file, config.engine_file, config.input_tensor_names, config.output_tensor_names),
          config_(config) {
    profile_min_k_ = config.min_keypoints;
    profile_opt_k_ = config.opt_keypoints;
    profile_max_k_ = config.max_keypoints;
    desc_dim_ = config.desc_dim;
    allow_fp16_ = config.fp16;
}

bool LightGlueMatcher::match(const Features &left, const Features &right, std::vector<cv::DMatch> &matches) {
    matches.clear();
    if (left.count <= 0 || right.count <= 0) {
        return true;
    }
    if (left.count > config_.max_keypoints || right.count > config_.max_keypoints) {
        std::cerr << "Keypoint count exceeds the LightGlue profile." << std::endl;
        return false;
    }
    auto set_shape = [&](const std::string &name, const Dims3 &dims) {
        return context_->setInputShape(name.c_str(), dims);
    };
    const int desc_dim = config_.desc_dim;
    if (!set_shape(config_.input_tensor_names[0], Dims3(1, left.count, 2)) ||
        !set_shape(config_.input_tensor_names[1], Dims3(1, left.count, desc_dim)) ||
        !set_shape(config_.input_tensor_names[2], Dims3(1, right.count, 2)) ||
        !set_shape(config_.input_tensor_names[3], Dims3(1, right.count, desc_dim))) {
        std::cerr << "Failed to set LightGlue input shapes." << std::endl;
        return false;
    }
    auto copy_points = [](const std::string &name, const Features &features, BufferManager &buffers) {
        auto *dst = static_cast<float *>(buffers.getHostBuffer(name));
        std::memcpy(dst, features.keypoints.data(), features.keypoints.size() * sizeof(float));
    };
    auto copy_desc = [](const std::string &name, const Features &features, BufferManager &buffers) {
        auto *dst = static_cast<float *>(buffers.getHostBuffer(name));
        std::memcpy(dst, features.descriptors.data(), features.descriptors.size() * sizeof(float));
    };
    copy_points(config_.input_tensor_names[0], left, *buffers_);
    copy_desc(config_.input_tensor_names[1], left, *buffers_);
    copy_points(config_.input_tensor_names[2], right, *buffers_);
    copy_desc(config_.input_tensor_names[3], right, *buffers_);
    if (!TrtModule::infer()) {
        return false;
    }
    auto *scores = static_cast<float *>(buffers_->getHostBuffer(config_.output_tensor_names[0]));
    const int m = left.count;
    const int n = right.count;
    const int cols = n + 1;
    std::vector<int> best_j(m), best_i(n);
    std::vector<float> best_jv(m);
    for (int i = 0; i < m; ++i) {
        int best = 0;
        float value = scores[i * cols];
        for (int j = 1; j < n; ++j) {
            const float candidate = scores[i * cols + j];
            if (candidate > value) {
                value = candidate;
                best = j;
            }
        }
        best_j[i] = best;
        best_jv[i] = value;
    }
    for (int j = 0; j < n; ++j) {
        int best = 0;
        float value = scores[j];
        for (int i = 1; i < m; ++i) {
            const float candidate = scores[i * cols + j];
            if (candidate > value) {
                value = candidate;
                best = i;
            }
        }
        best_i[j] = best;
    }
    for (int i = 0; i < m; ++i) {
        const int j = best_j[i];
        if (best_i[j] != i) {
            continue;
        }
        const float probability = std::exp(best_jv[i]);
        if (probability <= config_.match_threshold) {
            continue;
        }
        cv::DMatch match;
        match.queryIdx = i;
        match.trainIdx = j;
        match.distance = 1.f - probability;
        matches.push_back(match);
    }
    return true;
}
