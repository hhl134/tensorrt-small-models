#include "pipeline.h"

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

void read_rgb(const YAML::Node &node, float *values) {
    for (int i = 0; i < 3; ++i) {
        values[i] = node[i].as<float>();
    }
}

}  // namespace

PipelineConfig::PipelineConfig(const std::string &config_file, const std::string &model_dir) {
    YAML::Node root = YAML::LoadFile(config_file);
    YAML::Node node = root["mixvpr"];
    mixvpr.image_width = node["image_width"].as<int>();
    mixvpr.image_height = node["image_height"].as<int>();
    mixvpr.desc_dim = node["desc_dim"].as<int>();
    mixvpr.fp16 = !node["fp16"] || node["fp16"].as<bool>();
    if (node["force_norm_fp32"]) {
        mixvpr.force_norm_fp32 = node["force_norm_fp32"].as<bool>();
    }
    read_rgb(node["mean"], mixvpr.mean);
    read_rgb(node["std"], mixvpr.std);
    mixvpr.input_tensor_names = read_names(node["input_tensor_names"]);
    mixvpr.output_tensor_names = read_names(node["output_tensor_names"]);
    mixvpr.onnx_file = join_path(model_dir, node["onnx_file"].as<std::string>());
    mixvpr.engine_file = join_path(model_dir, node["engine_file"].as<std::string>());
}

MixVPR::MixVPR(const MixVPRConfig &config) : config_(config) {
    setReportableSeverity(Logger::Severity::kINFO);
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        stream_ = nullptr;
    }
}

MixVPR::~MixVPR() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

bool MixVPR::deserialize() {
    std::ifstream file(config_.engine_file, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.seekg(0, std::ifstream::end);
    const size_t size = file.tellg();
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

void MixVPR::save() {
    if (config_.engine_file.empty() || !engine_) {
        return;
    }
    TensorRTUniquePtr<IHostMemory> data{engine_->serialize()};
    if (!data) {
        return;
    }
    std::ofstream file(config_.engine_file, std::ios::binary);
    if (!file) {
        return;
    }
    file.write(reinterpret_cast<const char *>(data->data()), static_cast<std::streamsize>(data->size()));
}

bool MixVPR::construct() {
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
    if (!parser->parseFromFile(config_.onnx_file.c_str(), static_cast<int>(gLogger.getReportableSeverity()))) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << parser->getError(i)->desc() << std::endl;
        }
        return false;
    }
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 4096_MiB);
    if (config_.fp16) {
        config->setFlag(BuilderFlag::kFP16);
        if (config_.force_norm_fp32) {
            config->setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
        }
    }
    int fp32_layers = 0;
    for (int i = 0; i < network->getNbLayers() && config_.fp16 && config_.force_norm_fp32; ++i) {
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

bool MixVPR::bind() {
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char *name = engine_->getIOTensorName(i);
        if (!context_->setTensorAddress(name, buffers_->getDeviceBuffer(name))) {
            std::cerr << "Failed to bind tensor " << name << std::endl;
            return false;
        }
    }
    return true;
}

bool MixVPR::build() {
    if (!deserialize() && !construct()) {
        return false;
    }
    context_ = TensorRTUniquePtr<IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        return false;
    }
    buffers_ = std::make_unique<BufferManager>(engine_, 0, context_.get());
    return bind();
}

bool MixVPR::infer(const cv::Mat &bgr, std::vector<float> &descriptor) {
    if (!stream_ || bgr.empty()) {
        return false;
    }
    cv::Mat rgb, resized, fp32;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(config_.image_width, config_.image_height), 0, 0, cv::INTER_LINEAR);
    resized.convertTo(fp32, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> channels(3);
    cv::split(fp32, channels);
    auto *input = static_cast<float *>(buffers_->getHostBuffer(config_.input_tensor_names[0]));
    const int plane = config_.image_height * config_.image_width;
    for (int c = 0; c < 3; ++c) {
        float *dst = input + c * plane;
        const float *src = channels[c].ptr<float>();
        const float mean = config_.mean[c];
        const float inv_std = 1.0f / config_.std[c];
        for (int i = 0; i < plane; ++i) {
            dst[i] = (src[i] - mean) * inv_std;
        }
    }
    buffers_->copyInputToDeviceAsync(stream_);
    if (!context_->enqueueV3(stream_)) {
        return false;
    }
    buffers_->copyOutputToHostAsync(stream_);
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
        return false;
    }
    auto *output = static_cast<float *>(buffers_->getHostBuffer(config_.output_tensor_names[0]));
    descriptor.assign(output, output + config_.desc_dim);
    return true;
}
