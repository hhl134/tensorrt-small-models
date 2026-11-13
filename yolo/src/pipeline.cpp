#include "pipeline.h"
#include "int8_calibrator.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <yaml-cpp/yaml.h>

using namespace tensorrt_common;
using namespace tensorrt_log;
using namespace tensorrt_buffer;
using namespace nvinfer1;

namespace {

std::string join_path(const std::string &folder, const std::string &file) {
    if (folder.empty() || folder.back() == '/') {
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
    YAML::Node node = YAML::LoadFile(config_file)["yolo"];
    yolo.image_width = node["image_width"].as<int>();
    yolo.image_height = node["image_height"].as<int>();
    yolo.num_classes = node["num_classes"].as<int>();
    yolo.conf_threshold = node["conf_threshold"].as<float>();
    yolo.nms_threshold = node["nms_threshold"].as<float>();
    yolo.fp16 = !node["fp16"] || node["fp16"].as<bool>();
    yolo.int8 = node["int8"] && node["int8"].as<bool>();
    if (node["calib_dir"]) {
        yolo.calib_dir = node["calib_dir"].as<std::string>();
        if (!yolo.calib_dir.empty() && yolo.calib_dir.front() != '/') {
            yolo.calib_dir = join_path(model_dir, yolo.calib_dir);
        }
    }
    if (node["calib_cache"]) {
        yolo.calib_cache = join_path(model_dir, node["calib_cache"].as<std::string>());
    }
    yolo.input_tensor_names = read_names(node["input_tensor_names"]);
    yolo.output_tensor_names = read_names(node["output_tensor_names"]);
    yolo.onnx_file = join_path(model_dir, node["onnx_file"].as<std::string>());
    yolo.engine_file = join_path(model_dir, node["engine_file"].as<std::string>());
}

YoloOnnx::YoloOnnx(const YoloConfig &config) : config_(config) {
    setReportableSeverity(Logger::Severity::kINFO);
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        stream_ = nullptr;
    }
}

YoloOnnx::~YoloOnnx() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

bool YoloOnnx::deserialize() {
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

void YoloOnnx::save() {
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

bool YoloOnnx::construct() {
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
    std::unique_ptr<YoloInt8Calibrator> calibrator;
    if (config_.int8) {
        if (!builder->platformHasFastInt8()) {
            std::cerr << "Platform reports no fast INT8; continuing anyway." << std::endl;
        }
        std::vector<std::string> images;
        if (!config_.calib_dir.empty() && std::filesystem::is_directory(config_.calib_dir)) {
            for (const auto &entry : std::filesystem::directory_iterator(config_.calib_dir)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                const std::string ext = entry.path().extension().string();
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".JPG") {
                    images.push_back(entry.path().string());
                }
            }
            std::sort(images.begin(), images.end());
        }
        if (images.empty()) {
            std::cerr << "INT8 calibration image dir empty: " << config_.calib_dir << std::endl;
            return false;
        }
        const std::string cache =
                config_.calib_cache.empty() ? (config_.engine_file + ".calib") : config_.calib_cache;
        calibrator = std::make_unique<YoloInt8Calibrator>(images, config_.input_tensor_names[0], config_.image_width,
                                                          config_.image_height, cache, true);
        config->setFlag(BuilderFlag::kINT8);
        config->setFlag(BuilderFlag::kFP16);
        config->setInt8Calibrator(calibrator.get());
        std::cout << "Building YOLO ONNX INT8 engine with " << images.size() << " calib images." << std::endl;
    } else if (config_.fp16) {
        config->setFlag(BuilderFlag::kFP16);
    }
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

bool YoloOnnx::bind() {
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char *name = engine_->getIOTensorName(i);
        if (!context_->setTensorAddress(name, buffers_->getDeviceBuffer(name))) {
            std::cerr << "Failed to bind tensor " << name << std::endl;
            return false;
        }
    }
    return true;
}

bool YoloOnnx::build() {
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

void *YoloOnnx::deviceBuffer(const std::string &name) const {
    return buffers_->getDeviceBuffer(name);
}

Dims YoloOnnx::tensorShape(const std::string &name) const {
    return engine_->getTensorShape(name.c_str());
}

bool YoloOnnx::enqueue() {
    return stream_ && context_->enqueueV3(stream_);
}

bool YoloOnnx::infer(const float *nchw, std::vector<float> &output, Dims &output_dims) {
    if (!stream_ || nchw == nullptr) {
        return false;
    }
    const std::string &input_name = config_.input_tensor_names[0];
    const std::string &output_name = config_.output_tensor_names[0];
    auto *input = static_cast<float *>(buffers_->getHostBuffer(input_name));
    const int count = config_.image_height * config_.image_width * 3;
    std::copy(nchw, nchw + count, input);
    buffers_->copyInputToDeviceAsync(stream_);
    if (!context_->enqueueV3(stream_)) {
        return false;
    }
    buffers_->copyOutputToHostAsync(stream_);
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
        return false;
    }
    output_dims = engine_->getTensorShape(output_name.c_str());
    int64_t volume = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        volume *= output_dims.d[i];
    }
    auto *device_output = static_cast<float *>(buffers_->getHostBuffer(output_name));
    output.assign(device_output, device_output + volume);
    return true;
}
