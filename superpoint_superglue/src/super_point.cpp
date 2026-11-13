//
// Created by haoyuefan on 2021/9/22.
//
#include "super_point.h"
#include "grid_sample_cuda.h"
#include "int8_calibrator.h"
#include <utility>
#include <memory>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <dirent.h>
#include <unordered_map>
#include <opencv2/opencv.hpp>

using namespace tensorrt_common;
using namespace tensorrt_log;
using namespace tensorrt_buffer;

SuperPoint::SuperPoint(SuperPointConfig super_point_config)
        : super_point_config_(std::move(super_point_config)), engine_(
        nullptr) {
    setReportableSeverity(Logger::Severity::kINFO);
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        stream_ = nullptr;
    }
    if (const char *env = std::getenv("SPSG_GPU_SAMPLE")) {
        use_gpu_sample_ = std::atoi(env) != 0;
    }
}

SuperPoint::~SuperPoint() {
    if (d_grid_) {
        cudaFree(d_grid_);
        d_grid_ = nullptr;
    }
    if (d_sampled_) {
        cudaFree(d_sampled_);
        d_sampled_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

namespace {

std::vector<std::string> list_calib_images(const std::string &dir) {
    std::vector<std::string> paths;
    DIR *dp = opendir(dir.c_str());
    if (!dp) {
        return paths;
    }
    while (auto *ent = readdir(dp)) {
        std::string name = ent->d_name;
        if (name.size() < 4) {
            continue;
        }
        auto lower = name;
        for (char &c : lower) {
            c = static_cast<char>(::tolower(c));
        }
        if (lower.size() >= 4 &&
            (lower.compare(lower.size() - 4, 4, ".png") == 0 ||
             lower.compare(lower.size() - 4, 4, ".jpg") == 0 ||
             (lower.size() >= 5 && lower.compare(lower.size() - 5, 5, ".jpeg") == 0))) {
            paths.push_back(dir + "/" + name);
        }
    }
    closedir(dp);
    std::sort(paths.begin(), paths.end());
    return paths;
}

}  // namespace

bool SuperPoint::build() {
    if(deserialize_engine()){
        return true;
    }
    auto builder = TensorRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder) {
        return false;
    }
    const auto explicit_batch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TensorRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
    if (!network) {
        return false;
    }
    auto config = TensorRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }
    auto parser = TensorRTUniquePtr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser) {
        return false;
    }

    const int opt_h = super_point_config_.use_int8 ? super_point_config_.calib_height : 500;
    const int opt_w = super_point_config_.use_int8 ? super_point_config_.calib_width : 500;
    
    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        return false;
    }
    profile->setDimensions(super_point_config_.input_tensor_names[0].c_str(),
                           OptProfileSelector::kMIN, Dims4(1, 1, 100, 100));
    profile->setDimensions(super_point_config_.input_tensor_names[0].c_str(),
                           OptProfileSelector::kOPT, Dims4(1, 1, opt_h, opt_w));
    profile->setDimensions(super_point_config_.input_tensor_names[0].c_str(),
                           OptProfileSelector::kMAX, Dims4(1, 1, 1500, 1500));
    config->addOptimizationProfile(profile);
    
    auto constructed = construct_network(builder, network, config, parser);
    if (!constructed) {
        return false;
    }

    // Keep calibrator alive until buildSerializedNetwork returns.
    std::unique_ptr<SuperPointInt8Calibrator> calibrator;
    if (super_point_config_.use_int8) {
        if (!builder->platformHasFastInt8()) {
            std::cerr << "Platform reports no fast INT8; continuing anyway." << std::endl;
        }
        auto images = list_calib_images(super_point_config_.calib_image_dir);
        if (images.empty()) {
            std::cerr << "INT8 calibration image dir empty: "
                      << super_point_config_.calib_image_dir << std::endl;
            return false;
        }
        // Cap calibration set for build time; more images rarely help much for this CNN.
        constexpr size_t kMaxCalib = 64;
        if (images.size() > kMaxCalib) {
            images.resize(kMaxCalib);
        }
        std::string cache = super_point_config_.calib_cache_file;
        if (cache.empty()) {
            cache = super_point_config_.engine_file + ".calib";
        }
        calibrator = std::make_unique<SuperPointInt8Calibrator>(
                images, super_point_config_.input_tensor_names[0],
                super_point_config_.calib_width, super_point_config_.calib_height, cache, true);
        config->setFlag(BuilderFlag::kINT8);
        config->setFlag(BuilderFlag::kFP16);  // unsupported INT8 layers fall back to FP16
        config->setInt8Calibrator(calibrator.get());
        std::cout << "Building SuperPoint INT8 engine with " << images.size()
                  << " calibration images..." << std::endl;
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
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine_) {
        return false;
    }
    save_engine();
    ASSERT(network->getNbInputs() == 1);
    input_dims_ = network->getInput(0)->getDimensions();
    ASSERT(input_dims_.nbDims == 4);
    ASSERT(network->getNbOutputs() == 2);
    semi_dims_ = network->getOutput(0)->getDimensions();
    ASSERT(semi_dims_.nbDims == 3);
    desc_dims_ = network->getOutput(1)->getDimensions();
    ASSERT(desc_dims_.nbDims == 4);
    return true;
}

bool SuperPoint::construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                                   TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                   TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                                   TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
    auto parsed = parser->parseFromFile(super_point_config_.onnx_file.c_str(),
                                        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 4096_MiB);
    if (!super_point_config_.use_int8) {
        config->setFlag(BuilderFlag::kFP16);
    }
    enableDLA(builder.get(), config.get(), super_point_config_.dla_core);
    return true;
}


bool SuperPoint::bind_buffers() {
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char *name = engine_->getIOTensorName(i);
        if (!context_->setTensorAddress(name, buffers_->getDeviceBuffer(name))) {
            return false;
        }
    }
    return true;
}

bool SuperPoint::infer(const cv::Mat &image, Eigen::Matrix<double, 259, Eigen::Dynamic> &features) {
    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            return false;
        }
    }
    
    assert(engine_->getNbIOTensors() == 3);

    const char *input_name = super_point_config_.input_tensor_names[0].c_str();
    if (!context_->setInputShape(input_name, Dims4(1, 1, image.rows, image.cols))) {
        return false;
    }

    // Reuse host/device buffers across frames unless the image shape changes.
    if (!buffers_ || buffered_h_ != image.rows || buffered_w_ != image.cols) {
        buffers_ = std::make_unique<BufferManager>(engine_, 0, context_.get());
        buffered_h_ = image.rows;
        buffered_w_ = image.cols;
        if (!bind_buffers()) {
            return false;
        }
    }

    ASSERT(super_point_config_.input_tensor_names.size() == 1);
    if (!process_input(*buffers_, image)) {
        return false;
    }

    if (stream_) {
        buffers_->copyInputToDeviceAsync(stream_);
        if (!context_->enqueueV3(stream_)) {
            return false;
        }
        // Scores always needed on host for NMS. Descriptors stay on device when GPU sampling.
        const auto &score_name = super_point_config_.output_tensor_names[0];
        const auto &desc_name = super_point_config_.output_tensor_names[1];
        const size_t score_bytes = buffers_->size(score_name);
        cudaMemcpyAsync(buffers_->getHostBuffer(score_name), buffers_->getDeviceBuffer(score_name),
                        score_bytes, cudaMemcpyDeviceToHost, stream_);
        if (!use_gpu_sample_) {
            const size_t desc_bytes = buffers_->size(desc_name);
            cudaMemcpyAsync(buffers_->getHostBuffer(desc_name), buffers_->getDeviceBuffer(desc_name),
                            desc_bytes, cudaMemcpyDeviceToHost, stream_);
        }
        cudaStreamSynchronize(stream_);
    } else {
        buffers_->copyInputToDevice();
        if (!context_->executeV2(buffers_->getDeviceBindings().data())) {
            return false;
        }
        buffers_->copyOutputToHost();
    }

    if (!process_output(*buffers_, features)) {
        return false;
    }
    return true;
}

bool SuperPoint::process_input(const BufferManager &buffers, const cv::Mat &image) {
    input_dims_.d[2] = image.rows;
    input_dims_.d[3] = image.cols;
    semi_dims_.d[1] = image.rows;
    semi_dims_.d[2] = image.cols;
    desc_dims_.d[1] = 256;
    desc_dims_.d[2] = image.rows / 8;
    desc_dims_.d[3] = image.cols / 8;
    auto *host_data_buffer = static_cast<float *>(buffers.getHostBuffer(super_point_config_.input_tensor_names[0]));

    // Vectorized /255 conversion instead of per-pixel at<> loops.
    cv::Mat image_float;
    image.convertTo(image_float, CV_32FC1, 1.0 / 255.0);
    std::memcpy(host_data_buffer, image_float.ptr<float>(),
                static_cast<size_t>(image.rows) * static_cast<size_t>(image.cols) * sizeof(float));
    return true;
}

void SuperPoint::find_high_score_index(std::vector<float> &scores, std::vector<std::vector<int>> &keypoints,
                                       int h, int w, double threshold) {
    std::vector<float> new_scores;
    for (int i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            std::vector<int> location = {int(i / w), i % w};
            keypoints.emplace_back(location);
            new_scores.push_back(scores[i]);
        }
    }
    scores.swap(new_scores);
}


void SuperPoint::remove_borders(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int border,
                                int height,
                                int width) {
    std::vector<std::vector<int>> keypoints_selected;
    std::vector<float> scores_selected;
    for (int i = 0; i < keypoints.size(); ++i) {
        bool flag_h = (keypoints[i][0] >= border) && (keypoints[i][0] < (height - border));
        bool flag_w = (keypoints[i][1] >= border) && (keypoints[i][1] < (width - border));
        if (flag_h && flag_w) {
            keypoints_selected.push_back(std::vector<int>{keypoints[i][1], keypoints[i][0]});
            scores_selected.push_back(scores[i]);
        }
    }
    keypoints.swap(keypoints_selected);
    scores.swap(scores_selected);
}

std::vector<size_t> SuperPoint::sort_indexes(std::vector<float> &data) {
    std::vector<size_t> indexes(data.size());
    iota(indexes.begin(), indexes.end(), 0);
    sort(indexes.begin(), indexes.end(), [&data](size_t i1, size_t i2) { return data[i1] > data[i2]; });
    return indexes;
}

void SuperPoint::top_k_keypoints(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int k) {
    if (k < keypoints.size() && k != -1) {
        std::vector<std::vector<int>> keypoints_top_k;
        std::vector<float> scores_top_k;
        std::vector<size_t> indexes = sort_indexes(scores);
        for (int i = 0; i < k; ++i) {
            keypoints_top_k.push_back(keypoints[indexes[i]]);
            scores_top_k.push_back(scores[indexes[i]]);
        }
        keypoints.swap(keypoints_top_k);
        scores.swap(scores_top_k);
    }
}

int clip(int val, int max) {
    if (val < 0) return 0;
    return std::min(val, max - 1);
}

void normalize_keypoints(const std::vector<std::vector<int>> &keypoints, std::vector<float> &keypoints_norm,
                         int h, int w, int s) {
    keypoints_norm.resize(keypoints.size() * 2);
    const float denom_x = static_cast<float>(w * s - s / 2 - 0.5);
    const float denom_y = static_cast<float>(h * s - s / 2 - 0.5);
    for (size_t i = 0; i < keypoints.size(); ++i) {
        float kx = static_cast<float>(keypoints[i][0] - s / 2 + 0.5);
        float ky = static_cast<float>(keypoints[i][1] - s / 2 + 0.5);
        keypoints_norm[i * 2] = kx / denom_x * 2.f - 1.f;
        keypoints_norm[i * 2 + 1] = ky / denom_y * 2.f - 1.f;
    }
}

void grid_sample(const float *input, const std::vector<float> &grid,
                 std::vector<float> &output, int dim, int h, int w) {
    // descriptors: 1, 256, H/8, W/8  -> sample to N * 256 contiguous floats
    const size_t num = grid.size() / 2;
    output.resize(num * static_cast<size_t>(dim));
    for (size_t n = 0; n < num; ++n) {
        const float gx = grid[n * 2];
        const float gy = grid[n * 2 + 1];
        const float ix = ((gx + 1.f) / 2.f) * (w - 1);
        const float iy = ((gy + 1.f) / 2.f) * (h - 1);

        const int ix_nw = clip(static_cast<int>(std::floor(ix)), w);
        const int iy_nw = clip(static_cast<int>(std::floor(iy)), h);
        const int ix_ne = clip(ix_nw + 1, w);
        const int iy_ne = clip(iy_nw, h);
        const int ix_sw = clip(ix_nw, w);
        const int iy_sw = clip(iy_nw + 1, h);
        const int ix_se = clip(ix_nw + 1, w);
        const int iy_se = clip(iy_nw + 1, h);

        const float nw = (ix_se - ix) * (iy_se - iy);
        const float ne = (ix - ix_sw) * (iy_sw - iy);
        const float sw = (ix_ne - ix) * (iy - iy_ne);
        const float se = (ix - ix_nw) * (iy - iy_nw);

        float *dst = output.data() + n * dim;
        for (int i = 0; i < dim; ++i) {
            const float *plane = input + i * h * w;
            dst[i] = plane[iy_nw * w + ix_nw] * nw + plane[iy_ne * w + ix_ne] * ne +
                     plane[iy_sw * w + ix_sw] * sw + plane[iy_se * w + ix_se] * se;
        }
    }
}

void normalize_descriptors(std::vector<float> &dest_descriptors, int dim) {
    const size_t num = dest_descriptors.size() / static_cast<size_t>(dim);
    for (size_t n = 0; n < num; ++n) {
        float *desc = dest_descriptors.data() + n * dim;
        float sum = 0.f;
        for (int i = 0; i < dim; ++i) {
            sum += desc[i] * desc[i];
        }
        const float inv = 1.f / std::sqrt(sum + 1e-12f);
        for (int i = 0; i < dim; ++i) {
            desc[i] *= inv;
        }
    }
}

void SuperPoint::sample_descriptors(std::vector<std::vector<int>> &keypoints, float *descriptors,
                                    std::vector<float> &dest_descriptors, int dim, int h, int w, int s) {
    std::vector<float> keypoints_norm;
    normalize_keypoints(keypoints, keypoints_norm, h, w, s);
    grid_sample(descriptors, keypoints_norm, dest_descriptors, dim, h, w);
    normalize_descriptors(dest_descriptors, dim);
}

bool SuperPoint::sample_descriptors_gpu(std::vector<std::vector<int>> &keypoints, const float *device_descriptors,
                                        std::vector<float> &dest_descriptors, int dim, int h, int w, int s) {
    std::vector<float> keypoints_norm;
    normalize_keypoints(keypoints, keypoints_norm, h, w, s);
    const int num = static_cast<int>(keypoints.size());
    dest_descriptors.resize(static_cast<size_t>(num) * dim);

    const size_t grid_bytes = static_cast<size_t>(num) * 2 * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(num) * dim * sizeof(float);
    if (grid_bytes > d_grid_capacity_) {
        if (d_grid_) cudaFree(d_grid_);
        if (cudaMalloc(&d_grid_, grid_bytes) != cudaSuccess) return false;
        d_grid_capacity_ = grid_bytes;
    }
    if (out_bytes > d_sampled_capacity_) {
        if (d_sampled_) cudaFree(d_sampled_);
        if (cudaMalloc(&d_sampled_, out_bytes) != cudaSuccess) return false;
        d_sampled_capacity_ = out_bytes;
    }

    cudaStream_t stream = stream_ ? stream_ : nullptr;
    if (cudaMemcpyAsync(d_grid_, keypoints_norm.data(), grid_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return false;
    }
    if (spsg_cuda::launch_grid_sample_l2(device_descriptors, d_grid_, d_sampled_, num, dim, h, w, stream) !=
        cudaSuccess) {
        return false;
    }
    if (cudaMemcpyAsync(dest_descriptors.data(), d_sampled_, out_bytes, cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess) {
        return false;
    }
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
    return true;
}

bool SuperPoint::process_output(const BufferManager &buffers, Eigen::Matrix<double, 259, Eigen::Dynamic> &features) {
    keypoints_.clear();
    descriptors_.clear();
    auto *output_score = static_cast<float *>(buffers.getHostBuffer(super_point_config_.output_tensor_names[0]));
    int semi_feature_map_h = semi_dims_.d[1];
    int semi_feature_map_w = semi_dims_.d[2];
    std::vector<float> scores_vec(output_score, output_score + semi_feature_map_h * semi_feature_map_w);
    find_high_score_index(scores_vec, keypoints_, semi_feature_map_h, semi_feature_map_w,
                          super_point_config_.keypoint_threshold);
    remove_borders(keypoints_, scores_vec, super_point_config_.remove_borders, semi_feature_map_h, semi_feature_map_w);
    top_k_keypoints(keypoints_, scores_vec, super_point_config_.max_keypoints);

    const int num = static_cast<int>(scores_vec.size());
    features.resize(259, num);
    int desc_feature_dim = desc_dims_.d[1];
    int desc_feature_map_h = desc_dims_.d[2];
    int desc_feature_map_w = desc_dims_.d[3];

    bool ok = true;
    if (use_gpu_sample_) {
        const float *device_desc = static_cast<const float *>(
                buffers.getDeviceBuffer(super_point_config_.output_tensor_names[1]));
        ok = sample_descriptors_gpu(keypoints_, device_desc, descriptors_, desc_feature_dim,
                                    desc_feature_map_h, desc_feature_map_w);
    } else {
        auto *output_desc = static_cast<float *>(buffers.getHostBuffer(super_point_config_.output_tensor_names[1]));
        sample_descriptors(keypoints_, output_desc, descriptors_, desc_feature_dim, desc_feature_map_h,
                           desc_feature_map_w);
    }
    if (!ok) {
        return false;
    }

    for (int i = 0; i < num; ++i) {
        features(0, i) = scores_vec[i];
        features(1, i) = keypoints_[i][0];
        features(2, i) = keypoints_[i][1];
        const float *desc = descriptors_.data() + static_cast<size_t>(i) * desc_feature_dim;
        for (int d = 0; d < desc_feature_dim; ++d) {
            features(3 + d, i) = desc[d];
        }
    }
    return true;
}

void SuperPoint::visualization(const std::string &image_name, const cv::Mat &image) {
    cv::Mat image_display;
    if(image.channels() == 1)
        cv::cvtColor(image, image_display, cv::COLOR_GRAY2BGR);
    else
        image_display = image.clone();
    for (auto &keypoint : keypoints_) {
        cv::circle(image_display, cv::Point(int(keypoint[0]), int(keypoint[1])), 1, cv::Scalar(255, 0, 0), -1, 16);
    }
    cv::imwrite(image_name + ".jpg", image_display);
}

void SuperPoint::save_engine() {
    if (super_point_config_.engine_file.empty()) return;
    if (engine_ != nullptr) {
        nvinfer1::IHostMemory *data = engine_->serialize();
        std::ofstream file(super_point_config_.engine_file, std::ios::binary);
        if (!file) return;
        file.write(reinterpret_cast<const char *>(data->data()), data->size());
    }
}

bool SuperPoint::deserialize_engine() {
    std::ifstream file(super_point_config_.engine_file.c_str(), std::ios::binary);
    if (file.is_open()) {
        file.seekg(0, std::ifstream::end);
        size_t size = file.tellg();
        file.seekg(0, std::ifstream::beg);
        char *model_stream = new char[size];
        file.read(model_stream, size);
        file.close();
        IRuntime *runtime = createInferRuntime(gLogger);
        if (runtime == nullptr) {
            delete [] model_stream;
            return false;
        }
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(model_stream, size));
        if (engine_ == nullptr) {
            delete [] model_stream;
            return false;
        }
        delete [] model_stream;
        return true;
    }
    return false;
}

