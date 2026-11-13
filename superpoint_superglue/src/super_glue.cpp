//
// Created by haoyuefan on 2021/9/22.
//

#include "super_glue.h"
#include <cfloat>
#include <utility>
#include <unordered_map>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace tensorrt_common;
using namespace tensorrt_log;
using namespace tensorrt_buffer;

SuperGlue::SuperGlue(const SuperGlueConfig &superglue_config) : superglue_config_(superglue_config), engine_(nullptr) {
    setReportableSeverity(Logger::Severity::kINFO);
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        stream_ = nullptr;
    }
}

SuperGlue::~SuperGlue() {
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

bool SuperGlue::build() {
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

    auto profile = builder->createOptimizationProfile();
    if (!profile) {
        return false;
    }
    profile->setDimensions(superglue_config_.input_tensor_names[0].c_str(), OptProfileSelector::kMIN, Dims3(1, 1, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[0].c_str(), OptProfileSelector::kOPT, Dims3(1, 512, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[0].c_str(), OptProfileSelector::kMAX,
                           Dims3(1, 1024, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[1].c_str(), OptProfileSelector::kMIN, Dims2(1, 1));
    profile->setDimensions(superglue_config_.input_tensor_names[1].c_str(), OptProfileSelector::kOPT, Dims2(1, 512));
    profile->setDimensions(superglue_config_.input_tensor_names[1].c_str(), OptProfileSelector::kMAX, Dims2(1, 1024));
    profile->setDimensions(superglue_config_.input_tensor_names[2].c_str(), OptProfileSelector::kMIN, Dims3(1, 256, 1));
    profile->setDimensions(superglue_config_.input_tensor_names[2].c_str(), OptProfileSelector::kOPT,
                           Dims3(1, 256, 512));
    profile->setDimensions(superglue_config_.input_tensor_names[2].c_str(), OptProfileSelector::kMAX,
                           Dims3(1, 256, 1024));
    profile->setDimensions(superglue_config_.input_tensor_names[3].c_str(), OptProfileSelector::kMIN, Dims3(1, 1, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[3].c_str(), OptProfileSelector::kOPT, Dims3(1, 512, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[3].c_str(), OptProfileSelector::kMAX,
                           Dims3(1, 1024, 2));
    profile->setDimensions(superglue_config_.input_tensor_names[4].c_str(), OptProfileSelector::kMIN, Dims2(1, 1));
    profile->setDimensions(superglue_config_.input_tensor_names[4].c_str(), OptProfileSelector::kOPT, Dims2(1, 512));
    profile->setDimensions(superglue_config_.input_tensor_names[4].c_str(), OptProfileSelector::kMAX, Dims2(1, 1024));
    profile->setDimensions(superglue_config_.input_tensor_names[5].c_str(), OptProfileSelector::kMIN, Dims3(1, 256, 1));
    profile->setDimensions(superglue_config_.input_tensor_names[5].c_str(), OptProfileSelector::kOPT,
                           Dims3(1, 256, 512));
    profile->setDimensions(superglue_config_.input_tensor_names[5].c_str(), OptProfileSelector::kMAX,
                           Dims3(1, 256, 1024));
    config->addOptimizationProfile(profile);

    auto constructed = construct_network(builder, network, config, parser);
    if (!constructed) {
        return false;
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

    ASSERT(network->getNbInputs() == 6);
    keypoints_0_dims_ = network->getInput(0)->getDimensions();
    scores_0_dims_ = network->getInput(1)->getDimensions();
    descriptors_0_dims_ = network->getInput(2)->getDimensions();
    keypoints_1_dims_ = network->getInput(3)->getDimensions();
    scores_1_dims_ = network->getInput(4)->getDimensions();
    descriptors_1_dims_ = network->getInput(5)->getDimensions();
    assert(keypoints_0_dims_.d[1] == -1);
    assert(scores_0_dims_.d[1] == -1);
    assert(descriptors_0_dims_.d[2] == -1);
    assert(keypoints_1_dims_.d[1] == -1);
    assert(scores_1_dims_.d[1] == -1);
    assert(descriptors_1_dims_.d[2] == -1);
    return true;
}

bool SuperGlue::construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                                  TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                  TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                                  TensorRTUniquePtr<nvonnxparser::IParser> &parser) const {
    auto parsed = parser->parseFromFile(superglue_config_.onnx_file.c_str(),
                                        static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed) {
        return false;
    }
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 4096_MiB);
    config->setFlag(BuilderFlag::kFP16);
    // The ONNX Sinkhorn is 100 iterations of ReduceLogSumExp. TensorRT lowers each
    // one to Exp + ReduceSum + Log and does not subtract the max before Exp.
    // FP16 exp overflows above about 11, so the next iteration becomes NaN and
    // every match is rejected. These three ops stay in FP32; the rest of the
    // network can still run in FP16.
    config->setFlag(BuilderFlag::kOBEY_PRECISION_CONSTRAINTS);
    int fp32_layers = 0;
    for (int i = 0; i < network->getNbLayers(); ++i) {
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
    std::cout << "SuperGlue keeps " << fp32_layers << " Exp/Log/ReduceSum layers in FP32." << std::endl;
    enableDLA(builder.get(), config.get(), superglue_config_.dla_core);
    return true;
}

bool SuperGlue::infer(const Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                      const Eigen::Matrix<double, 259, Eigen::Dynamic> &features1,
                      Eigen::VectorXi &indices0,
                      Eigen::VectorXi &indices1,
                      Eigen::VectorXd &mscores0,
                      Eigen::VectorXd &mscores1) {
    if (!context_) {
        context_ = TensorRTUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
        if (!context_) {
            return false;
    }
    }

    assert(engine_->getNbIOTensors() == 7);

    const char *keypoints_0_name = superglue_config_.input_tensor_names[0].c_str();
    const char *scores_0_name = superglue_config_.input_tensor_names[1].c_str();
    const char *descriptors_0_name = superglue_config_.input_tensor_names[2].c_str();
    const char *keypoints_1_name = superglue_config_.input_tensor_names[3].c_str();
    const char *scores_1_name = superglue_config_.input_tensor_names[4].c_str();
    const char *descriptors_1_name = superglue_config_.input_tensor_names[5].c_str();
    const char *output_score_name = superglue_config_.output_tensor_names[0].c_str();

    if (!context_->setInputShape(keypoints_0_name, Dims3(1, features0.cols(), 2)) ||
        !context_->setInputShape(scores_0_name, Dims2(1, features0.cols())) ||
        !context_->setInputShape(descriptors_0_name, Dims3(1, 256, features0.cols())) ||
        !context_->setInputShape(keypoints_1_name, Dims3(1, features1.cols(), 2)) ||
        !context_->setInputShape(scores_1_name, Dims2(1, features1.cols())) ||
        !context_->setInputShape(descriptors_1_name, Dims3(1, 256, features1.cols()))) {
        return false;
    }

    keypoints_0_dims_ = context_->getTensorShape(keypoints_0_name);
    scores_0_dims_ = context_->getTensorShape(scores_0_name);
    descriptors_0_dims_ = context_->getTensorShape(descriptors_0_name);
    keypoints_1_dims_ = context_->getTensorShape(keypoints_1_name);
    scores_1_dims_ = context_->getTensorShape(scores_1_name);
    descriptors_1_dims_ = context_->getTensorShape(descriptors_1_name);
    output_scores_dims_ = context_->getTensorShape(output_score_name);

    const int n0 = static_cast<int>(features0.cols());
    const int n1 = static_cast<int>(features1.cols());
    // Reuse host/device buffers unless keypoint counts change.
    if (!buffers_ || buffered_keypoints0_ != n0 || buffered_keypoints1_ != n1) {
        buffers_ = std::make_unique<BufferManager>(engine_, 0, context_.get());
        buffered_keypoints0_ = n0;
        buffered_keypoints1_ = n1;
        if (!bind_buffers()) {
            return false;
        }
    }

    ASSERT(superglue_config_.input_tensor_names.size() == 6);
    if (!process_input(*buffers_, features0, features1)) {
        return false;
    }

    if (stream_) {
        buffers_->copyInputToDeviceAsync(stream_);
        if (!context_->enqueueV3(stream_)) {
            return false;
        }
        buffers_->copyOutputToHostAsync(stream_);
        cudaStreamSynchronize(stream_);
    } else {
        buffers_->copyInputToDevice();
        if (!context_->executeV2(buffers_->getDeviceBindings().data())) {
            return false;
        }
        buffers_->copyOutputToHost();
    }

    if (!process_output(*buffers_, indices0, indices1, mscores0, mscores1)) {
        return false;
    }

    return true;
}

bool SuperGlue::bind_buffers() {
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char *name = engine_->getIOTensorName(i);
        if (!context_->setTensorAddress(name, buffers_->getDeviceBuffer(name))) {
            return false;
        }
    }
    return true;
}

bool SuperGlue::process_input(const BufferManager &buffers,
                              const Eigen::Matrix<double, 259, Eigen::Dynamic> &features0,
                              const Eigen::Matrix<double, 259, Eigen::Dynamic> &features1) {
    auto *keypoints_0_buffer = static_cast<float *>(buffers.getHostBuffer(superglue_config_.input_tensor_names[0]));
    auto *scores_0_buffer = static_cast<float *>(buffers.getHostBuffer(superglue_config_.input_tensor_names[1]));
    auto *descriptors_0_buffer = static_cast<float *>(buffers.getHostBuffer(superglue_config_.input_tensor_names[2]));
    auto *keypoints_1_buffer = static_cast<float *>(buffers.getHostBuffer(superglue_config_.input_tensor_names[3]));
    auto *scores_1_buffer = static_cast<float *>(buffers.getHostBuffer(superglue_config_.input_tensor_names[4]));
    auto *descriptors_1_buffer = static_cast<float *>(buffers.getHostBuffer(superglue_config_.input_tensor_names[5]));

    const int n0 = static_cast<int>(features0.cols());
    const int n1 = static_cast<int>(features1.cols());

    // Eigen maps: descriptors must be row-major [256, N] to match ONNX layout (d * N + n).
    Eigen::Map<Eigen::VectorXf>(scores_0_buffer, n0) = features0.row(0).cast<float>();
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>>(keypoints_0_buffer, n0, 2) =
            features0.middleRows(1, 2).transpose().cast<float>();
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            descriptors_0_buffer, 256, n0) = features0.bottomRows(256).cast<float>();

    Eigen::Map<Eigen::VectorXf>(scores_1_buffer, n1) = features1.row(0).cast<float>();
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>>(keypoints_1_buffer, n1, 2) =
            features1.middleRows(1, 2).transpose().cast<float>();
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            descriptors_1_buffer, 256, n1) = features1.bottomRows(256).cast<float>();

    return true;
}

void where_negative_one(const int *flag_data, const int *data, int size, std::vector<int> &indices) {
    for (int i = 0; i < size; ++i) {
        if (flag_data[i] == 1) {
            indices.push_back(data[i]);
        } else {
            indices.push_back(-1);
        }
    }
}

void max_matrix(const float *data, int *indices, float *values, int h, int w, int dim) {
    if (dim == 2) {
        for (int i = 0; i < h - 1; ++i) {
            float max_value = -FLT_MAX;
            int max_indices = 0;
            for (int j = 0; j < w - 1; ++j) {
                if (max_value < data[i * w + j]) {
                    max_value = data[i * w + j];
                    max_indices = j;
                }
            }
            values[i] = max_value;
            indices[i] = max_indices;
        }
    } else if (dim == 1) {
        for (int i = 0; i < w - 1; ++i) {
            float max_value = -FLT_MAX;
            int max_indices = 0;
            for (int j = 0; j < h - 1; ++j) {
                if (max_value < data[j * w + i]) {
                    max_value = data[j * w + i];
                    max_indices = j;
                }
            }
            values[i] = max_value;
            indices[i] = max_indices;
        }
    }
}

void equal_gather(const int *indices0, const int *indices1, int *mutual, int size) {
    for (int i = 0; i < size; ++i) {
        if (indices0[indices1[i]] == i) {
            mutual[i] = 1;
        } else {
            mutual[i] = 0;
        }
    }
}

void where_exp(const int *flag_data, float *data, std::vector<double> &mscores0, int size) {
    for (int i = 0; i < size; ++i) {
        if (flag_data[i] == 1) {
            mscores0.push_back(std::exp(data[i]));
        } else {
            mscores0.push_back(0);
        }
    }
}

void where_gather(const int *flag_data, int *indices, std::vector<double> &mscores0, std::vector<double> &mscores1,
                  int size) {
    for (int i = 0; i < size; ++i) {
        if (flag_data[i] == 1) {
            mscores1.push_back(mscores0[indices[i]]);
        } else {
            mscores1.push_back(0);
        }
    }
}

void and_threshold(const int *mutual0, int *valid0, const std::vector<double> &mscores0, double threhold) {
    for (int i = 0; i < mscores0.size(); ++i) {
        if (mutual0[i] == 1 && mscores0[i] > threhold) {
            valid0[i] = 1;
        } else {
            valid0[i] = 0;
        }
    }
}

void and_gather(const int *mutual1, const int *valid0, const int *indices1, int *valid1, int size) {
    for (int i = 0; i < size; ++i) {
        if (mutual1[i] == 1 && valid0[indices1[i]] == 1) {
            valid1[i] = 1;
        } else {
            valid1[i] = 0;
        }
    }
}

void decode(float *scores, int h, int w, std::vector<int> &indices0, std::vector<int> &indices1,
            std::vector<double> &mscores0, std::vector<double> &mscores1) {
    auto *max_indices0 = new int[h - 1];
    auto *max_indices1 = new int[w - 1];
    auto *max_values0 = new float[h - 1];
    auto *max_values1 = new float[w - 1];
    max_matrix(scores, max_indices0, max_values0, h, w, 2);
    max_matrix(scores, max_indices1, max_values1, h, w, 1);
    auto *mutual0 = new int[h - 1];
    auto *mutual1 = new int[w - 1];
    equal_gather(max_indices1, max_indices0, mutual0, h - 1);
    equal_gather(max_indices0, max_indices1, mutual1, w - 1);
    where_exp(mutual0, max_values0, mscores0, h - 1);
    where_gather(mutual1, max_indices1, mscores0, mscores1, w - 1);
    auto *valid0 = new int[h - 1];
    auto *valid1 = new int[w - 1];
    and_threshold(mutual0, valid0, mscores0, 0.2);
    and_gather(mutual1, valid0, max_indices1, valid1, w - 1);
    where_negative_one(valid0, max_indices0, h - 1, indices0);
    where_negative_one(valid1, max_indices1, w - 1, indices1);
    delete[] max_indices0;
    delete[] max_indices1;
    delete[] max_values0;
    delete[] max_values1;
    delete[] mutual0;
    delete[] mutual1;
    delete[] valid0;
    delete[] valid1;
}

void log_sinkhorn_iterations(float *couplings, float *Z, int m, int n,
                             float *log_mu, float *log_nu, int iters) {
  auto *u = new float[m]();
  auto *v = new float[n]();
  for (int k = 0; k < iters; ++k) {
    for (int ki = 0; ki < m; ++ki) {
      float nu_expsum = 0.0;
      for (int kn = 0; kn < n; ++kn) {
        nu_expsum += std::exp(couplings[ki * n + kn] + v[kn]);
      }
      u[ki] = log_mu[ki] - std::log(nu_expsum);
    }
    for (int kj = 0; kj < n; ++kj) {
      float nu_expsum = 0.0;
      for (int km = 0; km < m; ++km) {
        nu_expsum += std::exp(couplings[km * n + kj] + u[km]);
      }
      v[kj] = log_nu[kj] - std::log(nu_expsum);
    }
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      Z[i * n + j] = couplings[i * n + j] + u[i] + v[j];
    }
  }
  delete[] u;
  delete[] v;
}

void log_optimal_transport(float *scores, float *Z, int m, int n,
                           float alpha = 2.3457, int iters = 100) {
  auto *couplings = new float[(m + 1) * (n + 1)];
  for (int i = 0; i < m + 1; ++i) {
    for (int j = 0; j < n + 1; ++j) {
      if (i == m || j == n) {
        couplings[i * (n + 1) + j] = alpha;
      } else {
        couplings[i * (n + 1) + j] = scores[i * n + j];
      }
    }
  }

  float norm = -std::log(m + n);

  auto *log_mu = new float[m + 1];
  auto *log_nu = new float[n + 1];
  for (int ii = 0; ii < m; ++ii) {
    log_mu[ii] = norm;
  }
  log_mu[m] = std::log(n) + norm;

  for (int jj = 0; jj < n; ++jj) {
    log_nu[jj] = norm;
  }
  log_nu[n] = std::log(m) + norm;

  log_sinkhorn_iterations(couplings, Z, m + 1, n + 1, log_mu, log_nu, iters);
  for (int ii = 0; ii < m + 1; ++ii) {
    for (int jj = 0; jj < n + 1; ++jj) {
      Z[ii * (n + 1) + jj] = Z[ii * (n + 1) + jj] - norm;
    }
  }
  delete[] couplings;
  delete[] log_mu;
  delete[] log_nu;
}

bool SuperGlue::process_output(const BufferManager &buffers,
                               Eigen::VectorXi &indices0,
                               Eigen::VectorXi &indices1,
                               Eigen::VectorXd &mscores0,
                               Eigen::VectorXd &mscores1) {
    indices0_.clear();
    indices1_.clear();
    mscores0_.clear();
    mscores1_.clear();
    auto *output_score = static_cast<float *>(buffers.getHostBuffer(superglue_config_.output_tensor_names[0]));
    int scores_map_h = output_scores_dims_.d[1];
    int scores_map_w = output_scores_dims_.d[2];
    auto *scores = new float[(scores_map_h + 1) * (scores_map_w + 1)];
    // log_optimal_transport(output_score, scores, scores_map_h, scores_map_w);
    // scores_map_h = scores_map_h + 1;
    // scores_map_w = scores_map_w + 1;
    decode(output_score, scores_map_h, scores_map_w, indices0_, indices1_, mscores0_, mscores1_);
    indices0.resize(indices0_.size());
    indices1.resize(indices1_.size());
    mscores0.resize(mscores0_.size());
    mscores1.resize(mscores1_.size());
    for (int i0 = 0; i0 < indices0_.size(); ++i0) {
        indices0(i0) = indices0_[i0];
    }
    for (int i1 = 0; i1 < indices1_.size(); ++i1) {
        indices1(i1) = indices1_[i1];
    }
    for (int j0 = 0; j0 < mscores0_.size(); ++j0) {
        mscores0(j0) = mscores0_[j0];
    }
    for (int j1 = 0; j1 < mscores1_.size(); ++j1) {
        mscores1(j1) = mscores1_[j1];
    }
    return true;
}

void SuperGlue::save_engine() {
    if (superglue_config_.engine_file.empty()) return;
    if (engine_ != nullptr) {
        nvinfer1::IHostMemory *data = engine_->serialize();
        std::ofstream file(superglue_config_.engine_file, std::ios::binary);;
        if (!file) return;
        file.write(reinterpret_cast<const char *>(data->data()), data->size());
    }
}

bool SuperGlue::deserialize_engine() {
    std::ifstream file(superglue_config_.engine_file, std::ios::binary);
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

int SuperGlue::matching_points(Eigen::Matrix<double, 259, Eigen::Dynamic>& features0,
                                  Eigen::Matrix<double, 259, Eigen::Dynamic>& features1, std::vector<cv::DMatch>& matches, bool outlier_rejection){
  matches.clear();
  Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features0 = normalize_keypoints(features0, superglue_config_.image_width, superglue_config_.image_height);
  Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features1 = normalize_keypoints(features1, superglue_config_.image_width, superglue_config_.image_height);
  Eigen::VectorXi indices0, indices1;
  Eigen::VectorXd mscores0, mscores1;
  infer(norm_features0, norm_features1, indices0, indices1, mscores0, mscores1);

  int num_match = 0;
  std::vector<cv::Point2f> points0, points1;
  std::vector<int> point_indexes;
  for(size_t i = 0; i < indices0.size(); i++){
    if(indices0(i) < indices1.size() && indices0(i) >= 0 && indices1(indices0(i)) == i){
      double d = 1.0 - (mscores0[i] + mscores1[indices0[i]]) / 2.0;
      matches.emplace_back(i, indices0[i], d);
      points0.emplace_back(features0(1, i), features0(2, i));
      points1.emplace_back(features1(1, indices0(i)), features1(2, indices0(i)));
      num_match++;
    }
  }

  if(outlier_rejection){
    std::vector<uchar> inliers;
    cv::findFundamentalMat(points0, points1, cv::FM_RANSAC, 3, 0.99, inliers);
    int j = 0;
    for(int i = 0; i < matches.size(); i++){
      if(inliers[i]){
        matches[j++] = matches[i];
      }
    }
    matches.resize(j);
  }

  return matches.size();
}

Eigen::Matrix<double, 259, Eigen::Dynamic> SuperGlue::normalize_keypoints(const Eigen::Matrix<double, 259, Eigen::Dynamic> &features,
                                                                             int width, int height) {
  Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features;
  norm_features.resize(259, features.cols());
  norm_features = features;
  for (int col = 0; col < features.cols(); ++col) {
    norm_features(1, col) =
        (features(1, col) - width / 2) / (std::max(width, height) * 0.7);
    norm_features(2, col) =
        (features(2, col) - height / 2) / (std::max(width, height) * 0.7);
  }
  return norm_features;
}


