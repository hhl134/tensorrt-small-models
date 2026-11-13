//
// Created by haoyuefan on 2021/9/22.
//

#ifndef SUPER_POINT_H_
#define SUPER_POINT_H_

#include <string>
#include <memory>
#include <Eigen/Core>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

#include "3rdparty/tensorrtbuffer/include/buffers.h"
#include "read_config.h"

using tensorrt_common::TensorRTUniquePtr;

class SuperPoint {
public:
    explicit SuperPoint(SuperPointConfig super_point_config);
    ~SuperPoint();

    bool build();

    bool infer(const cv::Mat &image, Eigen::Matrix<double, 259, Eigen::Dynamic> &features);

    void visualization(const std::string &image_name, const cv::Mat &image);

    void save_engine();

    bool deserialize_engine();

private:
    SuperPointConfig super_point_config_;
    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims semi_dims_{};
    nvinfer1::Dims desc_dims_{};
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::unique_ptr<tensorrt_buffer::BufferManager> buffers_;
    cudaStream_t stream_{nullptr};
    int buffered_h_{-1};
    int buffered_w_{-1};
    std::vector<std::vector<int>> keypoints_;
    // Contiguous descriptor storage: N * 256 floats.
    std::vector<float> descriptors_;
    float *d_grid_{nullptr};
    float *d_sampled_{nullptr};
    size_t d_grid_capacity_{0};
    size_t d_sampled_capacity_{0};
    bool use_gpu_sample_{true};

    bool construct_network(TensorRTUniquePtr<nvinfer1::IBuilder> &builder,
                           TensorRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                           TensorRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                           TensorRTUniquePtr<nvonnxparser::IParser> &parser) const;

    bool bind_buffers();

    bool process_input(const tensorrt_buffer::BufferManager &buffers, const cv::Mat &image);

    bool
    process_output(const tensorrt_buffer::BufferManager &buffers, Eigen::Matrix<double, 259, Eigen::Dynamic> &features);

    void remove_borders(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int border, int height,
                        int width);

    std::vector<size_t> sort_indexes(std::vector<float> &data);

    void top_k_keypoints(std::vector<std::vector<int>> &keypoints, std::vector<float> &scores, int k);

    void find_high_score_index(std::vector<float> &scores, std::vector<std::vector<int>> &keypoints, int h, int w,
                               double threshold);

    void sample_descriptors(std::vector<std::vector<int>> &keypoints, float *descriptors,
                            std::vector<float> &dest_descriptors, int dim, int h, int w, int s = 8);

    bool sample_descriptors_gpu(std::vector<std::vector<int>> &keypoints, const float *device_descriptors,
                                std::vector<float> &dest_descriptors, int dim, int h, int w, int s = 8);
};

typedef std::shared_ptr<SuperPoint> SuperPointPtr;

#endif //SUPER_POINT_H_
