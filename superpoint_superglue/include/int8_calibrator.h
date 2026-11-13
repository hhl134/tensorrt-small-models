#ifndef INT8_CALIBRATOR_H_
#define INT8_CALIBRATOR_H_

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

//! EntropyCalibrator2 for SuperPoint grayscale float input [1,1,H,W] in [0,1].
class SuperPointInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    SuperPointInt8Calibrator(std::vector<std::string> image_paths, std::string input_name,
                             int width, int height, std::string cache_file, bool read_cache = true)
            : image_paths_(std::move(image_paths)),
              input_name_(std::move(input_name)),
              width_(width),
              height_(height),
              cache_file_(std::move(cache_file)),
              read_cache_(read_cache),
              input_count_(static_cast<size_t>(width) * static_cast<size_t>(height)) {
        host_input_.resize(input_count_);
        if (cudaMalloc(&device_input_, input_count_ * sizeof(float)) != cudaSuccess) {
            device_input_ = nullptr;
            std::cerr << "SuperPointInt8Calibrator: cudaMalloc failed." << std::endl;
        }
        std::cout << "SuperPoint INT8 calibrator: " << image_paths_.size()
                  << " images at " << width_ << "x" << height_ << std::endl;
    }

    ~SuperPointInt8Calibrator() override {
        if (device_input_) {
            cudaFree(device_input_);
            device_input_ = nullptr;
        }
    }

    int32_t getBatchSize() const noexcept override {
        return 1;
    }

    bool getBatch(void *bindings[], const char *names[], int32_t nbBindings) noexcept override {
        if (!device_input_ || current_ >= image_paths_.size()) {
            return false;
        }
        if (nbBindings < 1 || std::string(names[0]) != input_name_) {
            std::cerr << "SuperPointInt8Calibrator: unexpected binding name "
                      << (nbBindings > 0 ? names[0] : "(none)") << std::endl;
            return false;
        }

        cv::Mat image = cv::imread(image_paths_[current_], cv::IMREAD_GRAYSCALE);
        ++current_;
        if (image.empty()) {
            std::cerr << "SuperPointInt8Calibrator: failed to read image, skip." << std::endl;
            return getBatch(bindings, names, nbBindings);
        }
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(width_, height_));
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC1, 1.0 / 255.0);
        std::memcpy(host_input_.data(), float_img.ptr<float>(), input_count_ * sizeof(float));
        if (cudaMemcpy(device_input_, host_input_.data(), input_count_ * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            return false;
        }
        bindings[0] = device_input_;
        return true;
    }

    const void *readCalibrationCache(std::size_t &length) noexcept override {
        calibration_cache_.clear();
        length = 0;
        if (!read_cache_) {
            return nullptr;
        }
        std::ifstream input(cache_file_, std::ios::binary);
        if (!input.good()) {
            return nullptr;
        }
        input >> std::noskipws;
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                  std::back_inserter(calibration_cache_));
        length = calibration_cache_.size();
        std::cout << "Loaded INT8 calibration cache: " << cache_file_
                  << " (" << length << " bytes)" << std::endl;
        return length ? calibration_cache_.data() : nullptr;
    }

    void writeCalibrationCache(const void *cache, std::size_t length) noexcept override {
        std::ofstream output(cache_file_, std::ios::binary);
        output.write(reinterpret_cast<const char *>(cache), static_cast<std::streamsize>(length));
        std::cout << "Wrote INT8 calibration cache: " << cache_file_
                  << " (" << length << " bytes)" << std::endl;
    }

private:
    std::vector<std::string> image_paths_;
    std::string input_name_;
    int width_;
    int height_;
    std::string cache_file_;
    bool read_cache_;
    size_t input_count_;
    size_t current_{0};
    std::vector<float> host_input_;
    void *device_input_{nullptr};
    std::vector<char> calibration_cache_;
};

#endif  // INT8_CALIBRATOR_H_
