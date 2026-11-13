#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

static double elapsed_ms(const std::chrono::steady_clock::time_point &start,
                         const std::chrono::steady_clock::time_point &end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static double cosine(const std::vector<float> &left, const std::vector<float> &right) {
    double dot = 0.0;
    double left_norm = 0.0;
    double right_norm = 0.0;
    const size_t count = std::min(left.size(), right.size());
    for (size_t i = 0; i < count; ++i) {
        dot += static_cast<double>(left[i]) * right[i];
        left_norm += static_cast<double>(left[i]) * left[i];
        right_norm += static_cast<double>(right[i]) * right[i];
    }
    return dot / std::sqrt(left_norm * right_norm);
}

static bool write_descriptor(const std::string &path, const std::vector<float> &descriptor) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file.write(reinterpret_cast<const char *>(descriptor.data()),
               static_cast<std::streamsize>(descriptor.size() * sizeof(float)));
    return static_cast<bool>(file);
}

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "./mixvpr_image config_path model_dir image0 image1" << std::endl;
        return 1;
    }
    const std::string config_path = argv[1];
    const std::string model_dir = argv[2];
    cv::Mat image0 = cv::imread(argv[3], cv::IMREAD_COLOR);
    cv::Mat image1 = cv::imread(argv[4], cv::IMREAD_COLOR);
    if (image0.empty() || image1.empty()) {
        std::cerr << "Failed to read input images." << std::endl;
        return 1;
    }

    int warmup = 5;
    int iters = 30;
    if (const char *env = std::getenv("MIX_WARMUP")) {
        warmup = std::atoi(env);
    }
    if (const char *env = std::getenv("MIX_ITERS")) {
        iters = std::atoi(env);
    }

    PipelineConfig configs(config_path, model_dir);
    std::cout << "Building MixVPR engine..." << std::endl;
    MixVPR model(configs.mixvpr);
    if (!model.build()) {
        std::cerr << "Failed to build the TensorRT engine." << std::endl;
        return 1;
    }

    std::vector<float> desc0, desc1;
    double infer_ms = 0.0;
    const int total = warmup + iters;
    for (int i = 0; i < total; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        if (!model.infer(image0, desc0) || !model.infer(image1, desc1)) {
            std::cerr << "MixVPR inference failed." << std::endl;
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        if (i >= warmup) {
            infer_ms += elapsed_ms(t0, t1);
        }
    }
    infer_ms /= iters;

    if (!write_descriptor(model_dir + "/trt_image0.bin", desc0) ||
        !write_descriptor(model_dir + "/trt_image1.bin", desc1)) {
        std::cerr << "Failed to write TensorRT descriptors." << std::endl;
        return 1;
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Descriptor dim: " << desc0.size() << " / " << desc1.size() << std::endl;
    std::cout << "Cosine: " << cosine(desc0, desc1) << std::endl;
    std::cout << "RESULT"
              << " size=" << configs.mixvpr.image_width << "x" << configs.mixvpr.image_height
              << " dim=" << desc0.size()
              << " pair_ms=" << infer_ms
              << " per_image_ms=" << (infer_ms / 2.0)
              << " cosine=" << cosine(desc0, desc1)
              << std::endl;
    return 0;
}
