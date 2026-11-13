#include "pipeline.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <cuda_runtime_api.h>

static double elapsed_ms(const std::chrono::steady_clock::time_point &start,
                         const std::chrono::steady_clock::time_point &end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static size_t gpu_free_bytes() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return free_bytes;
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) {
        std::cerr << "./aliked_lightglue_image config_path model_dir image0 image1 [output_png]" << std::endl;
        return 1;
    }
    const std::string config_path = argv[1];
    const std::string model_dir = argv[2];
    const std::string image0_path = argv[3];
    const std::string image1_path = argv[4];
    const std::string output_path = (argc == 6) ? argv[5] : "match_image.png";

    int warmup = 2;
    int iters = 10;
    if (const char *env = std::getenv("ALG_WARMUP")) {
        warmup = std::atoi(env);
    }
    if (const char *env = std::getenv("ALG_ITERS")) {
        iters = std::atoi(env);
    }

    cv::Mat image0 = cv::imread(image0_path, cv::IMREAD_COLOR);
    cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_COLOR);
    if (image0.empty() || image1.empty()) {
        std::cerr << "Failed to read input images." << std::endl;
        return 1;
    }

    PipelineConfig configs(config_path, model_dir);
    cv::resize(image0, image0, cv::Size(configs.aliked.image_width, configs.aliked.image_height));
    cv::resize(image1, image1, cv::Size(configs.aliked.image_width, configs.aliked.image_height));

    std::cout << "Building ALIKED and LightGlue engines..." << std::endl;
    const size_t free_before = gpu_free_bytes();
    Aliked aliked(configs.aliked);
    if (!aliked.build()) {
        std::cerr << "Failed to build TensorRT engines." << std::endl;
        return 1;
    }
    const size_t free_after_aliked = gpu_free_bytes();
    LightGlueMatcher lightglue(configs.lightglue);
    if (!lightglue.build()) {
        std::cerr << "Failed to build TensorRT engines." << std::endl;
        return 1;
    }
    const size_t free_after_glue = gpu_free_bytes();

    Features features0, features1;
    std::vector<cv::DMatch> matches;
    double extract_ms = 0;
    double match_ms = 0;
    const int total = warmup + iters;
    for (int i = 0; i < total; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        if (!aliked.infer(image0, features0) || !aliked.infer(image1, features1)) {
            std::cerr << "ALIKED inference failed." << std::endl;
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        if (!lightglue.match(features0, features1, matches)) {
            std::cerr << "LightGlue inference failed." << std::endl;
            return 1;
        }
        auto t2 = std::chrono::steady_clock::now();
        if (i >= warmup) {
            extract_ms += elapsed_ms(t0, t1);
            match_ms += elapsed_ms(t1, t2);
        }
    }
    extract_ms /= iters;
    match_ms /= iters;

    std::vector<cv::KeyPoint> keypoints0, keypoints1;
    keypoints0.reserve(features0.count);
    keypoints1.reserve(features1.count);
    for (int i = 0; i < features0.count; ++i) {
        keypoints0.emplace_back(features0.keypoints[i * 2], features0.keypoints[i * 2 + 1], 8, -1, features0.scores[i]);
    }
    for (int i = 0; i < features1.count; ++i) {
        keypoints1.emplace_back(features1.keypoints[i * 2], features1.keypoints[i * 2 + 1], 8, -1, features1.scores[i]);
    }
    cv::Mat canvas;
    cv::drawMatches(image0, keypoints0, image1, keypoints1, matches, canvas);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Keypoints: " << features0.count << " / " << features1.count << std::endl;
    std::cout << "Matches: " << matches.size() << std::endl;
    std::cout << "ALIKED x2: " << extract_ms << " MS" << std::endl;
    std::cout << "LightGlue: " << match_ms << " MS" << std::endl;
    const auto mb_delta = [](size_t before, size_t after) {
        const double delta = static_cast<double>(before) - static_cast<double>(after);
        return delta / (1024.0 * 1024.0);
    };
    std::cout << "RESULT"
              << " size=" << configs.aliked.image_width << "x" << configs.aliked.image_height
              << " cap=" << configs.aliked.max_keypoints
              << " glue_max=" << configs.lightglue.max_keypoints
              << " kpts=" << features0.count << "/" << features1.count
              << " matches=" << matches.size()
              << " aliked_ms=" << extract_ms
              << " lightglue_ms=" << match_ms
              << " total_ms=" << (extract_ms + match_ms)
              << " aliked_mb=" << mb_delta(free_before, free_after_aliked)
              << " lightglue_mb=" << mb_delta(free_after_aliked, free_after_glue)
              << " gpu_mb=" << mb_delta(free_before, free_after_glue)
              << std::endl;
    cv::imwrite(output_path, canvas);
    std::cout << "Saved match image to " << output_path << std::endl;
    return 0;
}
