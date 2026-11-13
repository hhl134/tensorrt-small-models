#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "./mixvpr_extract config_path model_dir image_list output.bin" << std::endl;
        return 1;
    }
    const std::string config_path = argv[1];
    const std::string model_dir = argv[2];
    const std::string list_path = argv[3];
    const std::string output_path = argv[4];

    std::ifstream list(list_path);
    if (!list) {
        std::cerr << "Failed to open image list." << std::endl;
        return 1;
    }
    std::vector<std::string> paths;
    std::string line;
    while (std::getline(list, line)) {
        if (!line.empty()) {
            paths.push_back(line);
        }
    }
    if (paths.empty()) {
        std::cerr << "Image list is empty." << std::endl;
        return 1;
    }

    PipelineConfig configs(config_path, model_dir);
    MixVPR model(configs.mixvpr);
    if (!model.build()) {
        std::cerr << "Failed to build the TensorRT engine." << std::endl;
        return 1;
    }

    std::vector<float> all(paths.size() * static_cast<size_t>(configs.mixvpr.desc_dim));
    std::vector<float> descriptor;
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < paths.size(); ++i) {
        cv::Mat image = cv::imread(paths[i], cv::IMREAD_COLOR);
        if (image.empty() || !model.infer(image, descriptor)) {
            std::cerr << "Failed on " << paths[i] << std::endl;
            return 1;
        }
        if (static_cast<int>(descriptor.size()) != configs.mixvpr.desc_dim) {
            std::cerr << "Unexpected descriptor length." << std::endl;
            return 1;
        }
        std::copy(descriptor.begin(), descriptor.end(),
                  all.begin() + static_cast<std::ptrdiff_t>(i * descriptor.size()));
        if ((i + 1) % 500 == 0 || i + 1 == paths.size()) {
            std::cout << "extracted " << (i + 1) << "/" << paths.size() << std::endl;
        }
    }
    auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::ofstream output(output_path, std::ios::binary);
    if (!output) {
        std::cerr << "Failed to open " << output_path << std::endl;
        return 1;
    }
    output.write(reinterpret_cast<const char *>(all.data()),
                 static_cast<std::streamsize>(all.size() * sizeof(float)));
    if (!output) {
        std::cerr << "Failed to write descriptors." << std::endl;
        return 1;
    }
    std::cout << "RESULT n=" << paths.size()
              << " dim=" << configs.mixvpr.desc_dim
              << " total_ms=" << ms
              << " per_image_ms=" << (ms / static_cast<double>(paths.size()))
              << std::endl;
    return 0;
}
