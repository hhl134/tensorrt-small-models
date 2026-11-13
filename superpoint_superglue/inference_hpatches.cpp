#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

#include "super_glue.h"
#include "super_point.h"
#include "utils.h"

namespace {

bool is_dir(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::vector<std::string> list_sequences(const std::string &root) {
    std::vector<std::string> names;
    DIR *dp = opendir(root.c_str());
    if (!dp) {
        return names;
    }
    while (auto *ent = readdir(dp)) {
        std::string name = ent->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        if (is_dir(root + "/" + name)) {
            names.push_back(name);
        }
    }
    closedir(dp);
    std::sort(names.begin(), names.end());
    return names;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "./superpointglue_hpatches config model_dir hpatches_root output.txt" << std::endl;
        return 1;
    }
    const std::string config_path = argv[1];
    const std::string model_dir = argv[2];
    const std::string root = argv[3];
    const std::string output_path = argv[4];

    int max_seq = 0;
    if (const char *env = std::getenv("HP_MAX_SEQ")) {
        max_seq = std::atoi(env);
    }

    Configs configs(config_path, model_dir);
    const int width = configs.superglue_config.image_width;
    const int height = configs.superglue_config.image_height;

    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
    if (!superpoint->build() || !superglue->build()) {
        std::cerr << "Failed to build SuperPoint or SuperGlue." << std::endl;
        return 1;
    }

    std::ofstream out(output_path);
    if (!out) {
        std::cerr << "Cannot write " << output_path << std::endl;
        return 1;
    }
    out.setf(std::ios::fixed);
    out.precision(3);

    const auto sequences = list_sequences(root);
    int written = 0;
    int seq_done = 0;
    for (const auto &name : sequences) {
        if (max_seq > 0 && seq_done >= max_seq) {
            break;
        }
        const std::string dir = root + "/" + name;
        cv::Mat image0 = cv::imread(dir + "/1.ppm", cv::IMREAD_GRAYSCALE);
        if (image0.empty()) {
            std::cerr << "Skip " << name << ": missing 1.ppm" << std::endl;
            continue;
        }
        const int w0 = image0.cols;
        const int h0 = image0.rows;
        cv::resize(image0, image0, cv::Size(width, height));
        Eigen::Matrix<double, 259, Eigen::Dynamic> features0;
        if (!superpoint->infer(image0, features0)) {
            std::cerr << "SuperPoint failed on " << name << "/1" << std::endl;
            continue;
        }
        const double sx0 = static_cast<double>(w0) / width;
        const double sy0 = static_cast<double>(h0) / height;

        for (int pair = 2; pair <= 6; ++pair) {
            cv::Mat image1 = cv::imread(dir + "/" + std::to_string(pair) + ".ppm", cv::IMREAD_GRAYSCALE);
            if (image1.empty()) {
                continue;
            }
            const int w1 = image1.cols;
            const int h1 = image1.rows;
            cv::resize(image1, image1, cv::Size(width, height));
            Eigen::Matrix<double, 259, Eigen::Dynamic> features1;
            if (!superpoint->infer(image1, features1)) {
                std::cerr << "SuperPoint failed on " << name << "/" << pair << std::endl;
                continue;
            }
            std::vector<cv::DMatch> matches;
            superglue->matching_points(features0, features1, matches);
            const double sx1 = static_cast<double>(w1) / width;
            const double sy1 = static_cast<double>(h1) / height;

            out << name << " " << pair << " " << w0 << " " << h0 << " " << matches.size();
            for (const auto &match : matches) {
                const double x0 = features0(1, match.queryIdx) * sx0;
                const double y0 = features0(2, match.queryIdx) * sy0;
                const double x1 = features1(1, match.trainIdx) * sx1;
                const double y1 = features1(2, match.trainIdx) * sy1;
                out << " " << x0 << " " << y0 << " " << x1 << " " << y1;
            }
            out << "\n";
            ++written;
        }
        ++seq_done;
        if (seq_done % 10 == 0) {
            std::cerr << "sequences " << seq_done << " pairs " << written << std::endl;
        }
    }
    std::cerr << "Wrote " << written << " pairs to " << output_path << std::endl;
    return written > 0 ? 0 : 1;
}
