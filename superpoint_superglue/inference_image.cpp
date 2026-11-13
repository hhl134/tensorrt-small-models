//
// Created by haoyuefan on 2021/9/22.
//

#include <memory>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <thread>
#include "utils.h"
#include "super_glue.h"
#include "super_point.h"

static double elapsed_ms(const std::chrono::steady_clock::time_point &start,
                         const std::chrono::steady_clock::time_point &end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv){
  if(argc != 5 && argc != 6){
    std::cerr << "./superpointglue_image config_path model_dir first_image second_image [output_match_png]" << std::endl;
    return 0;
  }

  std::string config_path = argv[1];
  std::string model_dir = argv[2];
  std::string image0_path = argv[3];
  std::string image1_path = argv[4];
  std::string output_path = (argc == 6) ? argv[5] : "match_image.png";

  int warmup = 5;
  int iters = 30;
  if (const char *env = std::getenv("SPSG_WARMUP")) {
    warmup = std::atoi(env);
  }
  if (const char *env = std::getenv("SPSG_ITERS")) {
    iters = std::atoi(env);
  }
  const bool parallel_sp = []() {
    const char *env = std::getenv("SPSG_PARALLEL_SP");
    return env && std::atoi(env) != 0;
  }();

  cv::Mat image0 = cv::imread(image0_path, cv::IMREAD_GRAYSCALE);
  cv::Mat image1 = cv::imread(image1_path, cv::IMREAD_GRAYSCALE);

  if(image0.empty() || image1.empty()){
    std::cerr << "Input image is empty. Please check the image path." << std::endl;
    return 0;
  }

  Configs configs(config_path, model_dir);
  int width = configs.superglue_config.image_width;
  int height = configs.superglue_config.image_height;

  cv::resize(image0, image0, cv::Size(width, height));
  cv::resize(image1, image1, cv::Size(width, height));
  std::cout << "First image size: " << image0.cols << "x" << image0.rows << std::endl;
  std::cout << "Second image size: " << image1.cols << "x" << image1.rows << std::endl;

  std::cout << "Building inference engine......" << std::endl;
  auto superpoint0 = std::make_shared<SuperPoint>(configs.superpoint_config);
  if (!superpoint0->build()){
    std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl;
    return 0;
  }
  SuperPointPtr superpoint1 = superpoint0;
  if (parallel_sp) {
    superpoint1 = std::make_shared<SuperPoint>(configs.superpoint_config);
    if (!superpoint1->build()){
      std::cerr << "Error in second SuperPoint building engine." << std::endl;
      return 0;
    }
  }
  auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
  if (!superglue->build()){
    std::cerr << "Error in SuperGlue building engine. Please check your onnx model path." << std::endl;
    return 0;
  }
  std::cout << "SuperPoint and SuperGlue inference engine build success."
            << " parallel_sp=" << (parallel_sp ? 1 : 0) << std::endl;

  Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0, feature_points1;
  std::vector<cv::DMatch> superglue_matches;

  double image0_tcount = 0;
  double image1_tcount = 0;
  double extract_tcount = 0;
  double match_tcount = 0;
  const int total = warmup + iters;
  std::cout << "Benchmark warmup=" << warmup << " iters=" << iters << std::endl;
  for (int i = 0; i < total; ++i){
    bool ok0 = true;
    bool ok1 = true;
    if (parallel_sp) {
      auto start = std::chrono::steady_clock::now();
      std::thread t0([&]() { ok0 = superpoint0->infer(image0, feature_points0); });
      std::thread t1([&]() { ok1 = superpoint1->infer(image1, feature_points1); });
      t0.join();
      t1.join();
      auto end = std::chrono::steady_clock::now();
      if(i >= warmup){
        const double e = elapsed_ms(start, end);
        extract_tcount += e;
        image0_tcount += e;
        image1_tcount += 0.0;
      }
    } else {
      auto start = std::chrono::steady_clock::now();
      ok0 = superpoint0->infer(image0, feature_points0);
      auto mid = std::chrono::steady_clock::now();
      ok1 = superpoint0->infer(image1, feature_points1);
      auto end = std::chrono::steady_clock::now();
      if(i >= warmup){
        image0_tcount += elapsed_ms(start, mid);
        image1_tcount += elapsed_ms(mid, end);
        extract_tcount += elapsed_ms(start, end);
      }
    }
    if (!ok0 || !ok1) {
      std::cerr << "Failed when extracting features." << std::endl;
      return 0;
    }

    auto start = std::chrono::steady_clock::now();
    superglue->matching_points(feature_points0, feature_points1, superglue_matches);
    auto end = std::chrono::steady_clock::now();
    if(i >= warmup){
      match_tcount += elapsed_ms(start, end);
    }
  }

  const double sp0_ms = image0_tcount / iters;
  const double sp1_ms = image1_tcount / iters;
  const double extract_ms = extract_tcount / iters;
  const double sg_ms = match_tcount / iters;
  const double total_ms = extract_ms + sg_ms;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "First image feature points number: " << feature_points0.cols() << std::endl;
  std::cout << "Second image feature points number: " << feature_points1.cols() << std::endl;
  std::cout << "First image infer cost " << sp0_ms << " MS" << std::endl;
  std::cout << "Second image infer cost " << sp1_ms << " MS" << std::endl;
  std::cout << "Feature extract cost " << extract_ms << " MS" << std::endl;
  std::cout << "Match image cost " << sg_ms << " MS" << std::endl;
  std::cout << "Matches: " << superglue_matches.size() << std::endl;

  // Machine-readable summary for experiment aggregation.
  std::cout << "RESULT"
            << " size=" << width << "x" << height
            << " kpts0=" << feature_points0.cols()
            << " kpts1=" << feature_points1.cols()
            << " matches=" << superglue_matches.size()
            << " sp0_ms=" << sp0_ms
            << " sp1_ms=" << sp1_ms
            << " sg_ms=" << sg_ms
            << " total_ms=" << total_ms
            << " extract_ms=" << extract_ms
            << " parallel_sp=" << (parallel_sp ? 1 : 0)
            << std::endl;

  cv::Mat match_image;
  std::vector<cv::KeyPoint> keypoints0, keypoints1;
  for(int i = 0; i < feature_points0.cols(); ++i){
    double score = feature_points0(0, i);
    double x = feature_points0(1, i);
    double y = feature_points0(2, i);
    keypoints0.emplace_back(x, y, 8, -1, score);
  }
  for(int i = 0; i < feature_points1.cols(); ++i){
    double score = feature_points1(0, i);
    double x = feature_points1(1, i);
    double y = feature_points1(2, i);
    keypoints1.emplace_back(x, y, 8, -1, score);
  }

  cv::drawMatches(image0, keypoints0, image1, keypoints1, superglue_matches, match_image);
  cv::imwrite(output_path, match_image);
  std::cout << "Saved match image to " << output_path << std::endl;

  return 0;
}
