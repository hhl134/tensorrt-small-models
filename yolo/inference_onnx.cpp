#include "pipeline.h"
#include "yolo_gpu.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>

struct OutputLayout {
    bool channel_major{};
    int anchors{};
};

static bool output_layout(const nvinfer1::Dims &dims, int num_classes, OutputLayout &layout) {
    if (dims.nbDims == 3 && dims.d[1] == 4 + num_classes) {
        layout.channel_major = true;
        layout.anchors = dims.d[2];
        return true;
    }
    if (dims.nbDims == 3 && dims.d[2] == 4 + num_classes) {
        layout.channel_major = false;
        layout.anchors = dims.d[1];
        return true;
    }
    std::cerr << "Unexpected YOLO output rank." << std::endl;
    return false;
}

static bool is_image(const std::filesystem::path &path) {
    const std::string ext = path.extension().string();
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".JPG";
}

static void print_boxes(const std::string &name, const std::vector<YoloBox> &boxes) {
    std::cout << "image " << name << " boxes=" << boxes.size() << std::endl;
    for (const YoloBox &item : boxes) {
        std::cout << "box " << item.x1 << " " << item.y1 << " " << item.x2 << " " << item.y2
                  << " conf " << item.score << " class " << item.class_id << std::endl;
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "./yolo_onnx config_path model_dir image_or_dir" << std::endl;
        return 1;
    }
    PipelineConfig configs(argv[1], argv[2]);
    if (const char *env = std::getenv("YOLO_CONF")) {
        configs.yolo.conf_threshold = std::atof(env);
    }
    if (const char *env = std::getenv("YOLO_NMS")) {
        configs.yolo.nms_threshold = std::atof(env);
    }
    YoloOnnx model(configs.yolo);
    if (!model.build()) {
        std::cerr << "Failed to build the ONNX TensorRT engine." << std::endl;
        return 1;
    }
    const std::string &input_name = configs.yolo.input_tensor_names[0];
    const std::string &output_name = configs.yolo.output_tensor_names[0];
    auto *device_input = static_cast<float *>(model.deviceBuffer(input_name));
    auto *device_output = static_cast<float *>(model.deviceBuffer(output_name));
    OutputLayout layout;
    if (!output_layout(model.tensorShape(output_name), configs.yolo.num_classes, layout)) {
        return 1;
    }
    YoloGpu gpu;

    std::vector<std::filesystem::path> images;
    const std::filesystem::path input_path(argv[3]);
    if (std::filesystem::is_directory(input_path)) {
        for (const auto &entry : std::filesystem::directory_iterator(input_path)) {
            if (entry.is_regular_file() && is_image(entry.path())) {
                images.push_back(entry.path());
            }
        }
        std::sort(images.begin(), images.end());
    } else {
        images.push_back(input_path);
    }
    if (images.empty()) {
        std::cerr << "No images in " << argv[3] << std::endl;
        return 1;
    }

    const int warmup = std::getenv("YOLO_WARMUP") ? std::atoi(std::getenv("YOLO_WARMUP")) : 0;
    const int iters = std::getenv("YOLO_ITERS") ? std::atoi(std::getenv("YOLO_ITERS")) : 0;
    auto run_one = [&](const cv::Mat &image, std::vector<YoloBox> &boxes) {
        if (!image.isContinuous()) {
            return false;
        }
        const LetterboxGeom geom = make_letterbox(image.cols, image.rows, configs.yolo.image_width, configs.yolo.image_height);
        if (!gpu.preprocess(image.data, image.cols, image.rows, device_input, configs.yolo.image_width, configs.yolo.image_height,
                            geom, model.stream())) {
            return false;
        }
        if (!model.enqueue()) {
            return false;
        }
        return gpu.decode_nms(device_output, layout.channel_major, layout.anchors, configs.yolo.num_classes, geom, image.cols,
                              image.rows, configs.yolo.conf_threshold, configs.yolo.nms_threshold, boxes, model.stream()) >= 0;
    };

    if (iters > 0) {
        cv::Mat image = cv::imread(images.front().string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to read " << images.front() << std::endl;
            return 1;
        }
        const LetterboxGeom geom = make_letterbox(image.cols, image.rows, configs.yolo.image_width, configs.yolo.image_height);
        double pre_ms = 0.0;
        double enq_ms = 0.0;
        double nms_ms = 0.0;
        std::vector<YoloBox> boxes;
        for (int n = 0; n < warmup + iters; ++n) {
            auto pre_start = std::chrono::steady_clock::now();
            if (!gpu.preprocess(image.data, image.cols, image.rows, device_input, configs.yolo.image_width,
                                configs.yolo.image_height, geom, model.stream())) {
                return 1;
            }
            auto pre_end = std::chrono::steady_clock::now();
            if (!model.enqueue() || cudaStreamSynchronize(model.stream()) != cudaSuccess) {
                return 1;
            }
            auto enq_end = std::chrono::steady_clock::now();
            if (gpu.decode_nms(device_output, layout.channel_major, layout.anchors, configs.yolo.num_classes, geom, image.cols,
                               image.rows, configs.yolo.conf_threshold, configs.yolo.nms_threshold, boxes, model.stream()) < 0) {
                return 1;
            }
            auto end = std::chrono::steady_clock::now();
            if (n >= warmup) {
                pre_ms += std::chrono::duration<double, std::milli>(pre_end - pre_start).count();
                enq_ms += std::chrono::duration<double, std::milli>(enq_end - pre_end).count();
                nms_ms += std::chrono::duration<double, std::milli>(end - enq_end).count();
            }
        }
        std::cout << "RESULT backend=onnx pre_ms=" << pre_ms / iters << " enqueue_ms=" << enq_ms / iters
                  << " nms_ms=" << nms_ms / iters << " infer_ms=" << (enq_ms + nms_ms) / iters << " iters=" << iters
                  << std::endl;
        return 0;
    }

    for (const auto &path : images) {
        cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to read " << path << std::endl;
            return 1;
        }
        std::vector<YoloBox> boxes;
        if (!run_one(image, boxes)) {
            std::cerr << "ONNX inference failed." << std::endl;
            return 1;
        }
        print_boxes(path.filename().string(), boxes);
    }
    return 0;
}
