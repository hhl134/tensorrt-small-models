#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

struct YoloBox {
    float x1{};
    float y1{};
    float x2{};
    float y2{};
    float score{};
    int class_id{};
};

struct LetterboxGeom {
    float scale{};
    float pad_x{};
    float pad_y{};
    int resized_w{};
    int resized_h{};
    int left{};
    int top{};
};

LetterboxGeom make_letterbox(int src_w, int src_h, int dst_w, int dst_h);

class YoloGpu {
public:
    YoloGpu();
    ~YoloGpu();

    bool preprocess(const std::uint8_t *host_bgr, int src_w, int src_h, float *device_nchw, int dst_w, int dst_h,
                    const LetterboxGeom &geom, cudaStream_t stream);
    int decode_nms(const float *device_output, bool channel_major, int anchors, int num_classes, const LetterboxGeom &geom,
                   int img_w, int img_h, float conf_threshold, float nms_threshold, std::vector<YoloBox> &boxes,
                   cudaStream_t stream);

private:
    std::uint8_t *image_{nullptr};
    size_t image_bytes_{0};
    void *candidates_{nullptr};
    int *candidate_count_{nullptr};
    void *kept_{nullptr};
    int *kept_count_{nullptr};
    static constexpr int kMaxCandidates = 8400;
};
