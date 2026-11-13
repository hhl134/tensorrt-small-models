#include "yolo_gpu.h"

#include <algorithm>
#include <cmath>

namespace {

struct Det {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int class_id;
};

__device__ int reflect101(int p, int length) {
    if (static_cast<unsigned>(p) < static_cast<unsigned>(length)) {
        return p;
    }
    if (length == 1) {
        return 0;
    }
    do {
        if (p < 0) {
            p = -p - 1;
        } else {
            p = length - 1 - (p - length);
        }
    } while (static_cast<unsigned>(p) >= static_cast<unsigned>(length));
    return p;
}

__device__ float at_bgr(const uint8_t *src, int width, int height, int x, int y, int channel) {
    x = reflect101(x, width);
    y = reflect101(y, height);
    return static_cast<float>(src[(y * width + x) * 3 + channel]);
}

__device__ void bilinear_rgb(const uint8_t *src, int width, int height, float sx, float sy, float &r, float &g, float &b) {
    const int x0 = static_cast<int>(floorf(sx));
    const int y0 = static_cast<int>(floorf(sy));
    const float lx = sx - static_cast<float>(x0);
    const float ly = sy - static_cast<float>(y0);
    const float hx = 1.0f - lx;
    const float hy = 1.0f - ly;
    const float w00 = hy * hx;
    const float w01 = hy * lx;
    const float w10 = ly * hx;
    const float w11 = ly * lx;
    b = w00 * at_bgr(src, width, height, x0, y0, 0) + w01 * at_bgr(src, width, height, x0 + 1, y0, 0) +
        w10 * at_bgr(src, width, height, x0, y0 + 1, 0) + w11 * at_bgr(src, width, height, x0 + 1, y0 + 1, 0);
    g = w00 * at_bgr(src, width, height, x0, y0, 1) + w01 * at_bgr(src, width, height, x0 + 1, y0, 1) +
        w10 * at_bgr(src, width, height, x0, y0 + 1, 1) + w11 * at_bgr(src, width, height, x0 + 1, y0 + 1, 1);
    r = w00 * at_bgr(src, width, height, x0, y0, 2) + w01 * at_bgr(src, width, height, x0 + 1, y0, 2) +
        w10 * at_bgr(src, width, height, x0, y0 + 1, 2) + w11 * at_bgr(src, width, height, x0 + 1, y0 + 1, 2);
}

__global__ void letterbox_kernel(const uint8_t *src, int src_w, int src_h, float *dst, int dst_w, int dst_h, int left,
                                 int top, int resized_w, int resized_h) {
    const int position = blockIdx.x * blockDim.x + threadIdx.x;
    const int edge = dst_w * dst_h;
    if (position >= edge) {
        return;
    }
    const int dx = position % dst_w;
    const int dy = position / dst_w;
    float r, g, b;
    if (dx < left || dx >= left + resized_w || dy < top || dy >= top + resized_h) {
        r = g = b = 114.0f;
    } else {
        const float sx = (static_cast<float>(dx - left) + 0.5f) * static_cast<float>(src_w) / static_cast<float>(resized_w) - 0.5f;
        const float sy = (static_cast<float>(dy - top) + 0.5f) * static_cast<float>(src_h) / static_cast<float>(resized_h) - 0.5f;
        bilinear_rgb(src, src_w, src_h, sx, sy, r, g, b);
    }
    const float scale = 1.0f / 255.0f;
    dst[position] = r * scale;
    dst[edge + position] = g * scale;
    dst[2 * edge + position] = b * scale;
}

__device__ float iou_det(const Det &left, const Det &right) {
    const float x1 = fmaxf(left.x1, right.x1);
    const float y1 = fmaxf(left.y1, right.y1);
    const float x2 = fminf(left.x2, right.x2);
    const float y2 = fminf(left.y2, right.y2);
    const float inter = fmaxf(0.0f, x2 - x1) * fmaxf(0.0f, y2 - y1);
    const float area = (left.x2 - left.x1) * (left.y2 - left.y1) + (right.x2 - right.x1) * (right.y2 - right.y1) - inter;
    return area > 0.0f ? inter / area : 0.0f;
}

__device__ float value_at(const float *output, int anchor, int channel, int stride, int channel_major) {
    return channel_major ? output[channel * stride + anchor] : output[anchor * stride + channel];
}

__global__ void decode_kernel(const float *output, int anchors, int num_classes, int channel_major, float scale, float pad_x,
                              float pad_y, int img_w, int img_h, float conf_threshold, Det *dets, int *count, int capacity) {
    const int anchor = blockIdx.x * blockDim.x + threadIdx.x;
    if (anchor >= anchors) {
        return;
    }
    const int stride = channel_major ? anchors : 4 + num_classes;
    int class_id = 0;
    float score = value_at(output, anchor, 4, stride, channel_major);
    for (int cls = 1; cls < num_classes; ++cls) {
        const float value = value_at(output, anchor, 4 + cls, stride, channel_major);
        if (value > score) {
            score = value;
            class_id = cls;
        }
    }
    if (score < conf_threshold) {
        return;
    }
    const int slot = atomicAdd(count, 1);
    if (slot >= capacity) {
        return;
    }
    const float cx = value_at(output, anchor, 0, stride, channel_major);
    const float cy = value_at(output, anchor, 1, stride, channel_major);
    const float w = value_at(output, anchor, 2, stride, channel_major);
    const float h = value_at(output, anchor, 3, stride, channel_major);
    Det det;
    det.x1 = fminf(fmaxf((cx - w * 0.5f - pad_x) / scale, 0.0f), static_cast<float>(img_w));
    det.y1 = fminf(fmaxf((cy - h * 0.5f - pad_y) / scale, 0.0f), static_cast<float>(img_h));
    det.x2 = fminf(fmaxf((cx + w * 0.5f - pad_x) / scale, 0.0f), static_cast<float>(img_w));
    det.y2 = fminf(fmaxf((cy + h * 0.5f - pad_y) / scale, 0.0f), static_cast<float>(img_h));
    det.score = score;
    det.class_id = class_id;
    dets[slot] = det;
}

__global__ void nms_kernel(Det *dets, const int *candidate_count, int capacity, float nms_threshold, Det *kept, int *kept_count) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    int count = *candidate_count;
    if (count > capacity) {
        count = capacity;
    }
    for (int i = 1; i < count; ++i) {
        const Det key = dets[i];
        int j = i - 1;
        while (j >= 0 && dets[j].score < key.score) {
            dets[j + 1] = dets[j];
            --j;
        }
        dets[j + 1] = key;
    }
    int written = 0;
    for (int i = 0; i < count; ++i) {
        bool suppress = false;
        for (int j = 0; j < written; ++j) {
            if (dets[i].class_id == kept[j].class_id && iou_det(dets[i], kept[j]) > nms_threshold) {
                suppress = true;
                break;
            }
        }
        if (!suppress) {
            kept[written++] = dets[i];
        }
    }
    *kept_count = written;
}

}  // namespace

LetterboxGeom make_letterbox(int src_w, int src_h, int dst_w, int dst_h) {
    LetterboxGeom geom;
    geom.scale = std::min(static_cast<float>(dst_w) / static_cast<float>(src_w),
                          static_cast<float>(dst_h) / static_cast<float>(src_h));
    geom.resized_w = std::lround(src_w * geom.scale);
    geom.resized_h = std::lround(src_h * geom.scale);
    const float pad_x = (dst_w - geom.resized_w) / 2.0f;
    const float pad_y = (dst_h - geom.resized_h) / 2.0f;
    geom.left = std::lround(pad_x - 0.1f);
    geom.top = std::lround(pad_y - 0.1f);
    geom.pad_x = static_cast<float>(geom.left);
    geom.pad_y = static_cast<float>(geom.top);
    return geom;
}

YoloGpu::YoloGpu() {
    cudaMalloc(&candidates_, sizeof(Det) * kMaxCandidates);
    cudaMalloc(&kept_, sizeof(Det) * kMaxCandidates);
    cudaMalloc(&candidate_count_, sizeof(int));
    cudaMalloc(&kept_count_, sizeof(int));
}

YoloGpu::~YoloGpu() {
    cudaFree(image_);
    cudaFree(candidates_);
    cudaFree(kept_);
    cudaFree(candidate_count_);
    cudaFree(kept_count_);
}

bool YoloGpu::preprocess(const uint8_t *host_bgr, int src_w, int src_h, float *device_nchw, int dst_w, int dst_h,
                         const LetterboxGeom &geom, cudaStream_t stream) {
    const size_t bytes = static_cast<size_t>(src_w) * static_cast<size_t>(src_h) * 3;
    if (bytes > image_bytes_) {
        cudaFree(image_);
        if (cudaMalloc(&image_, bytes) != cudaSuccess) {
            image_ = nullptr;
            image_bytes_ = 0;
            return false;
        }
        image_bytes_ = bytes;
    }
    if (cudaMemcpyAsync(image_, host_bgr, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        return false;
    }
    const int threads = 256;
    const int blocks = (dst_w * dst_h + threads - 1) / threads;
    letterbox_kernel<<<blocks, threads, 0, stream>>>(image_, src_w, src_h, device_nchw, dst_w, dst_h, geom.left, geom.top,
                                                     geom.resized_w, geom.resized_h);
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

int YoloGpu::decode_nms(const float *device_output, bool channel_major, int anchors, int num_classes,
                        const LetterboxGeom &geom, int img_w, int img_h, float conf_threshold, float nms_threshold,
                        std::vector<YoloBox> &boxes, cudaStream_t stream) {
    boxes.clear();
    if (cudaMemsetAsync(candidate_count_, 0, sizeof(int), stream) != cudaSuccess) {
        return -1;
    }
    const int threads = 256;
    const int blocks = (anchors + threads - 1) / threads;
    decode_kernel<<<blocks, threads, 0, stream>>>(device_output, anchors, num_classes, channel_major ? 1 : 0, geom.scale,
                                                  geom.pad_x, geom.pad_y, img_w, img_h, conf_threshold,
                                                  static_cast<Det *>(candidates_), candidate_count_, kMaxCandidates);
    nms_kernel<<<1, 1, 0, stream>>>(static_cast<Det *>(candidates_), candidate_count_, kMaxCandidates, nms_threshold,
                                   static_cast<Det *>(kept_), kept_count_);
    int kept_n = 0;
    if (cudaMemcpyAsync(&kept_n, kept_count_, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return -1;
    }
    std::vector<Det> host(static_cast<size_t>(kept_n));
    if (kept_n > 0 &&
        cudaMemcpy(host.data(), kept_, sizeof(Det) * static_cast<size_t>(kept_n), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return -1;
    }
    boxes.reserve(host.size());
    for (const Det &det : host) {
        boxes.push_back({det.x1, det.y1, det.x2, det.y2, det.score, det.class_id});
    }
    return kept_n;
}
