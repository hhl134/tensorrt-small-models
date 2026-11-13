#include <cuda_runtime.h>
#include <cmath>

namespace spsg_cuda {

__device__ inline int clip_dev(int val, int max_v) {
    if (val < 0) return 0;
    return val < max_v ? val : (max_v - 1);
}

// descriptors: [dim, h, w] on device
// grid: [N, 2] normalized xy on device
// output: [N, dim] on device
__global__ void grid_sample_l2_kernel(const float *input, const float *grid, float *output,
                                      int num, int dim, int h, int w) {
    const int n = blockIdx.x;
    if (n >= num) {
        return;
    }

    const float gx = grid[n * 2];
    const float gy = grid[n * 2 + 1];
    const float ix = ((gx + 1.f) * 0.5f) * (w - 1);
    const float iy = ((gy + 1.f) * 0.5f) * (h - 1);

    const int ix_nw = clip_dev(static_cast<int>(floorf(ix)), w);
    const int iy_nw = clip_dev(static_cast<int>(floorf(iy)), h);
    const int ix_ne = clip_dev(ix_nw + 1, w);
    const int iy_ne = clip_dev(iy_nw, h);
    const int ix_sw = clip_dev(ix_nw, w);
    const int iy_sw = clip_dev(iy_nw + 1, h);
    const int ix_se = clip_dev(ix_nw + 1, w);
    const int iy_se = clip_dev(iy_nw + 1, h);

    const float nw = (ix_se - ix) * (iy_se - iy);
    const float ne = (ix - ix_sw) * (iy_sw - iy);
    const float sw = (ix_ne - ix) * (iy - iy_ne);
    const float se = (ix - ix_nw) * (iy - iy_nw);

    float *dst = output + n * dim;
    float norm_sq = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        const float *plane = input + i * h * w;
        const float v = plane[iy_nw * w + ix_nw] * nw + plane[iy_ne * w + ix_ne] * ne +
                        plane[iy_sw * w + ix_sw] * sw + plane[iy_se * w + ix_se] * se;
        dst[i] = v;
        norm_sq += v * v;
    }

    // Reduce norm within block.
    __shared__ float shared_sum[256];
    shared_sum[threadIdx.x] = norm_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv = rsqrtf(shared_sum[0] + 1e-12f);
    __syncthreads();
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] *= inv;
    }
}

cudaError_t launch_grid_sample_l2(const float *input, const float *grid, float *output,
                                  int num, int dim, int h, int w, cudaStream_t stream) {
    if (num <= 0) {
        return cudaSuccess;
    }
    const int threads = 256;
    grid_sample_l2_kernel<<<num, threads, 0, stream>>>(input, grid, output, num, dim, h, w);
    return cudaGetLastError();
}

}  // namespace spsg_cuda
