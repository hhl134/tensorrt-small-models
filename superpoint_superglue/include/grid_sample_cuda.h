#ifndef GRID_SAMPLE_CUDA_H_
#define GRID_SAMPLE_CUDA_H_

#include <cuda_runtime.h>

namespace spsg_cuda {

cudaError_t launch_grid_sample_l2(const float *input, const float *grid, float *output,
                                  int num, int dim, int h, int w, cudaStream_t stream);

}  // namespace spsg_cuda

#endif  // GRID_SAMPLE_CUDA_H_
