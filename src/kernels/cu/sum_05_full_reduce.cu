#include <device_launch_parameters.h>
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#define WARP_SIZE 32

__global__ void sum_05_full_reduce(
    const unsigned int* a,
    unsigned int* b,
    unsigned int  n)
{
    // Подсказки:
    // const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    // const uint local_index = threadIdx.x;
    // __shared__ unsigned int local_data[GROUP_SIZE];
    // __syncthreads();

    int sum = 0;
    size_t start = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    for (size_t i = start; i<start+4; i++){
        sum += a[i];
        // sum += __ldg(a+i);
    }
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    __shared__ int shared_sum;
    shared_sum = 0;
    
    __syncthreads();

    if (threadIdx.x % 32 == 0)
        atomicAdd(&shared_sum, sum);

    __syncthreads();

    if (threadIdx.x == 0)
        atomicAdd(b, shared_sum);
}

namespace cuda {
void sum_05_full_reduce(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_05_full_reduce<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
