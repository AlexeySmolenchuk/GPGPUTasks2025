#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_multiply_via_local_memory(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w,
                       unsigned int h,
                       unsigned int k)
{
    __shared__ float local_a[GROUP_SIZE];
    __shared__ float local_b[GROUP_SIZE];
    
    const unsigned int D = 16; 
    
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    const unsigned int li = threadIdx.x + threadIdx.y * D;

    const unsigned int ci = x + w * y;

    float result = 0;

    for (unsigned int i = 0; i < k/D; i++)
    {
        const unsigned int ai = (blockIdx.y * blockDim.y + threadIdx.y) * k
                              + threadIdx.x + i * D;

        const unsigned int bi = (blockIdx.x * blockDim.x + threadIdx.x)
                              + (threadIdx.y + i * D) * w;

        __syncthreads();
        local_a[li] = a[ai];
        local_b[li] = b[bi];
        __syncthreads();

        for (unsigned int j = 0; j < D; j++)
            result += local_a[j + threadIdx.y * D]
                    * local_b[threadIdx.x + j * D];
    }

    c[ci] = result;
}

namespace cuda {
void matrix_multiply_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c,
             unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
