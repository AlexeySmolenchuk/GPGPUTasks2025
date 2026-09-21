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


// https://kharshit.github.io/blog/2024/06/07/matrix-multiplication-cuda#further-optimization
#define TILE_SIZE 16

// Kernel for matrix multiplication using tiling and shared memory
__global__ void matMulSharedMemoryKernel(const float* A,
                                        const float* B,
                                        float* C,
                                        unsigned int w,
                                        unsigned int h,
                                        unsigned int k)
{
    // Shared memory for tiles of A and B
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    // Calculate the global row and column index of the element
    int globalRow = blockIdx.y * blockDim.y + threadIdx.y;
    int globalCol = blockIdx.x * blockDim.x + threadIdx.x;

    float Cvalue = 0.0f;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over the tiles of the input matrices
    // A.width/TILE_SIZE and B.height/TILE_SIZE; take care of the last tile
    for (int m = 0; m < (k + TILE_SIZE - 1) / TILE_SIZE; ++m)
    {
        // Load elements of A into shared memory
        // if shared memory defined using 1d array, we'd have used shared_A[row * TILE_SIZE + col]
        if (row < m && (m * TILE_SIZE + col) < k) 
        {
            shared_A[row][col] = A[globalRow * k + m * TILE_SIZE + col];
        } else 
        {
            // When matrix dimensions are not exact multiples of the tile size,
            // some threads in the last blocks might access elements outside
            // the matrix boundaries. By setting out-of-bounds elements to zero,
            // we ensure that these threads do not contribute invalid values to final result.
            // e.g. Matrix A = [100x100] and TILE_SIZE = 16
            shared_A[row][col] = 0.0f;
        }
        // Load elements of B into shared memory
        if (col < w && (m * TILE_SIZE + row) < k) 
        {
            shared_B[row][col] = B[(m * TILE_SIZE + row) * w + globalCol];
        } else 
        {
            shared_B[row][col] = 0.0f;
        }
        // Synchronize to ensure all threads have loaded their elements
        __syncthreads();

        // Compute the partial result
        for (int k = 0; k < TILE_SIZE; ++k)
            Cvalue += shared_A[row][k] * shared_B[k][col];

        // Synchronize to ensure all threads have completed the computation
        __syncthreads();
    }

    // Write the result to global memory
    if (globalRow < h && globalCol < w)
        C[globalRow * w + globalCol] = Cvalue;
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
    // ::matMulSharedMemoryKernel<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
