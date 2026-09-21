#include <device_launch_parameters.h>
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

__global__ void matrix_transpose_coalesced_via_local_memory(
                       const float* matrix,            // w x h
                             float* transposed_matrix, // h x w
                             unsigned int w,
                             unsigned int h)
{
    constexpr bool fix = 1;

    // add extra elements to avoid bank conflicts 
    __shared__ float local_data[GROUP_SIZE_X * GROUP_SIZE_Y + 32*fix];
    
    // indexes
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // linear index in incoming array
    const unsigned int idx = x + w * y;

    // index within 16x16 cache
    const unsigned int i = threadIdx.x + threadIdx.y * GROUP_SIZE_Y;

    // write with offsets to avoid bank conflicts 
    local_data[i + (threadIdx.y/2)*2*fix] = matrix[idx];
    __syncthreads();

    // transposed indexes
    const unsigned int xi = blockIdx.y * blockDim.y + threadIdx.x;
    const unsigned int yi = blockIdx.x * blockDim.x + threadIdx.y;
    
    // transposed linear index in incoming array
    const unsigned int idxi = xi + h * yi;

    // transposed index within 16x16 cache with applied offset
    const unsigned int ii = threadIdx.y + (threadIdx.x) * GROUP_SIZE_X  + (threadIdx.x/2)*2*fix;

    transposed_matrix[idxi] = local_data[ii];
}


const unsigned int TILE_DIM = 32;
const unsigned int BLOCK_ROWS = 8;

// coalesced transpose
// Uses shared memory to achieve coalesing in both reads and writes
// Tile width == #banks causes shared memory bank conflicts.
__global__ void transposeCoalesced(float *odata,
                                    const float *idata,
                                    unsigned int width,
                                    unsigned int height)
{
  __shared__ float tile[TILE_DIM][TILE_DIM];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*height + x] = tile[threadIdx.x][threadIdx.y + j];
}
   

// No bank-conflict transpose
// Same as transposeCoalesced except the first tile dimension is padded 
// to avoid shared memory bank conflicts.
__global__ void transposeNoBankConflicts(float *odata,
                                        const float *idata,
                                        unsigned int width,
                                        unsigned int height)
{
  __shared__ float tile[TILE_DIM][TILE_DIM+1];
    
  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
     odata[(y+j)*height + x] = tile[threadIdx.x][threadIdx.y + j];
}

namespace cuda {
void matrix_transpose_coalesced_via_local_memory(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &matrix, gpu::gpu_mem_32f &transposed_matrix, unsigned int w, unsigned int h)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();

#if 1
    ::matrix_transpose_coalesced_via_local_memory<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(matrix.cuptr(), transposed_matrix.cuptr(), w, h);
#else

    // Override worksize for this examples
    gpu::WorkSize ws(32, 8, w, h/4);

    // ::transposeCoalesced<<<ws.cuGridSize(), ws.cuBlockSize(), 0, stream>>>( transposed_matrix.cuptr(), matrix.cuptr(), w, h );
    ::transposeNoBankConflicts<<<ws.cuGridSize(), ws.cuBlockSize(), 0, stream>>>( transposed_matrix.cuptr(), matrix.cuptr(), w, h );
#endif
    CUDA_CHECK_KERNEL(stream);
}

} // namespace cuda
