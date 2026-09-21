#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "helpers/rassert.cu"
#include "../defines.h"

#include <cublas_v2.h>

// Include WMMA header with nvcuda::wmma namespace
#include <mma.h>
using namespace nvcuda;

// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-matrix-functions

// The only dimensions currently supported by WMMA
const unsigned int D = 16; 

__global__ void matrix_multiply_wmma(
                       const float* a, // rows=h x cols=k
                       const float* b, // rows=k x cols=w
                             float* c, // rows=h x cols=w
                       unsigned int w, // N
                       unsigned int h, // M
                       unsigned int k) // K
{
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, D, D, D, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, D, D, D, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, D, D, D, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    
    __shared__ half local_a[GROUP_SIZE];
    __shared__ half local_b[GROUP_SIZE];


    // Loop over the K-dimension
    for (int j = 0; j < k; j += D)
    {
        // Seems syncthreads performed by wmma operations, so it's redunddant
        // __syncthreads();

        for (int i = 0; i < D; i+=2)
        {
            // per-thread index in shared memory
            const unsigned int li = threadIdx.x + (threadIdx.y + i) * D;

            const unsigned int ai = (blockIdx.y * blockDim.y * 8 + threadIdx.y + i) * k
                                  + threadIdx.x + j;
    
            const unsigned int bi = (threadIdx.y + j + i) * w
                                  + (blockIdx.x * blockDim.x + threadIdx.x);
    
            local_a[li] = a[ai];
            local_b[li] = b[bi];
        }

        // __syncthreads();

        wmma::load_matrix_sync(a_frag, local_a, D);
        wmma::load_matrix_sync(b_frag, local_b, D);

        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    }

    wmma::store_matrix_sync(c + blockIdx.x * blockDim.x
                              + blockIdx.y * blockDim.y * 8 * w,
                                c_frag, w, wmma::mem_row_major);

}


#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

namespace cuda {
void matrix_multiply_wmma(const gpu::WorkSize &workSize,
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
    ::matrix_multiply_wmma<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), b.cuptr(), c.cuptr(), w, h, k);
    CUDA_CHECK_KERNEL(stream);
}


void matrix_multiply_cuBLAS(
            const gpu::gpu_mem_32f &a, const gpu::gpu_mem_32f &b, gpu::gpu_mem_32f &c, unsigned int w, unsigned int h, unsigned int k)
{
    int MATRIX_M = h;
    int MATRIX_N = w;
    int MATRIX_K = k;

    float alpha = 1.0f;
    float beta = 0.0f;

    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 34523543124312, context.type());
    cudaStream_t stream = context.cudaStream();
   
    cublasHandle_t cublasHandle;
    cublasErrCheck(cublasCreate(&cublasHandle));
    
    // Match standard 
    // cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_PEDANTIC_MATH));

    // Tensor Cores will be used whenever possible.
    cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));


    // This mode is deprecated and will be removed in a future release. 
    // cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));


    // https://docs.nvidia.com/cuda/archive/12.3.1/cublas/index.html?highlight=cublasGemmEx#cublasgemmex
    // Order of operands changed because cuBLAS use column-major format
    
    cublasErrCheck(
    cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                MATRIX_N, MATRIX_M, MATRIX_K, 
                &alpha,
                b.cuptr(), CUDA_R_32F, MATRIX_N,
                a.cuptr(), CUDA_R_32F, MATRIX_K,
                &beta,
                c.cuptr(), CUDA_R_32F, MATRIX_N,
                CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUDA_CHECK_KERNEL(stream);
}

} // namespace cuda

