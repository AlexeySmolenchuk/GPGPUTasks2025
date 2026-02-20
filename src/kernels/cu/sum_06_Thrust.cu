#include <device_launch_parameters.h>
#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#define THRUST_WRAPPED_NAMESPACE course
#include <thrust/device_vector.h>

#define WARP_SIZE 32


unsigned int sum_06_Thrust(::course::thrust::device_vector<unsigned int> &vec)
{
    return ::course::thrust::reduce(vec.begin(), vec.end());
}

