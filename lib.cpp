#include <hip/hip_runtime.h>

__global__ void vector_add(float* a, const float* b, const int N)
{
    // Solution
    {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        printf("adding %d elements on indx %d\n", N, idx);

        if(idx < N) 
            a[idx] += b[idx];
    }
}