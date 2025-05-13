#include <hip/hip_runtime.h>
#include <iostream>
#include <math.h>
#include <omp.h> // TODO: What does OPen MP do?
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sstream>

// Computes ceil(numerator/divisor) for integer types.
template <typename intT1,
          class = typename std::enable_if<std::is_integral<intT1>::value>::type,
          typename intT2,
          class = typename std::enable_if<std::is_integral<intT2>::value>::type>
intT1 ceildiv(const intT1 numerator, const intT2 divisor)
{
    return (numerator + divisor - 1) / divisor;
}


int main()
{
    std::cout << "HIP vector addition example\n";

    const int N = 4;

    std::cout << "N: " << N << "\n";
    
    std::vector<float> vala(N);
    for(int i = 0; i < vala.size(); ++i)
    {
        vala[i] = i; // or whatever you want to fill it with
    }

    std::vector<float> valb(N);
    for(int i = 0; i < valb.size(); ++i)
    {
        valb[i] = i; // or whatever you want to fill it with
    }

    std::vector<float> valout(N, 0.0);
    
    {
        // Get the buffer size from the STL container:
        const size_t valbytes = vala.size() * sizeof(decltype(vala)::value_type);

        // Allocate and initialize device buffers:
        float* d_a = nullptr;
        if(hipMalloc(&d_a, valbytes) != hipSuccess)
        {
            throw std::runtime_error("hipMalloc failed");
        }
        if(hipMemcpy(d_a, vala.data(), valbytes, hipMemcpyHostToDevice) != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed");
        }
        float* d_b = nullptr;
        if(hipMalloc(&d_b, valbytes) != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed");
        }
        if(hipMemcpy(d_b, valb.data(), valbytes, hipMemcpyHostToDevice) != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed");
        }

        // Load the kernel        
        hipModule_t module;
        hipError_t err = hipModuleLoad(&module, "client_lib_bundle_tmp.cpp.o");
        if(err != hipSuccess) {
            std::cerr << "hipModuleLoad failed: " << hipGetErrorString(err) << "\n";
            return -1;
        }

        // Launch the kernel
        int blockSize = 8;
        const int gridSize    = ceildiv(N, blockSize);
        std::cout << "blockSize: " << blockSize << "\n";
        std::cout << "gridSize: " << gridSize << "\n";
        vector_add<<<dim3(gridSize), dim3(blockSize)>>>(d_a, d_b, N);

        auto ret = hipDeviceSynchronize();
        if(hipGetLastError() != hipSuccess)
        {
            throw std::runtime_error("kernel execution failed");
        }

        // Verify the result:
        if(hipMemcpy(valout.data(), d_a, valbytes, hipMemcpyDeviceToHost) != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed");
        }
        
        // Release device memory
        hipModuleUnload(module);
        if(hipFree(d_a) != hipSuccess)
        {
            throw std::runtime_error("hipFree failed");
        }
        if(hipFree(d_b) != hipSuccess)
        {
            throw std::runtime_error("hipFree failed");
        }
    }

    float maxerr = 0.0;
    for(int i = 0; i < valout.size(); ++i) {
        float diff = std::abs(vala[i] + valb[i] - valout[i]);
        if(diff > maxerr)
            maxerr = diff;
    }
    std::cout << "max error: " << maxerr << "\n";

    return 0;
}
