# Exercise: Loading and Executing a Separately Compiled HIP Code Object

**Objective:** This exercise will guide you through loading a pre-compiled HIP device code object using the HIP runtime and executing a kernel from it. This contrasts with the common approach where host and device code reside in the same source file and are compiled together. You'll learn how to compile device code into a standalone code object and then use hipModuleLoad() in your host application to execute it.

**What You'll Learn:**

- How to compile HIP device code separately into a relocatable code object (.co or .hsaco).
- How to use `hipModuleLoad()` to load this code object at runtime.
- How to retrieve a kernel function from the loaded module using `hipModuleGetFunction()`.
- How to launch the retrieved kernel.
- Basic CMake setup for HIP host applications.

**Why is this useful?**

- Modularity: Decouples device code from host code, allowing them to be developed and updated independently.
- Dynamic Loading: Kernels can be chosen or updated at runtime without recompiling the host application.
- IP Protection: Device code can be distributed as compiled binaries.
- Reduced Host Compilation Time: If only device code changes, only it needs recompilation.

**Prerequisites:**

- ROCm installed (e.g., in `/opt/rocm`).
- HIP supported compiler (e.g. `amdclang++` or `hipcc`).
- Basic understanding of C++ and HIP programming.
- CMake 3.25+ installed.

**Steps:**

1. Write a basic kernel that adds two numbers, call it `vector_add`.
2. Compile the kernel into a code object. You may start by using `hipcc` but to be considered complete, you should be able to conduct the exercise with `amdclang++` alone.
3. Verify the symbols in the generated object using `nm`.
4. Write a host client that uses `hipModuleLoad()` to load the kernel in the code object and launch it on the GPU. You will also need other HIP module API functions, which functions to use are left as part of the exercise.
5. Compile and run the host client.

---

## Hints

- HIP runtime functions are accessed through the include `#include <hip/hip_runtime>`.
- It's a good practice to ensure the kernel has C linkage if you're loading it by name to avoid C++ name mangling issues. As such, use `extern "C" __global__ void` to declare your kernel.
- `hipModuleLoad()` requires an un-linked relocatable code object compiled from device code. These objects are extended with .co/.hsaco and differ from a static (.a) or shared (.so) library.
- Key flags for `amdclang++` are:
    - `--offload-device-only`: Tells the compiler to only compile the device code and ignore host code.
    - `-x hip`: Specifies the input language is HIP.
    - `--offload-arch=<target-gpu>`: Specifies the target GPU architecture (e.g. gfx1201). You can determine the architecture with `rocminfo` or `hipinfo`.
    - `-c`: Compile and assemble but skip the link step, to produce a raw object file.
    - `-o <output-file.co>`: Specify the output file name.
- You can inspect the generated code object to ensure your kernel symbol is present: `nm -C kernel.co | grep vector_add`
- Create a C++ file for your host code (e.g., main.cpp). This application will:
    1. Load the `kernel.co` file using `hipModuleLoad()`.
    2. Get a function handle to the vector_add kernel using `hipModuleGetFunction()`.
    3. Allocate memory on the GPU.
    4. Copy data from host to GPU.
    5. Launch the kernel using `hipModuleLaunchKernel()`.
    6. Copy results back from GPU to host.
    7. Verify results and clean up.


## Troubleshooting & Key Insights 

- CMake can't find HIP (hip-config.cmake):
    - This usually means ROCm is installed in a non-standard location (like /opt/rocm).
    - Solution: Provide CMAKE_PREFIX_PATH to CMake:
    ```
    cmake -D CMAKE_PREFIX_PATH=/opt/rocm ..  
    ```

- `hipModuleLoad()` fails with "no kernel image is available for execution on the device":
    - This is a common error if the code object (.co file) was not compiled correctly for your specific GPU or if it wasn't compiled as a device-only object.
    - Solutions:
        - Correct GPU Architecture: Ensure --offload-arch=gfxXXXX in your amdclang++ command (Step 2) matches your GPU. Use rocminfo or hipinfo to find your GPU architecture (e.g., gfx90a, gfx1030, gfx1100).
        - Device-Only Compilation: Crucially, use the --offload-device-only flag when compiling your kernel.hip into kernel.co. This ensures it's a pure device code object suitable for hipModuleLoad.
        ```
        /opt/rocm/bin/amdclang++ --offload-device-only -x hip --offload-arch=gfxXXXX -c kernel.hip -o kernel.co  
        ```
        - ROCm Drivers/Installation: A very outdated or improperly installed ROCm stack (including kernel drivers) could cause issues. Ensure your ROCm installation is healthy.

- Understanding Code Object vs. Libraries (.a, .so):
    - When you compile device code normally and link it into a host application (the "bundled" approach), the build system handles embedding the device code.
    - For `hipModuleLoad()`, you don't want a static (.a) or shared (.so) library in the traditional sense. You need the "raw" compiled device code, which amdclang++ with `--offload-device-only -c` provides as a relocatable object file (often named .o by default, but .co or .hsaco are common conventions for HIP code objects).

- Inspecting Code Objects (nm):
    - The `nm -C <object_file>` command is very useful for listing symbols in an object file. The -C demangles C++ names. This helps verify that your kernel (e.g., `vector_add`) is indeed present in the kernel.co file and helps debug issues with `hipModuleGetFunction()` if it can't find the kernel by name.

## Helpful Commands

```bash
$ cmake -B build -S . -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=amdclang++
$ cmake --build build
```

```bash
$ /opt/rocm/bin/amdclang++                                   \
    -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1                \
    -isystem /opt/rocm/include -x hip --offload-arch=gfx1201 \
    -c client_lib_bundle.cpp                                 \
    -o client_lib_bundle_tmp.o 
```

```bash
$ nm -C client_lib_bundle_YEEHAH.cpp.o | grep vector_add
0000000000000000 R vector_add(float*, float const*, int)
0000000000000000 T __device_stub__vector_add(float*, float const*, int)
```

```bash
$ /opt/rocm/bin/amdclang++                                                         \
    -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -D__HIP_HCC_COMPAT_MODE__=1          \
    -isystem /opt/rocm/include --offload-device-only -x hip --offload-arch=gfx1201 \
    -c lib.cpp                                                                     \
    -o lib.hsaco
```