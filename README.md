# Simple Code Object Load

The objective of this exercise is the load a code object using the HIP runtime, and execute it on the kernel. This is different than the most basic application whereby both the device code and host code are contained in the same HIP file. Instead, the device code should be compiled separately from the host code, which should launch the kernel with a call to `hipModuleLoad()`.

Steps:

1. Write a basic kernel that adds two numbers.
2. Compile it into a code object using amdclang++.
3. Write a host client that uses `hipModuleLoad()` to load the kernel in the code object and launch it on the GPU.
4. Compile and run the host client.