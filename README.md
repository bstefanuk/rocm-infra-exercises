# Simple Code Object Load

The objective of this exercise is the load a code object using the HIP runtime, and execute it on the kernel. This is different than the most basic application whereby both the device code and host code are contained in the same HIP file. Instead, the device code should be compiled separately from the host code, which should launch the kernel with a call to `hipModuleLoad()`.

Steps:

1. Write a basic kernel that adds two numbers.
2. Compile it into a code object using amdclang++.
3. Write a host client that uses `hipModuleLoad()` to load the kernel in the code object and launch it on the GPU.
4. Compile and run the host client.


# My Quest
1. Okay, I've about set everything up. See `2291b3c9: first commit`. However, I can't find the HIP runtime with CMake, that is `hip-config.cmake`. This is a new machine, so it's not unexpected. Let's see if we can hunt that down. Okay, turns out I could `find` it in /opt/rocm already. Weird it didn't catch it the first time, but now it did... just the way of computers sometimes.
2. I've built the client library bundle, that shows a kernel compiled alongside a main function that executes it. This is basically straight out of the HIP training playbook.
3. I've broken out the device-side and host-side functions into a lib and client, respectively, and add them the the CMakeLists.txt. However, when I build, the library is compiled as a static .a library, and since I didn't link against it in CMake, the host code doesn't know anything about the `vector_add` function; it can't find it because it's in the device library.
4. This is the key, I need to find a way to load the symbols in the compiled device library into the host-side client so that I can launch the kernel. I believe this is what `hipModuleLoad()` is for. Here's the thing, I don't want my library compiled as an .so or .a, I just want the raw object, pre-linked. I can do this with `amdclang++` directly, but what exactly is the command? I checked the CMake build with `--verbose` to see how "client_lib_bundle" is compiled, and came up with this command to run outside of the cmake context.

```bash
/opt/rocm/bin/amdclang++ -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1 -isystem /opt/rocm/include -x hip --offload-arch=gfx1201 -MD -MT client
_lib_bundle_tmp.cpp.o -MF client_lib_bundle_tmp.d -o client_lib_bundle_tmp.cpp.o -c /home/bstefanu/dev/practice/simple-co-load/client_lib_bundle.cpp
```

This gives me the .o file I need, and if I run `file`: `client_lib_bundle_tmp.cpp.o: ELF 64-bit LSB relocatable, x86-64, version 1 (SYSV), not stripped`. Note that here the .o extension could just as well be .co, it won't change the behaviour of the file. Let's continue to inspect this file for a second. Knowing that the kernel name is `vector_add`, let's dump the symbols in the object file and search:

```
~/simple-co-load$ nm -C client_lib_bundle_YEEHAH.cpp.o | grep vector_add
0000000000000000 R vector_add(float*, float const*, int)
0000000000000000 T __device_stub__vector_add(float*, float const*, int)
```
