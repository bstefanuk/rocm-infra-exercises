# Custom Build of hipBLASLt

Stage: 🌱 Seed

---

Using the library logic files in [assets](../assets/), build hipblaslt, and create a hipblaslt-bench command to execute specific kernels from the set.

> [!TIP]
> Use set the `TENSILE_DB` environment variable to configure the output logging; see [Debug.cpp](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/tensilelite/src/Debug.cpp)