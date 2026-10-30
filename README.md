# gpuTT - GPU Tensor Transpose library

gpuTT is a high performance tensor transpose library for NVIDIA and AMD GPUs. It works with Kepler (SM 3.0) and above or gfx900 and above GPUs.

This code implements the following tensor transposing methods: `Trivial`, `Tiled`, `TiledCopy`, `Packed`, and `PackedSplit`. The fastest method is chosen for the given problem, either by measuring the performance, or by using a heuristic (the default).

## Building

Prerequisites:

 * CMake, 3.28+
 * C++ compiler with C++17 compatibility
 * CUDA toolkit for NVIDIA, or HIP/ROCm for AMD
 * GPU architecture supported by the selected toolkit

To compile the library, tests, benchmarks, and example for a GH200/Hopper GPU:

```sh
cmake -S . -B build -DGPUTT_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DGPUTT_BUILD_PYTHON=OFF
cmake --build build --parallel
cmake --install build --prefix /path/to/install
```

For an AMD MI250X, configure with `-DGPUTT_BACKEND=HIP -DCMAKE_HIP_ARCHITECTURES=gfx90a` instead. CUDA uses native NVCC; HIP uses the AMD compiler. The target architectures must be set explicitly. If `GPUTT_BACKEND` is omitted, it is inferred from the architecture variable; set the backend explicitly when both architecture variables are present.

`GPUTT_BUILD_TESTS`, `GPUTT_BUILD_BENCHMARKS`, `GPUTT_BUILD_EXAMPLES`, and `GPUTT_BUILD_PYTHON` default to `ON` for a standalone build and `OFF` when used as a subproject. Python bindings require the pybind11 submodule (`git submodule update --init --recursive`) and Python development files. `ENABLE_NVTOOLS` requires a CUDA toolkit providing the legacy nvToolsExt library (CUDA 12 or earlier).

Both an installed package and `add_subdirectory` provide the `gputt::gputt` target, including the selected GPU runtime dependency. Link it to your application:

```cmake
find_package(gputt CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE gputt::gputt)
```

For a source build, replace `find_package` with `add_subdirectory(path/to/gputt)`, setting the backend and architecture first. The installed public header is `include/gputt/gputt.h`, usable as either `<gputt.h>` or `<gputt/gputt.h>` through the CMake target, and the static library is installed under the configured library directory (`lib` or `lib64`). The package records its backend as `gputt_BACKEND`.

## Running tests and benchmarks

Tests and benchmark executables are in the build directory and they can be run without any options. Run the correctness suite with `ctest --test-dir build --output-on-failure`.
Options to the test executable lets you choose the device ID on which to run:

```
gputt_test [options]
Options:
-device gpuid : use GPU with ID gpuid
```

For the benchmark executable, we have an additional option that lets you run the benchmarks using
plans that are chosen optimally by measuring the performance of every possible implementation and
choosing the best one.

```
gputt_bench [options]
Options:
-device gpuid : use GPU with ID gpuid
-measure      : use gputtPlanMeasure (default is gputtPlan)
```

The following modern GPUs have been tested and passed the correctness test suite successfully:

* RTX 3080 LHR (GA104, CC 8.6)
* Radeon RX Vega 56 (gfx900)
* Radeon RX 7900 XTX (gfx1100)

## Performance

gpuTT was designed with performance as the main goal. Here are performance benchmarks for a random set of tensors with 200M `double` elements with ranks 2 to 7. The benchmarks were run with the measurement flag on `./gputt_bench -measure -bench 3`.

![k20x](doc/k20x_bench.png)

<!-- ![k40m](doc/bw_k40m_july1_2016.png)
 -->

<!-- ![titanx](doc/bw_titanx.png)
 -->

## Usage

gpuTT uses a "plan structure" similar to FFTW and cuFFT libraries, where the
user first creates a plan for the transpose and then executes that plan.
Here is an example code (please see a fully working example code in `src/example/example.cu`).

```c++
#include <gputt.h>

//
// Error checking wrapper for gputt
//
#define gputtCheck(stmt) do {                                 \
  gputtResult err = stmt;                            \
  if (err != GPUTT_SUCCESS) {                          \
    fprintf(stderr, "%s in file %s, function %s\n", #stmt,__FILE__,__FUNCTION__); \
    exit(1); \
  }                                                  \
} while(0)

int main() {

  // Four dimensional tensor
  // Transpose (31, 549, 2, 3) -> (3, 31, 2, 549)
  int dim[4] = {31, 549, 2, 3};
  int permutation[4] = {3, 0, 2, 1};

  .... input and output data is setup here ...
  // double* idata : size product(dim)
  // double* odata : size product(dim)

  // Option 1: Create plan on NULL stream and choose implementation based on heuristics
  gputtHandle plan;
  gputtCheck(gputtPlan(&plan, 4, dim, permutation, gputtDataTypeFloat64, nullptr));

  // Option 2: Create plan on NULL stream and choose implementation based on performance measurements
  // gputtCheck(gputtPlanMeasure(&plan, 4, dim, permutation, gputtDataTypeFloat64, nullptr, idata, odata));

  // Execute plan
  gputtCheck(gputtExecute(plan, idata, odata));

  ... do stuff with your output and deallocate data ...

  // Destroy plan
  gputtCheck(gputtDestroy(plan));

  return 0;
}
```

Input (idata) and output (odata) data are both in GPU memory and must point to different
memory areas for correct operation. That is, gpuTT only currently supports out-of-place
transposes. Note that using Option 2 to create the plan can take up some time especially
for high-rank tensors.

## gpuTT API

```c++
//
// Create plan
//
// Parameters
// handle            = Returned handle to gpuTT plan
// rank              = Rank of the tensor
// dim[rank]         = Dimensions of the tensor
// permutation[rank] = Transpose permutation
// dtype             = Element type, e.g. gputtDataTypeFloat64
// stream            = CUDA or HIP stream cast to gputtStream (nullptr for the default stream)
// method            = Transpose method (heuristic selection by default)
//
// Returns
// Success/unsuccess code
// 
gputtResult gputtPlan(gputtHandle* handle, int rank, const int* dim, const int* permutation,
  gputtDataType dtype, gputtStream stream,
  gputtTransposeMethod method = gputtTransposeMethodUnknown);

//
// Create plan and choose implementation by measuring performance
//
// Parameters
// handle            = Returned handle to gpuTT plan
// rank              = Rank of the tensor
// dim[rank]         = Dimensions of the tensor
// permutation[rank] = Transpose permutation
// dtype             = Element type, e.g. gputtDataTypeFloat64
// stream            = CUDA or HIP stream cast to gputtStream (nullptr for the default stream)
// idata             = Input data size product(dim)
// odata             = Output data size product(dim)
//
// Returns
// Success/unsuccess code
// 
gputtResult gputtPlanMeasure(gputtHandle* handle, int rank, const int* dim, const int* permutation,
  gputtDataType dtype, gputtStream stream, const void* idata, void* odata,
  const void* alpha = nullptr, const void* beta = nullptr);
  
//
// Destroy plan
//
// Parameters
// handle            = Handle to the gpuTT plan
// 
// Returns
// Success/unsuccess code
//
gputtResult gputtDestroy(gputtHandle handle);

//
// Execute plan out-of-place
//
// Parameters
// handle            = Returned handle to gpuTT plan
// idata             = Input data size product(dim)
// odata             = Output data size product(dim)
// alpha             = scaling-factor for input
// beta              = scaling-factor for output
// 
// Returns
// Success/unsuccess code
//
gputtResult gputtExecute(gputtHandle handle, const void* idata, void* odata,
  const void* alpha = nullptr, const void* beta = nullptr);
```

## Known Bugs

 * Benchmarks sometime fail due to the stupid algorithm I have now to create
 random tensors with fixed volume.

## TODO

 * Make "tiled" method work with sets of ranks (where ranks in `M_m` and `M_k` remain in same order)

## Troubleshooting

1. If CMake cannot find a HIP compiler, specify it explicitly with `-DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++` and inspect the compiler test error. A compiler installation can be found but still fail its initial test, for example [this issue](https://github.com/RadeonOpenCompute/ROCm/issues/1843).

2. The following error could be fixed by `sudo apt-get install libstdc++-12-dev`, as suggested in this issue:

```
  The HIP compiler

    "/opt/rocm/llvm/bin/clang++"

  is not able to compile a simple test program.

    /opt/rocm-5.5.0/llvm/lib/clang/16.0.0/include/__clang_hip_runtime_wrapper.h:50:10: fatal error: 'cmath' file not found
    #include <cmath>
             ^~~~~~~
```

## Credits

In memory of Antti-Pekka Hynninen.

