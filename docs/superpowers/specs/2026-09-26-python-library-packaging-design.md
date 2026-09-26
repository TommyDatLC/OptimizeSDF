# Design Specification: Python Library Packaging for OptimizeSDF

**Date**: 2026-09-26  
**Status**: Approved  
**Topic**: Packaging NVIDIA OptiX Shape Diameter Function (SDF) C++ engine into a lightweight Python library with a single entry-point function.

---

## 1. Objectives & Scope

### 1.1 Goals
- Package the existing high-performance C++/CUDA/OptiX SDF calculation engine into a pip-installable Python package named `optimize_sdf`.
- Provide a single, clean user-facing Python function:
  ```python
  compute_sdf(mesh, usePostProcessing: bool = True, rays: int = 64, coneAngle: float = 150.0) -> np.ndarray
  ```
- Support flexible 3D input: either a path to a `.obj` file (`str` / `pathlib.Path`) OR raw geometry arrays `(vertices, faces)` as NumPy `ndarray`.
- Honor the `usePostProcessing: bool` flag:
  - `True`: Returns normalized and smoothed SDF values (min-max logarithmic scaling $\alpha=4.0$ + 3 iterations of anisotropic bilateral smoothing on GPU CSR adjacency graph).
  - `False`: Returns raw physical travel distance values directly aggregated from the 64 Hammersley ray tracing intersections, skipping normalization and smoothing entirely.
- Ensure maximum Python portability (Python 3.10–3.13+) without CPython compiler ABI lock-in by using a C-ABI DLL (`optimize_sdf.dll`) + `ctypes` / `numpy.ctypeslib` wrapper.

---

## 2. System Architecture

```
                    +--------------------------------------------+
                    |                User Code                   |
                    |       import optimize_sdf                  |
                    |       sdf = optimize_sdf.compute_sdf(...)  |
                    +---------------------+----------------------+
                                          |
                                          v
+---------------------------------------------------------------------------------+
|                              python/optimize_sdf/                               |
|                                                                                 |
|   +-------------------------------------------------------------------------+   |
|   | __init__.py                                                             |   |
|   | Exposes single public function: compute_sdf(...)                        |   |
|   +------------------------------------+------------------------------------+   |
|                                        |                                        |
|   +------------------------------------v------------------------------------+   |
|   | core.py                                                                 |   |
|   | - Input Normalization: obj path parser OR numpy validation              |   |
|   | - Memory Alignment: contiguous float32 vertices, uint32 faces           |   |
|   | - ctypes C-ABI Bridge: loads optimize_sdf.dll and invokes C function    |   |
|   +------------------------------------+------------------------------------+   |
|                                        |                                        |
|   +------------------------------------v------------------------------------+   |
|   | lib/ (DLL & Shaders directory)                                          |   |
|   | - optimize_sdf.dll (Built via CMake MSVC)                               |   |
|   | - SDFOptix.ptx (Compiled PTX ray tracing shader)                        |   |
|   +------------------------------------+------------------------------------+   |
+----------------------------------------|----------------------------------------+
                                         | C-ABI call (Zero-Copy Pointers)
                                         v
+---------------------------------------------------------------------------------+
|                        optimize_sdf.dll (C++ / CUDA / OptiX)                    |
|                                                                                 |
|   +-------------------------------------------------------------------------+   |
|   | py_export.cu (extern "C" __declspec(dllexport) ComputeSDF_API)          |   |
|   +------------------------------------+------------------------------------+   |
|                                        |                                        |
|   +------------------------------------v------------------------------------+   |
|   | OptiX Ray Tracing Engine (OptixRunner.cuh & interface.cu)               |   |
|   | - BVH Acceleration Structure (GAS)                                      |   |
|   | - 64-Ray Hammersley Tangent Cone Ray Tracing (RT Cores)                 |   |
|   | - Angle-Weighted Raw Distance Aggregation                               |   |
|   +------------------------------------+------------------------------------+   |
|                                        |                                        |
|                         [usePostProcessing == true?]                            |
|                        /                            \                           |
|                    Yes/                              \No                        |
|                      v                                v                         |
|   +-----------------------------------+   +---------------------------------+   |
|   | GPU Post-Processing               |   | Copy Raw Distances Directly     |   |
|   | - Min-Max & Log Scaling           |   | into out_sdf                    |   |
|   | - CUB RadixSort CSR Graph         |   +---------------------------------+   |
|   | - 3x Anisotropic Bilateral Filter |                                         |
|   +-----------------------------------+                                         |
+---------------------------------------------------------------------------------+
```

---

## 3. Detailed Component Design

### 3.1 C-ABI Export (`src/Optix/py_export.cu`)
A dedicated export source file `py_export.cu` provides a clean C linkage function:

```cpp
extern "C" __declspec(dllexport) int ComputeSDF_API(
    const float* h_vertices,      // Flattened (V x 3) float32 coordinates on host
    int num_vertices,
    const unsigned int* h_faces,  // Flattened (F x 3) uint32 vertex indices on host
    int num_faces,
    bool use_post_processing,     // True: log-norm + bilateral smoothing; False: raw distance
    int rays_per_point,           // Default 64
    float cone_angle_deg,         // Default 150.0
    const char* ptx_directory,    // Directory containing SDFOptix.ptx
    float* h_out_sdf              // Caller-allocated buffer of size (V) float32
);
```

#### Memory & Execution Flow:
1. **Host Matrix Setup**: Wraps incoming contiguous pointers `h_vertices` and `h_faces` into `Matrix<float>` and `Matrix<unsigned int>` and uploads to device memory via existing `CopyToDevice()`.
2. **OptiX State**: Initializes OptiX global state using `SDFOptix.ptx` found in `ptx_directory` (or package directory).
3. **Ray Tracing**: Executes `RunOptixConeRayCasting()` on independent CUDA streams (`streamNorm` & `streamBVH`) to obtain device pointer `d_rawSDF`.
4. **Conditional Post-Processing**:
   - If `use_post_processing == true`:
     - Builds CSR graph via CUB radix sort on `streamCSR`.
     - Computes min-max bounding box and log-normalizes on `streamNorm`.
     - Executes 3 iterations of `AnisotropicSmoothingKernel` ping-pong buffering.
     - Copies smoothed result back to `h_out_sdf`.
   - If `use_post_processing == false`:
     - Copies `d_rawSDF` directly to `h_out_sdf` via `cudaMemcpy(..., cudaMemcpyDeviceToHost)`.
5. **Cleanup**: Frees all allocated GPU buffers and temporary buffers before returning code `0` (Success).

### 3.2 Python Package Structure (`python/optimize_sdf/`)

```
python/
├── pyproject.toml
├── setup.py
└── optimize_sdf/
    ├── __init__.py
    ├── core.py
    └── lib/
        ├── optimize_sdf.dll
        └── SDFOptix.ptx
```

#### `optimize_sdf/core.py` Logic:
1. **Dynamic Library Loading**:
   - Locates `optimize_sdf.dll` relative to `__file__`.
   - Adds library directory via `os.add_dll_directory` (required on Windows Python 3.8+).
   - Binds `ComputeSDF_API` with exact `ctypes` argument types (`ctypes.POINTER(ctypes.c_float)`, `ctypes.c_int`, etc.).
2. **Input Handling in `compute_sdf()`**:
   - **Case A (`str` or `pathlib.Path`)**:
     - Fast OBJ parsing into vertices `(V, 3)` `float32` and faces `(F, 3)` `uint32`.
   - **Case B (`tuple[np.ndarray, np.ndarray]`)**:
     - Ensures contiguous memory layout via `np.ascontiguousarray(..., dtype=np.float32)` and `np.ascontiguousarray(..., dtype=np.uint32)`.
3. **Buffer Allocation & Execution**:
   - Allocates `out_sdf = np.empty(num_vertices, dtype=np.float32)`.
   - Invokes `ComputeSDF_API`.
   - Checks return status code; raises `RuntimeError` if non-zero.
   - Returns `out_sdf`.

---

## 4. Build System & Integration

1. **`CMakeLists.txt` Target**:
   ```cmake
   add_library(optimize_sdf SHARED
       src/Optix/py_export.cu
       Core/Matrix.cu
       Core/MatrixMemoryManager.cu
       Core/MathHelper.cu
       Core/Model.cu
       Core/ModelHelper.cu
   )
   target_link_libraries(optimize_sdf PRIVATE cublas polyscope OpenCL::OpenCL)
   ```
2. **Post-Build Step**:
   - Copies `build/Release/optimize_sdf.dll` and `build/OptixShaders.dir/Release/SDFOptix.ptx` directly into `python/optimize_sdf/lib/`.
3. **Packaging (`setup.py` / `pyproject.toml`)**:
   - Configures `package_data={"optimize_sdf": ["lib/*.dll", "lib/*.ptx"]}`.
   - Enables standard `pip install -e python` installation.

---

## 5. Verification & Acceptance Criteria

1. **Test 1: OBJ File Input (`usePostProcessing=True`)**:
   ```python
   import optimize_sdf
   sdf = optimize_sdf.compute_sdf("Model/HighPeeling_Coil.obj", usePostProcessing=True)
   assert len(sdf) == 18000
   assert 0.0 <= sdf.min() and sdf.max() <= 1.0
   ```
2. **Test 2: OBJ File Input (`usePostProcessing=False`)**:
   ```python
   raw_sdf = optimize_sdf.compute_sdf("Model/HighPeeling_Coil.obj", usePostProcessing=False)
   assert len(raw_sdf) == 18000
   # Raw physical distance scale matches expected range (~0 to ~11.5)
   assert raw_sdf.max() > 5.0
   ```
3. **Test 3: NumPy Array Input (`vertices, faces`)**:
   ```python
   import pyvista as pv
   m = pv.read("Model/HighPeeling_Coil.obj")
   verts = m.points.astype(np.float32)
   faces = m.faces.reshape(-1, 4)[:, 1:4].astype(np.uint32)
   sdf = optimize_sdf.compute_sdf((verts, faces), usePostProcessing=True)
   assert len(sdf) == len(verts)
   ```
4. **Test 4: Backward Compatibility**:
   - `OptimizeSDF.exe` remains functional for standalone benchmark and preview workflows.
