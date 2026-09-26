# Python Library Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the NVIDIA OptiX Shape Diameter Function (SDF) C++ engine into a pip-installable Python library `optimize_sdf` with a single entry-point function `compute_sdf(mesh, usePostProcessing=True)`.

**Architecture:** A C-ABI shared library (`optimize_sdf.dll`) built via CMake/MSVC that exposes `ComputeSDF_API` with raw pointer interfaces, wrapped by a zero-dependency Python package (`python/optimize_sdf/`) via `ctypes` and NumPy for universal Python 3.10–3.13+ compatibility.

**Tech Stack:** C++20, CUDA 12+, NVIDIA OptiX 7.6+, CMake 3.24+, Python 3.10+, NumPy, ctypes.

## Global Constraints

- Library must expose a single user-facing function: `compute_sdf(mesh, usePostProcessing: bool = True, rays: int = 64, coneAngle: float = 150.0) -> np.ndarray`.
- `mesh` argument must accept either a filepath string (`.obj`) or a tuple of NumPy arrays `(vertices, faces)`.
- When `usePostProcessing=True`, return normalized and 3x smoothed SDF values in $[0, 1]$.
- When `usePostProcessing=False`, return physical raw ray distances directly without normalization or smoothing.
- No CPython ABI lock-in: C++ must export a pure C-ABI DLL (`extern "C" __declspec(dllexport)`).

---

### Task 1: C-ABI Export Source and CMake Shared Library Target

**Files:**
- Create: `src/Optix/py_export.cu`
- Modify: `CMakeLists.txt`
- Target Artifact: `build/Release/optimize_sdf.dll`

**Interfaces:**
- Produces: `ComputeSDF_API(const float*, int, const unsigned int*, int, bool, int, float, const char*, float*)`

- [ ] **Step 1: Create `src/Optix/py_export.cu`**

```cpp
#include <optix.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "../../Core/Matrix.cuh"
#include "../../Core/Helper.hpp"
#include "OptixHostUtils.cuh"
#include "SDFKernels.cuh"
#include "OptixRunner.cuh"

struct PyOptixState {
    OptixDeviceContext context;
    OptixModule module;
    OptixProgramGroup raygenProgGroup;
    OptixProgramGroup missProgGroup;
    OptixProgramGroup hitProgGroup;
    OptixPipeline pipeline;
    CUdeviceptr d_rgSbt;
    CUdeviceptr d_msSbt;
    CUdeviceptr d_hgSbt;
    OptixShaderBindingTable sbt;
    bool initialized = false;
};

static PyOptixState g_pyOptixState;

extern "C" __declspec(dllexport) int ComputeSDF_API(
    const float* h_vertices,
    int num_vertices,
    const unsigned int* h_faces,
    int num_faces,
    bool use_post_processing,
    int rays_per_point,
    float cone_angle_deg,
    const char* ptx_path,
    float* h_out_sdf
) {
    if (!h_vertices || num_vertices <= 0 || !h_faces || num_faces <= 0 || !h_out_sdf) {
        return -1;
    }

    try {
        if (!g_pyOptixState.initialized) {
            optixInit();
            std::string ptxCode = readFile(ptx_path ? ptx_path : "SDFOptix.ptx");
            g_pyOptixState.context = OptixRunner::InitContext();
            g_pyOptixState.module = nullptr;
            g_pyOptixState.pipeline = OptixRunner::CreatePipeline(
                g_pyOptixState.context, ptxCode,
                g_pyOptixState.raygenProgGroup, g_pyOptixState.missProgGroup,
                g_pyOptixState.hitProgGroup, g_pyOptixState.module
            );
            g_pyOptixState.sbt = OptixRunner::BuildSBT(
                g_pyOptixState.raygenProgGroup, g_pyOptixState.missProgGroup,
                g_pyOptixState.hitProgGroup, g_pyOptixState.d_rgSbt,
                g_pyOptixState.d_msSbt, g_pyOptixState.d_hgSbt
            );
            g_pyOptixState.initialized = true;
        }

        Matrix<float> vMat(num_vertices, 3, nullptr);
        for (int i = 0; i < num_vertices; ++i) {
            vMat.SetHost(i, 0, h_vertices[i * 3 + 0]);
            vMat.SetHost(i, 1, h_vertices[i * 3 + 1]);
            vMat.SetHost(i, 2, h_vertices[i * 3 + 2]);
        }
        vMat.CopyToDevice();

        Matrix<unsigned int> iMat(num_faces, 3, nullptr);
        for (int i = 0; i < num_faces; ++i) {
            iMat.SetHost(i, 0, h_faces[i * 3 + 0]);
            iMat.SetHost(i, 1, h_faces[i * 3 + 1]);
            iMat.SetHost(i, 2, h_faces[i * 3 + 2]);
        }
        iMat.CopyToDevice();

        Matrix<float> nMat(num_vertices, 3, nullptr);
        Matrix<float> fnMat(num_faces, 3, nullptr);
        
        cudaStream_t streamNormal, streamBVH;
        CUDA_CHECK(cudaStreamCreate(&streamNormal));
        CUDA_CHECK(cudaStreamCreate(&streamBVH));

        GPUNormalCaculation(
            (const uint3*)iMat.getDevicePtr(),
            (const float3*)vMat.getDevicePtr(),
            (float3*)fnMat.getDevicePtr(),
            (float3*)nMat.getDevicePtr(),
            numFaces, numVertices, streamNormal
        );
        GPUNormalizeVertexNormal((float3*)nMat.getDevicePtr(), numVertices, streamNormal);

        float coneAngleRadian = cone_angle_deg * (3.14159265f / 180.0f);
        CUdeviceptr d_tempBuffer, d_gasOutputBuffer;
        OptixTraversableHandle bvhHandle = OptixRunner::BuildBVH(
            g_pyOptixState.context,
            (CUdeviceptr)vMat.getDevicePtr(),
            (CUdeviceptr)iMat.getDevicePtr(),
            num_vertices, num_faces,
            d_tempBuffer, d_gasOutputBuffer, streamBVH
        );

        CUDA_CHECK(cudaStreamSynchronize(streamBVH));
        CUDA_CHECK(cudaStreamSynchronize(streamNormal));

        float* d_rawSDF = OptixRunner::LaunchOptixAndCUB(
            num_vertices, rays_per_point, coneAngleRadian,
            (CUdeviceptr)vMat.getDevicePtr(), (float3*)nMat.getDevicePtr(),
            bvhHandle, g_pyOptixState.pipeline, g_pyOptixState.sbt
        );

        CUDA_CHECK(cudaFree((void*)d_tempBuffer));
        CUDA_CHECK(cudaFree((void*)d_gasOutputBuffer));
        CUDA_CHECK(cudaStreamDestroy(streamNormal));
        CUDA_CHECK(cudaStreamDestroy(streamBVH));

        if (!use_post_processing) {
            CUDA_CHECK(cudaMemcpy(h_out_sdf, d_rawSDF, num_vertices * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_rawSDF));
            return 0;
        }

        // Post-Processing
        cudaStream_t streamCSR, streamNorm;
        CUDA_CHECK(cudaStreamCreate(&streamCSR));
        CUDA_CHECK(cudaStreamCreate(&streamNorm));

        int numEdges = num_faces * 6;
        uint64_t *d_edges, *d_sortedEdges, *d_uniqueEdges;
        int *d_numUniqueEdges, *d_nbrOffsets, *d_nbrLists;
        CUDA_CHECK(cudaMalloc(&d_edges, numEdges * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_sortedEdges, numEdges * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_uniqueEdges, numEdges * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_numUniqueEdges, sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_nbrOffsets, (num_vertices + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_nbrLists, numEdges * sizeof(int)));

        float* d_minMaxBox;
        CUDA_CHECK(cudaMalloc((void**)&d_minMaxBox, 6 * sizeof(float)));
        float initBox[6] = {1e15f, -1e15f, 1e15f, -1e15f, 1e15f, -1e15f};
        CUDA_CHECK(cudaMemcpyAsync(d_minMaxBox, initBox, 6 * sizeof(float), cudaMemcpyHostToDevice, streamNorm));

        float* d_sdfBuf1 = d_rawSDF;
        float* d_sdfBuf2;
        CUDA_CHECK(cudaMalloc((void**)&d_sdfBuf2, num_vertices * sizeof(float)));

        float* d_minMaxSDF;
        CUDA_CHECK(cudaMalloc((void**)&d_minMaxSDF, 2 * sizeof(float)));
        float initSDF[2] = {1e15f, -1e15f};
        CUDA_CHECK(cudaMemcpyAsync(d_minMaxSDF, initSDF, 2 * sizeof(float), cudaMemcpyHostToDevice, streamNorm));

        int blockSize = 256;
        int gridSizeFaces = (num_faces + blockSize - 1) / blockSize;
        GPUGenerateEdges<<<gridSizeFaces, blockSize, 0, streamCSR>>>((const uint3*)iMat.getDevicePtr(), num_faces, d_edges);

        void *d_temp_sort = nullptr; size_t temp_sort_bytes = 0;
        cub::DeviceRadixSort::SortKeys(d_temp_sort, temp_sort_bytes, d_edges, d_sortedEdges, numEdges, 0, sizeof(uint64_t)*8, streamCSR);
        CUDA_CHECK(cudaMalloc(&d_temp_sort, temp_sort_bytes));
        cub::DeviceRadixSort::SortKeys(d_temp_sort, temp_sort_bytes, d_edges, d_sortedEdges, numEdges, 0, sizeof(uint64_t)*8, streamCSR);

        void *d_temp_unique = nullptr; size_t temp_unique_bytes = 0;
        cub::DeviceSelect::Unique(d_temp_unique, temp_unique_bytes, d_sortedEdges, d_uniqueEdges, d_numUniqueEdges, numEdges, streamCSR);
        CUDA_CHECK(cudaMalloc(&d_temp_unique, temp_unique_bytes));
        cub::DeviceSelect::Unique(d_temp_unique, temp_unique_bytes, d_sortedEdges, d_uniqueEdges, d_numUniqueEdges, numEdges, streamCSR);

        int numUniqueEdges = 0;
        CUDA_CHECK(cudaMemcpyAsync(&numUniqueEdges, d_numUniqueEdges, sizeof(int), cudaMemcpyDeviceToHost, streamCSR));

        int gridSizeVerts = (num_vertices + blockSize - 1) / blockSize;
        GPUComputeBoundingBox<<<gridSizeVerts, blockSize, 0, streamNorm>>>((const float3*)vMat.getDevicePtr(), num_vertices, d_minMaxBox);
        float h_minMaxBox[6];
        CUDA_CHECK(cudaMemcpyAsync(h_minMaxBox, d_minMaxBox, 6 * sizeof(float), cudaMemcpyDeviceToHost, streamNorm));

        GPUComputeSDFMinMax<<<gridSizeVerts, blockSize, 0, streamNorm>>>(d_sdfBuf1, num_vertices, d_minMaxSDF);
        GPUApplySDFNormalization<<<gridSizeVerts, blockSize, 0, streamNorm>>>(d_sdfBuf1, num_vertices, d_minMaxSDF);

        CUDA_CHECK(cudaStreamSynchronize(streamCSR));
        CUDA_CHECK(cudaStreamSynchronize(streamNorm));

        int gridSizeUnique = (numUniqueEdges + blockSize - 1) / blockSize;
        GPUExtractCSR<<<gridSizeUnique, blockSize, 0, streamCSR>>>(d_uniqueEdges, numUniqueEdges, d_nbrOffsets, d_nbrLists, num_vertices);
        CUDA_CHECK(cudaStreamSynchronize(streamCSR));

        float dx = h_minMaxBox[1] - h_minMaxBox[0];
        float dy = h_minMaxBox[3] - h_minMaxBox[2];
        float dz = h_minMaxBox[5] - h_minMaxBox[4];
        float bboxDiagonal = std::sqrt(dx*dx + dy*dy + dz*dz);

        int numIterations = 3;
        float sigmaSpatial = bboxDiagonal * 0.02f;
        float sigmaRange = 0.1f;
        float3* d_vertices_direct = (float3*)vMat.getDevicePtr();

        for (int iter = 0; iter < numIterations; iter++) {
            float* d_in = (iter % 2 == 0) ? d_sdfBuf1 : d_sdfBuf2;
            float* d_out = (iter % 2 == 0) ? d_sdfBuf2 : d_sdfBuf1;
            AnisotropicSmoothingKernel<<<gridSizeVerts, blockSize>>>(
                d_vertices_direct, d_in, d_out, d_nbrOffsets, d_nbrLists,
                num_vertices, sigmaSpatial, sigmaRange
            );
            cudaDeviceSynchronize();
        }

        float* d_finalOut = (numIterations % 2 == 0) ? d_sdfBuf1 : d_sdfBuf2;
        CUDA_CHECK(cudaMemcpy(h_out_sdf, d_finalOut, num_vertices * sizeof(float), cudaMemcpyDeviceToHost));

        cudaFree(d_edges); cudaFree(d_sortedEdges); cudaFree(d_temp_sort);
        cudaFree(d_uniqueEdges); cudaFree(d_numUniqueEdges); cudaFree(d_temp_unique);
        cudaFree(d_minMaxBox); cudaFree(d_minMaxSDF);
        cudaFree(d_sdfBuf1); cudaFree(d_sdfBuf2);
        cudaFree(d_nbrOffsets); cudaFree(d_nbrLists);
        CUDA_CHECK(cudaStreamDestroy(streamCSR));
        CUDA_CHECK(cudaStreamDestroy(streamNorm));

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ComputeSDF_API Exception]: " << e.what() << std::endl;
        return -2;
    }
}
```

- [ ] **Step 2: Add `optimize_sdf` shared library target in `CMakeLists.txt`**

```cmake
add_library(optimize_sdf SHARED
    src/Optix/py_export.cu
    Core/Matrix.cu
    Core/MatrixMemoryManager.cu
    Core/MathHelper.cu
    Core/ModelHelper.cu
)
target_include_directories(optimize_sdf BEFORE PRIVATE ${CUDAToolkit_INCLUDE_DIRS} OptiX ${CMAKE_CURRENT_SOURCE_DIR}/Core)
target_link_libraries(optimize_sdf PRIVATE cublas)
add_dependencies(optimize_sdf OptixShaders)
```

- [ ] **Step 3: Build the DLL**

Run: `cmake --build build --config Release --target optimize_sdf -- /m:1`  
Expected: `optimize_sdf.vcxproj -> ...\optimize_sdf.dll`

- [ ] **Step 4: Commit**

```bash
git add src/Optix/py_export.cu CMakeLists.txt
git commit -m "feat: add C-ABI export and optimize_sdf shared library target"
```

---

### Task 2: Python Package Implementation (`python/optimize_sdf/`)

**Files:**
- Create: `python/optimize_sdf/__init__.py`
- Create: `python/optimize_sdf/core.py`
- Directory: `python/optimize_sdf/lib/` (copy `optimize_sdf.dll` and `SDFOptix.ptx`)

**Interfaces:**
- Produces: `optimize_sdf.compute_sdf(mesh, usePostProcessing=True, rays=64, coneAngle=150.0) -> np.ndarray`

- [ ] **Step 1: Create `python/optimize_sdf/core.py`**

```python
import os
import ctypes
import numpy as np
from pathlib import Path

LIB_DIR = Path(__file__).parent / "lib"
DLL_PATH = LIB_DIR / "optimize_sdf.dll"
PTX_PATH = LIB_DIR / "SDFOptix.ptx"

_lib = None

def _get_lib():
    global _lib
    if _lib is None:
        if not DLL_PATH.exists():
            raise FileNotFoundError(f"Cannot find optimize_sdf.dll at {DLL_PATH}")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(LIB_DIR.resolve()))
        _lib = ctypes.CDLL(str(DLL_PATH.resolve()))
        _lib.ComputeSDF_API.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint),
            ctypes.c_int,
            ctypes.c_bool,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_float),
        ]
        _lib.ComputeSDF_API.restype = ctypes.c_int
    return _lib

def _read_obj(file_path: str | Path):
    vertices = []
    faces = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()
                f_indices = []
                for p in parts[1:4]:
                    f_indices.append(int(p.split("/")[0]) - 1)
                faces.append(f_indices)
    return np.ascontiguousarray(vertices, dtype=np.float32), np.ascontiguousarray(faces, dtype=np.uint32)

def compute_sdf(
    mesh: str | Path | tuple[np.ndarray, np.ndarray],
    usePostProcessing: bool = True,
    rays: int = 64,
    coneAngle: float = 150.0
) -> np.ndarray:
    """Compute Shape Diameter Function (SDF) using NVIDIA OptiX RT Cores."""
    if isinstance(mesh, (str, Path)):
        verts, faces = _read_obj(mesh)
    elif isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        verts = np.ascontiguousarray(mesh[0], dtype=np.float32)
        faces = np.ascontiguousarray(mesh[1], dtype=np.uint32)
    else:
        raise TypeError("mesh must be a file path string or a tuple of (vertices, faces) numpy arrays")

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices array must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces array must have shape (M, 3)")

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]
    out_sdf = np.empty(num_verts, dtype=np.float32)

    lib = _get_lib()
    ptx_str = str(PTX_PATH.resolve()).encode("utf-8") if PTX_PATH.exists() else None

    v_ptr = verts.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    f_ptr = faces.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
    out_ptr = out_sdf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    ret = lib.ComputeSDF_API(
        v_ptr, num_verts,
        f_ptr, num_faces,
        bool(usePostProcessing),
        int(rays),
        float(coneAngle),
        ptx_str,
        out_ptr
    )
    if ret != 0:
        raise RuntimeError(f"ComputeSDF_API failed with error code: {ret}")

    return out_sdf
```

- [ ] **Step 2: Create `python/optimize_sdf/__init__.py`**

```python
"""OptimizeSDF: High-Performance GPU Shape Diameter Function computation using NVIDIA OptiX."""
from .core import compute_sdf

__all__ = ["compute_sdf"]
__version__ = "1.0.0"
```

- [ ] **Step 3: Copy DLL and PTX to `python/optimize_sdf/lib/`**

Run PowerShell script to copy binaries:
```powershell
New-Item -ItemType Directory -Force -Path "python/optimize_sdf/lib"
Copy-Item "build/Release/optimize_sdf.dll" "python/optimize_sdf/lib/optimize_sdf.dll"
Copy-Item "build/OptixShaders.dir/Release/SDFOptix.ptx" "python/optimize_sdf/lib/SDFOptix.ptx"
```

- [ ] **Step 4: Commit**

```bash
git add python/optimize_sdf
git commit -m "feat: implement python wrapper core and package structure"
```

---

### Task 3: Packaging Configuration (`setup.py` & `pyproject.toml`)

**Files:**
- Create: `python/setup.py`
- Create: `python/pyproject.toml`
- Create: `python/README.md`

- [ ] **Step 1: Create `python/setup.py`**

```python
from setuptools import setup, find_packages

setup(
    name="optimize_sdf",
    version="1.0.0",
    description="GPU-accelerated Shape Diameter Function using NVIDIA OptiX RT Cores",
    author="USTH Final Project Team",
    packages=find_packages(),
    package_data={"optimize_sdf": ["lib/*.dll", "lib/*.ptx"]},
    include_package_data=True,
    install_requires=["numpy"],
    python_requires=">=3.8",
)
```

- [ ] **Step 2: Create `python/pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "optimize_sdf"
version = "1.0.0"
description = "GPU-accelerated Shape Diameter Function using NVIDIA OptiX RT Cores"
dependencies = [
    "numpy",
]
requires-python = ">=3.8"
```

- [ ] **Step 3: Install in Editable Mode**

Run: `& "F:\Program\PythonMiniConda\python.exe" -m pip install -e python/`  
Expected: `Successfully installed optimize-sdf-1.0.0`

- [ ] **Step 4: Commit**

```bash
git add python/setup.py python/pyproject.toml python/README.md
git commit -m "feat: add packaging configuration for pip installation"
```

---

### Task 4: Verification & End-to-End Testing

**Files:**
- Create: `tests/test_library.py`

- [ ] **Step 1: Write test script `tests/test_library.py`**

```python
import numpy as np
import optimize_sdf

MODEL_PATH = "Model/HighPeeling_Coil.obj"

def test_obj_postprocessing_true():
    print("[TEST 1] Testing OBJ input with usePostProcessing=True...")
    sdf = optimize_sdf.compute_sdf(MODEL_PATH, usePostProcessing=True)
    assert isinstance(sdf, np.ndarray)
    assert len(sdf) == 18000
    assert 0.0 <= sdf.min() and sdf.max() <= 1.0
    print(f" -> PASSED: len={len(sdf)}, min={sdf.min():.4f}, max={sdf.max():.4f}")

def test_obj_postprocessing_false():
    print("[TEST 2] Testing OBJ input with usePostProcessing=False (raw distance)...")
    raw_sdf = optimize_sdf.compute_sdf(MODEL_PATH, usePostProcessing=False)
    assert isinstance(raw_sdf, np.ndarray)
    assert len(raw_sdf) == 18000
    assert raw_sdf.max() > 4.0
    print(f" -> PASSED: len={len(raw_sdf)}, min={raw_sdf.min():.4f}, max={raw_sdf.max():.4f}")

def test_numpy_arrays_input():
    print("[TEST 3] Testing direct NumPy arrays input (vertices, faces)...")
    from optimize_sdf.core import _read_obj
    verts, faces = _read_obj(MODEL_PATH)
    sdf = optimize_sdf.compute_sdf((verts, faces), usePostProcessing=True)
    assert len(sdf) == len(verts)
    print(f" -> PASSED: len={len(sdf)}, min={sdf.min():.4f}, max={sdf.max():.4f}")

if __name__ == "__main__":
    test_obj_postprocessing_true()
    test_obj_postprocessing_false()
    test_numpy_arrays_input()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
```

- [ ] **Step 2: Run test suite**

Run: `& "F:\Program\PythonMiniConda\python.exe" tests/test_library.py`  
Expected: `ALL TESTS PASSED SUCCESSFULLY!`

- [ ] **Step 3: Commit**

```bash
git add tests/test_library.py
git commit -m "test: add verification test suite for optimize_sdf python library"
```
