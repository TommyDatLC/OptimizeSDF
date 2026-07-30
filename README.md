# Optimizing Shape Diameter Function using High Performance Computing

![USTH](https://img.shields.io/badge/USTH-Final%20Thesis-blue)
![CUDA](https://img.shields.io/badge/CUDA-20.0-green)
![OptiX](https://img.shields.io/badge/OptiX-7.6-orange)

This thesis presents a **GPU-accelerated Shape Diameter Function (SDF) computation system** using NVIDIA OptiX. SDF assigns a scalar value to each vertex of a 3D mesh representing the local thickness of the model at that point — a fundamental geometric descriptor widely used in shape analysis, segmentation, and parameterization.

Unlike the traditional approach of casting rays on a CPU, our method uses **hardware-accelerated ray tracing (RT Cores)** to achieve speedups of **61.3x to 356.2x** over the PyMeshLab GPU reference implementation, while producing comparable output quality.

---

## What is the Shape Diameter Function?

Introduced by Shapira et al. [1], the SDF is a per-vertex scalar measure that captures the local "thickness" of a 3D model at any point on its surface. Conceptually, consider a point **p** on the mesh surface. If one looks along the inward normal direction **−n⃗**, the SDF quantifies how far one must travel before exiting the opposite side of the object.

However, a single ray along the inward normal may yield unreliable results on bumpy or irregular interior surfaces. To address this, the SDF casts a cone of **R rays** (typically R = 64) from point **p**, directed toward the interior of the model. Each ray records its travel distance **dᵢ** to the opposite wall, and the final SDF value is computed as a weighted average:

```
SDF(p) = Σ(dᵢ × wᵢ) / Σ(wᵢ)
```

where **wᵢ = 1/θᵢ** is the weight inversely proportional to the angle between the ray and the cone axis.

<p align="center">
  <img src="image/112_optix.png" width="400" alt="SDF heatmap example">
  <br>
  <em>A 3D model with SDF values visualized as a heatmap. Red = thick, blue = thin.</em>
</p>

A thick region (like a human torso) produces large SDF values, while a thin region (like a finger) produces small values. Because the SDF depends purely on the object's volume, it remains **invariant to rigid transformations** — rotating or bending the model does not alter the thickness values.

---

## Why SDF: Role in 3D Object Analysis

Understanding an object's thickness provides essential geometric cues for:
- **Mesh segmentation**: Cutting a 3D model into logical pieces (e.g., separating an arm from a body because the wrist is thinner)
- **Skeletonization**: Finding the "bones" inside a 3D character by tracing the thickest parts
- **Shape matching and retrieval**: Identifying similar shapes across databases using thickness-based signatures that are pose-oblivious
- **Character animation**: Rigging and skinning by understanding volumetric properties

---

## Thesis Structure

This repository contains the full implementation for the following thesis chapters:

| Chapter | Description |
|---------|-------------|
| **1. Introduction** | SDF definition, context, survey of existing work, objectives |
| **2. Materials and Technologies** | Dataset, CUDA, NVIDIA CUB, NVIDIA OptiX |
| **3. Methodology** | Pipeline: OBJ loading → normals → BVH → ray tracing → post-processing |
| **4. Experiment Results** | Performance benchmarks, quality comparison, quality analysis |
| **5. Conclusion** | Limitations, conclusion, future work |

---

## Performance Benchmark: PyMeshLab GPU (VCGlib) vs. NVIDIA OptiX

| Model | Vertices | PyMeshLab GPU (s) | OptiX (s) | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| 360.obj | 2,200 | 0.6134 | 0.0017 | **356.2x** |
| 9.obj | 2,639 | 0.6660 | 0.0025 | **269.5x** |
| 400.obj | 3,703 | 0.5736 | 0.0026 | **223.2x** |
| 76.obj | 5,923 | 0.5623 | 0.0028 | **200.2x** |
| 181.obj | 7,242 | 0.6296 | 0.0032 | **193.8x** |
| 118.obj | 9,153 | 0.5366 | 0.0042 | **128.4x** |
| 368.obj | 11,202 | 0.5933 | 0.0054 | **108.9x** |
| 112.obj | 13,628 | 1.4853 | 0.0229 | **64.9x** |
| 369.obj | 13,606 | 0.5912 | 0.0062 | **94.7x** |
| 158.obj | 14,587 | 0.6146 | 0.0077 | **79.4x** |
| 371.obj | 14,599 | 0.5464 | 0.0089 | **61.3x** |

**Key findings:**
- **OptiX is faster across all models**, ranging from **61.3x to 356.2x** speedup
- **PyMeshLab consistently requires >0.5 seconds**, even for small models, due to the fixed overhead of its 64-camera initialization and OpenGL context setup
- **OptiX processes models with up to 14,599 vertices within 8.9 milliseconds**, making it suitable for interactive applications

### Test Settings

| Parameter | OptiX (CUDA) | PyMeshLab GPU (VCGlib) |
| :--- | :--- | :--- |
| **Rays per vertex** | 64 | 64 |
| **Cone angle** | 150 degrees | 150 degrees |
| **Ray sampling** | Hammersley 2D (quasi-random) | Fibonacci sphere (uniform) |
| **Depth peeling layers** | N/A (single closest-hit via hardware BVH) | 10 layers |
| **Post-processing** | Log normalization + 3x anisotropic bilateral smoothing | None |
| **GPU** | NVIDIA RTX (OptiX RT Cores) | Any GPU with OpenGL 3.3+ |

---

## Methodology Overview

### Pipeline

```
Read OBJ Model
       │
       ▼
Parallel GPU Init
 ┌──────────────────┐
 │ Calculate Normals │  Build BVH
 └──────────────────┘
       │
       ▼
OptiX Ray Tracing Engine
 ┌──────────────────────┐
 │ Build Shader Binding │
 │     Table (SBT)      │
 ├──────────────────────┤
 │  OptiX Pipeline      │
 │  (raygen + closest-  │
 │       hit shaders)   │
 ├──────────────────────┤
 │  Calculate Weighted  │
 │        SDF           │
 └──────────────────────┘
       │
       ▼
GPU Post-Processing
 ┌──────────────────────┐
 │  Normalize SDF       │
 │  (log compression)   │
 ├──────────────────────┤
 │  Build CSR Graph     │
 │  (CUB radix sort)    │
 ├──────────────────────┤
 │  Anisotropic Bilateral│
 │  Smoothing (3x iter) │
 └──────────────────────┘
       │
       ▼
   Final SDF
```

### Key Components

1. **Read OBJ Model** (`Core/Model.cu`): Parses `.obj` files into vertex and face matrices on the GPU.

2. **Calculate Vertex Normals** (`Core/ModelHelper.cu`): Two CUDA kernels — one accumulates area-weighted face normals via `atomicAdd`, the other normalizes to unit length. Runs on its own CUDA stream in parallel with BVH construction.

3. **Build BVH** (`OptixRunner.cuh`): Constructs an OptiX Geometry Acceleration Structure (GAS) from the triangle mesh. Face culling is disabled to allow rays to hit triangles from both sides.

4. **Build Shader Binding Table** (`OptixRunner.cuh`): Maps three ray tracing events (ray generation, closest-hit, miss) to their corresponding GPU programs.

5. **OptiX Pipeline** (`SDFOptix.cu`):
   - **Ray Generator** (`__raygen__sdf_cone`): Distributes 64 rays within a cone using Hammersley 2D quasi-random sampling, maps them to world space via tangent-frame transformation, and traces them through the hardware BVH.
   - **Closest-Hit** (`__closesthit__sdf`): Records the travel distance using `optixGetRayTmax()`.
   - `OPTIX_RAY_FLAG_DISABLE_ANYHIT` is set for maximum RT Core throughput.

6. **Calculate Weighted SDF** (`SDFKernels.cuh`): The `GPUComputeRawSDF` kernel computes the weighted average per vertex, excluding rays that missed.

7. **Normalize SDF** (`SDFKernels.cuh`): Min-max scaling followed by logarithmic compression: `log(4.0 · x̂ + 1) / log(5.0)`.

8. **Build Adjacency Graph** (`SDFKernels.cuh`): Extracts 6 directed edges per triangle, sorts and deduplicates them via `cub::DeviceRadixSort` + `cub::DeviceSelect::Unique`, then converts to Compressed Sparse Row (CSR) format.

9. **Anisotropic Bilateral Smoothing** (`SDFKernels.cuh`): Three iterations of bilateral filtering with spatial Gaussian (σₛ = 2% of bounding box diagonal) and range Gaussian (σᵣ = 0.1). Uses ping-pong double-buffering.

---

## Quality Comparison

The following figures show the SDF heat map from both implementations and their absolute difference. Hot colors (red/yellow) indicate larger SDF values (thicker regions), while cool colors (blue) indicate thinner regions.

| Model | OptiX | PyMeshLab | Difference |
| :--- | :---: | :---: | :---: |
| **112** | <img src="image/112_optix.png" width="100%"> | <img src="image/112_pymeshlab.png" width="100%"> | <img src="image/112_diff.png" width="100%"> |
| **118** | <img src="image/118_optix.png" width="100%"> | <img src="image/118_pymeshlab.png" width="100%"> | <img src="image/118_diff.png" width="100%"> |
| **158** | <img src="image/158_optix.png" width="100%"> | <img src="image/158_pymeshlab.png" width="100%"> | <img src="image/158_diff.png" width="100%"> |
| **181** | <img src="image/181_optix.png" width="100%"> | <img src="image/181_pymeshlab.png" width="100%"> | <img src="image/181_diff.png" width="100%"> |
| **368** | <img src="image/368_optix.png" width="100%"> | <img src="image/368_pymeshlab.png" width="100%"> | <img src="image/368_diff.png" width="100%"> |
| **369** | <img src="image/369_optix.png" width="100%"> | <img src="image/369_pymeshlab.png" width="100%"> | <img src="image/369_diff.png" width="100%"> |
| **371** | <img src="image/371_optix.png" width="100%"> | <img src="image/371_pymeshlab.png" width="100%"> | <img src="image/371_diff.png" width="100%"> |
| **400** | <img src="image/400_optix.png" width="100%"> | <img src="image/400_pymeshlab.png" width="100%"> | <img src="image/400_diff.png" width="100%"> |
| **76** | <img src="image/76_optix.png" width="100%"> | <img src="image/76_pymeshlab.png" width="100%"> | <img src="image/76_diff.png" width="100%"> |
| **9** | <img src="image/9_optix.png" width="100%"> | <img src="image/9_pymeshlab.png" width="100%"> | <img src="image/9_diff.png" width="100%"> |

The OptiX bilateral filter produces smoother results in uniform thickness regions, while the PyMeshLab depth peeling approach exhibits high-frequency noise on constant-thickness surfaces. The overall thickness distribution captured by both methods is consistent.

---

## Limitations

- **Static meshes only**: The system currently does not handle animated or deforming geometry, as the BVH is built once and not updated during animation.
- **Single reference comparison**: Evaluation is limited to PyMeshLab; a broader comparison against other SDF methods would strengthen the assessment.

---

## Setup and Run

> **CAUTION: Linux Only**
> This project is strictly designed for **Linux environments** and will not compile or run correctly on Windows or macOS.

### Prerequisites

- **g++**: GNU C++ compiler
- **nvcc**: NVIDIA CUDA Compiler toolkit (CUDA 20.0+)
- **NVIDIA RTX GPU**: For hardware-accelerated ray tracing (RT Cores)
- **OptiX 7.6 SDK**: Available from NVIDIA Developer

### Build

```bash
git clone https://github.com/<your-username>/OptimizeSDF
cd OptimizeSDF
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Run

```bash
./OptimizeSDF
```

The program will iterate over all `.obj` models in the `Model/` directory, compute SDF using the OptiX pipeline, and compare against the PyMeshLab reference.

### Source Files

| File | Role |
|------|------|
| `main.cu` | Entry point: orchestrates full pipeline |
| `Core/Model.cu` | 3D mesh model reader (OBJ parser) |
| `Core/ModelHelper.cu` | GPU kernels for vertex normal computation |
| `src/Optix/SDFOptix.cu` | OptiX shaders: raygen + closest-hit (compiled to PTX) |
| `src/Optix/SDFKernels.cuh` | CUDA kernels: raw SDF, normalization, CSR graph, bilateral smoothing |
| `src/Optix/OptixRunner.cuh` | OptiX pipeline setup, BVH build, SBT construction |
| `src/Optix/interface.cu` | High-level orchestration of the full SDF pipeline |

---

## References

1. L. Shapira, A. Shamir, D. Cohen-Or, "Consistent Mesh Partitioning and Skeletonization using the Shape Diameter Function," *Visual Computer*, vol. 24, pp. 249–259, 2008.
2. J. Kamenický, "Parallelization of Shape Diameter Function Computation using OpenCL," *Proceedings of CESCG*, 204, 2014.
3. S. Chen, T. Liu, Z. Shu, S. Xin, Y. He, C. Tu, "Fast and robust shape diameter function," *PLoS ONE*, vol. 13, no. 1, e0190666, 2018.
4. P. Cignoni et al., "MeshLab: an Open-Source Mesh Processing Tool," *Sixth Eurographics Italian Chapter Conference*, pp. 129–136, 2008.
5. S. G. Parker et al., "OptiX: a general purpose ray tracing engine," *ACM SIGGRAPH 2010 papers*, pp. 1–13, 2010.
6. J. Nickolls et al., "Scalable parallel programming with CUDA," *Queue*, vol. 6, no. 2, pp. 40–53, 2008.
7. C. Tomasi and R. Manduchi, "Bilateral Filtering for Gray and Color Images," *ICCV*, pp. 839–846, 1998.
8. NVIDIA Corporation, "CUB: Cooperative primitives for CUDA C++."
9. X. Chen, A. Golovinskiy, T. Funkhouser, "A Benchmark for 3D Mesh Segmentation," *ACM SIGGRAPH*, 2009.
10. A. Baldacci et al., "GPU-based approaches for shape diameter function computation and its applications focused on skeleton extraction," *Computers & Graphics*, vol. 59, pp. 151–159, 2016.

---

## Author

**Nguyễn Đức Đạt** (22BA13065)  
University of Science and Technology of Hanoi  
*Supervised by:* Dr. Nguyễn Hoàng Hà, Prof. Lilian Aveneau
