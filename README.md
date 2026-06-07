# Optimizing Shape Diameter Function using HPC

The Shape Diameter Function (SDF) is a scalar function defined on a 3D mesh surface that measures the local volume thickness (diameter) around each point. It provides a pose-oblivious, informative signature for shape analysis, widely used in 3D Computer Vision for mesh segmentation, skeletonization, and identifying distinct shape components.

This project implements and benchmarks two fundamentally different GPU approaches for computing the SDF:

1. **NVIDIA OptiX** -- Hardware-accelerated ray tracing via CUDA and RT Cores
2. **VCGlib/MeshLab GPU** -- Rasterization-based "spawning camera" with depth peeling via OpenGL

We compare against the previous work: [Parallelization of Shape Diameter Function Computation using OpenCL](https://old.cescg.org/CESCG-2014/papers/Kamenicky-Parallelization_of_Shape_Diameter_Function_Computation_using_OpenCL.pdf)

---

## What is the Shape Diameter Function?

Given a point on a mesh surface, the SDF works by:

1. Shooting a cone of rays from the point inward (toward the opposite side of the mesh)
2. Measuring how far each ray travels before hitting the mesh on the other side
3. Computing a weighted average of those distances, where rays closer to the cone axis are weighted more heavily

The formula is:

```
SDF(p) = Σ(distance_i × weight_i) / Σ(weight_i)
```

where `weight_i = 1 / angle_i` (inverse of the angle between the ray and the cone axis).

A thick region (like a torso) will have large SDF values; a thin region (like a finger) will have small SDF values. The result is invariant to rigid body transformations and oblivious to local pose deformations.

---

## Performance Benchmark: PyMeshLab GPU (VCGlib) vs. NVIDIA OptiX

| Model | Vertices | PyMeshLab GPU (s) | OptiX (s) | Faster | Speedup |
| :--- | :---: | :---: | :---: | :--- | :---: |
| 360.obj | 2,200 | 0.3228 | 0.0019 | OptiX | 171.0x |
| 9.obj | 2,639 | 0.4359 | 0.0084 | OptiX | 51.6x |
| 400.obj | 3,703 | 0.4004 | 0.0121 | OptiX | 33.2x |
| 76.obj | 5,923 | 0.5027 | 0.0106 | OptiX | 47.6x |
| 181.obj | 7,242 | 0.3128 | 0.0039 | OptiX | 80.6x |
| 118.obj | 9,153 | 0.3141 | 0.0047 | OptiX | 66.3x |
| 368.obj | 11,202 | 0.3217 | 0.0163 | OptiX | 19.7x |
| 112.obj | 13,628 | 0.8288 | 0.0229 | OptiX | 36.1x |
| 369.obj | 13,606 | 0.3772 | 0.0517 | OptiX | 7.3x |
| 158.obj | 14,587 | 0.3450 | 0.0069 | OptiX | 50.0x |
| 371.obj | 14,599 | 0.3097 | 0.0583 | OptiX | 5.3x |
| Leaf.obj | 24,866 | 0.3827 | 0.1219 | OptiX | 3.1x |

### Technical Summary

- **OptiX is faster across all models**, ranging from **3.1x to 171x** speedup.
- **PyMeshLab GPU has a ~0.3s base overhead** regardless of model size, due to OpenGL context initialization, texture upload, and iterating over 128 camera directions.
- **OptiX scales better** -- its ray tracing time grows proportionally with vertex count, while PyMeshLab's overhead is dominated by fixed-cost OpenGL operations.
- **OptiX includes post-processing** (normalization + anisotropic smoothing) that PyMeshLab does not, making the comparison even more favorable for OptiX.

> **Note:** OptiX timings include the full pipeline: normal computation + BVH build + ray tracing + CSR graph construction + anisotropic smoothing. PyMeshLab timings are raw SDF computation only (no smoothing or normalization).

### Test Settings

| Parameter | OptiX (CUDA) | PyMeshLab GPU (VCGlib) |
| :--- | :--- | :--- |
| **Rays per vertex** | 64 | 64 |
| **Cone angle** | 120 degrees | 150 degrees |
| **Ray sampling** | Hammersley 2D (quasi-random) | Fibonacci sphere (uniform) |
| **Compute primitive** | Per-vertex | Per-vertex (`onprimitive=0`) |
| **Outlier removal** | No | No (`removeoutliers=False`) |
| **Post-processing** | Log normalization + 3x anisotropic bilateral smoothing | None |
| **Smoothing sigma spatial** | 2% of bbox diagonal | N/A |
| **Smoothing sigma range** | 0.1 | N/A |
| **Normalization** | `log(4 * normalized + 1) / log(5)` | None (raw distance values) |
| **GPU** | NVIDIA RTX (OptiX RT Cores) | Any GPU with OpenGL 3.3+ |
| **Timing method** | `std::chrono::high_resolution_clock` | `time.perf_counter()` |
| **Visualization** | Off (Polyscope disabled) | Off (PyVista disabled) |

> **Note:** The cone angle differs between the two implementations (120 vs 150 degrees). This is the default for each respective library. A wider cone captures more rays from oblique angles, which can affect SDF accuracy on thin structures.

### Time Complexity Analysis

Let **V** = number of vertices, **F** = number of faces, **R** = rays per vertex, **N** = number of camera directions, **P** = depth peeling layers, **I** = smoothing iterations, **k** = average neighbors per vertex.

#### OptiX (CUDA)

| Stage | Complexity | Explanation |
| :--- | :--- | :--- |
| Normal computation | O(V + F) | One pass over faces to accumulate, one pass over vertices to normalize |
| BVH build | O(F log F) | OptiX builds a bounding volume hierarchy from all triangles |
| Ray tracing | O(V x R x log F) | Each vertex shoots R rays; each ray traverses the BVH in O(log F) |
| Weighted average | O(V x R) | Simple accumulation of distance x weight per vertex |
| Normalization | O(V) | Single min-max pass + log transform |
| CSR graph build | O(F log F) | Edge extraction O(F), sort O(F log F), unique + CSR O(F) |
| Anisotropic smoothing | O(V x k x I) | I iterations, each vertex blends with k neighbors |
| **Total** | **O(V x R x log F + F log F)** | BVH build and ray tracing dominate |

#### VCGlib/MeshLab GPU (OpenGL)

| Stage | Complexity | Explanation |
| :--- | :--- | :--- |
| Preprocessing | O(V + F) | Normal computation, bounding box, compact arrays |
| Upload to texture | O(V) | Pack vertex data into RGBA32F textures |
| For each of N directions: | | |
| - Spawn camera | O(1) | Set orthographic projection matrix |
| - Depth peeling | O(F x P) | P layers, each layer renders all visible triangles |
| - SDF shader | O(V) | Per-vertex depth comparison and weight accumulation |
| - FBO blend | O(V) | Additive blending into result texture |
| Readback | O(V) | Single glReadPixels call |
| **Total** | **O(N x (F x P + V))** | Depth peeling across all directions dominates |

#### Comparison

| Aspect | OptiX | VCGlib/MeshLab GPU |
| :--- | :--- | :--- |
| **Asymptotic** | O(V x R x log F) | O(N x F x P) |
| **With typical values** | O(V x 64 x log F) | O(64 x F x P), where P scales with model size for accuracy |
| **Per-element work** | Per-vertex (R independent rays) | Per-face (entire mesh rendered per direction) |
| **BVH factor** | log F per ray traversal | None (rasterization, no BVH) |
| **Parallelism model** | One CUDA thread per vertex | One GPU draw call per direction |
| **Bottleneck** | Ray-BVH intersection (compute-bound) | Rasterization + depth peeling (bandwidth-bound) |
| **Fixed overhead** | Low (CUDA kernel launch) | High (OpenGL context, texture upload, N draw calls) |

#### Why OptiX Wins Despite log F Factor

1. **Per-vertex parallelism**: OptiX launches V threads in parallel, each independently shooting R rays. The GPU's RT Cores handle ray-BVH traversal in hardware.

2. **VCGlib's per-direction overhead**: Each of the 128 camera directions requires a full mesh render pass. Even though rasterization is fast, 128 draw calls + depth peeling iterations accumulate significant fixed overhead.

3. **Data transfer**: OptiX uses zero-copy CUDA device pointers (no host-device transfer during computation). VCGlib requires `glReadPixels` to read back the FBO result to the CPU.

4. **Post-processing included**: OptiX's O(V x k x I) smoothing adds negligible cost compared to its O(V x R x log F) ray tracing, while VCGlib has no post-processing at all.

---

## SDF via NVIDIA OptiX (Hardware Ray Tracing)

This approach runs the entire SDF pipeline on the GPU using CUDA and NVIDIA's hardware-accelerated ray tracing (RT Cores). Every stage -- from normal computation to ray tracing to post-processing -- executes on the GPU with zero CPU involvement.

### Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OPTIX SDF PIPELINE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                   │
│  │  Load Mesh   │                                                   │
│  └──────┬───────┘                                                   │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Compute Vertex Normals on GPU                           │       │
│  │  For each vertex, accumulate the normals of adjacent     │       │
│  │  faces weighted by face area, then normalize.           │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Build Bounding Volume Hierarchy (BVH)                   │       │
│  │  Construct an acceleration structure from all triangles  │       │
│  │  using NVIDIA RT Cores for hardware-accelerated          │       │
│  │  ray-triangle intersection.                              │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Launch One Thread Per Vertex                            │       │
│  │  Each GPU thread handles one vertex and its cone of rays.│       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Generate Ray Directions (Hammersley 2D Sampling)        │       │
│  │  For each vertex, shoot 64 rays inside a 120-degree     │       │
│  │  cone centered on the inward-facing normal. Ray          │       │
│  │  directions are generated using a low-discrepancy        │       │
│  │  sequence that ensures uniform coverage of the cone      │       │
│  │  with minimal clustering.                                │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Trace Rays Through Hardware BVH                          │       │
│  │  Each ray is traced through the mesh using OptiX RT      │       │
│  │  Cores. If it hits the mesh on the other side, the       │       │
│  │  distance and the inverse of the angle from the cone     │       │
│  │  axis are stored.                                         │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Compute Weighted Average (Raw SDF)                       │       │
│  │  For each vertex, divide the sum of (distance x weight)  │       │
│  │  by the sum of weights across all valid rays.            │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Normalize SDF Values                                     │       │
│  │  Scale all values to [0, 1] using min-max, then apply    │       │
│  │  logarithmic compression to reduce the dynamic range     │       │
│  │  and bring out detail in thin regions.                    │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Build Mesh Adjacency Graph (CSR) on GPU                  │       │
│  │  Extract all directed edges from triangle faces, sort    │       │
│  │  and deduplicate them, then build a Compressed Sparse     │       │
│  │  Row representation of the mesh connectivity.            │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Anisotropic Bilateral Smoothing (4 iterations)           │       │
│  │  For each vertex, blend its SDF value with its neighbors │       │
│  │  using bilateral weights that consider both spatial       │       │
│  │  distance and SDF value difference. This smooths noise   │       │
│  │  while preserving sharp features at shape boundaries.    │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│                   Final SDF per vertex                              │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Characteristics

- **Compute platform**: NVIDIA RTX GPU via CUDA + OptiX RT Cores
- **Ray tracing method**: Hardware BVH traversal (optixTrace)
- **Ray sampling**: Hammersley 2D quasi-random sequence (low-discrepancy)
- **Multi-layer handling**: Single closest-hit per ray (no depth peeling needed -- hardware BVH returns the first intersection)
- **Post-processing**: Full GPU pipeline including normalization and anisotropic bilateral smoothing via CSR adjacency graph
- **Granularity**: Per-vertex only

### Source Files

| File | Role |
|------|------|
| `src/Optix/SDFOptix.cu` | Ray generation shader (Hammersley sampling, optixTrace calls) |
| `src/Optix/SDFKernels.cuh` | CUDA kernels: raw SDF, normalization, bilateral smoothing, CSR graph |
| `src/Optix/OptixRunner.cuh` | OptiX pipeline setup, BVH build, SBT construction |
| `src/Optix/interface.cu` | High-level orchestration of the full SDF pipeline |

---

## SDF via VCGlib/MeshLab GPU (Spawning Camera + Depth Peeling)

This approach uses OpenGL rasterization instead of ray tracing. It "spawns" virtual orthographic cameras around the mesh and uses the GPU's rasterization pipeline with depth peeling to compute thickness. This is the implementation behind PyMeshLab's `compute_scalar_by_shape_diameter_function_per_vertex_gpu` filter.

### Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│               VCGlib/MeshLab GPU SDF PIPELINE                       │
│            (Spawning Camera + Depth Peeling)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                   │
│  │  Load Mesh   │                                                   │
│  └──────┬───────┘                                                   │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Preprocess Mesh                                         │       │
│  │  Compute per-vertex normals, per-face normals, bounding  │       │
│  │  box, and compact vertex/face arrays.                    │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Upload Mesh Data to GPU Textures                         │       │
│  │  Pack vertex positions into a RGBA32F texture and        │       │
│  │  vertex normals into a second RGBA32F texture. Each      │       │
│  │  pixel holds one vertex's data with a 1:1 mapping:      │       │
│  │  pixel[i] corresponds to vertex[i].                      │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Generate Camera Directions (Fibonacci Sphere)            │       │
│  │  Distribute N directions uniformly on a sphere using     │       │
│  │  the Fibonacci spiral method. Each direction becomes     │       │
│  │  a virtual camera that will look at the mesh.            │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  For Each Camera Direction:                               │       │
│  │                                                          │       │
│  │  ┌────────────────────────────────────────────────────┐  │       │
│  │  │ Spawn Orthographic Camera                          │  │       │
│  │  │ Place the camera at the direction vector offset    │  │       │
│  │  │ from the mesh center, looking back at the center.  │  │       │
│  │  │ The orthographic projection covers the entire      │  │       │
│  │  │ bounding box so no mesh part is clipped.           │  │       │
│  │  └────────────────────────┬───────────────────────────┘  │       │
│  │                           ▼                              │       │
│  │  ┌────────────────────────────────────────────────────┐  │       │
│  │  │ Depth Peeling Passes (up to P layers)              │  │       │
│  │  │                                                    │  │       │
│  │  │ Render the mesh from this camera view. The GPU     │  │       │
│  │  │ rasterizer produces a depth map of the closest     │  │       │
│  │  │ surface. Then, iteratively peel each layer:        │  │       │
│  │  │                                                    │  │       │
│  │  │  Layer 0: Render front-facing triangles only.      │  │       │
│  │  │           Store the front depth map.               │  │       │
│  │  │                                                    │  │       │
│  │  │  Layer 1: Render back-facing triangles only.       │  │       │
│  │  │           Store the back depth map.                │  │       │
│  │  │           This pair gives the first thickness.     │  │       │
│  │  │                                                    │  │       │
│  │  │  Layer 2+: Use depth peeling shader to skip        │  │       │
│  │  │            already-captured layers, revealing      │  │       │
│  │  │            deeper surfaces. Continue until no      │  │       │
│  │  │            more pixels pass the occlusion query.   │  │       │
│  │  │                                                    │  │       │
│  │  │ This handles meshes with hollow interiors or       │  │       │
│  │  │ multiple surfaces along a ray (e.g., sphere = 2,  │  │       │
│  │  │ torus = 4 layers).                                 │  │       │
│  │  │                                                    │  │       │
│  │  │ Three FBOs are used in a circular buffer to avoid  │  │       │
│  │  │ z-fighting: each layer's depth is compared against │  │       │
│  │  │ the previous two layers to ensure correct range.   │  │       │
│  │  └────────────────────────┬───────────────────────────┘  │       │
│  │                           ▼                              │       │
│  │  ┌────────────────────────────────────────────────────┐  │       │
│  │  │ SDF Fragment Shader                                │  │       │
│  │  │                                                    │  │       │
│  │  │ For each pixel (which maps 1:1 to a vertex):       │  │       │
│  │  │   - Look up vertex position and normal from the    │  │       │
│  │  │     vertex textures                                │  │       │
│  │  │   - Project the vertex into the depth peeling      │  │       │
│  │  │     viewport using the same camera matrices        │  │       │
│  │  │   - Sample front and back depth textures at the    │  │       │
│  │  │     projected screen position                      │  │       │
│  │  │   - Compute distance = backDepth - frontDepth      │  │       │
│  │  │   - Check angle between ray and vertex normal;     │  │       │
│  │  │     skip if outside the cone                       │  │       │
│  │  │   - Compute weight as inverse of the angle         │  │       │
│  │  │   - Accumulate distance x weight into the red      │  │       │
│  │  │     channel and weight into the green channel      │  │       │
│  │  │     using additive blending                       │  │       │
│  │  └────────────────────────┬───────────────────────────┘  │       │
│  │                           ▼                              │       │
│  │  ┌────────────────────────────────────────────────────┐  │       │
│  │  │ Write Results to FBO                               │  │       │
│  │  │ Red channel: accumulated distance x weight         │  │       │
│  │  │ Green channel: accumulated weight                  │  │       │
│  │  │ All N camera directions blend into the same FBO.   │  │       │
│  │  │ The FBO starts at zero; each direction adds its    │  │       │
│  │  │ contribution via additive blending.                │  │       │
│  │  └────────────────────────────────────────────────────┘  │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  Readback Results to CPU                                  │       │
│  │  Read the FBO pixel data. For each vertex, divide the    │       │
│  │  accumulated distance-weight sum (red) by the            │       │
│  │  accumulated weight sum (green) to get the final SDF.    │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│                   Final SDF per vertex                              │
└─────────────────────────────────────────────────────────────────────┘
```

### How the 1:1 Vertex-to-Pixel Mapping Works

The core mechanism that makes this approach work:

1. **Vertex data is packed into textures** where pixel `i` holds the data for vertex `i`
2. **The mesh is rendered** from each spawned camera using standard rasterization, producing depth textures
3. **The SDF shader runs** as a full-screen quad over the result FBO. For each pixel `i`:
   - It reads vertex `i`'s position and normal from the vertex textures
   - It projects that position into the depth peeling viewport using the same camera matrices
   - It samples the front and back depth textures at that projected position
   - It computes the thickness and accumulates the weighted result
4. **After all directions**, the result FBO contains the global sum. Dividing red by green gives the weighted average SDF

### Why 3 FBOs for Depth Peeling

The 3 FBOs form a sliding window of three consecutive depth layers. When computing SDF for a front/back pair, the shader needs to verify that each vertex's depth falls between the previous back layer and the current back layer. Without the third FBO, pixels near layer boundaries get assigned to the wrong layer due to z-fighting.

### Why Red and Green Channels Are Accumulated Separately

The SDF formula is a weighted average (division). GPU additive blending can only add, not divide. So the numerator (distance x weight) goes into the red channel and the denominator (weight) goes into the green channel. The division happens once on the CPU during readback.

### Why All Directions Blend into the Same FBO

Each camera direction contributes a small piece to the global sum. Additive blending naturally accumulates them. This means only one FBO is needed and only one `glReadPixels` call is needed at the end, regardless of how many directions are used.

### Key Characteristics

- **Compute platform**: Any GPU with OpenGL 3.3+, FBOs, FP32 textures, and shaders
- **Ray tracing method**: Rasterization + depth peeling (no hardware ray tracing)
- **Ray sampling**: Fibonacci sphere (uniform distribution on sphere)
- **Multi-layer handling**: Depth peeling (up to P layers, configurable)
- **Post-processing**: None -- raw weighted average only
- **Granularity**: Per-vertex or per-face (configurable)

### Source Files

| File | Role |
|------|------|
| `filter_sdfgpu.cpp` (MeshLab repo) | Plugin entry point, GL init, FBO management |
| `filter_sdfgpu.h` (MeshLab repo) | Plugin class definition |
| `calculateSdf.frag` (shader) | SDF fragment shader: depth comparison, weight accumulation |
| `shaderDepthPeeling.fs` (shader) | Depth peeling fragment shader |
| `vertexShaderDepthPeeling.vs` (shader) | Depth peeling vertex shader |

---

## Side-by-Side Pipeline Comparison

```
┌─────────────────────────────┬─────────────────────────────────────────┐
│      OPTIX (CUDA)           │    VCGlib/MeshLab GPU (OpenGL)          │
├─────────────────────────────┼─────────────────────────────────────────┤
│                             │                                         │
│  Load Mesh                  │  Load Mesh                              │
│      │                      │      │                                  │
│      ▼                      │      ▼                                  │
│  Compute normals on GPU     │  Compute normals + bbox                 │
│      │                      │      │                                  │
│      ▼                      │      ▼                                  │
│  Build hardware BVH         │  Upload vertex data to GPU textures     │
│  (RT Core acceleration)     │  (1:1 pixel-to-vertex mapping)          │
│      │                      │      │                                  │
│      ▼                      │      ▼                                  │
│  One thread per vertex      │  Generate N camera directions           │
│      │                      │  (Fibonacci sphere)                     │
│      ▼                      │      │                                  │
│  Generate 64 ray dirs       │      ▼                                  │
│  per vertex                 │  For each direction:                    │
│  (Hammersley sequence)      │      │                                  │
│      │                      │      ▼                                  │
│      ▼                      │  Spawn ortho camera at direction        │
│  Trace each ray through     │      │                                  │
│  hardware BVH               │      ▼                                  │
│  (single closest hit)       │  Render mesh, peel depth layers         │
│      │                      │  (multi-hit via rasterization)          │
│      ▼                      │      │                                  │
│  Compute weighted average   │      ▼                                  │
│  per vertex                 │  SDF shader: compute thickness           │
│      │                      │  between front/back depth layers        │
│      ▼                      │  (sample depth at vertex's position)    │
│  Normalize: log(4x+1)/log5 │      │                                  │
│      │                      │      ▼                                  │
│      ▼                      │  Accumulate into FBO                    │
│  Build CSR adjacency graph  │  (additive blending, all directions)    │
│      │                      │      │                                  │
│      ▼                      │      ▼                                  │
│  Smooth: 3x bilateral       │  Readback FBO, divide sums              │
│  filtering on GPU           │      │                                  │
│      │                      │      ▼                                  │
│      ▼                      │  Final SDF                              │
│  Final SDF                  │                                         │
│                             │                                         │
│  Key traits:                │  Key traits:                            │
│  - Ray tracing approach     │  - Rasterization approach               │
│  - HW BVH traversal         │  - Depth peeling for multi-layer        │
│  - Full GPU post-processing │  - No post-processing                  │
│  - NVIDIA RTX required      │  - Any GPU with OpenGL 3.3+            │
│  - Per-vertex only          │  - Per-vertex or per-face               │
└─────────────────────────────┴─────────────────────────────────────────┘
```

---

## Detailed Comparison

| Aspect | OptiX (this project) | VCGlib/MeshLab GPU |
|--------|---------------------|-------------------|
| **Compute Platform** | NVIDIA RTX GPU (CUDA + OptiX RT Cores) | Any GPU with OpenGL 3.3+ (FBOs, FP32 textures, shaders) |
| **Ray Tracing Method** | Hardware BVH traversal via `optixTrace` | Rasterization + depth peeling |
| **Granularity** | Per-vertex | Per-vertex or per-face (configurable) |
| **Ray Sampling** | Hammersley 2D (low-discrepancy quasi-random) | Fibonacci sphere (uniform on sphere) |
| **Default Rays per Point** | 64 | 128 |
| **Default Cone Angle** | 120 degrees | 120 degrees |
| **Multi-layer Handling** | Single closest-hit per ray (hardware BVH returns first intersection only) | Depth peeling (configurable up to P layers; handles hollow meshes, tori, etc.) |
| **BVH** | Hardware-accelerated via `optixAccelBuild` (RT Cores) | None -- rasterization-based, no ray tracing structure needed |
| **Weighting Formula** | `1 / angle_from_cone_axis` (same) | `1 / angle_from_cone_axis` (same) |
| **Post-processing** | Normalization (log compression) + Anisotropic bilateral smoothing (4 iterations on CSR graph) | None -- raw weighted average only |
| **Normalization** | Min-max scaling then `log(alpha * x + 1) / log(alpha + 1)` with alpha=4 | None (raw distance values) |
| **Graph Smoothing** | CSR bilateral filter on GPU: spatial weight + range weight, sigmaSpatial=2% of bbox diagonal, sigmaRange=0.1 | N/A |
| **False Intersection Removal** | No (all valid ray hits are used) | Yes (checks normal dot product at intersection; skips if ray points away from surface) |
| **Outlier Removal** | Optional (commented out in code) | Optional (shader-based supersampling of depth buffer; takes median of nearby depth values) |
| **FBOs for Depth Peeling** | N/A (hardware ray tracing, no rasterization) | 3 FBOs in circular buffer to avoid z-fighting between layers |
| **Result Accumulation** | Direct per-vertex computation in CUDA kernel | Additive blending into FBO: red = sum(distance x weight), green = sum(weight); division on CPU readback |
| **Vertex-to-Pixel Mapping** | N/A (each CUDA thread owns one vertex) | 1:1 mapping: vertex `i` stored at texture pixel `i`; SDF shader looks up vertex data by pixel index |
| **Language/Runtime** | CUDA C++ (compiled to PTX, JIT-loaded by OptiX) | OpenGL GLSL shaders + C++ host code |
| **Memory Model** | GPU VRAM (zero-copy via CUDA device pointers) | GPU VRAM (textures + FBOs) + CPU readback via `glReadPixels` |
| **Dependencies** | NVIDIA OptiX SDK, CUDA Toolkit | OpenGL, GLEW, Qt, MeshLab common library |
| **Hardware Requirement** | NVIDIA RTX GPU (RT Cores required for BVH acceleration) | Any GPU supporting OpenGL 3.3+ with FP32 texture and FBO support |

---

## Visual Quality Comparison

Each row shows the SDF heat map from both implementations and their absolute difference. Hot colors (red/yellow) indicate larger SDF values (thicker regions), while cool colors (blue) indicate thinner regions.

| Model | SDF OptiX | SDF PyMeshLab | Difference |
| :--- | :---: | :---: | :---: |
| **112** | <img src="image/112_optix.png" width="100%" alt="112 OptiX"> | <img src="image/112_pymeshlab.png" width="100%" alt="112 PyMeshLab"> | <img src="image/112_diff.png" width="100%" alt="112 Diff"> |
| **118** | <img src="image/118_optix.png" width="100%" alt="118 OptiX"> | <img src="image/118_pymeshlab.png" width="100%" alt="118 PyMeshLab"> | <img src="image/118_diff.png" width="100%" alt="118 Diff"> |
| **158** | <img src="image/158_optix.png" width="100%" alt="158 OptiX"> | <img src="image/158_pymeshlab.png" width="100%" alt="158 PyMeshLab"> | <img src="image/158_diff.png" width="100%" alt="158 Diff"> |
| **181** | <img src="image/181_optix.png" width="100%" alt="181 OptiX"> | <img src="image/181_pymeshlab.png" width="100%" alt="181 PyMeshLab"> | <img src="image/181_diff.png" width="100%" alt="181 Diff"> |
| **368** | <img src="image/368_optix.png" width="100%" alt="368 OptiX"> | <img src="image/368_pymeshlab.png" width="100%" alt="368 PyMeshLab"> | <img src="image/368_diff.png" width="100%" alt="368 Diff"> |
| **369** | <img src="image/369_optix.png" width="100%" alt="369 OptiX"> | <img src="image/369_pymeshlab.png" width="100%" alt="369 PyMeshLab"> | <img src="image/369_diff.png" width="100%" alt="369 Diff"> |
| **371** | <img src="image/371_optix.png" width="100%" alt="371 OptiX"> | <img src="image/371_pymeshlab.png" width="100%" alt="371 PyMeshLab"> | <img src="image/371_diff.png" width="100%" alt="371 Diff"> |
| **400** | <img src="image/400_optix.png" width="100%" alt="400 OptiX"> | <img src="image/400_pymeshlab.png" width="100%" alt="400 PyMeshLab"> | <img src="image/400_diff.png" width="100%" alt="400 Diff"> |
| **76** | <img src="image/76_optix.png" width="100%" alt="76 OptiX"> | <img src="image/76_pymeshlab.png" width="100%" alt="76 PyMeshLab"> | <img src="image/76_diff.png" width="100%" alt="76 Diff"> |
| **9** | <img src="image/9_optix.png" width="100%" alt="9 OptiX"> | <img src="image/9_pymeshlab.png" width="100%" alt="9 PyMeshLab"> | <img src="image/9_diff.png" width="100%" alt="9 Diff"> |

---

## Setup and Run

> **CAUTION: Linux Only**
> This project is strictly designed for **Linux environments** and will not compile or run correctly on Windows or macOS.

### Prerequisites

To successfully build and execute this project, ensure the following tools are installed on your system:

* **`g++`**: The GNU C++ compiler for building standard C++ source files.
* **`nvcc`**: The NVIDIA CUDA Compiler toolkit, required for compiling the OptiX/CUDA components.
* **`CLion`**: The recommended IDE to load the CMake project, resolve dependencies, and execute the builds properly.
