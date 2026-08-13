# Design Spec: Extract Methodologies and Architecture into README.md

**Date**: 2026-08-13  
**Status**: Approved by User  
**Topic**: Documenting Technical Methodologies and System Architecture in `README.md`

---

## 1. Goal & Context

The goal is to update the repository's [`README.md`](file:///e:/Code/FinalProject/README.md) file with a comprehensive, publication-grade documentation of:
1. **Mathematical & Algorithmic Methodologies**: Explicit mathematical formulas (LaTeX/KaTeX) for normal calculation, tangent frames, Hammersley 2D sampling, OptiX hardware ray tracing, weighted SDF aggregation, log compression, CSR graph generation, and anisotropic bilateral smoothing.
2. **System Architecture**: High-performance GPU pipeline, dual CUDA stream execution, OptiX Acceleration Structure (GAS) & Shader Binding Table (SBT) management, CUB parallel primitives, memory layout, and Mermaid component diagrams.

---

## 2. Proposed Changes & Content Structure

The [`README.md`](file:///e:/Code/FinalProject/README.md) file will be updated under the existing sections (specifically expanding **Methodology Overview** and adding a dedicated **System Architecture** section).

### 2.1 Mathematical & Algorithmic Methodologies Section

1. **Vertex Normal & Local Tangent Frame Generation (`Core/ModelHelper.cu`)**:
   - Area-weighted face normal accumulation via CUDA `atomicAdd`:
     $$\mathbf{N}_{\text{face}} = (\mathbf{v}_1 - \mathbf{v}_0) \times (\mathbf{v}_2 - \mathbf{v}_0)$$
     $$\mathbf{N}_v = \text{normalize}\left(\sum_{f \in \text{Faces}(v)} \mathbf{N}_{\text{face}}\right)$$
   - Local orthonormal frame ($\mathbf{T}, \mathbf{B}, \mathbf{N}$) construction to avoid numerical singularities when mapping sampling directions into 3D world space.

2. **Quasi-Random Sampling & Cone Direction Mapping (`src/Optix/SDFOptix.cu`)**:
   - Low-discrepancy 2D Hammersley sequence for $R = 64$ rays per vertex:
     $$x_i = \frac{i}{R}, \quad y_i = \Phi_2(i) = \sum_{k=0}^{\lfloor \log_2 i \rfloor} b_k \cdot 2^{-(k+1)}$$
   - Spherical cone mapping for aperture angle $\theta_{\text{max}} = 150^\circ$:
     $$\theta_i = x_i \cdot \frac{\theta_{\text{max}}}{2}, \quad \phi_i = 2\pi y_i$$
     $$\mathbf{d}_{\text{local}} = (\sin\theta_i \cos\phi_i, \sin\theta_i \sin\phi_i, \cos\theta_i)$$
     $$\mathbf{d}_{\text{world}} = d_x \mathbf{T} + d_y \mathbf{B} + d_z (-\mathbf{N}_v)$$

3. **OptiX Hardware Ray Tracing & Distance Aggregation (`src/Optix/OptixRunner.cuh`, `SDFKernels.cuh`)**:
   - Hardware-accelerated BVH traversal with RT Cores using `optixTrace` and ray payload distance extraction via `optixGetRayTmax()`.
   - Weighted average based on ray angle:
     $$w_i = \frac{1}{\theta_i + \epsilon}, \quad SDF_{\text{raw}}(v) = \frac{\sum_{i=1}^{R_{\text{hit}}} d_i \cdot w_i}{\sum_{i=1}^{R_{\text{hit}}} w_i}$$

4. **Logarithmic Compression & CSR Graph Bilateral Filtering (`src/Optix/SDFKernels.cuh`)**:
   - Range normalization and logarithmic scaling:
     $$SDF_{\text{log}}(v) = \frac{\ln(4.0 \cdot SDF_{\text{norm}}(v) + 1.0)}{\ln(5.0)}$$
   - **Graph Extraction**: Directed triangle edges converted into a Compressed Sparse Row (CSR) adjacency graph on GPU via `cub::DeviceRadixSort` and `cub::DeviceSelect::Unique`.
   - **3-Iteration Anisotropic Bilateral Smoothing**:
     $$SDF^{(k+1)}(i) = \frac{\sum_{j \in \mathcal{N}(i)} SDF^{(k)}(j) \cdot G_s(\|\mathbf{p}_i - \mathbf{p}_j\|) \cdot G_r(|SDF^{(k)}(i) - SDF^{(k)}(j)|)}{\sum_{j \in \mathcal{N}(i)} G_s(\|\mathbf{p}_i - \mathbf{p}_j\|) \cdot G_r(|SDF^{(k)}(i) - SDF^{(k)}(j)|)}$$
     where spatial Gaussian $G_s(r) = \exp\left(-\frac{r^2}{2\sigma_s^2}\right)$ with $\sigma_s = 0.02 \cdot \text{diag}(\text{bbox})$, and range Gaussian $G_r(d) = \exp\left(-\frac{d^2}{2\sigma_r^2}\right)$ with $\sigma_r = 0.1$. Double-buffered ping-pong memory prevents race conditions.

---

### 2.2 System & Software Architecture Section

1. **Pipeline & Hardware Execution Flow**:
   - Dual CUDA Stream design: Normal computation stream executes concurrently with OptiX GAS (Geometry Acceleration Structure) BVH building on the GPU.
   - OptiX Shader Binding Table (SBT) layout mapping raygen, closest-hit, and miss shaders.

2. **System Architecture Diagram (Mermaid)**:
   - Full `graph TD` flow linking Host application (`main.cu`), CUDA kernels (`ModelHelper.cu`, `SDFKernels.cuh`), OptiX pipeline (`SDFOptix.cu`), and CUB primitives.

3. **File Responsibilities Matrix**:
   - Comprehensive table mapping every core source file to its component responsibility and execution layer.

---

## 3. Verification & Review Plan

1. **Spec Review**: Self-review for placeholders, ambiguity, and mathematical consistency.
2. **Markdown Preview**: Validate KaTeX math block rendering and Mermaid diagram rendering.
3. **Commit**: Save design spec to git.

---

## 4. Next Step

After user approval of the written spec, invoke `writing-plans` to create the execution plan.
