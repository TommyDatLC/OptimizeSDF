# README Methodologies & Architecture Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand [`README.md`](file:///e:/Code/FinalProject/README.md) with comprehensive, publication-grade mathematical formulations and system architecture breakdowns extracted from the codebase.

**Architecture:** Update `README.md` in two structured stages: (1) Mathematical & Algorithmic Methodologies detailing equations for normals, tangent frames, Hammersley 2D sampling, OptiX ray tracing, weighted SDF aggregation, log compression, CSR graph generation, and anisotropic bilateral smoothing; (2) System & Software Architecture detailing dual CUDA streams, OptiX GAS/SBT pipeline, CUB GPU primitives, and dynamic Mermaid sequence/architecture diagrams.

**Tech Stack:** Markdown, KaTeX/LaTeX math, Mermaid diagrams.

## Global Constraints

- Preserve all existing benchmark performance tables, image links, setup/build instructions, and reference citations in [`README.md`](file:///e:/Code/FinalProject/README.md).
- Use standard GitHub Flavored Markdown and KaTeX math blocks (`$$...$$` and `$...$`).
- Ensure all file basenames in the documentation link to their absolute local file paths using `file:///e:/Code/FinalProject/...`.

---

### Task 1: Add Mathematical & Algorithmic Methodologies Section to README.md

**Files:**
- Modify: [`README.md`](file:///e:/Code/FinalProject/README.md:93-160)

**Interfaces:**
- Consumes: Code implementations in [`ModelHelper.cu`](file:///e:/Code/FinalProject/Core/ModelHelper.cu), [`SDFOptix.cu`](file:///e:/Code/FinalProject/src/Optix/SDFOptix.cu), [`OptixRunner.cuh`](file:///e:/Code/FinalProject/src/Optix/OptixRunner.cuh), [`SDFKernels.cuh`](file:///e:/Code/FinalProject/src/Optix/SDFKernels.cuh).
- Produces: Expanded "Methodology & Algorithmic Foundations" section in [`README.md`](file:///e:/Code/FinalProject/README.md).

- [ ] **Step 1: Draft the Mathematical & Algorithmic Methodologies markdown content**

Prepare the complete KaTeX formulas and detailed explanations for:
1. Vertex Normal Accumulation & Orthonormal Tangent Frame ($\mathbf{T}, \mathbf{B}, \mathbf{N}$)
2. Hammersley 2D Quasi-Random Sampling & $150^\circ$ Spherical Cone Mapping
3. OptiX Hardware Ray Tracing & Angle-Weighted Distance Aggregation
4. Logarithmic Range Compression
5. Topology Extraction & CSR Adjacency Graph Construction
6. 3-Iteration Anisotropic Bilateral Surface Filtering (Spatial $G_s$ & Range $G_r$ Gaussians with double-buffered ping-pong memory)

- [ ] **Step 2: Update README.md with the expanded Methodologies section**

Replace/expand the "Methodology Overview" section in [`README.md`](file:///e:/Code/FinalProject/README.md) with the drafted mathematical and algorithmic formulations.

- [ ] **Step 3: Commit Task 1 changes**

```powershell
git add README.md; git commit -m "docs: expand README.md with mathematical methodologies and LaTeX equations"
```

---

### Task 2: Add System & Software Architecture Section to README.md

**Files:**
- Modify: [`README.md`](file:///e:/Code/FinalProject/README.md:135-161)

**Interfaces:**
- Consumes: System workflow in [`main.cu`](file:///e:/Code/FinalProject/main.cu), [`interface.cu`](file:///e:/Code/FinalProject/src/Optix/interface.cu), [`OptixRunner.cuh`](file:///e:/Code/FinalProject/src/Optix/OptixRunner.cuh).
- Produces: New "System & Software Architecture" section in [`README.md`](file:///e:/Code/FinalProject/README.md) featuring Mermaid diagrams and CUDA/OptiX pipeline breakdowns.

- [ ] **Step 1: Draft the Mermaid Architecture and Pipeline diagrams**

Create two Mermaid diagrams:
1. `graph TD`: High-performance hardware & software system architecture (CPU Host, CUDA Streams, OptiX RT Cores, CUB primitives, and Memory Layout).
2. `flowchart TD`: Data flow from `.obj` model parsing to smoothed SDF vertex array output.

- [ ] **Step 2: Update README.md with the System & Software Architecture section**

Insert the architecture section into [`README.md`](file:///e:/Code/FinalProject/README.md), including:
- Dual CUDA Stream concurrency breakdown
- OptiX Acceleration Structure (GAS) & Shader Binding Table (SBT) setup
- CUB parallel radix sort & unique selection for CSR graph creation
- File responsibilities table with hyperlinked source files

- [ ] **Step 3: Commit Task 2 changes**

```powershell
git add README.md; git commit -m "docs: add system architecture breakdown and Mermaid diagrams to README.md"
```

---

### Task 3: Verify Formatting and Final Verification

**Files:**
- Verify: [`README.md`](file:///e:/Code/FinalProject/README.md)

- [ ] **Step 1: Check markdown syntax and links**

Ensure KaTeX formulas are valid, Mermaid block syntax is correct, and all file links use standard markdown syntax.

- [ ] **Step 2: Commit final verification**

```powershell
git add README.md; git commit -m "docs: finalize README.md structure and formatting"
```
