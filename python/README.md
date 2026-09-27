# optimize_sdf

GPU-accelerated Shape Diameter Function (SDF) calculation for 3D triangular meshes using NVIDIA OptiX Ray Tracing Cores and CUDA.

## Installation

```bash
pip install .
```

Or for development (editable mode):
```bash
pip install -e .
```

## Quick Start

```python
import optimize_sdf

# 1. From an OBJ file (with post-processing / bilateral smoothing normalized to [0, 1])
sdf = optimize_sdf.compute_sdf("path/to/model.obj", usePostProcessing=True)

# 2. Get raw physical ray travel distances without smoothing
raw_sdf = optimize_sdf.compute_sdf("path/to/model.obj", usePostProcessing=False)

# 3. From direct NumPy geometry arrays (vertices: (N, 3), faces: (M, 3))
import numpy as np
verts = np.array([...], dtype=np.float32)
faces = np.array([...], dtype=np.uint32)
sdf = optimize_sdf.compute_sdf((verts, faces), usePostProcessing=True, rays=64, coneAngle=150.0)
```

## API Reference

### `compute_sdf(mesh, usePostProcessing=True, rays=64, coneAngle=150.0) -> np.ndarray`

- **`mesh`**: `str`, `Path`, or `tuple(np.ndarray, np.ndarray)`. Either a file path to an `.obj` file or a 2-tuple of `(vertices, faces)` where `vertices` is `(N, 3)` `float32` and `faces` is `(M, 3)` `uint32`.
- **`usePostProcessing`**: `bool` (default: `True`).
  - If `True`: Returns normalized SDF in $[0, 1]$ after logarithmic compression and 3-iteration anisotropic bilateral smoothing on the mesh surface.
  - If `False`: Returns raw aggregated ray travel distances directly.
- **`rays`**: `int` (default: `64`). Number of cone rays traced per vertex into the interior of the mesh.
- **`coneAngle`**: `float` (default: `150.0`). Total cone opening angle in degrees.
- **Returns**: `np.ndarray` of shape `(N,)` and dtype `float32`.
