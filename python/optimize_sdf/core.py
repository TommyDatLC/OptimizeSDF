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
        
        # Add library directory and CUDA binary directory to DLL search path on Windows
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(LIB_DIR.resolve()))
            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                cuda_bin = Path(cuda_path) / "bin"
                if cuda_bin.exists():
                    os.add_dll_directory(str(cuda_bin))
        
        _lib = ctypes.CDLL(str(DLL_PATH.resolve()))
        _lib.ComputeSDF_API.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32),
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
                parts = line.strip().split()[1:]
                f_indices = [int(p.split("/")[0]) - 1 for p in parts]
                if len(f_indices) == 3:
                    faces.append(f_indices)
                elif len(f_indices) > 3:
                    for i in range(1, len(f_indices) - 1):
                        faces.append([f_indices[0], f_indices[i], f_indices[i + 1]])
    return np.ascontiguousarray(vertices, dtype=np.float32), np.ascontiguousarray(faces, dtype=np.uint32)

def compute_sdf(
    mesh: str | Path | tuple[np.ndarray, np.ndarray],
    usePostProcessing: bool = True,
    rays: int = 64,
    coneAngle: float = 150.0
) -> np.ndarray:
    """
    Compute Shape Diameter Function (SDF) using GPU ray tracing via NVIDIA OptiX.

    Parameters
    ----------
    mesh : str, Path, or tuple(np.ndarray, np.ndarray)
        Target 3D mesh. Either path to a .obj file or a tuple (vertices, faces)
        where vertices has shape (N, 3) float32 and faces has shape (M, 3) uint32.
    usePostProcessing : bool, default=True
        If True, applies logarithmic normalization and 3-iteration anisotropic bilateral smoothing,
        returning normalized SDF values in [0, 1].
        If False, returns raw aggregated ray travel distances directly.
    rays : int, default=64
        Number of cone rays traced per vertex into the interior of the mesh.
    coneAngle : float, default=150.0
        Total opening angle of the cone in degrees.

    Returns
    -------
    np.ndarray
        1D float32 array of shape (N,) containing the SDF value for each vertex.
    """
    if isinstance(mesh, (str, Path)):
        verts, faces = _read_obj(mesh)
    elif isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        verts = np.ascontiguousarray(mesh[0], dtype=np.float32)
        faces = np.ascontiguousarray(mesh[1], dtype=np.uint32)
    else:
        raise TypeError("mesh must be a file path string or a tuple of (vertices, faces) numpy arrays")

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices array must have shape (N, 3), got {verts.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces array must have shape (M, 3), got {faces.shape}")

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]
    out_sdf = np.empty(num_verts, dtype=np.float32)

    lib = _get_lib()
    ptx_str = str(PTX_PATH.resolve()).encode("utf-8") if PTX_PATH.exists() else None

    v_ptr = verts.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    f_ptr = faces.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
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
