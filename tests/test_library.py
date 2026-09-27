import numpy as np
import optimize_sdf

MODEL_PATH = "Model/HighPeeling_Coil.obj"

def test_obj_postprocessing_true():
    print("[TEST 1] Testing OBJ input with usePostProcessing=True...")
    sdf = optimize_sdf.compute_sdf(MODEL_PATH, usePostProcessing=True)
    assert isinstance(sdf, np.ndarray), "sdf must be a numpy ndarray"
    assert len(sdf) == 18000, f"Expected 18000 vertices, got {len(sdf)}"
    assert 0.0 <= sdf.min() and sdf.max() <= 1.0, f"Normalized SDF out of [0, 1]: [{sdf.min()}, {sdf.max()}]"
    print(f" -> PASSED: len={len(sdf)}, min={sdf.min():.4f}, max={sdf.max():.4f}")

def test_obj_postprocessing_false():
    print("[TEST 2] Testing OBJ input with usePostProcessing=False (raw distance)...")
    raw_sdf = optimize_sdf.compute_sdf(MODEL_PATH, usePostProcessing=False)
    assert isinstance(raw_sdf, np.ndarray), "raw_sdf must be a numpy ndarray"
    assert len(raw_sdf) == 18000, f"Expected 18000 vertices, got {len(raw_sdf)}"
    assert raw_sdf.max() > 4.0, f"Expected raw distance > 4.0, got max={raw_sdf.max()}"
    print(f" -> PASSED: len={len(raw_sdf)}, min={raw_sdf.min():.4f}, max={raw_sdf.max():.4f}")

def test_numpy_arrays_input():
    print("[TEST 3] Testing direct NumPy arrays input (vertices, faces)...")
    from optimize_sdf.core import _read_obj
    verts, faces = _read_obj(MODEL_PATH)
    sdf = optimize_sdf.compute_sdf((verts, faces), usePostProcessing=True)
    assert len(sdf) == len(verts), f"Expected {len(verts)} vertices, got {len(sdf)}"
    assert 0.0 <= sdf.min() and sdf.max() <= 1.0, f"Normalized SDF out of [0, 1]: [{sdf.min()}, {sdf.max()}]"
    print(f" -> PASSED: len={len(sdf)}, min={sdf.min():.4f}, max={sdf.max():.4f}")

if __name__ == "__main__":
    test_obj_postprocessing_true()
    test_obj_postprocessing_false()
    test_numpy_arrays_input()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
