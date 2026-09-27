"""
OptimizeSDF: High-Performance GPU Shape Diameter Function computation using NVIDIA OptiX.
"""

from .core import compute_sdf

__all__ = ["compute_sdf"]
__version__ = "1.0.0"
