"""
Core module - Mathematical foundations of _h_model_z_

Where pure mathematics meets elegant code.
"""

from .mathematical_core import MathematicalCore, HarmonicParameters
from .harmonic_engine import HarmonicEngine, EngineConfig, ErrorCollection

__all__ = [
    'MathematicalCore',
    'HarmonicParameters', 
    'HarmonicEngine',
    'EngineConfig',
    'ErrorCollection',
]