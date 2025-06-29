"""
Wells module - Softplus error trapping systems

Where high-energy errors find their peaceful rest.
"""

from .softplus_wells import SoftplusWells, softplus_well, multi_well_trap, adaptive_softplus_well

__all__ = [
    'SoftplusWells',
    'softplus_well', 
    'multi_well_trap',
    'adaptive_softplus_well',
]