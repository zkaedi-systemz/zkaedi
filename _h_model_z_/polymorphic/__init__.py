"""
Polymorphic module - Self-modifying, adaptive algorithms

Code that evolves, patterns that learn, algorithms that adapt.
"""

from .adaptive_flow import AdaptiveFlow, FlowState, Pattern, LinearPattern, HolomorphicPattern, SoftplusPattern

__all__ = [
    'AdaptiveFlow',
    'FlowState',
    'Pattern',
    'LinearPattern',
    'HolomorphicPattern',
    'SoftplusPattern',
]