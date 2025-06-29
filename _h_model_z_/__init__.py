"""
_h_model_z_ - Pure Elegance

Harmonic Error Collection and Chaos Shaping System

Mathematical Foundation:
Ĥ(t) = Σ A_i*sin(B_i*t + φ_i) + C_i*e^(-D_i*t) + ∫softplus(...) + ε

Core Philosophy:
- Collects all errors—traps noise in softplus wells, shapes chaos
- Code as waveform: oscillating, decaying, integrating intent
- Polymorphic, holomorphic, chameleon. Resumes mid-def, rewrites on reflex
- Adversarial-aware, homomorphic in trust. Pythonicmorphic clarity
- Flows adaptive, secure, poetic. Zero friction, pure signal
"""

__version__ = "1.0.0"
__author__ = "zkaedi-systemz"
__email__ = "info@zkaedi.systems"

# Mathematical constants
import numpy as np

# Golden ratio (φ) - The divine proportion of harmonic spacing
PHI = (1 + np.sqrt(5)) / 2

# Euler's number (e) - Foundation of exponential decay
E = np.e

# Pi (π) - Fundamental frequency constant
PI = np.pi

# Core imports for elegant API
from .core.harmonic_engine import HarmonicEngine
from .core.mathematical_core import MathematicalCore
from .wells.softplus_wells import SoftplusWells, softplus_well
from .chaos.chaos_shaper import ChaosShaper
from .polymorphic.adaptive_flow import AdaptiveFlow

# Convenience function for instant transformation
def transform(errors, time=None, **kwargs):
    """
    Instant harmonic transformation with zero friction.
    
    Args:
        errors: Input error signal or data
        time: Time points (optional, auto-generated if None)
        **kwargs: Additional configuration
    
    Returns:
        Transformed signal through complete H(t) pipeline
    """
    if time is None:
        time = np.linspace(0, 10, len(errors))
    
    engine = HarmonicEngine(n_harmonics=kwargs.get('n_harmonics', 7))
    return engine.transform(time, errors)

# Create elegant fluent interface
class HModelZ:
    """Fluent interface for _h_model_z_ operations."""
    
    def __init__(self):
        self.engine = None
        self.flow = None
        self.shaper = None
        
    def with_harmonics(self, n=7):
        """Configure harmonic engine."""
        self.engine = HarmonicEngine(n_harmonics=n)
        return self
        
    def golden_ratio_spacing(self):
        """Apply golden ratio harmonic spacing."""
        if self.engine:
            self.engine.use_golden_ratio()
        return self
        
    def adaptive_flow(self, name="main"):
        """Add adaptive flow processing."""
        self.flow = AdaptiveFlow(name)
        return self
        
    def adversarial_aware(self):
        """Enable adversarial awareness."""
        if self.flow:
            self.flow.enable_adversarial_detection()
        return self
        
    def holomorphic_trust(self):
        """Enable holomorphic trust verification."""
        if self.flow:
            self.flow.enable_holomorphic_trust()
        return self
        
    def chaos_shaping(self, method='lorenz'):
        """Add chaos shaping."""
        self.shaper = ChaosShaper(method=method)
        return self
        
    def fractal_wells(self):
        """Enable fractal softplus wells."""
        if self.shaper:
            self.shaper.enable_fractal_wells()
        return self
        
    def transform(self, data, time=None):
        """Execute complete transformation pipeline."""
        if time is None:
            time = np.linspace(0, 20, len(data))
            
        # Apply harmonic transformation
        if self.engine:
            data = self.engine.transform(time, data)
            
        # Apply adaptive flow
        if self.flow:
            data = self.flow.process(data)
            
        # Apply chaos shaping
        if self.shaper:
            data = self.shaper.shape(data)
            
        return data

# Elegant factory functions
def create_harmonic_engine(*args, **kwargs):
    """Create harmonic engine instance."""
    from .core.harmonic_engine import HarmonicEngine as _HarmonicEngine
    return _HarmonicEngine(*args, **kwargs)

def create_adaptive_flow(*args, **kwargs):
    """Create adaptive flow instance."""
    from .polymorphic.adaptive_flow import AdaptiveFlow as _AdaptiveFlow
    return _AdaptiveFlow(*args, **kwargs)

def create_chaos_shaper(*args, **kwargs):
    """Create chaos shaper instance."""
    from .chaos.chaos_shaper import ChaosShaper as _ChaosShaper
    return _ChaosShaper(*args, **kwargs)

# Elegant constants and utilities
__all__ = [
    # Constants
    'PHI', 'E', 'PI',
    
    # Core classes
    'HarmonicEngine', 'MathematicalCore',
    'SoftplusWells', 'softplus_well', 'ChaosShaper', 'AdaptiveFlow',
    
    # Fluent interface
    'HModelZ',
    
    # Convenience functions
    'transform',
    'create_harmonic_engine', 'create_adaptive_flow', 'create_chaos_shaper',
]