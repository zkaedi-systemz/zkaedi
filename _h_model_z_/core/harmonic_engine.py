"""
Harmonic Engine - The Heart of Pure Elegance

This is where the complete mathematical model comes alive:
Ĥ(t) = Σ A_i*sin(B_i*t + φ_i) + C_i*e^(-D_i*t) + ∫softplus(...) + ε

Every function oscillates with purpose, every line flows with intent.
"""

import numpy as np
from typing import Union, Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
import warnings

from .mathematical_core import MathematicalCore, HarmonicParameters, PHI


@dataclass
class EngineConfig:
    """Configuration for the harmonic engine with elegant defaults."""
    n_harmonics: int = 7  # Lucky number 7, pleasing to the eye
    use_golden_ratio: bool = False
    noise_level: float = 0.1
    softplus_beta: float = 1.0
    convergence_tolerance: float = 1e-6
    max_frequency: float = 10.0
    min_frequency: float = 0.1
    

class HarmonicEngine:
    """
    The Harmonic Engine - where errors transform into beauty.
    
    This engine implements the complete mathematical model with elegant simplicity.
    It embodies the philosophy that code should be as beautiful as the mathematics it implements.
    """
    
    def __init__(self, 
                 n_harmonics: int = 7,
                 config: Optional[EngineConfig] = None):
        """
        Initialize the harmonic engine with specified number of harmonics.
        
        Args:
            n_harmonics: Number of harmonic components (default: 7)
            config: Optional configuration object
        """
        self.config = config or EngineConfig(n_harmonics=n_harmonics)
        self.config.n_harmonics = n_harmonics  # Override if provided
        
        self.math_core = MathematicalCore()
        self.harmonics: List[HarmonicParameters] = []
        self.transformation_history: List[np.ndarray] = []
        self._use_golden_ratio = False
        
        # Initialize harmonics
        self._initialize_harmonics()
        
    def _initialize_harmonics(self):
        """Initialize harmonic parameters with mathematical beauty."""
        if self.config.use_golden_ratio or self._use_golden_ratio:
            self.harmonics = self.math_core.golden_ratio_harmonics(self.config.n_harmonics)
        else:
            self.harmonics = self._generate_standard_harmonics()
    
    def _generate_standard_harmonics(self) -> List[HarmonicParameters]:
        """Generate standard harmonic parameters with musical intervals."""
        harmonics = []
        
        # Use musical intervals for pleasing harmonic relationships
        musical_ratios = [1, 2, 3/2, 4/3, 5/4, 6/5, 7/6, 8/7, 9/8]
        
        for i in range(self.config.n_harmonics):
            # Use musical ratios for frequencies
            ratio_idx = i % len(musical_ratios)
            base_freq = 1.0
            freq = base_freq * musical_ratios[ratio_idx] * (1 + i // len(musical_ratios))
            
            # Ensure frequency is within bounds
            freq = np.clip(freq, self.config.min_frequency, self.config.max_frequency)
            
            # Amplitude decreases with frequency for natural sound
            amp = 1.0 / (1 + i * 0.2)
            
            # Phase based on harmonic number
            phase = i * np.pi / 4
            
            # Decay rate for stability
            decay = 0.05 + i * 0.02
            
            harmonics.append(HarmonicParameters(amp, freq, phase, decay))
            
        return harmonics
    
    def use_golden_ratio(self) -> 'HarmonicEngine':
        """
        Enable golden ratio harmonic spacing for maximum mathematical beauty.
        
        Returns:
            Self for fluent interface
        """
        self._use_golden_ratio = True
        self.config.use_golden_ratio = True
        self._initialize_harmonics()
        return self
    
    def with_harmonics(self, n: int) -> 'HarmonicEngine':
        """
        Set number of harmonics and reinitialize.
        
        Args:
            n: Number of harmonics
            
        Returns:
            Self for fluent interface
        """
        self.config.n_harmonics = n
        self._initialize_harmonics()
        return self
    
    def set_noise_level(self, level: float) -> 'HarmonicEngine':
        """
        Set chaos noise level.
        
        Args:
            level: Noise level (0.0 to 1.0)
            
        Returns:
            Self for fluent interface
        """
        self.config.noise_level = np.clip(level, 0.0, 1.0)
        return self
    
    def transform(self, 
                  t: Union[np.ndarray, List[float]], 
                  error_signal: Union[np.ndarray, List[float]],
                  **kwargs) -> np.ndarray:
        """
        Apply complete harmonic transformation to error signal.
        
        This is where the magic happens - errors become harmony.
        
        Args:
            t: Time points
            error_signal: Input error signal to transform
            **kwargs: Additional transformation parameters
            
        Returns:
            Transformed signal through complete Ĥ(t) model
        """
        # Convert inputs to numpy arrays
        t = np.asarray(t, dtype=float)
        error_signal = np.asarray(error_signal, dtype=float)
        
        # Validate inputs
        if len(t) != len(error_signal):
            # If time and signal lengths don't match, create time array
            if len(t) == 1:
                t = np.linspace(0, t[0], len(error_signal))
            else:
                warnings.warn("Time and signal lengths don't match. Truncating to shorter length.")
                min_len = min(len(t), len(error_signal))
                t = t[:min_len]
                error_signal = error_signal[:min_len]
        
        # Override config with kwargs
        noise_level = kwargs.get('noise_level', self.config.noise_level)
        beta = kwargs.get('softplus_beta', self.config.softplus_beta)
        
        # Apply the complete harmonic model
        transformed = self.math_core.compute_full_harmonic_model(
            t=t,
            signal=error_signal,
            harmonics=self.harmonics,
            noise_level=noise_level,
            beta=beta
        )
        
        # Store transformation history
        self.transformation_history.append(transformed.copy())
        
        # Trim history to prevent memory bloat
        if len(self.transformation_history) > 100:
            self.transformation_history = self.transformation_history[-50:]
            
        return transformed
    
    def collect_errors(self, 
                      raw_data: Union[np.ndarray, List[float]]) -> 'ErrorCollection':
        """
        Collect and prepare errors for transformation.
        
        Args:
            raw_data: Raw error data
            
        Returns:
            ErrorCollection object for fluent processing
        """
        return ErrorCollection(self, raw_data)
    
    def analyze_convergence(self) -> Dict[str, Any]:
        """
        Analyze convergence properties of recent transformations.
        
        Returns:
            Dictionary with convergence analysis
        """
        if not self.transformation_history:
            return {"status": "no_data", "converged": False}
            
        latest = self.transformation_history[-1]
        converged = self.math_core.convergence_test(latest, self.config.convergence_tolerance)
        
        # Calculate additional metrics
        lyapunov = self.math_core.lyapunov_exponent(latest)
        
        return {
            "status": "converged" if converged else "diverging",
            "converged": converged,
            "lyapunov_exponent": lyapunov,
            "is_chaotic": lyapunov > 0,
            "signal_length": len(latest),
            "signal_variance": np.var(latest),
            "signal_mean": np.mean(latest),
        }
    
    def phase_space_analysis(self, 
                           dimension: int = 3) -> Dict[str, Any]:
        """
        Perform phase space analysis on latest transformation.
        
        Args:
            dimension: Embedding dimension
            
        Returns:
            Phase space analysis results
        """
        if not self.transformation_history:
            return {"status": "no_data"}
            
        latest = self.transformation_history[-1]
        embedded = self.math_core.phase_space_embedding(latest, dimension)
        
        return {
            "status": "analyzed",
            "dimension": dimension,
            "embedding_shape": embedded.shape,
            "attractor_bounds": {
                "min": np.min(embedded, axis=0).tolist(),
                "max": np.max(embedded, axis=0).tolist(),
                "center": np.mean(embedded, axis=0).tolist(),
            }
        }
    
    def holomorphic_extension(self) -> np.ndarray:
        """
        Create holomorphic extension of latest transformation.
        
        Returns:
            Complex analytic signal
        """
        if not self.transformation_history:
            raise ValueError("No transformation history available")
            
        latest = self.transformation_history[-1]
        return self.math_core.holomorphic_extension(latest)
    
    def get_harmonic_summary(self) -> Dict[str, Any]:
        """
        Get summary of current harmonic configuration.
        
        Returns:
            Summary dictionary
        """
        return {
            "n_harmonics": len(self.harmonics),
            "use_golden_ratio": self.config.use_golden_ratio,
            "frequency_range": {
                "min": min(h.frequency for h in self.harmonics),
                "max": max(h.frequency for h in self.harmonics),
                "mean": np.mean([h.frequency for h in self.harmonics]),
            },
            "amplitude_range": {
                "min": min(h.amplitude for h in self.harmonics),
                "max": max(h.amplitude for h in self.harmonics),
                "mean": np.mean([h.amplitude for h in self.harmonics]),
            },
            "harmonics": [
                {
                    "amplitude": h.amplitude,
                    "frequency": h.frequency,
                    "phase": h.phase,
                    "decay": h.decay,
                }
                for h in self.harmonics
            ]
        }


class ErrorCollection:
    """
    Fluent interface for error collection and processing.
    
    Enables elegant chaining: engine.collect_errors(data).through(flow).shape_chaos(shaper)
    """
    
    def __init__(self, engine: HarmonicEngine, data: Union[np.ndarray, List[float]]):
        """Initialize error collection."""
        self.engine = engine
        self.data = np.asarray(data, dtype=float)
        self.time = np.linspace(0, 20, len(self.data))  # Default time range
        
    def with_time(self, t: Union[np.ndarray, List[float]]) -> 'ErrorCollection':
        """
        Set time points for the error signal.
        
        Args:
            t: Time points
            
        Returns:
            Self for fluent interface
        """
        self.time = np.asarray(t, dtype=float)
        return self
    
    def through(self, processor) -> 'ErrorCollection':
        """
        Pass through adaptive flow processor.
        
        Args:
            processor: Adaptive flow or other processor
            
        Returns:
            Self for fluent interface
        """
        if hasattr(processor, 'process'):
            self.data = processor.process(self.data)
        elif hasattr(processor, 'flow'):
            self.data = processor.flow(self.data)
        elif callable(processor):
            self.data = processor(self.data)
        return self
    
    def shape_chaos(self, shaper) -> 'ErrorCollection':
        """
        Apply chaos shaping.
        
        Args:
            shaper: Chaos shaper instance
            
        Returns:
            Self for fluent interface
        """
        if hasattr(shaper, 'shape'):
            self.data = shaper.shape(self.data)
        elif callable(shaper):
            self.data = shaper(self.data)
        return self
    
    def render_live(self) -> np.ndarray:
        """
        Execute transformation and return final result.
        
        Returns:
            Transformed signal
        """
        return self.engine.transform(self.time, self.data)