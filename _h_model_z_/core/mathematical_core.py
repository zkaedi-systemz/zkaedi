"""
Mathematical Core - Foundations of Harmonic Beauty

This module implements the core mathematical framework that underlies
the entire _h_model_z_ system. Every function embodies mathematical elegance.
"""

import numpy as np
from typing import Union, Optional, Callable, Tuple, List
import scipy.special as special
from dataclasses import dataclass
import cmath

# Mathematical constants for harmonic beauty
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant


@dataclass
class HarmonicParameters:
    """Parameters for harmonic decomposition with mathematical meaning."""
    amplitude: float  # A_i - Wave amplitude
    frequency: float  # B_i - Angular frequency
    phase: float      # φ_i - Phase offset
    decay: float      # D_i - Exponential decay rate
    
    def __post_init__(self):
        """Validate parameters maintain mathematical stability."""
        if self.amplitude < 0:
            raise ValueError("Amplitude must be non-negative")
        if self.frequency <= 0:
            raise ValueError("Frequency must be positive")
        if self.decay < 0:
            raise ValueError("Decay rate must be non-negative")


class MathematicalCore:
    """
    Core mathematical engine implementing the fundamental equation:
    
    Ĥ(t) = Σ A_i*sin(B_i*t + φ_i) + C_i*e^(-D_i*t) + ∫softplus(...) + ε(t)
    
    This is where chaos becomes harmony, where noise becomes signal.
    """
    
    def __init__(self, precision: str = "double"):
        """Initialize mathematical core with specified precision."""
        self.precision = precision
        self.epsilon = 1e-12 if precision == "double" else 1e-6
        self._golden_ratio_harmonics = None
        
    def oscillatory_term(self, 
                        t: np.ndarray, 
                        params: HarmonicParameters) -> np.ndarray:
        """
        Compute A_i * sin(B_i * t + φ_i)
        
        The oscillatory heart of the model - where periodic patterns emerge.
        """
        return params.amplitude * np.sin(params.frequency * t + params.phase)
    
    def decay_term(self, 
                   t: np.ndarray, 
                   signal: np.ndarray, 
                   decay_rate: float) -> np.ndarray:
        """
        Compute C_i * e^(-D_i * t) * signal
        
        Exponential decay that ensures stability and convergence.
        """
        return signal * np.exp(-decay_rate * t)
    
    def softplus_integration(self, 
                           x: np.ndarray, 
                           beta: float = 1.0) -> np.ndarray:
        """
        Compute ∫ softplus(β * x) dx = (1/β) * log(1 + e^(β * x))
        
        The softplus function creates smooth potential wells that trap errors
        without harsh discontinuities. Mathematical beauty in action.
        """
        # Numerically stable softplus computation
        return np.where(
            beta * x > 700,  # Prevent overflow
            x + np.log(beta) / beta,  # Linear approximation for large values
            np.log1p(np.exp(beta * x)) / beta  # Standard softplus
        )
    
    def chaos_residual(self, 
                      t: np.ndarray, 
                      noise_level: float = 0.1,
                      seed: Optional[int] = None) -> np.ndarray:
        """
        Generate ε(t) - chaotic residual that captures unpredictable components.
        
        Even chaos has structure when viewed through the right lens.
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Multi-scale noise with fractal characteristics
        n = len(t)
        noise = np.zeros(n)
        
        # Add noise at multiple scales
        for scale in [1, 2, 4, 8]:
            if n // scale > 0:
                noise += noise_level * np.random.normal(0, 1/scale, n)
                
        return noise
    
    def golden_ratio_harmonics(self, n_harmonics: int) -> List[HarmonicParameters]:
        """
        Generate harmonics with golden ratio spacing for maximum beauty.
        
        The golden ratio appears in nature's most elegant forms - 
        from nautilus shells to galaxy spirals. Here it shapes our frequencies.
        """
        if self._golden_ratio_harmonics is None or len(self._golden_ratio_harmonics) != n_harmonics:
            harmonics = []
            for i in range(n_harmonics):
                # Golden ratio progression for frequencies
                freq = PHI ** (i - n_harmonics/2)
                
                # Amplitude decreases with higher frequencies
                amp = 1.0 / (1 + i * 0.3)
                
                # Phase follows fibonacci sequence converted to radians
                fib_n = self._fibonacci(i + 1)
                phase = (fib_n % 8) * np.pi / 4
                
                # Decay rate increases with frequency for stability
                decay = 0.1 * freq
                
                harmonics.append(HarmonicParameters(amp, freq, phase, decay))
                
            self._golden_ratio_harmonics = harmonics
            
        return self._golden_ratio_harmonics
    
    def _fibonacci(self, n: int) -> int:
        """Compute nth Fibonacci number efficiently."""
        if n <= 1:
            return n
        
        # Using Binet's formula for efficiency
        phi_n = PHI ** n
        psi_n = ((-1/PHI) ** n)
        return int((phi_n - psi_n) / np.sqrt(5))
    
    def compute_full_harmonic_model(self, 
                                   t: np.ndarray,
                                   signal: np.ndarray,
                                   harmonics: List[HarmonicParameters],
                                   noise_level: float = 0.1,
                                   beta: float = 1.0) -> np.ndarray:
        """
        Compute the complete harmonic model:
        Ĥ(t) = Σ A_i*sin(B_i*t + φ_i) + C_i*e^(-D_i*t) + ∫softplus(...) + ε(t)
        
        This is the heart of the transformation - where errors become harmony.
        """
        result = np.zeros_like(t, dtype=float)
        
        # Oscillatory components: Σ A_i*sin(B_i*t + φ_i)
        for params in harmonics:
            result += self.oscillatory_term(t, params)
        
        # Decay components: C_i*e^(-D_i*t) applied to signal
        for params in harmonics:
            result += self.decay_term(t, signal, params.decay)
        
        # Softplus integration: ∫softplus(...)
        result += self.softplus_integration(signal, beta)
        
        # Chaos residual: ε(t)
        result += self.chaos_residual(t, noise_level)
        
        return result
    
    def holomorphic_extension(self, 
                            real_signal: np.ndarray) -> np.ndarray:
        """
        Extend real signal to complex plane maintaining holomorphic properties.
        
        Holomorphic functions are complex differentiable everywhere - 
        the most elegant functions in mathematics.
        """
        # Use Hilbert transform to create analytic signal
        analytic_signal = special.hilbert(real_signal)
        return analytic_signal
    
    def convergence_test(self, 
                        signal: np.ndarray, 
                        tolerance: float = 1e-6) -> bool:
        """
        Test if the harmonic model converges to stable solution.
        
        Mathematics demands convergence - chaos must yield to order.
        """
        if len(signal) < 10:
            return False
            
        # Check if the signal is stabilizing
        recent_variance = np.var(signal[-10:])
        early_variance = np.var(signal[:10])
        
        # Convergence if recent variance is much smaller
        return recent_variance < tolerance * max(early_variance, tolerance)
    
    def phase_space_embedding(self, 
                             signal: np.ndarray, 
                             dimension: int = 3, 
                             delay: int = 1) -> np.ndarray:
        """
        Create phase space embedding for chaos analysis.
        
        Phase space reveals the hidden structure in apparent chaos.
        """
        n = len(signal)
        embedded = np.zeros((n - (dimension - 1) * delay, dimension))
        
        for i in range(dimension):
            embedded[:, i] = signal[i * delay : n - (dimension - 1 - i) * delay]
            
        return embedded
    
    def lyapunov_exponent(self, 
                         signal: np.ndarray, 
                         dimension: int = 3) -> float:
        """
        Estimate Lyapunov exponent to quantify chaos.
        
        The Lyapunov exponent measures how quickly nearby trajectories diverge.
        Positive values indicate chaos, negative indicate stability.
        """
        # Simplified Lyapunov estimation
        embedded = self.phase_space_embedding(signal, dimension)
        
        if len(embedded) < 20:
            return 0.0
            
        # Estimate divergence rate
        divergences = []
        for i in range(len(embedded) - 10):
            current = embedded[i]
            future = embedded[i + 1]
            
            # Find nearest neighbors
            distances = np.linalg.norm(embedded - current, axis=1)
            nearest_idx = np.argsort(distances)[1]  # Exclude self
            
            # Measure divergence
            d0 = distances[nearest_idx]
            if d0 > 0 and nearest_idx + 1 < len(embedded):
                d1 = np.linalg.norm(embedded[nearest_idx + 1] - future)
                if d1 > 0:
                    divergences.append(np.log(d1 / d0))
        
        return np.mean(divergences) if divergences else 0.0