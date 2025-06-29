"""
Chaos Shaper - Where Chaos Becomes Beauty

Transforms unpredictable chaotic systems into manageable, beautiful patterns.
Uses strange attractors, fractal landscapes, and nonlinear dynamics to
shape chaos into harmony.
"""

import numpy as np
from typing import Union, List, Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass
import scipy.integrate as integrate
from abc import ABC, abstractmethod


@dataclass
class AttractorParameters:
    """Parameters for strange attractor systems."""
    sigma: float = 10.0   # Prandtl number (Lorenz)
    rho: float = 28.0     # Rayleigh number (Lorenz)
    beta: float = 8.0/3.0 # Physical parameter (Lorenz)
    a: float = 0.2        # Rössler parameter
    b: float = 0.2        # Rössler parameter  
    c: float = 5.7        # Rössler parameter


class StrangeAttractor(ABC):
    """Abstract base class for strange attractor systems."""
    
    @abstractmethod
    def equations(self, state: np.ndarray, t: float) -> np.ndarray:
        """Differential equations defining the attractor."""
        pass
    
    @abstractmethod
    def integrate(self, 
                 initial_state: np.ndarray,
                 time_span: Tuple[float, float],
                 n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate the attractor equations."""
        pass


class LorenzAttractor(StrangeAttractor):
    """
    The famous Lorenz attractor - the butterfly effect incarnate.
    
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y  
    dz/dt = xy - βz
    
    This system exhibits exquisite sensitivity to initial conditions,
    creating the butterfly-shaped strange attractor.
    """
    
    def __init__(self, params: Optional[AttractorParameters] = None):
        """Initialize Lorenz attractor with parameters."""
        self.params = params or AttractorParameters()
    
    def equations(self, state: np.ndarray, t: float) -> np.ndarray:
        """Lorenz equations."""
        x, y, z = state
        
        dx_dt = self.params.sigma * (y - x)
        dy_dt = x * (self.params.rho - z) - y
        dz_dt = x * y - self.params.beta * z
        
        return np.array([dx_dt, dy_dt, dz_dt])
    
    def integrate(self, 
                 initial_state: np.ndarray,
                 time_span: Tuple[float, float] = (0, 30),
                 n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate Lorenz equations."""
        t = np.linspace(time_span[0], time_span[1], n_points)
        
        # Use scipy's ODE solver
        solution = integrate.odeint(self.equations, initial_state, t)
        
        return t, solution


class RosslerAttractor(StrangeAttractor):
    """
    Rössler attractor - elegantly simple yet chaotic.
    
    dx/dt = -y - z
    dy/dt = x + ay
    dz/dt = b + z(x - c)
    
    Simpler than Lorenz but still exhibits rich chaotic behavior.
    """
    
    def __init__(self, params: Optional[AttractorParameters] = None):
        """Initialize Rössler attractor with parameters."""
        self.params = params or AttractorParameters()
    
    def equations(self, state: np.ndarray, t: float) -> np.ndarray:
        """Rössler equations."""
        x, y, z = state
        
        dx_dt = -y - z
        dy_dt = x + self.params.a * y
        dz_dt = self.params.b + z * (x - self.params.c)
        
        return np.array([dx_dt, dy_dt, dz_dt])
    
    def integrate(self, 
                 initial_state: np.ndarray,
                 time_span: Tuple[float, float] = (0, 50),
                 n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate Rössler equations."""
        t = np.linspace(time_span[0], time_span[1], n_points)
        
        solution = integrate.odeint(self.equations, initial_state, t)
        
        return t, solution


class ChuaAttractor(StrangeAttractor):
    """
    Chua's circuit attractor - chaos from electronics.
    
    A beautiful example of how simple nonlinear circuits can create chaos.
    """
    
    def __init__(self, params: Optional[AttractorParameters] = None):
        """Initialize Chua attractor."""
        self.params = params or AttractorParameters()
        self.alpha = 15.6
        self.beta = 28.0
        self.m0 = -1.143
        self.m1 = -0.714
    
    def _chua_nonlinearity(self, x: float) -> float:
        """Chua's nonlinear function."""
        return self.m1 * x + 0.5 * (self.m0 - self.m1) * (np.abs(x + 1) - np.abs(x - 1))
    
    def equations(self, state: np.ndarray, t: float) -> np.ndarray:
        """Chua circuit equations."""
        x, y, z = state
        
        dx_dt = self.alpha * (y - x - self._chua_nonlinearity(x))
        dy_dt = x - y + z
        dz_dt = -self.beta * y
        
        return np.array([dx_dt, dy_dt, dz_dt])
    
    def integrate(self, 
                 initial_state: np.ndarray,
                 time_span: Tuple[float, float] = (0, 100),
                 n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate Chua equations."""
        t = np.linspace(time_span[0], time_span[1], n_points)
        
        solution = integrate.odeint(self.equations, initial_state, t)
        
        return t, solution


class ChaosShaper:
    """
    Main chaos shaping engine that transforms chaos into beauty.
    
    Takes chaotic signals and applies attractor-based transformations
    to create structured, beautiful patterns from apparent randomness.
    """
    
    def __init__(self, method: str = 'lorenz'):
        """
        Initialize chaos shaper with specified method.
        
        Args:
            method: Shaping method ('lorenz', 'rossler', 'chua', 'hybrid')
        """
        self.method = method
        self.attractor = self._create_attractor(method)
        self.fractal_wells_enabled = False
        self.hybrid_mode = False
        
        # Chaos shaping history
        self.shaping_history: List[np.ndarray] = []
        
    def _create_attractor(self, method: str) -> StrangeAttractor:
        """Create appropriate attractor based on method."""
        if method == 'lorenz':
            return LorenzAttractor()
        elif method == 'rossler':
            return RosslerAttractor()
        elif method == 'chua':
            return ChuaAttractor()
        else:
            return LorenzAttractor()  # Default
    
    def enable_fractal_wells(self) -> 'ChaosShaper':
        """Enable fractal softplus wells for enhanced shaping."""
        self.fractal_wells_enabled = True
        return self
    
    def shape(self, 
              chaos_signal: Union[np.ndarray, List[float]],
              **kwargs) -> np.ndarray:
        """
        Shape chaotic signal into beautiful patterns.
        
        Args:
            chaos_signal: Input chaotic/noisy signal
            **kwargs: Additional shaping parameters
            
        Returns:
            Beautifully shaped signal
        """
        chaos_signal = np.asarray(chaos_signal, dtype=float)
        
        if self.method == 'hybrid':
            return self._hybrid_shaping(chaos_signal, **kwargs)
        else:
            return self._attractor_shaping(chaos_signal, **kwargs)
    
    def _attractor_shaping(self, 
                          chaos_signal: np.ndarray,
                          **kwargs) -> np.ndarray:
        """Shape signal using strange attractor dynamics."""
        # Use signal statistics to determine initial conditions
        signal_mean = np.mean(chaos_signal)
        signal_std = np.std(chaos_signal)
        signal_range = np.max(chaos_signal) - np.min(chaos_signal)
        
        # Create initial state from signal characteristics
        initial_state = np.array([
            signal_mean,
            signal_std,
            signal_range * 0.1
        ])
        
        # Integrate attractor
        n_points = len(chaos_signal)
        time_span = kwargs.get('time_span', (0, 30))
        
        t, attractor_trajectory = self.attractor.integrate(
            initial_state, time_span, n_points
        )
        
        # Extract relevant component (usually x-component)
        component = kwargs.get('component', 0)
        shaped_signal = attractor_trajectory[:, component]
        
        # Scale and blend with original signal
        blend_factor = kwargs.get('blend_factor', 0.3)
        
        # Normalize shaped signal to match original scale
        shaped_signal = self._normalize_to_match(shaped_signal, chaos_signal)
        
        # Blend original chaos with attractor dynamics
        result = (1 - blend_factor) * chaos_signal + blend_factor * shaped_signal
        
        # Apply fractal wells if enabled
        if self.fractal_wells_enabled:
            result = self._apply_fractal_wells(result)
        
        # Store history
        self.shaping_history.append(result.copy())
        if len(self.shaping_history) > 50:
            self.shaping_history = self.shaping_history[-25:]
        
        return result
    
    def _hybrid_shaping(self, 
                       chaos_signal: np.ndarray,
                       **kwargs) -> np.ndarray:
        """Hybrid shaping using multiple attractors."""
        # Shape with multiple attractors
        lorenz_shaper = ChaosShaper('lorenz')
        rossler_shaper = ChaosShaper('rossler')
        
        lorenz_shaped = lorenz_shaper._attractor_shaping(chaos_signal, **kwargs)
        rossler_shaped = rossler_shaper._attractor_shaping(chaos_signal, **kwargs)
        
        # Adaptive weighting based on signal characteristics
        signal_complexity = self._estimate_complexity(chaos_signal)
        
        if signal_complexity > 0.5:
            # High complexity: favor Lorenz (more chaotic)
            weight_lorenz = 0.7
            weight_rossler = 0.3
        else:
            # Low complexity: favor Rössler (simpler)
            weight_lorenz = 0.3
            weight_rossler = 0.7
        
        result = weight_lorenz * lorenz_shaped + weight_rossler * rossler_shaped
        
        return result
    
    def _apply_fractal_wells(self, signal: np.ndarray) -> np.ndarray:
        """Apply fractal softplus wells for additional smoothing."""
        # Multi-scale softplus application
        result = signal.copy()
        
        scales = [1.0, 0.5, 0.25, 0.125]  # Multiple scales
        weights = [0.4, 0.3, 0.2, 0.1]   # Decreasing weights
        
        for scale, weight in zip(scales, weights):
            beta = 1.0 / scale
            well_contribution = weight * np.log1p(np.exp(beta * signal)) / beta
            result += well_contribution
        
        return result
    
    def _normalize_to_match(self, 
                           source: np.ndarray, 
                           target: np.ndarray) -> np.ndarray:
        """Normalize source signal to match target signal statistics."""
        source_mean = np.mean(source)
        source_std = np.std(source)
        target_mean = np.mean(target)
        target_std = np.std(target)
        
        if source_std == 0:
            return np.full_like(source, target_mean)
        
        # Standardize and rescale
        normalized = (source - source_mean) / source_std
        matched = normalized * target_std + target_mean
        
        return matched
    
    def _estimate_complexity(self, signal: np.ndarray) -> float:
        """Estimate signal complexity using approximate entropy."""
        def _maxdist(xi: np.ndarray, xj: np.ndarray, N: int) -> float:
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m: int) -> float:
            patterns = np.array([signal[i:i + m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)
            
            for i in range(N - m + 1):
                template = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template, patterns[j], m) <= r:
                        C[i] += 1.0
            
            phi = np.mean([np.log(c / (N - m + 1.0)) for c in C if c > 0])
            return phi
        
        N = len(signal)
        if N < 20:
            return 0.5  # Default for short signals
        
        r = 0.2 * np.std(signal)  # Tolerance
        
        try:
            return abs(_phi(2) - _phi(1))
        except:
            return 0.5  # Fallback
    
    def analyze_attractor_properties(self) -> Dict[str, Any]:
        """Analyze properties of the current attractor."""
        if not self.shaping_history:
            return {"status": "no_data"}
        
        latest = self.shaping_history[-1]
        
        # Generate attractor trajectory for analysis
        initial_state = np.array([1.0, 1.0, 1.0])
        t, trajectory = self.attractor.integrate(initial_state, (0, 30), 1000)
        
        # Calculate Lyapunov exponent estimate
        lyapunov = self._estimate_lyapunov(trajectory)
        
        # Calculate correlation dimension estimate
        correlation_dim = self._estimate_correlation_dimension(trajectory)
        
        return {
            "method": self.method,
            "lyapunov_exponent": lyapunov,
            "correlation_dimension": correlation_dim,
            "trajectory_bounds": {
                "x": (np.min(trajectory[:, 0]), np.max(trajectory[:, 0])),
                "y": (np.min(trajectory[:, 1]), np.max(trajectory[:, 1])),
                "z": (np.min(trajectory[:, 2]), np.max(trajectory[:, 2])),
            },
            "fractal_wells_enabled": self.fractal_wells_enabled,
            "shaping_history_length": len(self.shaping_history),
        }
    
    def _estimate_lyapunov(self, trajectory: np.ndarray) -> float:
        """Estimate largest Lyapunov exponent."""
        # Simplified estimation
        n_points = len(trajectory)
        if n_points < 100:
            return 0.0
        
        # Look at trajectory divergence
        divergences = []
        for i in range(n_points - 50):
            current = trajectory[i]
            future = trajectory[i + 1]
            
            # Find nearest neighbor
            distances = np.linalg.norm(trajectory - current, axis=1)
            nearest_idx = np.argsort(distances)[1]  # Exclude self
            
            if nearest_idx + 1 < n_points:
                d0 = distances[nearest_idx]
                d1 = np.linalg.norm(trajectory[nearest_idx + 1] - future)
                
                if d0 > 0 and d1 > 0:
                    divergences.append(np.log(d1 / d0))
        
        return np.mean(divergences) if divergences else 0.0
    
    def _estimate_correlation_dimension(self, trajectory: np.ndarray) -> float:
        """Estimate correlation dimension of the attractor."""
        # Simplified correlation dimension estimation
        n_points = min(len(trajectory), 500)  # Limit for computational efficiency
        sample = trajectory[:n_points]
        
        # Calculate pairwise distances
        distances = []
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = np.linalg.norm(sample[i] - sample[j])
                distances.append(dist)
        
        distances = np.array(distances)
        
        # Estimate dimension using box-counting approach
        if len(distances) == 0:
            return 2.0
        
        min_dist = np.min(distances[distances > 0])
        max_dist = np.max(distances)
        
        # Simple estimation
        log_ratio = np.log(max_dist / min_dist) if min_dist > 0 else 1.0
        estimated_dim = 2.0 + 0.1 * log_ratio  # Rough approximation
        
        return min(estimated_dim, 3.0)  # Cap at 3D
    
    def get_shaping_summary(self) -> Dict[str, Any]:
        """Get summary of chaos shaping configuration."""
        return {
            "method": self.method,
            "attractor_type": type(self.attractor).__name__,
            "fractal_wells_enabled": self.fractal_wells_enabled,
            "hybrid_mode": self.hybrid_mode,
            "shaping_history_length": len(self.shaping_history),
            "attractor_parameters": {
                "sigma": self.attractor.params.sigma if hasattr(self.attractor, 'params') else None,
                "rho": self.attractor.params.rho if hasattr(self.attractor, 'params') else None,
                "beta": self.attractor.params.beta if hasattr(self.attractor, 'params') else None,
            } if hasattr(self.attractor, 'params') else {}
        }