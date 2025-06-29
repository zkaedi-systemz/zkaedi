"""
Adaptive Flow - Self-Modifying Code that Rewrites on Reflex

Polymorphic, holomorphic, chameleon. Resumes mid-def, rewrites on reflex.
This is where code becomes liquid, where algorithms evolve in real-time.
"""

import numpy as np
from typing import Union, List, Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect
import copy
import hashlib
import time


@dataclass
class FlowState:
    """State of an adaptive flow process."""
    name: str
    iteration: int = 0
    pattern_hash: str = ""
    last_error: float = 0.0
    adaptation_count: int = 0
    trust_level: float = 1.0
    adversarial_alerts: int = 0
    performance_history: List[float] = field(default_factory=list)
    pattern_evolution: List[str] = field(default_factory=list)


class Pattern(ABC):
    """Abstract base class for adaptive patterns."""
    
    @abstractmethod
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply pattern transformation to data."""
        pass
    
    @abstractmethod
    def evolve(self, error_signal: float, performance: float) -> 'Pattern':
        """Evolve pattern based on feedback."""
        pass
    
    @abstractmethod
    def get_signature(self) -> str:
        """Get unique signature for pattern identification."""
        pass


class LinearPattern(Pattern):
    """Linear transformation pattern."""
    
    def __init__(self, weight: float = 1.0, bias: float = 0.0):
        self.weight = weight
        self.bias = bias
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply linear transformation."""
        return self.weight * data + self.bias
    
    def evolve(self, error_signal: float, performance: float) -> 'LinearPattern':
        """Evolve linear pattern based on performance."""
        # Adaptive learning rate
        learning_rate = 0.01 * (1.0 / (1.0 + abs(error_signal)))
        
        # Adjust weight based on error
        weight_adjustment = -learning_rate * error_signal
        new_weight = self.weight + weight_adjustment
        
        # Bias adjustment for centering
        bias_adjustment = learning_rate * error_signal * 0.1
        new_bias = self.bias + bias_adjustment
        
        return LinearPattern(new_weight, new_bias)
    
    def get_signature(self) -> str:
        """Get pattern signature."""
        return f"linear_w{self.weight:.3f}_b{self.bias:.3f}"


class HolomorphicPattern(Pattern):
    """Complex-analytic pattern maintaining holomorphic properties."""
    
    def __init__(self, alpha: complex = 1.0+0j, beta: complex = 0.0+0j):
        self.alpha = alpha
        self.beta = beta
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply holomorphic transformation."""
        # Convert to complex representation
        complex_data = data.astype(complex)
        
        # Holomorphic transformation: f(z) = α*z + β
        transformed = self.alpha * complex_data + self.beta
        
        # Return real part for real-valued output
        return np.real(transformed)
    
    def evolve(self, error_signal: float, performance: float) -> 'HolomorphicPattern':
        """Evolve holomorphic pattern."""
        learning_rate = 0.01 * performance
        
        # Complex gradient descent
        alpha_real_adj = -learning_rate * error_signal
        alpha_imag_adj = learning_rate * error_signal * 0.1
        
        new_alpha = self.alpha + complex(alpha_real_adj, alpha_imag_adj)
        new_beta = self.beta + complex(learning_rate * error_signal * 0.05, 0)
        
        return HolomorphicPattern(new_alpha, new_beta)
    
    def get_signature(self) -> str:
        """Get pattern signature."""
        return f"holomorphic_a{abs(self.alpha):.3f}_b{abs(self.beta):.3f}"


class SoftplusPattern(Pattern):
    """Softplus-based nonlinear pattern."""
    
    def __init__(self, beta: float = 1.0, shift: float = 0.0):
        self.beta = beta
        self.shift = shift
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply softplus transformation."""
        shifted_data = data - self.shift
        return np.log1p(np.exp(self.beta * shifted_data)) / self.beta
    
    def evolve(self, error_signal: float, performance: float) -> 'SoftplusPattern':
        """Evolve softplus pattern."""
        learning_rate = 0.01
        
        # Adjust beta (steepness)
        beta_adj = learning_rate * error_signal * performance
        new_beta = max(0.1, self.beta - beta_adj)
        
        # Adjust shift (center)
        shift_adj = learning_rate * error_signal
        new_shift = self.shift + shift_adj
        
        return SoftplusPattern(new_beta, new_shift)
    
    def get_signature(self) -> str:
        """Get pattern signature."""
        return f"softplus_b{self.beta:.3f}_s{self.shift:.3f}"


class AdaptiveFlow:
    """
    Self-modifying algorithmic flow that adapts in real-time.
    
    This is where the magic happens - code that rewrites itself,
    patterns that evolve, algorithms that learn from their mistakes.
    """
    
    def __init__(self, name: str = "main_flow"):
        """
        Initialize adaptive flow.
        
        Args:
            name: Flow identifier
        """
        self.state = FlowState(name)
        self.current_pattern: Pattern = LinearPattern()
        self.pattern_library: List[Pattern] = [
            LinearPattern(),
            HolomorphicPattern(),
            SoftplusPattern(),
        ]
        
        # Adaptation settings
        self.adaptation_threshold = 0.1
        self.performance_window = 10
        self.trust_decay_rate = 0.01
        
        # Security features
        self.adversarial_detection_enabled = False
        self.holomorphic_trust_enabled = False
        self.max_adaptations_per_session = 100
        
        # Evolution tracking
        self.pattern_genealogy: Dict[str, List[str]] = {}
        
    def enable_adversarial_detection(self) -> 'AdaptiveFlow':
        """Enable adversarial input detection."""
        self.adversarial_detection_enabled = True
        return self
    
    def enable_holomorphic_trust(self) -> 'AdaptiveFlow':
        """Enable holomorphic trust verification."""
        self.holomorphic_trust_enabled = True
        return self
    
    def process(self, data: Union[np.ndarray, List[float]]) -> np.ndarray:
        """
        Process data through adaptive flow.
        
        This is where the self-modification happens - the flow analyzes
        its own performance and evolves its processing pattern.
        """
        data = np.asarray(data, dtype=float)
        
        # Security checks
        if self.adversarial_detection_enabled:
            if self._detect_adversarial_input(data):
                return self._handle_adversarial_input(data)
        
        # Apply current pattern
        start_time = time.time()
        result = self.current_pattern.apply(data)
        processing_time = time.time() - start_time
        
        # Calculate performance metrics
        performance = self._calculate_performance(data, result, processing_time)
        error_signal = self._calculate_error_signal(data, result)
        
        # Update state
        self.state.iteration += 1
        self.state.last_error = error_signal
        self.state.performance_history.append(performance)
        
        # Trim performance history
        if len(self.state.performance_history) > self.performance_window:
            self.state.performance_history = self.state.performance_history[-self.performance_window:]
        
        # Decide if adaptation is needed
        if self._should_adapt(error_signal, performance):
            self._adapt_pattern(error_signal, performance)
        
        # Update trust level
        self._update_trust_level(performance)
        
        # Holomorphic trust verification
        if self.holomorphic_trust_enabled:
            trust_verified = self._verify_holomorphic_trust(data, result)
            if not trust_verified:
                self.state.trust_level *= 0.9  # Reduce trust
        
        return result
    
    def flow(self, data: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Alias for process method to match API in README examples."""
        return self.process(data)
    
    def _detect_adversarial_input(self, data: np.ndarray) -> bool:
        """Detect potentially adversarial inputs."""
        # Simple adversarial detection heuristics
        data_std = np.std(data)
        data_mean = np.mean(data)
        data_range = np.max(data) - np.min(data)
        
        # Check for suspicious patterns
        if data_std > 100 * abs(data_mean):  # Extremely high variance
            return True
        
        if data_range > 1000:  # Extremely large range
            return True
        
        # Check for repeating patterns (potential injection)
        if len(data) > 10:
            autocorr = np.correlate(data, data, mode='full')
            autocorr_normalized = autocorr / np.max(autocorr)
            max_autocorr = np.max(autocorr_normalized[len(autocorr_normalized)//2+1:])
            if max_autocorr > 0.99:  # Nearly perfect repetition
                return True
        
        return False
    
    def _handle_adversarial_input(self, data: np.ndarray) -> np.ndarray:
        """Handle detected adversarial input."""
        self.state.adversarial_alerts += 1
        self.state.trust_level *= 0.5  # Significantly reduce trust
        
        # Apply defensive transformation
        clipped_data = np.clip(data, -10, 10)  # Clip extreme values
        smoothed_data = np.convolve(clipped_data, np.ones(3)/3, mode='same')  # Smooth
        
        return self.current_pattern.apply(smoothed_data)
    
    def _calculate_performance(self, 
                              input_data: np.ndarray, 
                              output_data: np.ndarray, 
                              processing_time: float) -> float:
        """Calculate overall performance score."""
        # Performance based on multiple factors
        
        # Stability score (lower variance is better)
        stability = 1.0 / (1.0 + np.var(output_data))
        
        # Efficiency score (faster processing is better)
        efficiency = 1.0 / (1.0 + processing_time * 1000)  # Penalize slow processing
        
        # Signal preservation score
        input_energy = np.sum(input_data ** 2)
        output_energy = np.sum(output_data ** 2)
        energy_ratio = min(output_energy / max(input_energy, 1e-10), 1.0)
        
        # Smoothness score (penalize sudden changes)
        if len(output_data) > 1:
            smoothness = 1.0 / (1.0 + np.sum(np.diff(output_data) ** 2))
        else:
            smoothness = 1.0
        
        # Weighted combination
        performance = 0.3 * stability + 0.2 * efficiency + 0.3 * energy_ratio + 0.2 * smoothness
        
        return performance
    
    def _calculate_error_signal(self, 
                               input_data: np.ndarray, 
                               output_data: np.ndarray) -> float:
        """Calculate error signal for adaptation."""
        # Mean squared error between input and output
        mse = np.mean((input_data - output_data) ** 2)
        
        # Add penalty for extreme outputs
        extreme_penalty = np.sum(np.abs(output_data) > 10) * 0.1
        
        return mse + extreme_penalty
    
    def _should_adapt(self, error_signal: float, performance: float) -> bool:
        """Decide whether pattern adaptation is needed."""
        # Don't adapt too frequently
        if self.state.adaptation_count >= self.max_adaptations_per_session:
            return False
        
        # Adapt if error is high
        if error_signal > self.adaptation_threshold:
            return True
        
        # Adapt if performance is declining
        if len(self.state.performance_history) >= 3:
            recent_performance = np.mean(self.state.performance_history[-3:])
            if recent_performance < 0.5:  # Poor performance
                return True
        
        # Random adaptation for exploration (with low probability)
        if np.random.random() < 0.01 * self.state.trust_level:  # Trust-modulated exploration
            return True
        
        return False
    
    def _adapt_pattern(self, error_signal: float, performance: float):
        """Adapt the current pattern or select a new one."""
        self.state.adaptation_count += 1
        
        # Try evolving current pattern
        evolved_pattern = self.current_pattern.evolve(error_signal, performance)
        
        # Decide between evolution and pattern switching
        if performance < 0.3:  # Very poor performance - try different pattern
            self._switch_pattern(error_signal, performance)
        else:  # Moderate performance - evolve current pattern
            old_signature = self.current_pattern.get_signature()
            self.current_pattern = evolved_pattern
            new_signature = self.current_pattern.get_signature()
            
            # Track evolution
            if old_signature not in self.pattern_genealogy:
                self.pattern_genealogy[old_signature] = []
            self.pattern_genealogy[old_signature].append(new_signature)
            
            self.state.pattern_evolution.append(f"evolved: {old_signature} -> {new_signature}")
    
    def _switch_pattern(self, error_signal: float, performance: float):
        """Switch to a different pattern type."""
        # Score all patterns in library
        pattern_scores = []
        current_type = type(self.current_pattern)
        
        for pattern in self.pattern_library:
            if type(pattern) == current_type:
                continue  # Skip current pattern type
            
            # Simple scoring based on error and performance
            score = 1.0 / (1.0 + error_signal) + performance
            pattern_scores.append((pattern, score))
        
        if pattern_scores:
            # Select best scoring pattern
            best_pattern, best_score = max(pattern_scores, key=lambda x: x[1])
            
            old_signature = self.current_pattern.get_signature()
            self.current_pattern = copy.deepcopy(best_pattern)
            new_signature = self.current_pattern.get_signature()
            
            self.state.pattern_evolution.append(f"switched: {old_signature} -> {new_signature}")
    
    def _update_trust_level(self, performance: float):
        """Update trust level based on performance."""
        if performance > 0.8:
            # Good performance increases trust
            self.state.trust_level = min(1.0, self.state.trust_level + 0.01)
        elif performance < 0.3:
            # Poor performance decreases trust
            self.state.trust_level *= (1.0 - self.trust_decay_rate)
        
        # Natural trust decay over time
        self.state.trust_level *= (1.0 - self.trust_decay_rate * 0.1)
        
        # Ensure trust doesn't go below minimum
        self.state.trust_level = max(0.1, self.state.trust_level)
    
    def _verify_holomorphic_trust(self, 
                                 input_data: np.ndarray, 
                                 output_data: np.ndarray) -> bool:
        """Verify holomorphic properties for trust validation."""
        if not isinstance(self.current_pattern, HolomorphicPattern):
            return True  # Only verify holomorphic patterns
        
        # Check if transformation preserves complex differentiability
        # Simplified check: verify Cauchy-Riemann equations approximately
        
        if len(input_data) < 4:
            return True  # Too small for verification
        
        # Create complex representation
        z = input_data[::2] + 1j * input_data[1::2] if len(input_data) % 2 == 0 else input_data[:-1:2] + 1j * input_data[1::2]
        w = output_data[::2] + 1j * output_data[1::2] if len(output_data) % 2 == 0 else output_data[:-1:2] + 1j * output_data[1::2]
        
        if len(z) < 2 or len(w) < 2:
            return True
        
        # Check approximate derivative consistency
        dz = np.diff(z)
        dw = np.diff(w)
        
        # Avoid division by zero
        valid_indices = np.abs(dz) > 1e-10
        if not np.any(valid_indices):
            return True
        
        derivatives = dw[valid_indices] / dz[valid_indices]
        
        # Check if derivatives are approximately consistent (holomorphic property)
        if len(derivatives) > 1:
            derivative_std = np.std(derivatives)
            derivative_mean = np.mean(np.abs(derivatives))
            
            # Consistent derivatives indicate holomorphic behavior
            consistency = derivative_std / max(derivative_mean, 1e-10)
            return consistency < 1.0  # Reasonable consistency threshold
        
        return True
    
    def get_flow_state(self) -> Dict[str, Any]:
        """Get current flow state information."""
        return {
            "name": self.state.name,
            "iteration": self.state.iteration,
            "adaptation_count": self.state.adaptation_count,
            "trust_level": self.state.trust_level,
            "adversarial_alerts": self.state.adversarial_alerts,
            "current_pattern": {
                "type": type(self.current_pattern).__name__,
                "signature": self.current_pattern.get_signature(),
            },
            "performance_stats": {
                "mean": np.mean(self.state.performance_history) if self.state.performance_history else 0.0,
                "std": np.std(self.state.performance_history) if self.state.performance_history else 0.0,
                "trend": self._calculate_performance_trend(),
            },
            "security": {
                "adversarial_detection": self.adversarial_detection_enabled,
                "holomorphic_trust": self.holomorphic_trust_enabled,
            },
            "evolution_history": self.state.pattern_evolution[-10:],  # Last 10 evolutions
        }
    
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend (improving/declining/stable)."""
        if len(self.state.performance_history) < 3:
            return "insufficient_data"
        
        recent = self.state.performance_history[-3:]
        older = self.state.performance_history[-6:-3] if len(self.state.performance_history) >= 6 else self.state.performance_history[:-3]
        
        if not older:
            return "insufficient_data"
        
        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        
        if recent_mean > older_mean + 0.05:
            return "improving"
        elif recent_mean < older_mean - 0.05:
            return "declining"
        else:
            return "stable"
    
    def reset_adaptation(self):
        """Reset adaptation counters and state."""
        self.state.adaptation_count = 0
        self.state.adversarial_alerts = 0
        self.state.trust_level = 1.0
        self.state.performance_history = []
        self.state.pattern_evolution = []
    
    def export_pattern_genealogy(self) -> Dict[str, Any]:
        """Export pattern evolution genealogy."""
        return {
            "genealogy": self.pattern_genealogy,
            "current_lineage": self._trace_current_lineage(),
            "evolution_count": len(self.state.pattern_evolution),
            "pattern_diversity": len(set(evo.split(" -> ")[1] for evo in self.state.pattern_evolution if " -> " in evo)),
        }
    
    def _trace_current_lineage(self) -> List[str]:
        """Trace the lineage of the current pattern."""
        current_sig = self.current_pattern.get_signature()
        lineage = [current_sig]
        
        # Trace backwards through genealogy
        for parent, children in self.pattern_genealogy.items():
            if current_sig in children:
                lineage.insert(0, parent)
                current_sig = parent
        
        return lineage