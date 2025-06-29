"""
Softplus Wells - Error Trapping in Smooth Potential Landscapes

The softplus function creates smooth, differentiable wells that trap high-energy errors
without the harsh discontinuities of traditional approaches. Mathematical elegance
in error handling.

∫ softplus(β * x) dx = (1/β) * log(1 + e^(β * x))
"""

import numpy as np
from typing import Union, List, Optional, Dict, Any, Callable
from dataclasses import dataclass
import scipy.optimize as optimize
import matplotlib.pyplot as plt


def softplus_well(x: Union[np.ndarray, float], 
                  beta: float = 1.0) -> Union[np.ndarray, float]:
    """
    The fundamental softplus well function.
    
    Creates smooth potential wells that trap errors without discontinuities.
    The mathematical beauty lies in its infinite differentiability.
    
    Args:
        x: Input values (errors to be trapped)
        beta: Well steepness parameter (higher = steeper walls)
        
    Returns:
        Softplus-transformed values
    """
    # Numerically stable implementation
    beta_x = beta * x
    
    # For large positive values, use linear approximation to prevent overflow
    return np.where(
        beta_x > 700,
        x + np.log(beta) / beta,  # Linear approximation
        np.log1p(np.exp(beta_x)) / beta  # Standard softplus
    )


def softplus_derivative(x: Union[np.ndarray, float], 
                       beta: float = 1.0) -> Union[np.ndarray, float]:
    """
    Derivative of softplus function: σ(β * x) = 1 / (1 + e^(-β * x))
    
    This is the sigmoid function - the gentle slope of the well walls.
    """
    return 1.0 / (1.0 + np.exp(-beta * x))


def softplus_inverse(y: Union[np.ndarray, float], 
                    beta: float = 1.0) -> Union[np.ndarray, float]:
    """
    Inverse softplus function for exact error recovery when needed.
    
    softplus_inverse(softplus(x)) = x
    """
    return np.log(np.expm1(beta * y)) / beta


@dataclass
class WellParameters:
    """Parameters defining a single softplus well."""
    center: float  # Well center position
    depth: float   # Well depth (beta parameter)
    width: float   # Well width scaling
    strength: float  # Overall well strength
    
    def __post_init__(self):
        """Validate well parameters."""
        if self.depth <= 0:
            raise ValueError("Well depth must be positive")
        if self.width <= 0:
            raise ValueError("Well width must be positive")
        if self.strength < 0:
            raise ValueError("Well strength must be non-negative")


class SoftplusWells:
    """
    Multi-well softplus system for comprehensive error trapping.
    
    Creates a landscape of smooth potential wells, each designed to capture
    specific types of errors. The system adapts the well configuration
    based on the error distribution.
    """
    
    def __init__(self, n_wells: int = 3):
        """
        Initialize softplus well system.
        
        Args:
            n_wells: Number of wells in the system
        """
        self.n_wells = n_wells
        self.wells: List[WellParameters] = []
        self.adaptive_mode = False
        self.error_history: List[np.ndarray] = []
        
        # Initialize default well configuration
        self._initialize_wells()
    
    def _initialize_wells(self):
        """Initialize wells with sensible defaults."""
        self.wells = []
        
        # Create wells at different scales for multi-scale error capture
        for i in range(self.n_wells):
            center = -2.0 + i * 2.0  # Spread wells across error space
            depth = 1.0 + i * 0.5    # Increasing depth for larger errors
            width = 1.0              # Standard width
            strength = 1.0 / (i + 1) # Decreasing strength for higher order wells
            
            self.wells.append(WellParameters(center, depth, width, strength))
    
    def trap_errors(self, 
                   errors: Union[np.ndarray, List[float]],
                   adaptive: bool = False) -> np.ndarray:
        """
        Trap errors in the multi-well potential landscape.
        
        Args:
            errors: Input error signal
            adaptive: Whether to adapt wells to error distribution
            
        Returns:
            Well-trapped error signal
        """
        errors = np.asarray(errors, dtype=float)
        
        if adaptive:
            self._adapt_wells_to_errors(errors)
        
        # Store error history for adaptation
        self.error_history.append(errors.copy())
        if len(self.error_history) > 50:
            self.error_history = self.error_history[-25:]
        
        trapped = np.zeros_like(errors)
        
        # Apply each well to the errors
        for well in self.wells:
            # Shift errors relative to well center
            shifted_errors = (errors - well.center) / well.width
            
            # Apply softplus well
            well_contribution = well.strength * softplus_well(shifted_errors, well.depth)
            
            trapped += well_contribution
        
        return trapped
    
    def _adapt_wells_to_errors(self, errors: np.ndarray):
        """
        Adapt well configuration based on error distribution.
        
        This is where the system becomes intelligent - wells move and reshape
        to better capture the specific error patterns they encounter.
        """
        # Analyze error distribution
        error_mean = np.mean(errors)
        error_std = np.std(errors)
        error_range = np.max(errors) - np.min(errors)
        
        # Find error clusters using simple binning
        n_bins = min(self.n_wells * 2, 10)
        hist, bin_edges = np.histogram(errors, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find the most significant error concentrations
        significant_bins = np.argsort(hist)[-self.n_wells:]
        
        # Adapt wells to error concentrations
        for i, (well, bin_idx) in enumerate(zip(self.wells, significant_bins)):
            well.center = bin_centers[bin_idx]
            well.depth = 1.0 + hist[bin_idx] / np.max(hist)  # Deeper for higher concentrations
            well.width = error_std * 0.5  # Width based on error spread
            well.strength = hist[bin_idx] / np.sum(hist)  # Strength proportional to error density
    
    def potential_landscape(self, 
                          x_range: tuple = (-5, 5), 
                          n_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the complete potential landscape.
        
        Args:
            x_range: Range of x values to evaluate
            n_points: Number of evaluation points
            
        Returns:
            Tuple of (x_values, potential_values)
        """
        x = np.linspace(x_range[0], x_range[1], n_points)
        potential = self.trap_errors(x)
        return x, potential
    
    def well_capacity(self, well_idx: int, error_range: tuple = (-10, 10)) -> float:
        """
        Calculate the capacity of a specific well.
        
        Args:
            well_idx: Index of the well
            error_range: Range of errors to consider
            
        Returns:
            Well capacity (area under the well curve)
        """
        if well_idx >= len(self.wells):
            raise IndexError(f"Well index {well_idx} out of range")
        
        well = self.wells[well_idx]
        x = np.linspace(error_range[0], error_range[1], 1000)
        shifted_x = (x - well.center) / well.width
        well_values = well.strength * softplus_well(shifted_x, well.depth)
        
        # Integrate using trapezoidal rule
        return np.trapz(well_values, x)
    
    def optimize_wells(self, 
                      target_errors: np.ndarray,
                      method: str = "minimize_variance") -> Dict[str, Any]:
        """
        Optimize well configuration for specific error patterns.
        
        Args:
            target_errors: Target error distribution to optimize for
            method: Optimization method ("minimize_variance", "maximize_capture")
            
        Returns:
            Optimization results
        """
        target_errors = np.asarray(target_errors)
        
        def objective_function(params):
            """Objective function for well optimization."""
            # Reshape parameters into well configuration
            n_params_per_well = 4  # center, depth, width, strength
            wells = []
            
            for i in range(self.n_wells):
                start_idx = i * n_params_per_well
                center = params[start_idx]
                depth = max(params[start_idx + 1], 0.1)  # Ensure positive
                width = max(params[start_idx + 2], 0.1)  # Ensure positive
                strength = max(params[start_idx + 3], 0.0)  # Ensure non-negative
                
                wells.append(WellParameters(center, depth, width, strength))
            
            # Temporarily set wells and evaluate
            old_wells = self.wells.copy()
            self.wells = wells
            
            trapped = self.trap_errors(target_errors)
            
            if method == "minimize_variance":
                objective = np.var(trapped)
            elif method == "maximize_capture":
                objective = -np.sum(trapped)  # Negative for maximization
            else:
                objective = np.var(trapped)
            
            # Restore original wells
            self.wells = old_wells
            
            return objective
        
        # Initial parameters from current well configuration
        initial_params = []
        for well in self.wells:
            initial_params.extend([well.center, well.depth, well.width, well.strength])
        
        # Optimize
        result = optimize.minimize(
            objective_function,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        # Update wells with optimized parameters if successful
        if result.success:
            optimized_params = result.x
            new_wells = []
            
            for i in range(self.n_wells):
                start_idx = i * 4
                center = optimized_params[start_idx]
                depth = max(optimized_params[start_idx + 1], 0.1)
                width = max(optimized_params[start_idx + 2], 0.1)
                strength = max(optimized_params[start_idx + 3], 0.0)
                
                new_wells.append(WellParameters(center, depth, width, strength))
            
            self.wells = new_wells
        
        return {
            "success": result.success,
            "objective_value": result.fun,
            "n_iterations": result.nit,
            "message": result.message,
            "optimized_wells": [
                {
                    "center": well.center,
                    "depth": well.depth,
                    "width": well.width,
                    "strength": well.strength,
                }
                for well in self.wells
            ]
        }
    
    def visualize_wells(self, 
                       x_range: tuple = (-5, 5),
                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create visualization of the well landscape.
        
        Args:
            x_range: Range of x values to plot
            save_path: Optional path to save the plot
            
        Returns:
            Plotting information
        """
        x, potential = self.potential_landscape(x_range)
        
        plt.figure(figsize=(12, 8))
        
        # Plot overall potential landscape
        plt.subplot(2, 1, 1)
        plt.plot(x, potential, 'b-', linewidth=2, label='Total Potential')
        
        # Plot individual wells
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        for i, well in enumerate(self.wells):
            shifted_x = (x - well.center) / well.width
            individual_well = well.strength * softplus_well(shifted_x, well.depth)
            color = colors[i % len(colors)]
            plt.plot(x, individual_well, '--', color=color, alpha=0.7, 
                    label=f'Well {i+1} (center={well.center:.2f})')
        
        plt.xlabel('Error Value')
        plt.ylabel('Potential')
        plt.title('Softplus Well Landscape')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot well derivatives (forces)
        plt.subplot(2, 1, 2)
        potential_derivative = np.gradient(potential, x[1] - x[0])
        plt.plot(x, -potential_derivative, 'r-', linewidth=2, label='Force (negative gradient)')
        plt.xlabel('Error Value')
        plt.ylabel('Force')
        plt.title('Well Forces (Error Trapping Strength)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return {
            "x_range": x_range,
            "n_points": len(x),
            "potential_range": (np.min(potential), np.max(potential)),
            "max_force": np.max(np.abs(potential_derivative)),
            "well_centers": [well.center for well in self.wells],
        }
    
    def get_well_summary(self) -> Dict[str, Any]:
        """Get summary of current well configuration."""
        return {
            "n_wells": len(self.wells),
            "adaptive_mode": self.adaptive_mode,
            "wells": [
                {
                    "center": well.center,
                    "depth": well.depth,
                    "width": well.width,
                    "strength": well.strength,
                    "capacity": self.well_capacity(i),
                }
                for i, well in enumerate(self.wells)
            ],
            "total_capacity": sum(self.well_capacity(i) for i in range(len(self.wells))),
            "error_history_length": len(self.error_history),
        }


# Convenience functions for elegant API
def multi_well_trap(errors: Union[np.ndarray, List[float]], 
                   n_wells: int = 3,
                   adaptive: bool = False) -> np.ndarray:
    """
    Convenience function for multi-well error trapping.
    
    Args:
        errors: Input errors
        n_wells: Number of wells
        adaptive: Whether to use adaptive wells
        
    Returns:
        Trapped errors
    """
    wells = SoftplusWells(n_wells)
    return wells.trap_errors(errors, adaptive)


def adaptive_softplus_well(errors: Union[np.ndarray, List[float]]) -> np.ndarray:
    """
    Adaptive single-well softplus trapping.
    
    Automatically adjusts beta parameter based on error distribution.
    """
    errors = np.asarray(errors)
    
    # Adaptive beta based on error variance
    error_std = np.std(errors)
    beta = 1.0 / max(error_std, 0.1)  # Inverse relationship with spread
    
    return softplus_well(errors, beta)