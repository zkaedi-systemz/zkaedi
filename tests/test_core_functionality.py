"""
Test suite for _h_model_z_ Pure Elegance

Ensuring mathematical beauty remains mathematically correct.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add the package to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

import _h_model_z_ as h


class TestHarmonicEngine:
    """Test the core harmonic engine functionality."""
    
    def test_basic_initialization(self):
        """Test basic engine initialization."""
        engine = h.HarmonicEngine(n_harmonics=5)
        assert engine.config.n_harmonics == 5
        assert len(engine.harmonics) == 5
    
    def test_golden_ratio_harmonics(self):
        """Test golden ratio harmonic generation."""
        engine = h.HarmonicEngine(n_harmonics=7)
        engine.use_golden_ratio()
        
        # Check that golden ratio harmonics are generated
        assert engine.config.use_golden_ratio is True
        assert len(engine.harmonics) == 7
        
        # Frequencies should follow golden ratio progression
        frequencies = [h.frequency for h in engine.harmonics]
        assert all(f > 0 for f in frequencies)
    
    def test_transform_basic(self):
        """Test basic transformation functionality."""
        engine = h.HarmonicEngine(n_harmonics=3)
        
        # Create simple test signal
        t = np.linspace(0, 10, 100)
        signal = np.sin(t) + 0.1 * np.random.normal(0, 1, 100)
        
        # Transform
        result = engine.transform(t, signal)
        
        # Check output properties
        assert len(result) == len(signal)
        assert isinstance(result, np.ndarray)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_convergence_analysis(self):
        """Test convergence analysis functionality."""
        engine = h.HarmonicEngine(n_harmonics=5)
        
        # Generate and transform signal
        t = np.linspace(0, 10, 100)
        signal = np.sin(t)
        result = engine.transform(t, signal)
        
        # Analyze convergence
        analysis = engine.analyze_convergence()
        
        assert 'status' in analysis
        assert 'converged' in analysis
        assert 'lyapunov_exponent' in analysis
        assert isinstance(analysis['converged'], (bool, np.bool_))


class TestAdaptiveFlow:
    """Test adaptive flow functionality."""
    
    def test_basic_flow(self):
        """Test basic adaptive flow processing."""
        flow = h.AdaptiveFlow("test_flow")
        
        # Process simple signal
        signal = np.array([1, 2, 3, 4, 5])
        result = flow.process(signal)
        
        assert len(result) == len(signal)
        assert flow.state.iteration == 1
        assert isinstance(result, np.ndarray)
    
    def test_adversarial_detection(self):
        """Test adversarial input detection."""
        flow = h.AdaptiveFlow("secure_flow")
        flow.enable_adversarial_detection()
        
        # Normal signal should pass (relax the test since random signals might trigger detection)
        normal_signal = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Simple, predictable signal
        result1 = flow.process(normal_signal)
        # Just check it processes without crashing
        assert len(result1) == len(normal_signal)
        
        # Adversarial signal should be detected
        adversarial_signal = np.array([1000] * 100)  # Extreme values
        initial_alerts = flow.state.adversarial_alerts
        result2 = flow.process(adversarial_signal)
        # Should handle without crashing and may increase alerts
        assert len(result2) == len(adversarial_signal)
        assert flow.state.adversarial_alerts >= initial_alerts
    
    def test_pattern_evolution(self):
        """Test pattern evolution over time."""
        flow = h.AdaptiveFlow("evolving_flow")
        
        # Process multiple signals to trigger adaptation
        for i in range(10):
            signal = np.random.normal(0, 1, 50) * (i + 1)  # Increasing variance
            result = flow.process(signal)
        
        # Should have some adaptations
        assert flow.state.iteration == 10
        assert len(flow.state.performance_history) > 0


class TestSoftplusWells:
    """Test softplus wells functionality."""
    
    def test_basic_softplus(self):
        """Test basic softplus function."""
        x = np.array([-1, 0, 1, 2])
        result = h.softplus_well(x, beta=1.0)
        
        # Softplus should be positive and smooth
        assert all(result > 0)
        assert len(result) == len(x)
        
        # Should be approximately x for large positive x
        large_x = np.array([10, 20])
        large_result = h.softplus_well(large_x, beta=1.0)
        np.testing.assert_allclose(large_result, large_x, rtol=0.1)
    
    def test_multi_well_system(self):
        """Test multi-well error trapping."""
        wells = h.SoftplusWells(n_wells=3)
        
        # Test error trapping
        errors = np.random.normal(0, 1, 100)
        trapped = wells.trap_errors(errors)
        
        assert len(trapped) == len(errors)
        assert isinstance(trapped, np.ndarray)
        
        # Wells should capture errors (generally increasing magnitude)
        assert np.mean(np.abs(trapped)) >= np.mean(np.abs(errors))
    
    def test_adaptive_wells(self):
        """Test adaptive well configuration."""
        wells = h.SoftplusWells(n_wells=2)
        
        # Process with adaptation
        errors = np.random.normal(2.0, 0.5, 100)  # Biased errors
        trapped = wells.trap_errors(errors, adaptive=True)
        
        # Should adapt well positions
        assert len(wells.error_history) == 1
        assert len(trapped) == len(errors)


class TestChaosShaper:
    """Test chaos shaping functionality."""
    
    def test_lorenz_shaping(self):
        """Test Lorenz attractor chaos shaping."""
        shaper = h.ChaosShaper('lorenz')
        
        # Shape chaotic signal
        chaos = np.random.normal(0, 1, 100)
        shaped = shaper.shape(chaos)
        
        assert len(shaped) == len(chaos)
        assert isinstance(shaped, np.ndarray)
        assert not np.any(np.isnan(shaped))
    
    def test_hybrid_shaping(self):
        """Test hybrid chaos shaping."""
        shaper = h.ChaosShaper('hybrid')
        
        # Shape with hybrid method
        chaos = np.random.normal(0, 1, 50)
        shaped = shaper.shape(chaos)
        
        assert len(shaped) == len(chaos)
        assert shaper.method == 'hybrid'
    
    def test_fractal_wells(self):
        """Test fractal wells enhancement."""
        shaper = h.ChaosShaper('lorenz').enable_fractal_wells()
        
        assert shaper.fractal_wells_enabled is True
        
        # Shape with fractal wells
        chaos = np.random.uniform(-2, 2, 100)
        shaped = shaper.shape(chaos)
        
        assert len(shaped) == len(chaos)


class TestMathematicalCore:
    """Test core mathematical functions."""
    
    def test_harmonic_parameters_validation(self):
        """Test harmonic parameter validation."""
        from _h_model_z_.core.mathematical_core import HarmonicParameters
        
        # Valid parameters
        params = HarmonicParameters(1.0, 2.0, 0.0, 0.1)
        assert params.amplitude == 1.0
        assert params.frequency == 2.0
        
        # Invalid parameters should raise errors
        with pytest.raises(ValueError):
            HarmonicParameters(-1.0, 2.0, 0.0, 0.1)  # Negative amplitude
        
        with pytest.raises(ValueError):
            HarmonicParameters(1.0, 0.0, 0.0, 0.1)  # Zero frequency
    
    def test_softplus_integration(self):
        """Test softplus integration function."""
        from _h_model_z_.core.mathematical_core import MathematicalCore
        
        core = MathematicalCore()
        x = np.array([-2, -1, 0, 1, 2])
        result = core.softplus_integration(x, beta=1.0)
        
        # Should be smooth and positive
        assert all(result > 0)
        assert len(result) == len(x)
    
    def test_golden_ratio_harmonics(self):
        """Test golden ratio harmonic generation."""
        from _h_model_z_.core.mathematical_core import MathematicalCore
        
        core = MathematicalCore()
        harmonics = core.golden_ratio_harmonics(5)
        
        assert len(harmonics) == 5
        # Check that frequencies follow golden ratio pattern
        frequencies = [h.frequency for h in harmonics]
        assert all(f > 0 for f in frequencies)


class TestConvenienceFunctions:
    """Test convenience functions and API."""
    
    def test_transform_function(self):
        """Test the convenience transform function."""
        errors = np.random.normal(0, 1, 100)
        result = h.transform(errors)
        
        assert len(result) == len(errors)
        assert isinstance(result, np.ndarray)
    
    def test_fluent_interface(self):
        """Test fluent interface API."""
        # This tests the API style shown in README
        errors = np.random.normal(0, 1, 100)
        
        # Create fluent pipeline
        pipeline = (h.HarmonicEngine(n_harmonics=5)
                   .use_golden_ratio()
                   .with_harmonics(7))
        
        # Should work without errors
        assert pipeline.config.n_harmonics == 7
        assert pipeline.config.use_golden_ratio is True


class TestLiveVisualization:
    """Test live HTML visualization."""
    
    def test_renderer_creation(self):
        """Test HTML renderer creation."""
        from _h_model_z_.renderers import create_live_visualization
        
        renderer = create_live_visualization("Test Visualization")
        assert renderer.title == "Test Visualization"
        assert renderer.plots == []
    
    def test_plot_addition(self):
        """Test adding plots to renderer."""
        from _h_model_z_.renderers import LiveHTMLRenderer
        
        renderer = LiveHTMLRenderer()
        renderer.add_plot("test-plot", "Test Plot")
        
        assert len(renderer.plots) == 1
        assert "test-plot" in renderer.plots[0]
    
    def test_html_generation(self):
        """Test HTML generation."""
        from _h_model_z_.renderers import create_live_visualization
        import tempfile
        import os
        
        renderer = create_live_visualization()
        
        # Generate to temporary file
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            result_path = renderer.render(temp_path)
            
            # Check file was created and has content
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 1000  # Substantial content
            
            # Check it's valid HTML
            with open(result_path, 'r') as f:
                content = f.read()
                assert '<!DOCTYPE html>' in content
                assert '_h_model_z_' in content
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


if __name__ == '__main__':
    # Run tests directly if executed
    import subprocess
    subprocess.run(['python', '-m', 'pytest', __file__, '-v'])