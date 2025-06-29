"""
Live HTML Renderer - Interactive Visualization of Pure Elegance

Creates beautiful, interactive HTML visualizations of harmonic transformations.
Real-time updates, responsive design, cyberpunk aesthetics.
"""

import numpy as np
from typing import Union, List, Optional, Dict, Any
import json
from pathlib import Path
import datetime


class LiveHTMLRenderer:
    """
    Interactive HTML renderer for harmonic transformations.
    
    Creates stunning visualizations with:
    - Real-time signal plots using Plotly
    - 3D phase space rendering
    - Cyberpunk color scheme
    - Responsive design
    """
    
    def __init__(self, title: str = "_h_model_z_ Live Visualization"):
        """Initialize HTML renderer."""
        self.title = title
        self.plots = []
        self.data_sets = {}
        self.template = self._create_base_template()
        
    def _create_base_template(self) -> str:
        """Create base HTML template with cyberpunk styling."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 50%, #16213e 100%);
            color: #06ffa5;
            font-family: 'Courier New', monospace;
            min-height: 100vh;
            padding: 20px;
            overflow-x: hidden;
        }}
        
        .header {{
            text-align: center;
            padding: 40px 0;
            background: rgba(6, 255, 165, 0.1);
            border-radius: 20px;
            margin-bottom: 30px;
            border: 2px solid rgba(6, 255, 165, 0.3);
            box-shadow: 0 0 30px rgba(6, 255, 165, 0.2);
        }}
        
        .header h1 {{
            font-size: 3rem;
            background: linear-gradient(45deg, #06ffa5, #ff006e, #00b4d8);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: gradient-shift 3s ease-in-out infinite;
            text-shadow: 0 0 20px rgba(6, 255, 165, 0.5);
        }}
        
        .header .subtitle {{
            font-size: 1.2rem;
            color: #ff006e;
            margin-top: 10px;
            opacity: 0.8;
        }}
        
        @keyframes gradient-shift {{
            0%, 100% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            gap: 30px;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
        }}
        
        .plot-container {{
            background: rgba(16, 21, 62, 0.8);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(6, 255, 165, 0.3);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            transition: all 0.3s ease;
        }}
        
        .plot-container:hover {{
            border-color: rgba(255, 0, 110, 0.6);
            box-shadow: 0 15px 40px rgba(255, 0, 110, 0.2);
            transform: translateY(-5px);
        }}
        
        .plot-title {{
            font-size: 1.5rem;
            color: #00b4d8;
            margin-bottom: 15px;
            text-align: center;
            text-shadow: 0 0 10px rgba(0, 180, 216, 0.5);
        }}
        
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: rgba(6, 255, 165, 0.1);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(6, 255, 165, 0.3);
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #06ffa5;
            display: block;
        }}
        
        .metric-label {{
            color: #ff006e;
            font-size: 0.9rem;
            margin-top: 5px;
        }}
        
        .controls {{
            background: rgba(16, 21, 62, 0.9);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 30px;
            border: 1px solid rgba(255, 0, 110, 0.3);
        }}
        
        .controls h3 {{
            color: #ff006e;
            margin-bottom: 15px;
        }}
        
        .control-group {{
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
            align-items: center;
        }}
        
        .control-group label {{
            color: #00b4d8;
            min-width: 120px;
        }}
        
        .control-group input, .control-group select {{
            background: rgba(6, 255, 165, 0.1);
            border: 1px solid rgba(6, 255, 165, 0.3);
            color: #06ffa5;
            padding: 8px 12px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
        }}
        
        .control-group button {{
            background: linear-gradient(45deg, #ff006e, #06ffa5);
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            transition: all 0.3s ease;
        }}
        
        .control-group button:hover {{
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(255, 0, 110, 0.4);
        }}
        
        .equation {{
            background: rgba(0, 180, 216, 0.1);
            border: 1px solid rgba(0, 180, 216, 0.3);
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            text-align: center;
            font-size: 1.2rem;
            color: #00b4d8;
        }}
        
        .status-indicator {{
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 20px;
            background: rgba(6, 255, 165, 0.9);
            color: #0f0f0f;
            border-radius: 20px;
            font-weight: bold;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
        
        .footer {{
            text-align: center;
            margin-top: 50px;
            color: rgba(6, 255, 165, 0.6);
            font-style: italic;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .container {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 2rem;
            }}
            
            .metrics {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="status-indicator">LIVE</div>
    
    <div class="header">
        <h1>{title}</h1>
        <div class="subtitle">Pure Elegance • Harmonic Error Collection • Chaos Shaping</div>
    </div>
    
    <div class="equation">
        Ĥ(t) = Σ A<sub>i</sub>sin(B<sub>i</sub>t + φ<sub>i</sub>) + C<sub>i</sub>e<sup>-D<sub>i</sub>t</sup> + ∫softplus(·) + ε(t)
    </div>
    
    <div class="metrics">
        <div class="metric-card">
            <span class="metric-value" id="harmonics-count">7</span>
            <div class="metric-label">Harmonics</div>
        </div>
        <div class="metric-card">
            <span class="metric-value" id="signal-length">1000</span>
            <div class="metric-label">Signal Points</div>
        </div>
        <div class="metric-card">
            <span class="metric-value" id="variance-reduction">0%</span>
            <div class="metric-label">Variance Reduction</div>
        </div>
        <div class="metric-card">
            <span class="metric-value" id="processing-time">0ms</span>
            <div class="metric-label">Processing Time</div>
        </div>
    </div>
    
    <div class="controls">
        <h3>🎛️ Real-time Controls</h3>
        <div class="control-group">
            <label>Harmonics:</label>
            <input type="range" id="harmonics-slider" min="3" max="15" value="7" oninput="updateHarmonics(this.value)">
            <span id="harmonics-display">7</span>
        </div>
        <div class="control-group">
            <label>Noise Level:</label>
            <input type="range" id="noise-slider" min="0" max="1" step="0.01" value="0.1" oninput="updateNoise(this.value)">
            <span id="noise-display">0.1</span>
        </div>
        <div class="control-group">
            <label>Golden Ratio:</label>
            <input type="checkbox" id="golden-ratio" onchange="updateGoldenRatio(this.checked)">
        </div>
        <div class="control-group">
            <button onclick="generateNewSignal()">🎲 Generate New Signal</button>
            <button onclick="resetTransformation()">🔄 Reset</button>
            <button onclick="exportData()">💾 Export Data</button>
        </div>
    </div>
    
    <div class="container">
        {plot_containers}
    </div>
    
    <div class="footer">
        <p>"Code as waveform, errors as harmony, chaos shaped into clarity."</p>
        <p>Generated at {timestamp}</p>
    </div>
    
    <script>
        // Global state
        let currentData = {{}};
        let harmonicEngine = null;
        let adaptiveFlow = null;
        let chaosShaper = null;
        
        // Initialize visualization
        function initializeVisualization() {{
            generateNewSignal();
        }}
        
        function updateHarmonics(value) {{
            document.getElementById('harmonics-display').textContent = value;
            document.getElementById('harmonics-count').textContent = value;
            updateTransformation();
        }}
        
        function updateNoise(value) {{
            document.getElementById('noise-display').textContent = value;
            updateTransformation();
        }}
        
        function updateGoldenRatio(checked) {{
            updateTransformation();
        }}
        
        function generateNewSignal() {{
            const signalLength = 1000;
            const t = Array.from({{length: signalLength}}, (_, i) => i * 20 / signalLength);
            
            // Generate synthetic error signal
            const noise = document.getElementById('noise-slider').value;
            const errors = t.map(time => {{
                return parseFloat(noise) * (Math.random() - 0.5) * 2 +
                       2 * Math.sin(0.5 * time) +
                       Math.sin(2 * time) * Math.exp(-0.1 * time);
            }});
            
            currentData = {{ t, errors }};
            document.getElementById('signal-length').textContent = signalLength;
            
            updateTransformation();
        }}
        
        function updateTransformation() {{
            const startTime = performance.now();
            
            // Simulate harmonic transformation
            const harmonics = parseInt(document.getElementById('harmonics-slider').value);
            const noise = parseFloat(document.getElementById('noise-slider').value);
            const goldenRatio = document.getElementById('golden-ratio').checked;
            
            // Simple transformation simulation
            const transformed = currentData.errors.map((error, i) => {{
                let result = 0;
                for (let h = 1; h <= harmonics; h++) {{
                    const freq = goldenRatio ? Math.pow(1.618, h - harmonics/2) : h;
                    const amp = 1.0 / (1 + h * 0.3);
                    const phase = h * Math.PI / 4;
                    result += amp * Math.sin(freq * currentData.t[i] + phase);
                }}
                
                // Add decay term
                result += error * Math.exp(-0.1 * currentData.t[i]);
                
                // Add softplus
                result += Math.log(1 + Math.exp(error));
                
                return result;
            }});
            
            const processingTime = performance.now() - startTime;
            document.getElementById('processing-time').textContent = Math.round(processingTime) + 'ms';
            
            // Calculate variance reduction
            const originalVar = variance(currentData.errors);
            const transformedVar = variance(transformed);
            const reduction = ((originalVar - transformedVar) / originalVar * 100).toFixed(1);
            document.getElementById('variance-reduction').textContent = reduction + '%';
            
            // Update plots
            updatePlots(currentData.t, currentData.errors, transformed);
        }}
        
        function variance(arr) {{
            const mean = arr.reduce((a, b) => a + b) / arr.length;
            return arr.reduce((sum, x) => sum + Math.pow(x - mean, 2), 0) / arr.length;
        }}
        
        function updatePlots(t, original, transformed) {{
            // Update original signal plot
            if (document.getElementById('plot-original')) {{
                Plotly.newPlot('plot-original', [{{
                    x: t,
                    y: original,
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#ff006e', width: 2 }},
                    name: 'Original Errors'
                }}], {{
                    title: 'Original Error Signal',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: {{ color: '#06ffa5', family: 'Courier New' }},
                    xaxis: {{ gridcolor: 'rgba(6, 255, 165, 0.2)', title: 'Time' }},
                    yaxis: {{ gridcolor: 'rgba(6, 255, 165, 0.2)', title: 'Amplitude' }}
                }});
            }}
            
            // Update transformed signal plot
            if (document.getElementById('plot-transformed')) {{
                Plotly.newPlot('plot-transformed', [{{
                    x: t,
                    y: transformed,
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#06ffa5', width: 2 }},
                    name: 'Harmonic Transform'
                }}], {{
                    title: 'Harmonic Transformation',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: {{ color: '#06ffa5', family: 'Courier New' }},
                    xaxis: {{ gridcolor: 'rgba(6, 255, 165, 0.2)', title: 'Time' }},
                    yaxis: {{ gridcolor: 'rgba(6, 255, 165, 0.2)', title: 'Amplitude' }}
                }});
            }}
            
            // Update comparison plot
            if (document.getElementById('plot-comparison')) {{
                Plotly.newPlot('plot-comparison', [
                    {{
                        x: t,
                        y: original,
                        type: 'scatter',
                        mode: 'lines',
                        line: {{ color: '#ff006e', width: 1.5 }},
                        name: 'Original',
                        opacity: 0.7
                    }},
                    {{
                        x: t,
                        y: transformed,
                        type: 'scatter',
                        mode: 'lines',
                        line: {{ color: '#06ffa5', width: 2 }},
                        name: 'Transformed'
                    }}
                ], {{
                    title: 'Before vs After Comparison',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    font: {{ color: '#06ffa5', family: 'Courier New' }},
                    xaxis: {{ gridcolor: 'rgba(6, 255, 165, 0.2)', title: 'Time' }},
                    yaxis: {{ gridcolor: 'rgba(6, 255, 165, 0.2)', title: 'Amplitude' }}
                }});
            }}
        }}
        
        function resetTransformation() {{
            document.getElementById('harmonics-slider').value = 7;
            document.getElementById('noise-slider').value = 0.1;
            document.getElementById('golden-ratio').checked = false;
            document.getElementById('harmonics-display').textContent = '7';
            document.getElementById('noise-display').textContent = '0.1';
            generateNewSignal();
        }}
        
        function exportData() {{
            const data = {{
                timestamp: new Date().toISOString(),
                configuration: {{
                    harmonics: parseInt(document.getElementById('harmonics-slider').value),
                    noise: parseFloat(document.getElementById('noise-slider').value),
                    golden_ratio: document.getElementById('golden-ratio').checked
                }},
                data: currentData
            }};
            
            const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'h_model_z_data.json';
            a.click();
            URL.revokeObjectURL(url);
        }}
        
        // Initialize on page load
        window.addEventListener('load', initializeVisualization);
        
        // Auto-refresh every 30 seconds for live feel
        setInterval(() => {{
            const indicator = document.querySelector('.status-indicator');
            indicator.style.animation = 'none';
            setTimeout(() => indicator.style.animation = 'pulse 2s infinite', 10);
        }}, 30000);
    </script>
</body>
</html>
        '''
    
    def add_plot(self, 
                plot_id: str, 
                title: str, 
                data: Dict[str, Any] = None) -> 'LiveHTMLRenderer':
        """Add a plot container to the visualization."""
        plot_container = f'''
        <div class="plot-container">
            <div class="plot-title">{title}</div>
            <div id="{plot_id}" style="height: 400px;"></div>
        </div>
        '''
        self.plots.append(plot_container)
        
        if data:
            self.data_sets[plot_id] = data
            
        return self
    
    def render(self, 
              output_path: Union[str, Path] = "h_model_z_live.html",
              live: bool = True) -> str:
        """
        Render the complete HTML visualization.
        
        Args:
            output_path: Path to save the HTML file
            live: Whether to enable live updates
            
        Returns:
            Path to the generated HTML file
        """
        # Create plot containers
        plot_containers = "\n".join(self.plots) if self.plots else self._default_plots()
        
        # Prepare template variables
        template_vars = {
            'title': self.title,
            'plot_containers': plot_containers,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        
        # Generate final HTML
        html_content = self.template.format(**template_vars)
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path.absolute())
    
    def _default_plots(self) -> str:
        """Create default plot containers."""
        return '''
        <div class="plot-container">
            <div class="plot-title">📊 Original Error Signal</div>
            <div id="plot-original" style="height: 400px;"></div>
        </div>
        
        <div class="plot-container">
            <div class="plot-title">🌊 Harmonic Transformation</div>
            <div id="plot-transformed" style="height: 400px;"></div>
        </div>
        
        <div class="plot-container">
            <div class="plot-title">⚖️ Before vs After Comparison</div>
            <div id="plot-comparison" style="height: 400px;"></div>
        </div>
        
        <div class="plot-container">
            <div class="plot-title">🔄 Phase Space Embedding</div>
            <div id="plot-phase" style="height: 400px;"></div>
        </div>
        '''
    
    def plot_waveform(self, 
                     data: np.ndarray,
                     time: Optional[np.ndarray] = None,
                     title: str = "Waveform",
                     color: str = "#06ffa5") -> 'LiveHTMLRenderer':
        """Add waveform plot to visualization."""
        if time is None:
            time = np.linspace(0, 20, len(data))
        
        plot_id = f"plot-{len(self.plots)}"
        self.add_plot(plot_id, title, {
            'x': time.tolist(),
            'y': data.tolist(),
            'color': color
        })
        
        return self
    
    def add_wells(self, 
                 trapped_data: np.ndarray,
                 label: str = "Softplus Wells") -> 'LiveHTMLRenderer':
        """Add wells visualization."""
        # This would add well visualization
        # For now, just add as another waveform
        return self.plot_waveform(trapped_data, title=label, color="#ff006e")


# Convenience function
def create_live_visualization(title: str = "_h_model_z_ Live Demo") -> LiveHTMLRenderer:
    """Create a live HTML visualization renderer."""
    return LiveHTMLRenderer(title)