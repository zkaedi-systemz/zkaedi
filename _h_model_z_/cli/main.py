"""
CLI Main - Command-line Interface for _h_model_z_

Zero friction, pure signal. One command to beauty.
"""

import click
import numpy as np
import sys
import time
from pathlib import Path
from typing import Optional

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    rprint = print

from .. import HarmonicEngine, AdaptiveFlow, ChaosShaper, transform
from ..wells.softplus_wells import SoftplusWells


def create_console():
    """Create rich console if available."""
    if RICH_AVAILABLE:
        return Console()
    return None


@click.group()
@click.version_option()
def main():
    """
    _h_model_z_ - Pure Elegance
    
    Harmonic Error Collection and Chaos Shaping System
    
    Mathematical Foundation:
    Ĥ(t) = Σ A_i*sin(B_i*t + φ_i) + C_i*e^(-D_i*t) + ∫softplus(...) + ε
    """
    pass


@main.command()
@click.option('--n-harmonics', '-n', default=7, help='Number of harmonics')
@click.option('--signal-length', '-l', default=1000, help='Signal length')
@click.option('--noise-level', '-noise', default=0.1, help='Noise level')
@click.option('--golden-ratio', '-g', is_flag=True, help='Use golden ratio spacing')
@click.option('--show-plot', '-p', is_flag=True, help='Show interactive plot')
@click.option('--output', '-o', help='Output file path')
def demo(n_harmonics, signal_length, noise_level, golden_ratio, show_plot, output):
    """
    Instant demonstration of harmonic transformation.
    
    Experience the beauty of error collection and chaos shaping.
    """
    console = create_console()
    
    if console:
        console.print(Panel.fit(
            "[bold cyan]_h_model_z_[/bold cyan] [dim]Pure Elegance Demo[/dim]\n"
            "Where errors become harmony, chaos becomes beauty.",
            border_style="cyan"
        ))
    else:
        print("=== _h_model_z_ Pure Elegance Demo ===")
        print("Where errors become harmony, chaos becomes beauty.")
    
    # Generate sample error signal
    t = np.linspace(0, 20, signal_length)
    errors = (np.random.normal(0, noise_level, signal_length) + 
              2 * np.sin(0.5 * t) + 
              np.sin(2 * t) * np.exp(-0.1 * t))
    
    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            
            # Initialize engine
            task1 = progress.add_task("Initializing harmonic engine...", total=100)
            engine = HarmonicEngine(n_harmonics=n_harmonics)
            
            if golden_ratio:
                engine.use_golden_ratio()
                progress.update(task1, description="Applying golden ratio spacing...")
            
            progress.update(task1, completed=30)
            
            # Transform signal
            progress.update(task1, description="Transforming error signal...")
            transformed = engine.transform(t, errors)
            progress.update(task1, completed=60)
            
            # Apply adaptive flow
            progress.update(task1, description="Applying adaptive flow...")
            flow = AdaptiveFlow("demo_flow").enable_adversarial_detection()
            flow_result = flow.process(transformed)
            progress.update(task1, completed=80)
            
            # Apply chaos shaping
            progress.update(task1, description="Shaping chaos...")
            shaper = ChaosShaper('lorenz').enable_fractal_wells()
            final_result = shaper.shape(flow_result)
            progress.update(task1, completed=100)
    else:
        print("Initializing harmonic engine...")
        engine = HarmonicEngine(n_harmonics=n_harmonics)
        if golden_ratio:
            engine.use_golden_ratio()
        
        print("Transforming error signal...")
        transformed = engine.transform(t, errors)
        
        print("Applying adaptive flow...")
        flow = AdaptiveFlow("demo_flow")
        flow_result = flow.process(transformed)
        
        print("Shaping chaos...")
        shaper = ChaosShaper('lorenz')
        final_result = shaper.shape(flow_result)
    
    # Display results
    if console:
        # Create results table
        table = Table(title="Transformation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Original", style="red")
        table.add_column("Transformed", style="green")
        table.add_column("Improvement", style="yellow")
        
        original_var = np.var(errors)
        final_var = np.var(final_result)
        var_improvement = ((original_var - final_var) / original_var * 100) if original_var > 0 else 0
        
        table.add_row(
            "Variance",
            f"{original_var:.6f}",
            f"{final_var:.6f}",
            f"{var_improvement:.1f}%"
        )
        
        original_std = np.std(errors)
        final_std = np.std(final_result)
        std_improvement = ((original_std - final_std) / original_std * 100) if original_std > 0 else 0
        
        table.add_row(
            "Std Dev",
            f"{original_std:.6f}",
            f"{final_std:.6f}",
            f"{std_improvement:.1f}%"
        )
        
        original_mean = np.mean(np.abs(errors))
        final_mean = np.mean(np.abs(final_result))
        mean_improvement = ((original_mean - final_mean) / original_mean * 100) if original_mean > 0 else 0
        
        table.add_row(
            "Mean Abs",
            f"{original_mean:.6f}",
            f"{final_mean:.6f}",
            f"{mean_improvement:.1f}%"
        )
        
        console.print(table)
        
        # Show configuration
        config_panel = Panel(
            f"Harmonics: {n_harmonics}\n"
            f"Golden Ratio: {golden_ratio}\n" 
            f"Signal Length: {signal_length}\n"
            f"Noise Level: {noise_level}\n"
            f"Flow Adaptations: {flow.state.adaptation_count}\n"
            f"Trust Level: {flow.state.trust_level:.3f}",
            title="Configuration",
            border_style="blue"
        )
        console.print(config_panel)
        
    else:
        print(f"\n--- Results ---")
        print(f"Original variance: {np.var(errors):.6f}")
        print(f"Final variance: {np.var(final_result):.6f}")
        print(f"Improvement: {((np.var(errors) - np.var(final_result)) / np.var(errors) * 100):.1f}%")
    
    # Save output if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez(output_path, 
                 time=t,
                 original=errors,
                 transformed=transformed,
                 flow_result=flow_result,
                 final=final_result,
                 config={
                     'n_harmonics': n_harmonics,
                     'golden_ratio': golden_ratio,
                     'noise_level': noise_level
                 })
        
        if console:
            console.print(f"[green]Results saved to {output_path}[/green]")
        else:
            print(f"Results saved to {output_path}")
    
    # Show plot if requested
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            plt.plot(t, errors, 'r-', alpha=0.7, label='Original Errors')
            plt.title('Original Error Signal')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.subplot(2, 2, 2)
            plt.plot(t, transformed, 'b-', alpha=0.7, label='Harmonic Transform')
            plt.title('Harmonic Transformation')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.subplot(2, 2, 3)
            plt.plot(t, flow_result, 'g-', alpha=0.7, label='Adaptive Flow')
            plt.title('Adaptive Flow Processing')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.subplot(2, 2, 4)
            plt.plot(t, final_result, 'm-', alpha=0.7, label='Chaos Shaped')
            plt.title('Final Chaos-Shaped Result')
            plt.xlabel('Time')
            plt.ylabel('Amplitude')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            if console:
                console.print("[yellow]matplotlib not available - skipping plot[/yellow]")
            else:
                print("matplotlib not available - skipping plot")


@main.command()
@click.option('--port', '-p', default=8765, help='WebSocket server port')
@click.option('--host', '-h', default='localhost', help='Server host')
@click.option('--data-file', '-d', help='Data file to process')
def serve(port, host, data_file):
    """
    Start WebSocket server for real-time harmonic processing.
    """
    console = create_console()
    
    if console:
        console.print(Panel.fit(
            f"[bold green]Starting WebSocket Server[/bold green]\n"
            f"Host: {host}\n"
            f"Port: {port}\n"
            f"Connect for real-time harmonic processing",
            border_style="green"
        ))
    else:
        print(f"Starting WebSocket server on {host}:{port}")
    
    # This would start the actual WebSocket server
    # For now, just show the configuration
    if console:
        console.print("[yellow]WebSocket server implementation coming soon![/yellow]")
        console.print(f"[dim]Would listen on ws://{host}:{port}[/dim]")
    else:
        print("WebSocket server implementation coming soon!")
        print(f"Would listen on ws://{host}:{port}")


@main.command()
@click.option('--input-file', '-i', required=True, help='Input data file')
@click.option('--output-file', '-o', required=True, help='Output file')
@click.option('--config', '-c', help='Configuration file')
@click.option('--format', '-f', default='numpy', help='Output format (numpy, csv, json)')
def process(input_file, output_file, config, format):
    """
    Process data file through complete harmonic pipeline.
    """
    console = create_console()
    
    if console:
        console.print(f"[cyan]Processing {input_file} -> {output_file}[/cyan]")
    else:
        print(f"Processing {input_file} -> {output_file}")
    
    try:
        # Load input data
        if input_file.endswith('.npy'):
            data = np.load(input_file)
        elif input_file.endswith('.csv'):
            data = np.loadtxt(input_file, delimiter=',')
        else:
            raise ValueError(f"Unsupported input format: {input_file}")
        
        # Process through pipeline
        engine = HarmonicEngine(n_harmonics=7).use_golden_ratio()
        flow = AdaptiveFlow("batch_process")
        shaper = ChaosShaper('hybrid').enable_fractal_wells()
        
        # Generate time array
        t = np.linspace(0, 20, len(data))
        
        # Transform
        transformed = engine.transform(t, data)
        flow_result = flow.process(transformed)
        final_result = shaper.shape(flow_result)
        
        # Save output
        if format == 'numpy':
            np.save(output_file, final_result)
        elif format == 'csv':
            np.savetxt(output_file, final_result, delimiter=',')
        elif format == 'json':
            import json
            with open(output_file, 'w') as f:
                json.dump(final_result.tolist(), f)
        
        if console:
            console.print(f"[green]Successfully processed and saved to {output_file}[/green]")
        else:
            print(f"Successfully processed and saved to {output_file}")
            
    except Exception as e:
        if console:
            console.print(f"[red]Error: {e}[/red]")
        else:
            print(f"Error: {e}")
        sys.exit(1)


@main.command()
def interactive():
    """
    Start interactive REPL environment.
    """
    console = create_console()
    
    if console:
        console.print(Panel.fit(
            "[bold magenta]_h_model_z_ Interactive Environment[/bold magenta]\n"
            "Type 'help' for commands, 'exit' to quit.",
            border_style="magenta"
        ))
    else:
        print("=== _h_model_z_ Interactive Environment ===")
        print("Type 'help' for commands, 'exit' to quit.")
    
    # Initialize components
    engine = HarmonicEngine(n_harmonics=7)
    flow = AdaptiveFlow("interactive")
    shaper = ChaosShaper('lorenz')
    
    while True:
        try:
            if console:
                command = console.input("[bold cyan]h-model-z>[/bold cyan] ")
            else:
                command = input("h-model-z> ")
            
            if command.lower() in ['exit', 'quit']:
                break
            elif command.lower() == 'help':
                if console:
                    help_text = """
[bold]Available Commands:[/bold]
  demo          - Run demonstration
  generate      - Generate sample data
  transform     - Transform current data
  status        - Show system status
  reset         - Reset components
  help          - Show this help
  exit/quit     - Exit interactive mode
                    """
                    console.print(Panel(help_text, title="Help", border_style="blue"))
                else:
                    print("Available commands: demo, generate, transform, status, reset, help, exit")
            elif command.lower() == 'demo':
                # Run mini demo
                t = np.linspace(0, 10, 100)
                errors = np.random.normal(0, 0.1, 100) + np.sin(t)
                result = transform(errors, t)
                if console:
                    console.print(f"[green]Demo complete. Transformed {len(errors)} points.[/green]")
                else:
                    print(f"Demo complete. Transformed {len(errors)} points.")
            elif command.lower() == 'status':
                if console:
                    status = Table(title="System Status")
                    status.add_column("Component", style="cyan")
                    status.add_column("Status", style="green")
                    
                    status.add_row("Harmonic Engine", f"{engine.config.n_harmonics} harmonics")
                    status.add_row("Adaptive Flow", f"Trust: {flow.state.trust_level:.3f}")
                    status.add_row("Chaos Shaper", f"Method: {shaper.method}")
                    
                    console.print(status)
                else:
                    print(f"Harmonic Engine: {engine.config.n_harmonics} harmonics")
                    print(f"Adaptive Flow: Trust {flow.state.trust_level:.3f}")
                    print(f"Chaos Shaper: {shaper.method}")
            else:
                if console:
                    console.print(f"[red]Unknown command: {command}[/red]")
                else:
                    print(f"Unknown command: {command}")
                    
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    if console:
        console.print("[dim]Goodbye! Pure signal, zero friction.[/dim]")
    else:
        print("Goodbye! Pure signal, zero friction.")


if __name__ == '__main__':
    main()