"""
Visualization module for HVT v3 training pipeline.

Generates real-time plots, phase diagrams, sync evolution charts,
and comprehensive training dashboards.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
import math
import time
from pathlib import Path
import json
from collections import deque
import threading
import queue


class TrainingDashboard:
    """
    Real-time training dashboard with live updates.
    
    Displays sync order, loss curves, phase diagrams, and
    other oscillator-specific metrics in real-time.
    """
    
    def __init__(
        self,
        update_interval: float = 2.0,  # Update every 2 seconds
        max_history: int = 1000,
        output_dir: str = "results/plots"
    ):
        self.update_interval = update_interval
        self.max_history = max_history
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.metrics_history = deque(maxlen=max_history)
        self.sync_history = deque(maxlen=max_history)
        self.phase_history = deque(maxlen=100)  # Store recent phases
        self.energy_history = deque(maxlen=max_history)
        
        # Plot configuration
        self.fig = None
        self.axes = None
        self.lines = {}
        self.initialized = False
        
        # Threading for non-blocking updates
        self.update_queue = queue.Queue()
        self.update_thread = None
        self.running = False
        
        # Color scheme
        self.colors = {
            'sync': '#2E8B57',      # Sea Green
            'target': '#FF6B6B',    # Coral Red
            'loss': '#4ECDC4',      # Turquoise
            'energy': '#FFE66D',    # Yellow
            'phase': '#95E1D3',     # Mint
            'background': '#F7F9FC', # Light Blue-Gray
        }
    
    def initialize_plots(self):
        """Initialize the dashboard plots."""
        if self.initialized:
            return
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.patch.set_facecolor(self.colors['background'])
        
        # Grid layout
        gs = self.fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Sync Order Evolution (top left, spanning 2 columns)
        self.axes['sync'] = self.fig.add_subplot(gs[0, :2])
        self.lines['sync'], = self.axes['sync'].plot([], [], color=self.colors['sync'], linewidth=2, label='Sync Order')
        self.lines['sync_target'] = self.axes['sync'].axhline(y=0.618, color=self.colors['target'], 
                                                              linestyle='--', alpha=0.8, label='Target (φ⁻¹)')
        self.axes['sync'].set_ylim(0, 1)
        self.axes['sync'].set_xlabel('Training Steps')
        self.axes['sync'].set_ylabel('Sync Order')
        self.axes['sync'].set_title('Oscillator Synchronization Evolution', fontsize=14, fontweight='bold')
        self.axes['sync'].legend()
        self.axes['sync'].grid(True, alpha=0.3)
        
        # 2. Loss Curves (top right, spanning 2 columns)
        self.axes['loss'] = self.fig.add_subplot(gs[0, 2:])
        self.lines['total_loss'], = self.axes['loss'].plot([], [], color=self.colors['loss'], linewidth=2, label='Total Loss')
        self.lines['task_loss'], = self.axes['loss'].plot([], [], color=self.colors['loss'], alpha=0.7, linewidth=1, label='Task Loss')
        self.lines['sync_loss'], = self.axes['loss'].plot([], [], color=self.colors['target'], alpha=0.7, linewidth=1, label='Sync Loss')
        self.axes['loss'].set_xlabel('Training Steps')
        self.axes['loss'].set_ylabel('Loss')
        self.axes['loss'].set_title('Training Loss Evolution', fontsize=14, fontweight='bold')
        self.axes['loss'].set_yscale('log')
        self.axes['loss'].legend()
        self.axes['loss'].grid(True, alpha=0.3)
        
        # 3. Phase Space Diagram (middle left)
        self.axes['phase'] = self.fig.add_subplot(gs[1, 0])
        self.scatter_phase = self.axes['phase'].scatter([], [], c=[], cmap='hsv', alpha=0.6, s=20)
        self.axes['phase'].set_xlim(-math.pi, math.pi)
        self.axes['phase'].set_ylim(-math.pi, math.pi)
        self.axes['phase'].set_xlabel('Phase (Real Part)')
        self.axes['phase'].set_ylabel('Phase (Imaginary Part)')
        self.axes['phase'].set_title('Phase Space', fontsize=14, fontweight='bold')
        
        # Add unit circle
        circle = Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
        self.axes['phase'].add_patch(circle)
        
        # 4. Energy Distribution (middle center)
        self.axes['energy'] = self.fig.add_subplot(gs[1, 1])
        self.lines['energy'], = self.axes['energy'].plot([], [], color=self.colors['energy'], linewidth=2)
        self.axes['energy'].set_xlabel('Training Steps')
        self.axes['energy'].set_ylabel('Energy Stability')
        self.axes['energy'].set_title('Energy Stability', fontsize=14, fontweight='bold')
        self.axes['energy'].grid(True, alpha=0.3)
        
        # 5. Frequency Response (middle right)
        self.axes['freq'] = self.fig.add_subplot(gs[1, 2])
        self.lines['freq'], = self.axes['freq'].plot([], [], color='purple', linewidth=2)
        self.axes['freq'].set_xlabel('Frequency Bin')
        self.axes['freq'].set_ylabel('Magnitude')
        self.axes['freq'].set_title('Frequency Response', fontsize=14, fontweight='bold')
        self.axes['freq'].grid(True, alpha=0.3)
        
        # 6. Accuracy (bottom left)
        self.axes['accuracy'] = self.fig.add_subplot(gs[1, 3])
        self.lines['accuracy'], = self.axes['accuracy'].plot([], [], color='green', linewidth=2)
        self.axes['accuracy'].set_xlabel('Training Steps')
        self.axes['accuracy'].set_ylabel('Accuracy')
        self.axes['accuracy'].set_title('Training Accuracy', fontsize=14, fontweight='bold')
        self.axes['accuracy'].set_ylim(0, 1)
        self.axes['accuracy'].grid(True, alpha=0.3)
        
        # 7. Learning Rate Schedule (bottom left)
        self.axes['lr'] = self.fig.add_subplot(gs[2, 0])
        self.lines['lr'], = self.axes['lr'].plot([], [], color='orange', linewidth=2)
        self.axes['lr'].set_xlabel('Training Steps')
        self.axes['lr'].set_ylabel('Learning Rate')
        self.axes['lr'].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        self.axes['lr'].set_yscale('log')
        self.axes['lr'].grid(True, alpha=0.3)
        
        # 8. Sync Distribution (bottom center)
        self.axes['sync_dist'] = self.fig.add_subplot(gs[2, 1])
        self.hist_sync = None
        self.axes['sync_dist'].set_xlabel('Sync Order')
        self.axes['sync_dist'].set_ylabel('Frequency')
        self.axes['sync_dist'].set_title('Sync Order Distribution', fontsize=14, fontweight='bold')
        
        # 9. Breathing Indicator (bottom right)
        self.axes['breathing'] = self.fig.add_subplot(gs[2, 2])
        self.breathing_indicator = Circle((0.5, 0.5), 0.3, 
                                          facecolor='green' if not hasattr(self, 'is_breathing') or not self.is_breathing else 'red',
                                          transform=self.axes['breathing'].transAxes)
        self.axes['breathing'].add_patch(self.breathing_indicator)
        self.axes['breathing'].set_xlim(0, 1)
        self.axes['breathing'].set_ylim(0, 1)
        self.axes['breathing'].set_aspect('equal')
        self.axes['breathing'].axis('off')
        self.axes['breathing'].set_title('Breathing Phase', fontsize=14, fontweight='bold')
        
        # 10. Status Text (bottom right)
        self.axes['status'] = self.fig.add_subplot(gs[2, 3])
        self.axes['status'].axis('off')
        self.status_text = self.axes['status'].text(0.05, 0.95, '', 
                                                   transform=self.axes['status'].transAxes,
                                                   fontsize=12, verticalalignment='top',
                                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.initialized = True
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update metrics from training."""
        self.metrics_history.append(metrics)
        
        if 'sync_order' in metrics:
            self.sync_history.append(metrics['sync_order'])
        
        if 'energy_stability' in metrics:
            self.energy_history.append(metrics['energy_stability'])
        
        # Update queue for async processing
        self.update_queue.put(metrics)
    
    def update_sync_data(self, sync_order: float, is_breathing: bool = False):
        """Update sync-specific data."""
        self.sync_history.append(sync_order)
        self.is_breathing = is_breathing
    
    def update_phase_data(self, phase: torch.Tensor):
        """Update phase data for visualization."""
        # Store recent phase data (limit memory usage)
        if len(self.phase_history) >= 100:
            self.phase_history.popleft()
        self.phase_history.append(phase.cpu())
    
    def _update_plots(self):
        """Update all plots with current data."""
        if not self.initialized or not self.metrics_history:
            return
        
        steps = list(range(len(self.metrics_history)))
        
        # Update sync order plot
        if len(self.sync_history) > 0:
            self.lines['sync'].set_data(steps[-len(self.sync_history):], list(self.sync_history))
            self.axes['sync'].set_xlim(0, max(1000, len(self.sync_history)))
        
        # Update loss plots
        if len(self.metrics_history) > 0:
            total_losses = [m.get('loss', 0) for m in self.metrics_history]
            self.lines['total_loss'].set_data(steps, total_losses)
            self.axes['loss'].set_xlim(0, len(steps))
            
            if 'task_loss' in self.metrics_history[0]:
                task_losses = [m.get('task_loss', 0) for m in self.metrics_history]
                self.lines['task_loss'].set_data(steps, task_losses)
            
            if 'sync_loss' in self.metrics_history[0]:
                sync_losses = [m.get('sync_loss', 0) for m in self.metrics_history]
                self.lines['sync_loss'].set_data(steps, sync_losses)
        
        # Update energy plot
        if len(self.energy_history) > 0:
            self.lines['energy'].set_data(steps[-len(self.energy_history):], list(self.energy_history))
            self.axes['energy'].set_xlim(0, max(1000, len(self.energy_history)))
        
        # Update accuracy plot
        if len(self.metrics_history) > 0 and 'accuracy' in self.metrics_history[0]:
            accuracies = [m.get('accuracy', 0) for m in self.metrics_history]
            self.lines['accuracy'].set_data(steps, accuracies)
            self.axes['accuracy'].set_xlim(0, len(steps))
        
        # Update learning rate plot
        if len(self.metrics_history) > 0 and 'learning_rate' in self.metrics_history[0]:
            lrs = [m.get('learning_rate', 0) for m in self.metrics_history]
            self.lines['lr'].set_data(steps, lrs)
            self.axes['lr'].set_xlim(0, len(steps))
        
        # Update phase space
        if len(self.phase_history) > 0:
            latest_phase = self.phase_history[-1]
            if latest_phase.dim() == 3:
                phase_2d = latest_phase[0, :, 0]  # First sample, first band
            else:
                phase_2d = latest_phase
            
            # Convert to complex representation
            phase_complex = torch.exp(1j * phase_2d)
            x = phase_complex.real.numpy()
            y = phase_complex.imag.numpy()
            
            # Update scatter plot
            if hasattr(self, 'scatter_phase'):
                self.scatter_phase.set_offsets(np.c_[x, y])
                self.scatter_phase.set_array(np.angle(phase_2d).numpy())
        
        # Update sync distribution
        if len(self.sync_history) > 10:
            self.axes['sync_dist'].clear()
            self.axes['sync_dist'].hist(list(self.sync_history), bins=20, alpha=0.7, 
                                       color=self.colors['sync'], edgecolor='black')
            self.axes['sync_dist'].set_xlabel('Sync Order')
            self.axes['sync_dist'].set_ylabel('Frequency')
            self.axes['sync_dist'].set_title('Sync Order Distribution', fontsize=14, fontweight='bold')
        
        # Update breathing indicator
        if hasattr(self, 'is_breathing'):
            color = 'red' if self.is_breathing else 'green'
            self.breathing_indicator.set_facecolor(color)
        
        # Update status text
        if len(self.metrics_history) > 0:
            latest = self.metrics_history[-1]
            status_text = f""
            status_text += f"Step: {latest.get('step', 'N/A')}\n"
            status_text += f"Loss: {latest.get('loss', 'N/A'):.4f}\n"
            status_text += f"Sync: {latest.get('sync_order', 'N/A'):.3f}\n"
            status_text += f"Acc: {latest.get('accuracy', 'N/A')*100:.1f}%\n"
            status_text += f"LR: {latest.get('learning_rate', 'N/A'):.2e}\n"
            status_text += f"Energy: {latest.get('energy_stability', 'N/A'):.3f}"
            
            self.status_text.set_text(status_text)
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def start_live_updates(self):
        """Start live updating thread."""
        if not self.running:
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop)
            self.update_thread.daemon = True
            self.update_thread.start()
    
    def _update_loop(self):
        """Background update loop."""
        while self.running:
            try:
                # Process any pending updates
                while not self.update_queue.empty():
                    metrics = self.update_queue.get_nowait()
                    
                # Update plots
                self._update_plots()
                
                # Sleep
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Dashboard update error: {e}")
                time.sleep(1)
    
    def stop_live_updates(self):
        """Stop live updating thread."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=1)
    
    def save_snapshot(self, filename: str = None):
        """Save current dashboard as image."""
        if not self.initialized:
            return
        
        if filename is None:
            filename = f"dashboard_step_{len(self.metrics_history)}.png"
        
        save_path = self.output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=self.colors['background'])
        print(f"Dashboard saved to {save_path}")
    
    def show(self):
        """Show the dashboard."""
        if not self.initialized:
            self.initialize_plots()
        
        plt.ion()  # Interactive mode
        plt.show()
        
        # Start live updates
        self.start_live_updates()


class PhaseAnimator:
    """
    Animate phase evolution over time.
    """
    
    def __init__(self, output_dir: str = "results/animations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fig = None
        self.ax = None
        self.animation = None
    
    def create_phase_animation(
        self,
        phase_evolution: torch.Tensor,
        filename: str = "phase_evolution.mp4",
        fps: int = 10
    ):
        """Create animated phase evolution video."""
        if phase_evolution.dim() != 4:
            raise ValueError("Expected phase_evolution to be 4D: [batch, time, oscillators, bands]")
        
        # Use first sample
        phase_data = phase_evolution[0].cpu().numpy()  # [time, oscillators, bands]
        
        # Initialize figure
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(-1.5, 1.5)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Oscillator Phase Evolution', fontsize=16, fontweight='bold')
        
        # Initialize scatter plot
        self.scatter = self.ax.scatter([], [], s=100, alpha=0.7)
        
        # Add unit circle
        circle = Circle((0, 0), 1, fill=False, color='gray', linestyle='--', linewidth=2)
        self.ax.add_patch(circle)
        
        # Animation function
        def animate(frame):
            phase_at_t = phase_data[frame, :, 0]  # First band
            
            # Convert to complex
            x = np.cos(phase_at_t)
            y = np.sin(phase_at_t)
            
            # Update scatter
            self.scatter.set_offsets(np.c_[x, y])
            
            # Color by sync order
            sync_order = np.abs(np.mean(np.exp(1j * phase_at_t)))
            colors = plt.cm.viridis(np.linspace(0, 1, len(phase_at_t)))
            self.scatter.set_color(colors)
            
            # Update title
            self.ax.set_title(f'Oscillator Phase Evolution (Step {frame}, Sync: {sync_order:.3f})', 
                             fontsize=16, fontweight='bold')
            
            return self.scatter,
        
        # Create animation
        self.animation = animation.FuncAnimation(
            self.fig, animate,
            frames=len(phase_data),
            interval=1000//fps,
            blit=True,
            repeat=True
        )
        
        # Save animation
        save_path = self.output_dir / filename
        self.animation.save(str(save_path), fps=fps, dpi=150)
        print(f"Phase animation saved to {save_path}")
        
        plt.close()
        return save_path
    
    def create_sync_animation(
        self,
        sync_history: List[float],
        filename: str = "sync_evolution.mp4",
        fps: int = 10
    ):
        """Create animated sync order evolution."""
        # Initialize figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Animation function
        def animate(frame):
            ax.clear()
            
            # Plot up to current frame
            steps = range(frame + 1)
            sync_values = sync_history[:frame + 1]
            
            ax.plot(steps, sync_values, color='#2E8B57', linewidth=2)
            ax.axhline(y=0.618, color='#FF6B6B', linestyle='--', alpha=0.8, label='Target (φ⁻¹)')
            ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Perfect Sync')
            
            ax.set_xlim(0, max(1000, len(sync_history)))
            ax.set_ylim(0, 1)
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Sync Order')
            ax.set_title(f'Oscillator Synchronization Evolution (Step {frame})', 
                        fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            return ax.lines
        
        # Create animation
        animation = animation.FuncAnimation(
            fig, animate,
            frames=len(sync_history),
            interval=1000//fps,
            blit=False,
            repeat=True
        )
        
        # Save animation
        save_path = self.output_dir / filename
        animation.save(str(save_path), fps=fps, dpi=150)
        print(f"Sync animation saved to {save_path}")
        
        plt.close()
        return save_path


class MetricsPlotter:
    """
    Generate various plots for training analysis.
    """
    
    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color palette
        self.colors = {
            'primary': '#2E8B57',
            'secondary': '#FF6B6B',
            'accent': '#4ECDC4',
            'background': '#F7F9FC',
            'text': '#2C3E50'
        }
    
    def plot_ablation_study(
        self,
        results: Dict[str, Dict[str, List[float]]],
        save_path: Optional[str] = None
    ):
        """Plot ablation study results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.patch.set_facecolor(self.colors['background'])
        
        methods = list(results.keys())
        
        # Accuracy comparison
        accuracies = [results[method]['accuracy'][-1] for method in methods]
        axes[0, 0].bar(methods, accuracies, color=self.colors['primary'], alpha=0.8)
        axes[0, 0].set_title('Final Accuracy by Method', fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Sync order comparison
        sync_orders = [results[method]['sync_order'][-1] for method in methods]
        axes[0, 1].bar(methods, sync_orders, color=self.colors['secondary'], alpha=0.8)
        axes[0, 1].set_title('Final Sync Order by Method', fontweight='bold')
        axes[0, 1].set_ylabel('Sync Order')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].axhline(y=0.618, color='red', linestyle='--', alpha=0.7, label='Target')
        axes[0, 1].legend()
        
        # Training curves for each method
        for method in methods:
            steps = range(len(results[method]['loss']))
            axes[1, 0].plot(steps, results[method]['loss'], label=method, alpha=0.8)
        
        axes[1, 0].set_title('Training Loss Curves', fontweight='bold')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sync evolution for each method
        for method in methods:
            steps = range(len(results[method]['sync_order']))
            axes[1, 1].plot(steps, results[method]['sync_order'], label=method, alpha=0.8)
        
        axes[1, 1].set_title('Sync Order Evolution', fontweight='bold')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Sync Order')
        axes[1, 1].axhline(y=0.618, color='red', linestyle='--', alpha=0.7, label='Target')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=self.colors['background'])
        else:
            plt.savefig(self.output_dir / "ablation_study.png", dpi=150, bbox_inches='tight', 
                       facecolor=self.colors['background'])
        
        plt.close()
    
    def plot_hyperparameter_sensitivity(
        self,
        param_values: List[float],
        metric_values: List[float],
        param_name: str,
        metric_name: str,
        save_path: Optional[str] = None
    ):
        """Plot hyperparameter sensitivity analysis."""
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(self.colors['background'])
        
        ax.plot(param_values, metric_values, 'o-', color=self.colors['primary'], linewidth=2, markersize=8)
        ax.set_xlabel(param_name)
        ax.set_ylabel(metric_name)
        ax.set_title(f'Hyperparameter Sensitivity: {param_name} vs {metric_name}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Highlight best value
        best_idx = np.argmax(metric_values)
        ax.axvline(x=param_values[best_idx], color=self.colors['secondary'], linestyle='--', alpha=0.7)
        ax.scatter(param_values[best_idx], metric_values[best_idx], 
                  color=self.colors['secondary'], s=100, zorder=5)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=self.colors['background'])
        else:
            plt.savefig(self.output_dir / f"sensitivity_{param_name}.png", dpi=150, bbox_inches='tight',
                       facecolor=self.colors['background'])
        
        plt.close()
    
    def plot_model_comparison(
        self,
        models: List[str],
        metrics: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """Compare multiple models."""
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
        fig.patch.set_facecolor(self.colors['background'])
        
        if len(metrics) == 1:
            axes = [axes]
        
        for i, (metric_name, values) in enumerate(metrics.items()):
            axes[i].bar(models, values, color=self.colors['primary'], alpha=0.8)
            axes[i].set_title(f'{metric_name} Comparison', fontweight='bold')
            axes[i].set_ylabel(metric_name)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=self.colors['background'])
        else:
            plt.savefig(self.output_dir / "model_comparison.png", dpi=150, bbox_inches='tight',
                       facecolor=self.colors['background'])
        
        plt.close()


# Convenience functions for quick visualization
def plot_sync_evolution(sync_history: List[float], save_path: str):
    """Quick plot of sync order evolution."""
    plotter = MetricsPlotter()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(sync_history, color='#2E8B57', linewidth=2)
    ax.axhline(y=0.618, color='#FF6B6B', linestyle='--', alpha=0.8, label='Target (φ⁻¹)')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Sync Order')
    ax.set_title('Oscillator Synchronization Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(history: List[Dict[str, float]], save_path: str):
    """Quick plot of training curves."""
    plotter = MetricsPlotter()
    
    steps = list(range(len(history)))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Loss
    axes[0].plot(steps, [h['loss'] for h in history], color='#2E8B57', linewidth=2)
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    
    # Sync order
    axes[1].plot(steps, [h['sync_order'] for h in history], color='#4ECDC4', linewidth=2)
    axes[1].axhline(y=0.618, color='#FF6B6B', linestyle='--', alpha=0.8)
    axes[1].set_xlabel('Steps')
    axes[1].set_ylabel('Sync Order')
    axes[1].set_title('Sync Order Evolution')
    axes[1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[2].plot(steps, [h['accuracy'] for h in history], color='green', linewidth=2)
    axes[2].set_xlabel('Steps')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Training Accuracy')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()