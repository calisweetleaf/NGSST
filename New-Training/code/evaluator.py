"""
Evaluation module for HVT v3 training pipeline.

Provides comprehensive metrics for oscillator dynamics,
synchronization analysis, and model performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import sys

from .config import TrainingConfig
from .datasets import get_dataloader

# Import HVT model - handle both package and standalone usage
_parent = Path(__file__).resolve().parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
from hvt_v2 import HarmonicVisionTransformer


class OscillatorMetrics:
    """
    Comprehensive metrics for oscillator dynamics analysis.
    """
    
    def __init__(self, model: HarmonicVisionTransformer, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def compute_sync_order_stats(self, sync_order: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive sync order statistics."""
        stats = {}
        
        # Basic statistics
        stats['mean'] = sync_order.mean().item()
        stats['std'] = sync_order.std().item()
        stats['min'] = sync_order.min().item()
        stats['max'] = sync_order.max().item()
        
        # Distribution statistics
        stats['median'] = sync_order.median().item()
        stats['q25'] = torch.quantile(sync_order, 0.25).item()
        stats['q75'] = torch.quantile(sync_order, 0.75).item()
        
        # Sync quality metrics
        target = 0.618  # Golden ratio complement
        sync_error = abs(sync_order - target)
        stats['target_error'] = sync_error.mean().item()
        stats['target_error_std'] = sync_error.std().item()
        
        # Entropy (measure of randomness)
        hist = torch.histogram(sync_order.flatten().cpu(), bins=20, range=(0, 1))[0]
        hist = hist / hist.sum() + 1e-8
        entropy = -(hist * torch.log(hist)).sum()
        stats['entropy'] = entropy.item()
        
        return stats
    
    def compute_phase_metrics(self, phase: torch.Tensor) -> Dict[str, float]:
        """Compute phase space metrics."""
        stats = {}
        
        # Phase coverage
        phase_range = phase.max() - phase.min()
        stats['phase_range'] = phase_range.item()
        stats['phase_coverage'] = (phase_range / (2 * math.pi)).item()
        
        # Phase coherence
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        
        coherence = torch.sqrt(cos_phase.mean()**2 + sin_phase.mean()**2)
        stats['phase_coherence'] = coherence.item()
        
        # Phase clustering
        phase_flat = phase.flatten()
        hist = torch.histogram(phase_flat.cpu(), bins=50, range=(-math.pi, math.pi))[0]
        hist = hist / hist.sum() + 1e-8
        phase_entropy = -(hist * torch.log(hist)).sum()
        stats['phase_entropy'] = phase_entropy.item()
        
        return stats
    
    def compute_energy_metrics(self, amplitude: torch.Tensor) -> Dict[str, float]:
        """Compute energy stability metrics."""
        energy = amplitude**2
        
        stats = {}
        stats['energy_mean'] = energy.mean().item()
        stats['energy_std'] = energy.std().item()
        stats['energy_min'] = energy.min().item()
        stats['energy_max'] = energy.max().item()
        
        # Energy stability (coefficient of variation)
        stability = energy.std() / (energy.mean() + 1e-8)
        stats['energy_stability'] = stability.item()
        
        # Energy distribution
        log_energy = torch.log(energy + 1e-8)
        stats['log_energy_mean'] = log_energy.mean().item()
        stats['log_energy_std'] = log_energy.std().item()
        
        return stats
    
    def compute_coupling_metrics(self, coupling: torch.Tensor) -> Dict[str, float]:
        """Compute coupling matrix metrics."""
        stats = {}
        
        # Basic statistics
        stats['coupling_mean'] = coupling.mean().item()
        stats['coupling_std'] = coupling.std().item()
        stats['coupling_min'] = coupling.min().item()
        stats['coupling_max'] = coupling.max().item()
        
        # Spectral properties
        eigenvals = torch.linalg.eigvals(coupling).real
        stats['spectral_radius'] = eigenvals.abs().max().item()
        stats['largest_eigenval'] = eigenvals.max().item()
        stats['smallest_eigenval'] = eigenvals.min().item()
        
        # Network properties
        coupling_binary = (coupling > coupling.mean()).float()
        density = coupling_binary.sum() / coupling.numel()
        stats['network_density'] = density.item()
        
        return stats
    
    def compute_geometric_metrics(self, xi: torch.Tensor) -> Dict[str, float]:
        """Compute SE(3) geometric metrics."""
        stats = {}
        
        # Split into rotation and translation
        omega = xi[:, :3]  # Rotation part
        v = xi[:, 3:]      # Translation part
        
        # Rotation metrics
        stats['rotation_magnitude'] = torch.norm(omega, dim=-1).mean().item()
        stats['rotation_std'] = torch.norm(omega, dim=-1).std().item()
        
        # Translation metrics
        stats['translation_magnitude'] = torch.norm(v, dim=-1).mean().item()
        stats['translation_std'] = torch.norm(v, dim=-1).std().item()
        
        # Geometric consistency (simplified)
        consistency = torch.norm(omega, dim=-1) / (torch.norm(v, dim=-1) + 1e-8)
        stats['geometric_consistency'] = consistency.mean().item()
        
        return stats
    
    def analyze_sample(self, image: torch.Tensor) -> Dict[str, Any]:
        """Analyze a single sample's oscillator dynamics."""
        image = image.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image, return_intermediates=True)
        
        analysis = {}
        
        # Sync order analysis
        if 'sync_order' in outputs:
            sync_order = outputs['sync_order']
            analysis['sync_stats'] = self.compute_sync_order_stats(sync_order)
            analysis['sync_order'] = sync_order.cpu()
        
        # Phase analysis
        if 'phase_evolution' in outputs:
            phase = outputs['phase_evolution']
            analysis['phase_stats'] = self.compute_phase_metrics(phase)
            analysis['phase_evolution'] = phase.cpu()
        
        # Energy analysis
        if 'amplitude_evolution' in outputs:
            amplitude = outputs['amplitude_evolution']
            analysis['energy_stats'] = self.compute_energy_metrics(amplitude)
            analysis['amplitude_evolution'] = amplitude.cpu()
        
        # Geometric analysis
        if 'xi_motion' in outputs:
            xi = outputs['xi_motion']
            analysis['geometric_stats'] = self.compute_geometric_metrics(xi)
            analysis['xi_motion'] = xi.cpu()
        
        return analysis
    
    def analyze_dataset(self, dataset: Any, num_samples: int = 1000) -> Dict[str, Any]:
        """Analyze oscillator dynamics across a dataset."""
        all_metrics = {
            'sync_stats': [],
            'phase_stats': [],
            'energy_stats': [],
            'geometric_stats': [],
        }
        
        for i in range(min(num_samples, len(dataset))):
            if i % 100 == 0:
                print(f"Processing sample {i}/{num_samples}")
            
            if hasattr(dataset, '__getitem__'):
                image, _ = dataset[i]
            else:
                # Handle iterable dataset
                try:
                    image, _ = next(iter(dataset))
                except StopIteration:
                    break
            
            analysis = self.analyze_sample(image)
            
            for key in all_metrics:
                if key in analysis:
                    all_metrics[key].append(analysis[key])
        
        # Aggregate statistics
        aggregated = {}
        for metric_type, samples in all_metrics.items():
            if not samples:
                continue
            
            # Average across samples
            aggregated[metric_type] = {}
            for key in samples[0]:
                if isinstance(samples[0][key], (int, float)):
                    values = [s[key] for s in samples]
                    aggregated[metric_type][key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                    }
        
        return aggregated


class PerformanceEvaluator:
    """
    Model performance evaluation with oscillator-specific metrics.
    """
    
    def __init__(self, model: HarmonicVisionTransformer, device: str = "cpu"):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_classification(
        self, 
        dataloader: Any, 
        num_classes: int = 10
    ) -> Dict[str, Any]:
        """Evaluate classification performance."""
        all_preds = []
        all_labels = []
        all_sync_orders = []
        
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images, return_intermediates=True)
                
                if 'logits' in outputs:
                    preds = outputs['logits'].argmax(dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                if 'sync_order' in outputs:
                    sync_arr = outputs['sync_order'].mean(dim=-1).cpu().numpy()
                    # Ensure we extend with scalar per-sample sync values
                    sync_vals = np.atleast_1d(sync_arr).ravel()
                    all_sync_orders.extend([float(v) for v in sync_vals])
        
        # Classification metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        # Sync order correlation with accuracy - be defensive against mismatched lengths
        try:
            pred_matches = np.array(all_preds) == np.array(all_labels)
            if len(all_sync_orders) != len(pred_matches) or len(all_sync_orders) < 2:
                sync_acc_corr = float('nan')
            else:
                sync_acc_corr = np.corrcoef(np.array(all_sync_orders), pred_matches)[0, 1]
        except Exception:
            sync_acc_corr = float('nan')
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'per_class_accuracy': per_class_acc,
            'sync_accuracy_correlation': sync_acc_corr,
            'num_samples': len(all_preds),
        }
    
    def evaluate_robustness(
        self,
        dataloader: Any,
        noise_levels: List[float] = [0.1, 0.2, 0.3]
    ) -> Dict[str, Any]:
        """Evaluate robustness to noise."""
        results = {}
        
        for noise_level in noise_levels:
            correct = 0
            total = 0
            sync_orders = []
            
            with torch.no_grad():
                for batch in dataloader:
                    if len(batch) == 3:
                        images, labels, _ = batch
                    else:
                        images, labels = batch
                    
                    # Add Gaussian noise
                    noise = torch.randn_like(images) * noise_level
                    noisy_images = images + noise
                    noisy_images = torch.clamp(noisy_images, 0, 1)
                    
                    noisy_images = noisy_images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(noisy_images, return_intermediates=True)
                    
                    if 'logits' in outputs:
                        preds = outputs['logits'].argmax(dim=-1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)
                    
                    if 'sync_order' in outputs:
                        sync_order = outputs['sync_order'].mean().item()
                        sync_orders.append(sync_order)
            
            accuracy = correct / total if total > 0 else 0.0
            avg_sync = sum(sync_orders) / len(sync_orders) if sync_orders else 0.0
            
            results[f'noise_{noise_level}'] = {
                'accuracy': accuracy,
                'avg_sync_order': avg_sync,
            }
        
        return results
    
    def evaluate_efficiency(self, input_size: Tuple[int, int, int, int] = (1, 3, 32, 32)) -> Dict[str, Any]:
        """Evaluate computational efficiency."""
        dummy_input = torch.randn(*input_size).to(self.device)
        
        # Warm up
        for _ in range(10):
            _ = self.model(dummy_input)
        
        # Timing
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        end = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        
        if self.device.type == 'cuda':
            start.record()
            for _ in range(100):
                _ = self.model(dummy_input)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end) / 1000  # Convert to seconds
        else:
            import time
            start_time = time.time()
            for _ in range(100):
                _ = self.model(dummy_input)
            elapsed = time.time() - start_time
        
        # Compute metrics
        batch_size = input_size[0]
        samples_per_second = (100 * batch_size) / elapsed
        ms_per_sample = (elapsed / (100 * batch_size)) * 1000
        
        # Count parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            'samples_per_second': samples_per_second,
            'ms_per_sample': ms_per_sample,
            'num_parameters': num_params,
            'flops_per_sample': num_params * 2,  # Rough estimate
        }


class VisualizationGenerator:
    """
    Generate visualizations for oscillator dynamics analysis.
    """
    
    def __init__(self, output_dir: str = "results/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_palette("husl")
    
    def plot_sync_evolution(
        self,
        sync_history: List[float],
        target_sync: float = 0.618,
        save_path: Optional[str] = None
    ):
        """Plot sync order evolution over training."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        steps = range(len(sync_history))
        ax.plot(steps, sync_history, alpha=0.7, linewidth=2, label='Sync Order')
        ax.axhline(y=target_sync, color='r', linestyle='--', alpha=0.7, label=f'Target ({target_sync})')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label='Perfect Sync')
        
        # Highlight breathing phases if any
        breathing_threshold = 0.1
        breathing_phases = []
        for i in range(1, len(sync_history)):
            if abs(sync_history[i] - sync_history[i-1]) > breathing_threshold:
                breathing_phases.append(i)
        
        for phase in breathing_phases:
            ax.axvspan(phase-2, phase+2, alpha=0.2, color='yellow', label='Breathing Phase' if phase == breathing_phases[0] else "")
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Sync Order')
        ax.set_title('Oscillator Synchronization Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "sync_evolution.png", dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_phase_space(
        self,
        phase_evolution: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """Plot phase space visualization."""
        if phase_evolution.dim() == 4:
            # Take first sample and first frequency band
            phase = phase_evolution[0, :, :, 0].cpu().numpy()
        else:
            phase = phase_evolution.cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Phase evolution over time
        im1 = axes[0].imshow(phase, aspect='auto', cmap='hsv', vmin=-np.pi, vmax=np.pi)
        axes[0].set_title('Phase Evolution')
        axes[0].set_xlabel('Oscillator Index')
        axes[0].set_ylabel('Time Step')
        plt.colorbar(im1, ax=axes[0], label='Phase')
        
        # Phase distribution
        axes[1].hist(phase.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_title('Phase Distribution')
        axes[1].set_xlabel('Phase')
        axes[1].set_ylabel('Frequency')
        
        # Phase coherence over time
        if phase.shape[0] > 1:
            coherence = []
            for t in range(phase.shape[0]):
                cos_phase = np.cos(phase[t])
                sin_phase = np.sin(phase[t])
                coh = np.sqrt(cos_phase.mean()**2 + sin_phase.mean()**2)
                coherence.append(coh)
            
            axes[2].plot(coherence)
            axes[2].set_title('Phase Coherence Over Time')
            axes[2].set_xlabel('Time Step')
            axes[2].set_ylabel('Coherence')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "phase_space.png", dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_energy_landscape(
        self,
        amplitude_evolution: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """Plot energy landscape visualization."""
        if amplitude_evolution.dim() == 4:
            amplitude = amplitude_evolution[0, :, :, 0].cpu().numpy()
        else:
            amplitude = amplitude_evolution.cpu().numpy()
        
        energy = amplitude**2
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Energy heatmap
        im1 = axes[0].imshow(energy, aspect='auto', cmap='hot')
        axes[0].set_title('Energy Landscape')
        axes[0].set_xlabel('Oscillator Index')
        axes[0].set_ylabel('Time Step')
        plt.colorbar(im1, ax=axes[0], label='Energy')
        
        # Energy distribution
        axes[1].hist(energy.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1].set_title('Energy Distribution')
        axes[1].set_xlabel('Energy')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "energy_landscape.png", dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_training_curves(
        self,
        training_history: List[Dict[str, float]],
        save_path: Optional[str] = None
    ):
        """Plot comprehensive training curves."""
        if not training_history:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        steps = [h['step'] for h in training_history]
        
        # Loss curves
        axes[0, 0].plot(steps, [h['loss'] for h in training_history], label='Total Loss')
        if 'task_loss' in training_history[0]:
            axes[0, 0].plot(steps, [h['task_loss'] for h in training_history], label='Task Loss')
        if 'sync_loss' in training_history[0]:
            axes[0, 0].plot(steps, [h['sync_loss'] for h in training_history], label='Sync Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(steps, [h['accuracy'] for h in training_history])
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sync order
        axes[0, 2].plot(steps, [h['sync_order'] for h in training_history])
        axes[0, 2].axhline(y=0.618, color='r', linestyle='--', alpha=0.7, label='Target')
        axes[0, 2].set_title('Sync Order Evolution')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Sync Order')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 0].plot(steps, [h['learning_rate'] for h in training_history])
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Energy stability
        axes[1, 1].plot(steps, [h['energy_stability'] for h in training_history])
        axes[1, 1].set_title('Energy Stability')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Stability')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Phase coherence
        axes[1, 2].plot(steps, [h['phase_coherence'] for h in training_history])
        axes[1, 2].set_title('Phase Coherence')
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Coherence')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_frequency_analysis(
        self,
        frequency_response: torch.Tensor,
        save_path: Optional[str] = None
    ):
        """Plot frequency domain analysis."""
        if frequency_response.dim() > 2:
            freq = frequency_response[0].cpu().numpy()
        else:
            freq = frequency_response.cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Frequency response
        axes[0].plot(freq)
        axes[0].set_title('Frequency Response')
        axes[0].set_xlabel('Frequency Bin')
        axes[0].set_ylabel('Magnitude')
        axes[0].grid(True, alpha=0.3)
        
        # Power spectrum
        power = np.abs(freq)**2
        axes[1].plot(power)
        axes[1].set_title('Power Spectrum')
        axes[1].set_xlabel('Frequency Bin')
        axes[1].set_ylabel('Power')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "frequency_analysis.png", dpi=150, bbox_inches='tight')
        
        plt.close()


def run_comprehensive_evaluation(
    model_path: str,
    config: TrainingConfig,
    test_dataset: Any,
    output_dir: str = "results/evaluation"
):
    """
    Run comprehensive evaluation of trained HVT model.
    
    Args:
        model_path: Path to trained model checkpoint
        config: Training configuration
        test_dataset: Test dataset
        output_dir: Directory to save evaluation results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = HarmonicVisionTransformer(
        num_classes=config.num_classes,
        num_freq_bands=config.num_freq_bands,
        base_omega=config.base_omega,
        num_evolution_layers=config.num_evolution_layers,
        hidden_dim=config.hidden_dim,
        patch_size=getattr(config, 'patch_size', 16),
        learnable_physics=True,
    )
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Initialize evaluators
    oscillator_metrics = OscillatorMetrics(model, config.device)
    performance_evaluator = PerformanceEvaluator(model, config.device)
    visualizer = VisualizationGenerator(str(output_path / "plots"))
    
    print("Running comprehensive evaluation...")
    
    # 1. Oscillator dynamics analysis
    print("1. Analyzing oscillator dynamics...")
    dynamics_analysis = oscillator_metrics.analyze_dataset(test_dataset, num_samples=500)
    
    with open(output_path / "dynamics_analysis.json", 'w') as f:
        json.dump(dynamics_analysis, f, indent=2, default=str)
    
    # 2. Classification performance
    print("2. Evaluating classification performance...")
    test_loader = get_dataloader(test_dataset, config, is_training=False)
    classification_results = performance_evaluator.evaluate_classification(test_loader, config.num_classes)
    
    with open(output_path / "classification_results.json", 'w') as f:
        json.dump(classification_results, f, indent=2, default=str)
    
    # 3. Robustness evaluation
    print("3. Evaluating robustness...")
    robustness_results = performance_evaluator.evaluate_robustness(test_loader)
    
    with open(output_path / "robustness_results.json", 'w') as f:
        json.dump(robustness_results, f, indent=2, default=str)
    
    # 4. Efficiency evaluation
    print("4. Evaluating efficiency...")
    efficiency_results = performance_evaluator.evaluate_efficiency()
    
    with open(output_path / "efficiency_results.json", 'w') as f:
        json.dump(efficiency_results, f, indent=2, default=str)
    
    # 5. Generate visualizations
    print("5. Generating visualizations...")
    
    # Sample some data for visualization
    sample_idx = 0
    sample_image, sample_label = test_dataset[sample_idx]
    sample_analysis = oscillator_metrics.analyze_sample(sample_image)
    
    # Sync evolution (need training history for this)
    # visualizer.plot_sync_evolution(sync_history)
    
    # Phase space
    if 'phase_evolution' in sample_analysis:
        visualizer.plot_phase_space(sample_analysis['phase_evolution'])
    
    # Energy landscape
    if 'amplitude_evolution' in sample_analysis:
        visualizer.plot_energy_landscape(sample_analysis['amplitude_evolution'])
    
    # Frequency analysis (if available)
    if hasattr(model, 'get_frequency_response'):
        freq_response = model.get_frequency_response()
        visualizer.plot_frequency_analysis(freq_response)
    
    # 6. Generate summary report
    print("6. Generating summary report...")
    
    summary = {
        'model_path': model_path,
        'config': config.to_dict(),
        'dynamics_analysis': dynamics_analysis,
        'classification_results': classification_results,
        'robustness_results': robustness_results,
        'efficiency_results': efficiency_results,
        'summary': {
            'final_accuracy': classification_results['accuracy'],
            'best_sync_order': max(dynamics_analysis.get('sync_stats', {}).get('mean', {}).get('mean', 0), 0),
            'energy_stability': dynamics_analysis.get('energy_stats', {}).get('energy_stability', {}).get('mean', 0),
            'samples_per_second': efficiency_results['samples_per_second'],
        }
    }
    
    with open(output_path / "evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"Evaluation complete! Results saved to {output_path}")
    
    return summary