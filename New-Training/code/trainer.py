"""
Main training module for HVT v3 training pipeline.

Implements the novel training methods with sync monitoring,
adaptive breathing, and physics-informed optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import math
import time
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict

from .config import TrainingConfig
from .losses import HarmonicLoss
from .optimizers import get_optimizer
from .schedulers import get_scheduler, SyncTriggeredBreathingScheduler
from .datasets import get_datasets, get_dataloader

# Import HVT model - handle both package and standalone usage
import sys
from pathlib import Path
_parent = Path(__file__).resolve().parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
from hvt_v2 import HarmonicVisionTransformer


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    epoch: int
    loss: float
    accuracy: float
    sync_order: float
    sync_variance: float
    energy_stability: float
    learning_rate: float
    phase_coherence: float
    geometric_consistency: float
    training_time: float
    samples_per_second: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return asdict(self)


class HVTTrainer:
    """
    Main trainer class for HVT v3 with novel training methods.
    
    Integrates all the novel approaches:
    - Sync-aware loss functions
    - Physics-informed optimization
    - Adaptive breathing schedules
    - Multi-scale curriculum learning
    - Geometric consistency regularization
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        model: Optional[HarmonicVisionTransformer] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = model or self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optimizer or self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = scheduler or self._create_scheduler()
        
        # Initialize loss function
        self.loss_fn = HarmonicLoss(config)
        
        # Initialize datasets
        self.train_dataset, self.val_dataset = get_datasets(config)
        self.train_loader = get_dataloader(self.train_dataset, config, is_training=True)
        
        if isinstance(self.val_dataset, list):
            # Multi-dataset validation
            self.val_loaders = [get_dataloader(vd, config, is_training=False) for vd in self.val_dataset]
        else:
            # Single validation dataset
            self.val_loaders = [get_dataloader(self.val_dataset, config, is_training=False)]
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_metrics = {}
        self.training_history = []
        
        # Sync monitoring
        self.sync_history = []
        self.energy_history = []
        self.phase_coverage = 0.0
        
        # Setup logging
        self._setup_logging()
        
        # Mixed precision
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def _create_model(self) -> HarmonicVisionTransformer:
        """Create HVT model from configuration."""
        return HarmonicVisionTransformer(
            num_classes=self.config.num_classes,
            num_freq_bands=self.config.num_freq_bands,
            base_omega=self.config.base_omega,
            num_evolution_layers=self.config.num_evolution_layers,
            hidden_dim=self.config.hidden_dim,
            patch_size=getattr(self.config, 'patch_size', 16),
            use_phase_routing=True,
            learnable_physics=True,
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from configuration."""
        # Use AdamW by default - proven to work with hvt_v2.py (see train_multi.py)
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler from configuration."""
        if self.config.novel_methods.use_adaptive_breathing:
            scheduler_type = "SyncTriggeredBreathing"
        elif self.config.novel_methods.use_multi_scale_curriculum:
            scheduler_type = "Curriculum"
        else:
            scheduler_type = "Cosine"
        
        return get_scheduler(self.optimizer, self.config, scheduler_type)
    
    def _setup_logging(self):
        """Setup logging for training."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # WandB integration
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.experiment_name,
                    config=self.config.to_dict()
                )
                self.wandb = wandb
            except ImportError:
                self.logger.warning("wandb not available, skipping logging")
                self.wandb = None
        else:
            self.wandb = None
    
    def _compute_sync_metrics(self, sync_order: torch.Tensor) -> Dict[str, float]:
        """Compute synchronization metrics."""
        metrics = {}
        
        # Basic sync statistics
        metrics['sync_order_mean'] = sync_order.mean().item()
        metrics['sync_order_std'] = sync_order.std().item()
        metrics['sync_order_min'] = sync_order.min().item()
        metrics['sync_order_max'] = sync_order.max().item()
        
        # Sync quality metrics
        target = self.config.sync_target
        sync_error = abs(sync_order.mean() - target)
        metrics['sync_error'] = sync_error.item()
        
        # Sync diversity (want some diversity, not complete uniformity)
        sync_diversity = sync_order.std() / (sync_order.mean() + 1e-8)
        metrics['sync_diversity'] = sync_diversity.item()
        
        return metrics
    
    def _compute_energy_metrics(self, amplitude: torch.Tensor) -> Dict[str, float]:
        """Compute energy stability metrics."""
        energy = (amplitude**2).mean(dim=[1, 2])  # Per sample
        
        metrics = {}
        metrics['energy_mean'] = energy.mean().item()
        metrics['energy_std'] = energy.std().item()
        
        # Energy stability (coefficient of variation)
        stability = energy.std() / (energy.mean() + 1e-8)
        metrics['energy_stability'] = stability.item()
        
        return metrics
    
    def _compute_phase_metrics(self, phase: torch.Tensor) -> Dict[str, float]:
        """Compute phase space metrics."""
        # Phase coverage (how much of [-π, π] is covered)
        phase_range = phase.max() - phase.min()
        coverage = phase_range / (2 * math.pi)
        
        metrics = {}
        metrics['phase_coverage'] = coverage.item()
        metrics['phase_range'] = phase_range.item()
        
        return metrics
    
    def _training_step(self, batch: Tuple[torch.Tensor, ...]) -> TrainingMetrics:
        """Perform a single training step."""
        self.model.train()
        
        # Unpack batch (handle different batch formats)
        if len(batch) == 3:
            images, labels, curriculum_info = batch
        else:
            images, labels = batch
            curriculum_info = {}
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Mixed precision forward pass
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(images, return_intermediates=True)
                losses = self.loss_fn(
                    predictions=outputs,
                    targets=images,
                    sync_order=outputs.get('sync_order'),
                    phase_evolution=outputs.get('phase_evolution'),
                    amplitude_evolution=outputs.get('amplitude_evolution'),
                    xi_motion=outputs.get('xi_motion'),
                    labels=labels,
                )
            
            # Backward pass with scaling
            self.scaler.scale(losses['total']).backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard forward pass
            outputs = self.model(images, return_intermediates=True)
            losses = self.loss_fn(
                predictions=outputs,
                targets=images,
                sync_order=outputs.get('sync_order'),
                phase_evolution=outputs.get('phase_evolution'),
                amplitude_evolution=outputs.get('amplitude_evolution'),
                xi_motion=outputs.get('xi_motion'),
                labels=labels,
            )
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
            
            # Optimizer step
            self.optimizer.step()
        
        # Learning rate scheduling
        if isinstance(self.scheduler, SyncTriggeredBreathingScheduler):
            # Update scheduler with sync order
            if 'sync_order' in outputs:
                sync_order = outputs['sync_order'].mean().item()
                self.scheduler.update_sync_order(sync_order)
        
        self.scheduler.step()
        
        # Compute metrics
        with torch.no_grad():
            # Accuracy
            if 'logits' in outputs:
                preds = outputs['logits'].argmax(dim=-1)
                accuracy = (preds == labels).float().mean().item()
            else:
                accuracy = 0.0
            
            # Sync metrics
            sync_metrics = {}
            if 'sync_order' in outputs:
                sync_metrics = self._compute_sync_metrics(outputs['sync_order'])
            
            # Energy metrics
            energy_metrics = {}
            if 'amplitude_evolution' in outputs:
                energy_metrics = self._compute_energy_metrics(outputs['amplitude_evolution'])
            
            # Phase metrics
            phase_metrics = {}
            if 'phase_evolution' in outputs:
                phase_metrics = self._compute_phase_metrics(outputs['phase_evolution'])
        
        # Compile metrics
        metrics = TrainingMetrics(
            step=self.step,
            epoch=self.epoch,
            loss=losses['total'].item(),
            accuracy=accuracy,
            sync_order=sync_metrics.get('sync_order_mean', 0.0),
            sync_variance=sync_metrics.get('sync_order_std', 0.0),
            energy_stability=energy_metrics.get('energy_stability', 0.0),
            learning_rate=self.optimizer.param_groups[0]['lr'],
            phase_coherence=phase_metrics.get('phase_coverage', 0.0),
            geometric_consistency=0.0,  # TODO: Implement geometric metrics
            training_time=0.0,
            samples_per_second=0.0,
        )
        
        return metrics
    
    def _validation_step(self, val_loader: Any) -> Dict[str, float]:
        """Perform validation on given loader."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_sync_orders = []
        all_energy_stability = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images, return_intermediates=True)
                
                # Compute loss
                losses = self.loss_fn(
                    predictions=outputs,
                    targets=images,
                    sync_order=outputs.get('sync_order'),
                    labels=labels,
                )
                total_loss += losses['total'].item() * images.size(0)
                
                # Compute accuracy
                if 'logits' in outputs:
                    preds = outputs['logits'].argmax(dim=-1)
                    total_correct += (preds == labels).sum().item()
                
                total_samples += images.size(0)
                
                # Collect sync order
                if 'sync_order' in outputs:
                    all_sync_orders.append(outputs['sync_order'].mean().item())
                
                # Collect energy stability
                if 'amplitude_evolution' in outputs:
                    energy = (outputs['amplitude_evolution']**2).mean()
                    all_energy_stability.append(energy.item())
        
        metrics = {
            'val_loss': total_loss / total_samples,
            'val_accuracy': total_correct / total_samples,
            'val_sync_order': sum(all_sync_orders) / len(all_sync_orders) if all_sync_orders else 0.0,
            'val_energy_stability': sum(all_energy_stability) / len(all_energy_stability) if all_energy_stability else 0.0,
        }
        
        return metrics
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        start_time = time.time()
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.logger.info("Starting training...")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed precision: {self.config.mixed_precision}")
        
        # Training loop
        for epoch in range(self.epoch, self.config.total_steps // len(self.train_loader) + 1):
            self.epoch = epoch
            epoch_start = time.time()
            
            for batch_idx, batch in enumerate(self.train_loader):
                step_start = time.time()
                
                # Training step
                metrics = self._training_step(batch)
                metrics.training_time = time.time() - step_start
                
                # Update sync and energy history
                self.sync_history.append(metrics.sync_order)
                self.energy_history.append(metrics.energy_stability)
                
                # Limit history size
                if len(self.sync_history) > 100:
                    self.sync_history.pop(0)
                if len(self.energy_history) > 100:
                    self.energy_history.pop(0)
                
                # Logging
                if self.step % self.config.log_every == 0:
                    self._log_step(metrics)
                
                # Validation
                if self.step % self.config.val_every == 0 and self.step > 0:
                    self._run_validation()
                
                # NO mid-training checkpoints - using simple in-memory approach
                # Will save at end only
                
                self.step += 1
                
                # Check if we've reached total steps
                if self.step >= self.config.total_steps:
                    break
            
            # End of epoch logging
            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            
            if self.step >= self.config.total_steps:
                break
        
        # Final checkpoint
        self.save_checkpoint(is_final=True)
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
    
    def _log_step(self, metrics: TrainingMetrics):
        """Log training step metrics."""
        # Progress percentage
        progress = (metrics.step / self.config.total_steps) * 100
        
        # ASCII indicators for sync (no emoji to avoid encoding issues)
        if metrics.sync_order > 0.6:
            sync_indicator = "[HIGH]"
        elif metrics.sync_order > 0.4:
            sync_indicator = "[MED] "
        else:
            sync_indicator = "[LOW] "
        
        self.logger.info(
            f"[{progress:5.1f}%] Step {metrics.step:>6} | "
            f"Loss: {metrics.loss:.4f} | "
            f"Acc: {metrics.accuracy*100:>5.1f}% | "
            f"Sync: {sync_indicator} {metrics.sync_order:.3f} | "
            f"LR: {metrics.learning_rate:.2e}"
        )
        
        # WandB logging
        if self.wandb:
            self.wandb.log(metrics.to_dict())
        
        # Store in history
        self.training_history.append(metrics.to_dict())
    
    def _run_validation(self):
        """Run validation on all validation sets."""
        self.logger.info("Running validation...")
        
        for i, val_loader in enumerate(self.val_loaders):
            val_metrics = self._validation_step(val_loader)
            
            dataset_name = f"val_{i}" if len(self.val_loaders) > 1 else "val"
            self.logger.info(
                f"{dataset_name}: Loss={val_metrics['val_loss']:.4f}, "
                f"Acc={val_metrics['val_accuracy']*100:.1f}%, "
                f"Sync={val_metrics['val_sync_order']:.3f}"
            )
            
            # Update best metrics
            if dataset_name not in self.best_metrics or val_metrics['val_accuracy'] > self.best_metrics[dataset_name].get('val_accuracy', 0):
                self.best_metrics[dataset_name] = val_metrics
        
        # Log to WandB
        if self.wandb:
            flat_metrics = {}
            for dataset, metrics in self.best_metrics.items():
                for key, value in metrics.items():
                    flat_metrics[f"{dataset}/{key}"] = value
            self.wandb.log(flat_metrics)
    
    def save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_path = checkpoint_dir / f"model_step_{self.step}.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save optimizer and scheduler state
        optimizer_path = checkpoint_dir / f"optimizer_step_{self.step}.pt"
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        if hasattr(self.scheduler, 'state_dict'):
            scheduler_path = checkpoint_dir / f"scheduler_step_{self.step}.pt"
            torch.save(self.scheduler.state_dict(), scheduler_path)
        
        # Save training state
        state = {
            'step': self.step,
            'epoch': self.epoch,
            'config': self.config.to_dict(),
            'best_metrics': self.best_metrics,
            'training_history': self.training_history[-1000:],  # Keep last 1000
            'sync_history': self.sync_history,
            'energy_history': self.energy_history,
        }
        
        state_path = checkpoint_dir / f"state_step_{self.step}.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Save best model separately
        if is_final or any(
            'val_accuracy' in metrics and metrics['val_accuracy'] > 0.9
            for metrics in self.best_metrics.values()
        ):
            best_path = checkpoint_dir / "best_model.pt"
            torch.save(self.model.state_dict(), best_path)
        
        self.logger.info(f"Checkpoint saved at step {self.step}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        # Find latest checkpoint
        model_files = list(checkpoint_dir.glob("model_step_*.pt"))
        if not model_files:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        
        latest_model = max(model_files, key=lambda p: int(p.stem.split('_')[-1]))
        step = int(latest_model.stem.split('_')[-1])
        
        # Load model
        self.model.load_state_dict(torch.load(latest_model, map_location=self.device))
        
        # Load optimizer
        optimizer_path = checkpoint_dir / f"optimizer_step_{step}.pt"
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        
        # Load scheduler
        scheduler_path = checkpoint_dir / f"scheduler_step_{step}.pt"
        if scheduler_path.exists():
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
        
        # Load training state
        state_path = checkpoint_dir / f"state_step_{step}.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            
            self.step = state.get('step', step)
            self.epoch = state.get('epoch', 0)
            self.best_metrics = state.get('best_metrics', {})
            self.training_history = state.get('training_history', [])
            self.sync_history = state.get('sync_history', [])
            self.energy_history = state.get('energy_history', [])
        
        self.logger.info(f"Resumed from step {self.step}")
    
    def get_model(self) -> HarmonicVisionTransformer:
        """Get the trained model."""
        return self.model
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history for analysis."""
        return self.training_history
    
    def get_best_metrics(self) -> Dict[str, Any]:
        """Get best validation metrics."""
        return self.best_metrics