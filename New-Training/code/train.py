"""
HVT Production Trainer - Simple load→train→save pattern.

No checkpoints, no complexity. Just:
1. Load .pt → in-memory state_dict
2. Train on in-memory weights
3. Optional per-epoch backups
4. Final save back to .pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import math
import time
import logging
import json

# Local imports
from .losses import HarmonicLoss
from .optimizers import SymplecticAdam, get_optimizer
from .schedulers import get_scheduler
from .datasets import get_datasets, get_dataloader
from .config import TrainingConfig

# Import HVT model
import sys
parent_dir = Path(__file__).resolve().parent.parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))
from hvt_v2 import HarmonicVisionTransformer


@dataclass
class TrainingMetrics:
    """Simple metrics container."""
    step: int
    epoch: int
    loss: float
    accuracy: float
    sync_order: float
    learning_rate: float
    samples_per_second: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class HVTProductionTrainer:
    """
    Simple, production-grade trainer for HVT.
    
    Uses user's RAM analogy:
    - Load .pt → in-memory state_dict (layers.bin metaphor)
    - Release .pt file handle
    - Train on in-memory weights
    - Optional per-epoch backups
    - Final serialize back to .pt
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load or create model (Step 1 of RAM pattern)
        self.model = self._load_or_create_model()
        self.model.to(self.device)
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
        
        # Create scheduler
        self.scheduler = self._create_scheduler()
        
        # Create loss function
        self.loss_fn = HarmonicLoss(config)
        
        # Load datasets
        self.train_dataset, self.val_dataset = get_datasets(config)
        self.train_loader = get_dataloader(self.train_dataset, config, is_training=True)
        self.val_loader = get_dataloader(self.val_dataset, config, is_training=False)
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.training_history = []
        self.best_accuracy = 0.0
        
        self.logger.info(f"Trainer initialized on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _load_or_create_model(self) -> HarmonicVisionTransformer:
        """
        Load model from .pt or create blank.
        After load, the .pt file is "released" - we work on in-memory weights.
        """
        model = HarmonicVisionTransformer(
            num_classes=self.config.num_classes,
            num_freq_bands=self.config.num_freq_bands,
            base_omega=self.config.base_omega,
            coupling_strength=self.config.coupling_strength,
            learnable_physics=self.config.learnable_physics,
            patch_size=self.config.patch_size,
            hidden_dim=self.config.hidden_dim,
            num_routing_heads=self.config.num_routing_heads,
            num_evolution_layers=self.config.num_evolution_layers,
            use_phase_routing=self.config.use_phase_routing,
            use_spectral_norm=self.config.use_spectral_norm,
        )
        
        model_path = Path(self.config.checkpoint_dir) / "hvt_model.pt"
        
        if model_path.exists():
            self.logger.info(f"Loading existing model from {model_path}")
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            self.logger.info("Model loaded. File released, working on in-memory weights.")
        else:
            self.logger.info("No existing model found. Creating blank untrained model.")
            Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        return model
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer - AdamW for stable training (matches train_multi.py)."""
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.total_steps,
            eta_min=self.config.min_learning_rate,
        )
    
    def _training_step(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single training step."""
        self.model.train()
        
        # Unpack batch
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(images, return_intermediates=True)
        
        # Compute loss
        losses = self.loss_fn(
            predictions=outputs,
            targets=images,
            sync_order=outputs.get('sync_order'),
            phase_evolution=outputs.get('phase_evolution'),
            amplitude_evolution=outputs.get('amplitude_evolution'),
            xi_motion=outputs.get('xi_motion'),
            labels=labels,
        )
        
        total_loss = losses['total']
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
        
        # Optimizer step
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            if 'logits' in outputs and outputs['logits'] is not None:
                preds = outputs['logits'].argmax(dim=-1)
                accuracy = (preds == labels).float().mean().item()
            else:
                accuracy = 0.0
            
            if 'sync_order' in outputs:
                sync_order = outputs['sync_order'].mean().item()
            else:
                sync_order = 0.0
        
        metrics = {
            'loss': total_loss.item(),
            'accuracy': accuracy,
            'sync_order': sync_order,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return total_loss, metrics
    
    def _validation_step(self) -> Dict[str, float]:
        """Run validation on entire validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_sync = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if len(batch) == 3:
                    images, labels, _ = batch
                else:
                    images, labels = batch
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images, return_intermediates=True)
                
                # Loss
                losses = self.loss_fn(
                    predictions=outputs,
                    targets=images,
                    sync_order=outputs.get('sync_order'),
                    labels=labels,
                )
                total_loss += losses['total'].item() * images.size(0)
                
                # Accuracy
                if 'logits' in outputs and outputs['logits'] is not None:
                    preds = outputs['logits'].argmax(dim=-1)
                    total_correct += (preds == labels).sum().item()
                
                # Sync order
                if 'sync_order' in outputs:
                    total_sync += outputs['sync_order'].mean().item() * images.size(0)
                
                total_samples += images.size(0)
        
        return {
            'val_loss': total_loss / max(total_samples, 1),
            'val_accuracy': total_correct / max(total_samples, 1),
            'val_sync_order': total_sync / max(total_samples, 1),
        }
    
    def _backup_current_state(self, epoch: int):
        """Optional per-epoch backup (NOT checkpoint)."""
        backup_dir = Path(self.config.checkpoint_dir) / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_path = backup_dir / f"backup_epoch_{epoch:04d}.pt"
        torch.save(self.model.state_dict(), backup_path)
        self.logger.info(f"Backup saved: {backup_path}")
    
    def _save_final_model(self):
        """Serialize in-memory weights back to .pt"""
        model_path = Path(self.config.checkpoint_dir) / "hvt_model.pt"
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Final model saved: {model_path}")
        
        # Also save training history
        history_path = Path(self.config.checkpoint_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        self.logger.info(f"Training history saved: {history_path}")
    
    def train(self, backup_every_n_epochs: int = 0):
        """
        Main training loop.
        
        Args:
            backup_every_n_epochs: If > 0, save backup every N epochs
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting HVT Training")
        self.logger.info(f"Total steps: {self.config.total_steps}")
        self.logger.info(f"Batch size: {self.config.batch_size}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        epoch_start_time = start_time
        
        while self.step < self.config.total_steps:
            self.epoch += 1
            epoch_metrics = []
            
            for batch in self.train_loader:
                if self.step >= self.config.total_steps:
                    break
                
                self.step += 1
                step_start = time.time()
                
                # Training step
                loss, metrics = self._training_step(batch)
                
                # LR schedule
                self.scheduler.step()
                
                # Track metrics
                step_time = time.time() - step_start
                metrics['samples_per_second'] = self.config.batch_size / step_time
                epoch_metrics.append(metrics)
                
                # Log progress
                if self.step % self.config.log_every == 0:
                    avg_loss = sum(m['loss'] for m in epoch_metrics[-self.config.log_every:]) / min(len(epoch_metrics), self.config.log_every)
                    avg_acc = sum(m['accuracy'] for m in epoch_metrics[-self.config.log_every:]) / min(len(epoch_metrics), self.config.log_every)
                    avg_sync = sum(m['sync_order'] for m in epoch_metrics[-self.config.log_every:]) / min(len(epoch_metrics), self.config.log_every)
                    
                    self.logger.info(
                        f"Step {self.step:>6}/{self.config.total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Acc: {avg_acc*100:.1f}% | "
                        f"Sync: {avg_sync:.3f} | "
                        f"LR: {metrics['lr']:.2e}"
                    )
                
                # Validation
                if self.step % self.config.val_every == 0:
                    val_metrics = self._validation_step()
                    self.logger.info(
                        f"[VAL] Loss: {val_metrics['val_loss']:.4f} | "
                        f"Acc: {val_metrics['val_accuracy']*100:.1f}% | "
                        f"Sync: {val_metrics['val_sync_order']:.3f}"
                    )
                    
                    # Track best
                    if val_metrics['val_accuracy'] > self.best_accuracy:
                        self.best_accuracy = val_metrics['val_accuracy']
                        self.logger.info(f"New best accuracy: {self.best_accuracy*100:.1f}%")
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            epoch_start_time = time.time()
            
            # Store epoch summary
            epoch_summary = {
                'epoch': self.epoch,
                'step': self.step,
                'avg_loss': sum(m['loss'] for m in epoch_metrics) / len(epoch_metrics),
                'avg_accuracy': sum(m['accuracy'] for m in epoch_metrics) / len(epoch_metrics),
                'avg_sync_order': sum(m['sync_order'] for m in epoch_metrics) / len(epoch_metrics),
                'epoch_time': epoch_time,
            }
            self.training_history.append(epoch_summary)
            
            self.logger.info(f"Epoch {self.epoch} completed in {epoch_time:.1f}s")
            
            # Optional backup
            if backup_every_n_epochs > 0 and self.epoch % backup_every_n_epochs == 0:
                self._backup_current_state(self.epoch)
        
        # Training complete - save final model
        total_time = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info(f"Training complete!")
        self.logger.info(f"Total time: {total_time/60:.1f} minutes")
        self.logger.info(f"Best accuracy: {self.best_accuracy*100:.1f}%")
        self.logger.info("=" * 60)
        
        self._save_final_model()
        
        return self.training_history
    
    def get_model(self) -> HarmonicVisionTransformer:
        """Return the trained model."""
        return self.model


def train_hvt(config_path: str = None, backup_epochs: int = 0) -> Dict[str, Any]:
    """
    Convenience function to train HVT.
    
    Args:
        config_path: Path to YAML config file
        backup_epochs: Save backup every N epochs (0 = disabled)
    
    Returns:
        Training history
    """
    if config_path:
        config = TrainingConfig.from_yaml(config_path)
    else:
        config = TrainingConfig()
    
    trainer = HVTProductionTrainer(config)
    history = trainer.train(backup_every_n_epochs=backup_epochs)
    
    return {
        'history': history,
        'best_accuracy': trainer.best_accuracy,
        'model_path': str(Path(config.checkpoint_dir) / "hvt_model.pt"),
    }
