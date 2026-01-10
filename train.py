"""
Harmonic Vision Transformer v2.0 - Training Script

World-class training designed for oscillator-based architectures.
NOT a standard ML training loop - designed to let the model "breathe".

Features:
- Streaming dataset loading (no RAM explosion)
- Memory-mapped checkpoints (.safetensors)
- Harmonic learning rate schedule (golden ratio modulation)
- Oscillator warmup (let phases stabilize before classification pressure)
- Sync order monitoring (the heartbeat of the model)
- Breathing cycles (periodic consolidation phases)

Author: Vision Modality Research Initiative
Date: January 2026
"""

import os
import sys
import time
import math
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hvt_v2 import (
    HarmonicVisionTransformer,
    HarmonicLoss,
    PHI,
    SACRED_RATIO,
)

# Optional imports
try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("⚠ safetensors not installed. Using .pt format. Install with: pip install safetensors")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("⚠ datasets not installed. Install with: pip install datasets")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for harmonic training."""
    
    # Dataset
    dataset_name: str = "uoft-cs/cifar10"
    dataset_config: str = None
    dataset_split: str = "train"
    streaming: bool = True
    image_key: str = "img"
    label_key: str = "label"
    
    # Model
    num_freq_bands: int = 4
    num_evolution_layers: int = 3
    hidden_dim: int = 64
    num_classes: int = 10
    image_size: int = 32
    
    # Training phases
    total_steps: int = 10000
    warmup_steps: int = 500         # Let oscillators stabilize
    breathing_interval: int = 100    # Steps between "breathing" cycles
    breathing_duration: int = 10     # Steps of reduced pressure
    
    # Optimization
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    batch_size: int = 32
    gradient_clip: float = 1.0
    
    # Harmonic-specific
    sync_target: float = 0.618       # Golden ratio complement
    oscillator_warmup_weight: float = 0.1  # Low classification pressure initially
    
    # Memory management
    num_workers: int = 0             # 0 for streaming
    prefetch_factor: int = 2
    pin_memory: bool = False         # CPU training
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 500
    log_every: int = 25
    
    # Device
    device: str = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Streaming Dataset Wrapper
# ============================================================================

class StreamingImageDataset(IterableDataset):
    """
    Memory-efficient streaming dataset wrapper.
    
    Streams data from HuggingFace without loading to RAM.
    Applies minimal preprocessing on-the-fly.
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_key: str = "img",
        label_key: str = "label",
        image_size: int = 32,
        streaming: bool = True,
        dataset_config: str = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.image_key = image_key
        self.label_key = label_key
        self.image_size = image_size
        self.streaming = streaming
        self.dataset_config = dataset_config
        
        # Mean/std for normalization (CIFAR-10 defaults)
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        
    def _load_dataset(self):
        """Lazy load the dataset."""
        if self.dataset_config:
            return load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.split,
                streaming=self.streaming,
            )
        return load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=self.streaming,
        )
    
    def _process_sample(self, sample: Dict) -> Tuple[torch.Tensor, int]:
        """Process a single sample."""
        # Get image
        img = sample[self.image_key]
        
        # Convert PIL to tensor if needed
        if hasattr(img, 'convert'):
            img = img.convert('RGB')
            img = torch.from_numpy(np.array(img)).float() / 255.0
            img = img.permute(2, 0, 1)  # HWC -> CHW
        elif isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
            if img.max() > 1.0:
                img = img / 255.0
            if img.dim() == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)
        
        # Resize if needed
        if img.shape[-1] != self.image_size or img.shape[-2] != self.image_size:
            img = F.interpolate(
                img.unsqueeze(0), 
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Normalize
        img = (img - self.mean) / self.std
        
        # Get label
        label = sample[self.label_key]
        if isinstance(label, (list, np.ndarray)):
            label = label[0]
        
        return img, int(label)
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        dataset = self._load_dataset()
        for sample in dataset:
            try:
                yield self._process_sample(sample)
            except Exception as e:
                # Skip bad samples silently
                continue


def collate_streaming(batch):
    """Collate function for streaming batches."""
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)


# ============================================================================
# Harmonic Learning Rate Schedule
# ============================================================================

class HarmonicScheduler:
    """
    Learning rate scheduler with golden ratio modulation.
    
    Not cosine annealing - harmonic breathing with φ-based cycles.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int,
        min_lr: float,
        breathing_interval: int = 100,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.breathing_interval = breathing_interval
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.current_step = 0
        
    def step(self) -> float:
        """Update learning rate and return current value."""
        self.current_step += 1
        lr = self._compute_lr()
        
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = lr * (base_lr / self.base_lrs[0])
        
        return lr
    
    def _compute_lr(self) -> float:
        step = self.current_step
        
        # Phase 1: Linear warmup
        if step < self.warmup_steps:
            return self.base_lrs[0] * (step / self.warmup_steps)
        
        # Phase 2: Harmonic decay with breathing
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        # Base cosine decay
        base_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        # Golden ratio breathing modulation
        breathing_phase = (step % self.breathing_interval) / self.breathing_interval
        breathing_mod = 1.0 + 0.1 * math.sin(2 * math.pi * PHI * breathing_phase)
        
        # Combine
        lr = self.min_lr + (self.base_lrs[0] - self.min_lr) * base_decay * breathing_mod
        
        return lr


# ============================================================================
# Training State & Checkpointing
# ============================================================================

@dataclass
class TrainingState:
    """Mutable training state."""
    step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    best_sync: float = 0.0
    total_samples: int = 0
    
    # Running metrics
    loss_ema: float = 0.0
    sync_ema: float = 0.0
    accuracy_ema: float = 0.0
    
    # History (last N values for analysis)
    loss_history: list = None
    sync_history: list = None
    
    def __post_init__(self):
        if self.loss_history is None:
            self.loss_history = []
        if self.sync_history is None:
            self.sync_history = []
    
    def update_ema(self, loss: float, sync: float, accuracy: float, alpha: float = 0.99):
        """Update exponential moving averages."""
        self.loss_ema = alpha * self.loss_ema + (1 - alpha) * loss
        self.sync_ema = alpha * self.sync_ema + (1 - alpha) * sync
        self.accuracy_ema = alpha * self.accuracy_ema + (1 - alpha) * accuracy
        
        # Keep last 100 values
        self.loss_history.append(loss)
        self.sync_history.append(sync)
        if len(self.loss_history) > 100:
            self.loss_history.pop(0)
            self.sync_history.pop(0)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainingState,
    config: TrainingConfig,
    path: Path,
):
    """Save checkpoint with memory-mapped safetensors."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Model weights
    model_path = path / f"model_step_{state.step}"
    
    if HAS_SAFETENSORS:
        # Save as safetensors (memory-mapped, secure)
        save_file(model.state_dict(), str(model_path) + ".safetensors")
    else:
        # Fallback to .pt
        torch.save(model.state_dict(), str(model_path) + ".pt")
    
    # Optimizer and state as JSON/PT (small)
    meta = {
        'step': state.step,
        'epoch': state.epoch,
        'best_loss': state.best_loss,
        'best_sync': state.best_sync,
        'total_samples': state.total_samples,
        'loss_ema': state.loss_ema,
        'sync_ema': state.sync_ema,
        'accuracy_ema': state.accuracy_ema,
        'config': config.to_dict(),
    }
    
    with open(path / f"meta_step_{state.step}.json", 'w') as f:
        json.dump(meta, f, indent=2)
    
    # Optimizer state (for resume)
    torch.save(optimizer.state_dict(), path / f"optim_step_{state.step}.pt")
    
    return model_path


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: Path,
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], TrainingState]:
    """Load checkpoint from safetensors or .pt."""
    path = Path(path)
    
    # Find latest checkpoint
    safetensor_files = list(path.glob("model_step_*.safetensors"))
    pt_files = list(path.glob("model_step_*.pt"))
    
    if safetensor_files and HAS_SAFETENSORS:
        latest = max(safetensor_files, key=lambda p: int(p.stem.split('_')[-1]))
        state_dict = load_file(str(latest))
    elif pt_files:
        latest = max(pt_files, key=lambda p: int(p.stem.split('_')[-1]))
        state_dict = torch.load(latest, map_location='cpu')
    else:
        raise FileNotFoundError(f"No checkpoints found in {path}")
    
    model.load_state_dict(state_dict)
    
    # Load meta
    step = int(latest.stem.split('_')[-1])
    meta_path = path / f"meta_step_{step}.json"
    
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        state = TrainingState(
            step=meta['step'],
            epoch=meta.get('epoch', 0),
            best_loss=meta.get('best_loss', float('inf')),
            best_sync=meta.get('best_sync', 0.0),
            total_samples=meta.get('total_samples', 0),
            loss_ema=meta.get('loss_ema', 0.0),
            sync_ema=meta.get('sync_ema', 0.0),
            accuracy_ema=meta.get('accuracy_ema', 0.0),
        )
    else:
        state = TrainingState(step=step)
    
    # Load optimizer
    if optimizer is not None:
        optim_path = path / f"optim_step_{step}.pt"
        if optim_path.exists():
            optimizer.load_state_dict(torch.load(optim_path, map_location='cpu'))
    
    return model, optimizer, state


# ============================================================================
# Training Loop
# ============================================================================

class HarmonicTrainer:
    """
    Training loop designed for oscillator-based architectures.
    
    Key differences from standard training:
    1. Oscillator warmup - let phases stabilize before classification pressure
    2. Breathing cycles - periodic reduced-pressure consolidation
    3. Sync monitoring - the "heartbeat" of the model
    4. Adaptive pressure - based on sync order stability
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = HarmonicVisionTransformer(
            num_freq_bands=config.num_freq_bands,
            base_omega=SACRED_RATIO,
            num_evolution_layers=config.num_evolution_layers,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            use_phase_routing=True,
        ).to(self.device)
        
        # Loss function
        self.loss_fn = HarmonicLoss(
            reconstruction_weight=0.5,
            sync_regularization=0.1,
            phase_smoothness=0.01,
            energy_conservation=0.05,
            target_sync=config.sync_target,
        )
        
        # Optimizer - use Adam (not AdamW) to avoid in-place weight decay conflicts
        # with spectral norm. Apply weight decay manually in backward if needed.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
        )
        
        # Scheduler
        self.scheduler = HarmonicScheduler(
            self.optimizer,
            total_steps=config.total_steps,
            warmup_steps=config.warmup_steps,
            min_lr=config.min_learning_rate,
            breathing_interval=config.breathing_interval,
        )
        
        # State
        self.state = TrainingState()
        
        # Dataset
        self.dataset = None
        self.dataloader = None
        
        print(self._header())
        print(f"\n  Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Device: {self.device}")
        print(f"  Streaming: {config.streaming}")
        print()
    
    def _header(self) -> str:
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    HARMONIC VISION TRANSFORMER TRAINING                      ║
║                                                                              ║
║  "The model breathes. Let it."                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    
    def setup_data(self):
        """Initialize streaming dataset."""
        if not HAS_DATASETS:
            raise ImportError("datasets library required. pip install datasets")
        
        self.dataset = StreamingImageDataset(
            dataset_name=self.config.dataset_name,
            split=self.config.dataset_split,
            image_key=self.config.image_key,
            label_key=self.config.label_key,
            image_size=self.config.image_size,
            streaming=self.config.streaming,
            dataset_config=self.config.dataset_config,
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=collate_streaming,
            pin_memory=self.config.pin_memory,
        )
        
        dataset_str = self.config.dataset_name
        if self.config.dataset_config:
            dataset_str += f" ({self.config.dataset_config})"
        print(f"  Dataset: {dataset_str}")
        print(f"  Batch size: {self.config.batch_size}")
        print()
    
    def _is_breathing(self) -> bool:
        """Check if we're in a breathing (consolidation) phase."""
        cycle_pos = self.state.step % self.config.breathing_interval
        return cycle_pos >= (self.config.breathing_interval - self.config.breathing_duration)
    
    def _compute_classification_weight(self) -> float:
        """
        Compute classification loss weight based on training phase.
        
        During warmup: low weight (let oscillators stabilize)
        During breathing: reduced weight (consolidation)
        Normal: full weight
        """
        # Warmup ramp
        if self.state.step < self.config.warmup_steps:
            warmup_progress = self.state.step / self.config.warmup_steps
            base_weight = self.config.oscillator_warmup_weight + \
                         (1.0 - self.config.oscillator_warmup_weight) * warmup_progress
        else:
            base_weight = 1.0
        
        # Breathing reduction
        if self._is_breathing():
            base_weight *= 0.5
        
        return base_weight
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        outputs = self.model(images, return_intermediates=True)
        
        # Compute losses
        losses = self.loss_fn(
            predictions=outputs['output'],
            targets=images,  # Reconstruction target
            sync_order=outputs['sync_order'],
            phase_evolution=outputs.get('phase_evolution'),
            amplitude_evolution=outputs.get('amplitude_evolution'),
        )
        
        # Classification loss (with adaptive weight)
        cls_weight = self._compute_classification_weight()
        cls_loss = F.cross_entropy(outputs['logits'], labels)
        
        # Total loss
        total_loss = losses['total'] + cls_weight * cls_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip
            )
        
        self.optimizer.step()
        lr = self.scheduler.step()
        
        # Compute accuracy
        with torch.no_grad():
            preds = outputs['logits'].argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item()
        
        # Sync order
        sync_order = outputs['sync_order'].mean().item()
        
        return {
            'loss': total_loss.item(),
            'cls_loss': cls_loss.item(),
            'recon_loss': losses['reconstruction'].item(),
            'sync_loss': losses['sync'].item(),
            'sync_order': sync_order,
            'accuracy': accuracy,
            'lr': lr,
            'cls_weight': cls_weight,
        }
    
    def train(self, resume_from: Optional[Path] = None):
        """Main training loop."""
        
        # Setup data
        self.setup_data()
        
        # Resume if requested
        if resume_from is not None:
            self.model, self.optimizer, self.state = load_checkpoint(
                self.model, self.optimizer, resume_from
            )
            print(f"  Resumed from step {self.state.step}")
        
        # Training loop
        print("  Starting training...")
        print("  " + "─" * 70)
        print()
        
        start_time = time.time()
        data_iter = iter(self.dataloader)
        
        while self.state.step < self.config.total_steps:
            # Get batch (restart iterator if exhausted)
            try:
                images, labels = next(data_iter)
            except StopIteration:
                self.state.epoch += 1
                data_iter = iter(self.dataloader)
                images, labels = next(data_iter)
            
            # Training step
            metrics = self.train_step(images, labels)
            
            # Update state
            self.state.step += 1
            self.state.total_samples += len(labels)
            self.state.update_ema(
                metrics['loss'], 
                metrics['sync_order'], 
                metrics['accuracy']
            )
            
            # Logging
            if self.state.step % self.config.log_every == 0:
                self._log_step(metrics, start_time)
            
            # Checkpointing
            if self.state.step % self.config.save_every == 0:
                self._save_checkpoint()
        
        # Final checkpoint
        self._save_checkpoint()
        
        # Save to model file if specified
        if hasattr(self, 'model_path') and self.model_path:
            import torch
            torch.save(self.model.state_dict(), self.model_path)
            print(f"  💾 Saved to {self.model_path}")
        
        print("\n  Training complete! 🌊")
    
    def _log_step(self, metrics: Dict[str, float], start_time: float):
        """Log training progress."""
        elapsed = time.time() - start_time
        steps_per_sec = self.state.step / elapsed if elapsed > 0 else 0
        
        # Phase indicator
        if self.state.step < self.config.warmup_steps:
            phase = "🔥 WARMUP"
        elif self._is_breathing():
            phase = "🌬️ BREATHING"
        else:
            phase = "⚡ TRAINING"
        
        # Sync health indicator
        sync = metrics['sync_order']
        if sync > 0.7:
            sync_indicator = "🟢"
        elif sync > 0.4:
            sync_indicator = "🟡"
        else:
            sync_indicator = "🔴"
        
        print(f"  Step {self.state.step:>6} │ {phase:12} │ "
              f"Loss: {metrics['loss']:.4f} │ "
              f"Acc: {metrics['accuracy']*100:>5.1f}% │ "
              f"Sync: {sync_indicator} {sync:.3f} │ "
              f"LR: {metrics['lr']:.2e} │ "
              f"{steps_per_sec:.1f} steps/s")
    
    def _save_checkpoint(self):
        """Save checkpoint."""
        ckpt_path = Path(self.config.checkpoint_dir)
        save_checkpoint(
            self.model, 
            self.optimizer, 
            self.state, 
            self.config, 
            ckpt_path
        )
        
        # Update best
        if self.state.loss_ema < self.state.best_loss:
            self.state.best_loss = self.state.loss_ema
            
            # Save as 'best'
            if HAS_SAFETENSORS:
                save_file(self.model.state_dict(), str(ckpt_path / "best.safetensors"))
            else:
                torch.save(self.model.state_dict(), str(ckpt_path / "best.pt"))
        
        print(f"  💾 Checkpoint saved at step {self.state.step}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Harmonic Vision Transformer v2.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='uoft-cs/cifar10',
                       help='HuggingFace dataset name')
    parser.add_argument('--dataset-config', type=str, default=None,
                       help='Dataset config name (e.g., cropped_digits for SVHN)')
    parser.add_argument('--image-key', type=str, default='img',
                       help='Key for image in dataset')
    parser.add_argument('--label-key', type=str, default='label',
                       help='Key for label in dataset')
    
    # Model
    parser.add_argument('--num-bands', type=int, default=4,
                       help='Number of frequency bands')
    parser.add_argument('--num-layers', type=int, default=3,
                       help='Number of evolution layers')
    parser.add_argument('--hidden-dim', type=int, default=64,
                       help='Hidden dimension')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of classes')
    parser.add_argument('--image-size', type=int, default=32,
                       help='Input image size')
    
    # Training
    parser.add_argument('--steps', type=int, default=10000,
                       help='Total training steps')
    parser.add_argument('--warmup', type=int, default=500,
                       help='Warmup steps')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--save-every', type=int, default=500,
                       help='Save checkpoint every N steps')
    parser.add_argument('--log-every', type=int, default=25,
                       help='Log every N steps')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint directory')
    parser.add_argument('--model', type=str, default='hvt_model.pt',
                       help='Path to model file (loads from and saves to this file)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='[DEPRECATED] Use --model instead')
    parser.add_argument('--auto-resume', action='store_true',
                       help='Auto-resume from checkpoint-dir if exists')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Train for N epochs (50k samples/epoch for CIFAR-10)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Calculate steps from epochs if provided (CIFAR-10 = 50,000 samples)
    if args.epochs is not None:
        samples_per_epoch = 50000  # CIFAR-10
        steps_per_epoch = samples_per_epoch // args.batch_size
        total_steps = args.epochs * steps_per_epoch
        print(f"  📊 Training for {args.epochs} epochs = {total_steps} steps")
    else:
        total_steps = args.steps
    
    config = TrainingConfig(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        image_key=args.image_key,
        label_key=args.label_key,
        num_freq_bands=args.num_bands,
        num_evolution_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        image_size=args.image_size,
        total_steps=total_steps,
        warmup_steps=args.warmup,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        log_every=args.log_every,
    )
    
    trainer = HarmonicTrainer(config)
    
    # Determine model path (prefer --model, fallback to deprecated --checkpoint)
    model_path = args.model
    if args.checkpoint:
        print("  ⚠️  --checkpoint is deprecated, use --model instead")
        model_path = args.checkpoint
    
    # Load from model file if it exists
    model_file = Path(model_path)
    if model_file.exists():
        print(f"  📦 Loading weights from {model_file}")
        if HAS_SAFETENSORS and str(model_file).endswith('.safetensors'):
            state_dict = load_file(str(model_file))
        else:
            state_dict = torch.load(model_file, map_location='cpu')
        trainer.model.load_state_dict(state_dict, strict=False)
        print(f"  ✅ Loaded model weights ({len(state_dict)} keys)")
        resume_path = None  # Don't resume training state, just weights
    else:
        print(f"  🆕 Starting fresh (no {model_file} found)")
        resume_path = None
    
    # Store model_path for saving
    trainer.model_path = model_path
    
    # Auto-resume: check if checkpoint exists
    if args.auto_resume and ckpt_dir.exists():
        ckpts = list(ckpt_dir.glob("model_step_*.safetensors")) + list(ckpt_dir.glob("model_step_*.pt"))
        if ckpts:
            resume_path = ckpt_dir
            print(f"  🔄 Auto-resuming from {ckpt_dir}")
        else:
            resume_path = None
    elif args.resume:
        resume_path = Path(args.resume)
    else:
        resume_path = None
    
    # Define ckpt_dir for non-checkpoint cases
    ckpt_dir = Path(args.checkpoint_dir)
    
    trainer.train(resume_from=resume_path)


if __name__ == "__main__":
    main()
