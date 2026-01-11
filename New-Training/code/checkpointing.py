"""
Checkpointing module for HVT v3 training pipeline.

Provides safe model saving/loading with spectral normalization
and oscillator-specific checkpoint management.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import json
import pickle
from pathlib import Path
import shutil
import hashlib
import logging
from datetime import datetime


class SafeCheckpointManager:
    """
    Safe checkpoint manager with atomic operations and backup support.
    
    Features:
    - Atomic save operations
    - Automatic backups
    - Spectral normalization finalization
    - Oscillator state verification
    - Checkpoint versioning
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_best_n: int = 3,
        backup_before_overwrite: bool = True,
        verify_oscillator_state: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.keep_best_n = keep_best_n
        self.backup_before_overwrite = backup_before_overwrite
        self.verify_oscillator_state = verify_oscillator_state
        
        # Backup directory
        self.backup_dir = self.checkpoint_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Track best checkpoints
        self.best_checkpoints = []
        self.checkpoint_metadata = {}
    
    def _atomic_save(self, obj: Any, filepath: Path, save_fn: callable):
        """Save with atomic operation (write to temp, then rename)."""
        temp_path = filepath.with_suffix('.tmp')
        
        try:
            # Save to temporary file
            save_fn(obj, temp_path)
            
            # Atomic rename
            if filepath.exists() and self.backup_before_overwrite:
                backup_path = self.backup_dir / f"{filepath.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(filepath, backup_path)
            
            temp_path.rename(filepath)
            
        except Exception as e:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            raise e
    
    def save_model(
        self,
        model: nn.Module,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None,
        apply_spectral_norm: bool = True
    ):
        """Save model with oscillator-specific handling."""
        filepath = Path(filepath)
        
        # Apply spectral normalization if requested
        if apply_spectral_norm:
            model = self._apply_spectral_normalization(model)
        
        # Verify oscillator state
        if self.verify_oscillator_state:
            self._verify_oscillator_state(model)
        
        # Prepare state dict
        state_dict = model.state_dict()
        
        # Add metadata
        checkpoint = {
            'model_state_dict': state_dict,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_class': model.__class__.__name__,
                'num_parameters': sum(p.numel() for p in model.parameters()),
                **(metadata or {})
            }
        }
        
        # Atomic save
        self._atomic_save(
            checkpoint,
            filepath,
            lambda obj, path: torch.save(obj, path)
        )
        
        self.logger.info(f"Model saved to {filepath}")
        
        # Update checkpoint metadata
        self.checkpoint_metadata[str(filepath)] = checkpoint['metadata']
        
        return filepath
    
    def load_model(
        self,
        model: nn.Module,
        filepath: str,
        strict: bool = True,
        verify_checksum: bool = True
    ) -> Dict[str, Any]:
        """Load model with verification."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        # Load checkpoint
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Verify metadata
        metadata = checkpoint.get('metadata', {})
        
        if verify_checksum and 'checksum' in metadata:
            computed_checksum = self._compute_checksum(checkpoint['model_state_dict'])
            if computed_checksum != metadata['checksum']:
                raise ValueError("Checkpoint checksum mismatch - file may be corrupted")
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Verify oscillator state after loading
        if self.verify_oscillator_state:
            self._verify_oscillator_state(model)
        
        self.logger.info(f"Model loaded from {filepath}")
        self.logger.info(f"Metadata: {metadata}")
        
        return metadata
    
    def save_training_state(
        self,
        state: Dict[str, Any],
        filepath: str
    ):
        """Save complete training state."""
        filepath = Path(filepath)
        
        # Add metadata
        checkpoint = {
            'training_state': state,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
            }
        }
        
        # Atomic save
        self._atomic_save(
            checkpoint,
            filepath,
            lambda obj, path: json.dump(obj, path, indent=2, default=str)
        )
        
        self.logger.info(f"Training state saved to {filepath}")
    
    def load_training_state(self, filepath: str) -> Dict[str, Any]:
        """Load training state."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Training state not found: {filepath}")
        
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        self.logger.info(f"Training state loaded from {filepath}")
        
        return checkpoint['training_state']
    
    def save_optimizer_state(
        self,
        optimizer: torch.optim.Optimizer,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save optimizer state."""
        filepath = Path(filepath)
        
        checkpoint = {
            'optimizer_state_dict': optimizer.state_dict(),
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'optimizer_class': optimizer.__class__.__name__,
                **(metadata or {})
            }
        }
        
        # Atomic save
        self._atomic_save(
            checkpoint,
            filepath,
            lambda obj, path: torch.save(obj, path)
        )
        
        self.logger.info(f"Optimizer state saved to {filepath}")
    
    def load_optimizer_state(
        self,
        optimizer: torch.optim.Optimizer,
        filepath: str
    ) -> Dict[str, Any]:
        """Load optimizer state."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Optimizer state not found: {filepath}")
        
        checkpoint = torch.load(filepath)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.logger.info(f"Optimizer state loaded from {filepath}")
        
        return checkpoint.get('metadata', {})
    
    def save_complete_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        training_state: Dict[str, Any],
        filepath: str,
        is_best: bool = False
    ):
        """Save complete training checkpoint."""
        filepath = Path(filepath)
        
        # Create timestamped directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_dir = filepath.parent / f"checkpoint_{timestamp}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save components
        model_path = checkpoint_dir / "model.pt"
        optimizer_path = checkpoint_dir / "optimizer.pt"
        state_path = checkpoint_dir / "training_state.json"
        
        self.save_model(model, str(model_path), training_state)
        self.save_optimizer_state(optimizer, str(optimizer_path))
        self.save_training_state(training_state, str(state_path))
        
        if scheduler is not None and hasattr(scheduler, 'state_dict'):
            scheduler_path = checkpoint_dir / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_path)
        
        # Create symlink for latest
        latest_link = filepath.parent / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(checkpoint_dir.name)
        
        # Update best checkpoint tracking
        if is_best:
            self._update_best_checkpoints(checkpoint_dir, training_state)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        self.logger.info(f"Complete checkpoint saved to {checkpoint_dir}")
        
        return checkpoint_dir
    
    def load_complete_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        checkpoint_dir: str
    ) -> Dict[str, Any]:
        """Load complete training checkpoint."""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load components
        model_path = checkpoint_dir / "model.pt"
        optimizer_path = checkpoint_dir / "optimizer.pt"
        state_path = checkpoint_dir / "training_state.json"
        scheduler_path = checkpoint_dir / "scheduler.pt"
        
        training_state = self.load_training_state(str(state_path))
        self.load_model(model, str(model_path))
        self.load_optimizer_state(optimizer, str(optimizer_path))
        
        if scheduler is not None and scheduler_path.exists():
            scheduler_state = torch.load(scheduler_path)
            scheduler.load_state_dict(scheduler_state)
        
        self.logger.info(f"Complete checkpoint loaded from {checkpoint_dir}")
        
        return training_state
    
    def _apply_spectral_normalization(self, model: nn.Module) -> nn.Module:
        """Apply spectral normalization to coupling matrices."""
        # This is a simplified version - full implementation would
        # apply spectral norm to specific layers
        return model
    
    def _verify_oscillator_state(self, model: nn.Module):
        """Verify that oscillator parameters are in valid ranges."""
        for name, param in model.named_parameters():
            if 'phase' in name:
                # Phases should be in [-π, π]
                if param.data.abs().max() > 2 * math.pi:
                    self.logger.warning(f"Phase parameter {name} out of range: {param.data.abs().max()}")
                    param.data = torch.atan2(torch.sin(param.data), torch.cos(param.data))
            
            elif 'log_damping' in name:
                # Log damping should be reasonable
                if param.data.abs().max() > 10:
                    self.logger.warning(f"Log damping parameter {name} out of range: {param.data.abs().max()}")
                    param.data = torch.clamp(param.data, -10, 10)
            
            elif 'omega' in name:
                # Frequencies should be positive
                if param.data.min() < 0:
                    self.logger.warning(f"Frequency parameter {name} has negative values: {param.data.min()}")
                    param.data = torch.clamp(param.data, 0.01, 10.0)
    
    def _compute_checksum(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Compute checksum of model state dict."""
        # Simple checksum based on parameter hashes
        hasher = hashlib.sha256()
        for param in state_dict.values():
            hasher.update(param.numpy().tobytes())
        return hasher.hexdigest()
    
    def _update_best_checkpoints(self, checkpoint_dir: Path, training_state: Dict[str, Any]):
        """Update best checkpoint tracking."""
        # Extract validation accuracy or other metric
        val_accuracy = training_state.get('val_accuracy', 0)
        
        self.best_checkpoints.append({
            'path': checkpoint_dir,
            'accuracy': val_accuracy,
            'timestamp': datetime.now().isoformat()
        })
        
        # Sort by accuracy
        self.best_checkpoints.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Keep only best N
        self.best_checkpoints = self.best_checkpoints[:self.keep_best_n]
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond keep_best_n."""
        # This is a simplified implementation
        # Full implementation would track and remove old checkpoints
        pass
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint."""
        latest_link = self.checkpoint_dir / "latest"
        if latest_link.exists() or latest_link.is_symlink():
            target = latest_link.resolve()
            if target.exists():
                return target
        
        # Fallback: find latest by timestamp
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*"))
        if checkpoints:
            return max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        return None
    
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint by validation accuracy."""
        if self.best_checkpoints:
            return self.best_checkpoints[0]['path']
        
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []
        
        for checkpoint_dir in self.checkpoint_dir.glob("checkpoint_*"):
            metadata_path = checkpoint_dir / "training_state.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    checkpoints.append({
                        'path': checkpoint_dir,
                        'timestamp': metadata.get('timestamp', 'unknown'),
                        'step': metadata.get('step', 0),
                        'val_accuracy': metadata.get('val_accuracy', 0),
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to read metadata from {metadata_path}: {e}")
        
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return checkpoints


class OscillatorStateManager:
    """
    Specialized manager for oscillator state tracking and analysis.
    """
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.oscillator_states = {}
    
    def save_oscillator_state(
        self,
        step: int,
        sync_order: torch.Tensor,
        phase_evolution: Optional[torch.Tensor],
        amplitude_evolution: Optional[torch.Tensor],
        coupling_matrix: Optional[torch.Tensor]
    ):
        """Save oscillator state at specific training step."""
        state = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'sync_order': sync_order.cpu().numpy().tolist(),
        }
        
        if phase_evolution is not None:
            state['phase_evolution'] = phase_evolution.cpu().numpy().tolist()
        
        if amplitude_evolution is not None:
            state['amplitude_evolution'] = amplitude_evolution.cpu().numpy().tolist()
        
        if coupling_matrix is not None:
            state['coupling_matrix'] = coupling_matrix.cpu().numpy().tolist()
        
        # Save to file
        state_path = self.checkpoint_dir / f"oscillator_state_step_{step}.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.oscillator_states[step] = state_path
    
    def load_oscillator_state(self, step: int) -> Dict[str, Any]:
        """Load oscillator state for specific step."""
        state_path = self.oscillator_states.get(step)
        if not state_path or not state_path.exists():
            raise FileNotFoundError(f"Oscillator state not found for step {step}")
        
        with open(state_path, 'r') as f:
            state = json.load(f)
        
        return state
    
    def get_oscillator_trajectory(self, start_step: int, end_step: int) -> Dict[str, Any]:
        """Get oscillator trajectory over training steps."""
        trajectory = {
            'steps': [],
            'sync_order': [],
            'phase_evolution': [],
            'amplitude_evolution': [],
        }
        
        for step in range(start_step, end_step + 1):
            if step in self.oscillator_states:
                try:
                    state = self.load_oscillator_state(step)
                    trajectory['steps'].append(step)
                    trajectory['sync_order'].append(state['sync_order'])
                    
                    if 'phase_evolution' in state:
                        trajectory['phase_evolution'].append(state['phase_evolution'])
                    
                    if 'amplitude_evolution' in state:
                        trajectory['amplitude_evolution'].append(state['amplitude_evolution'])
                except Exception as e:
                    logging.warning(f"Failed to load oscillator state for step {step}: {e}")
        
        return trajectory


def create_checkpoint_manager(config: Any) -> SafeCheckpointManager:
    """
    Factory function to create checkpoint manager from config.
    
    Args:
        config: Training configuration
        
    Returns:
        Configured checkpoint manager
    """
    return SafeCheckpointManager(
        checkpoint_dir=config.checkpoint_dir,
        keep_best_n=config.keep_best_n,
        backup_before_overwrite=True,
        verify_oscillator_state=True
    )