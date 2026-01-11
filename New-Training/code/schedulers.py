"""
Adaptive learning rate schedulers for HVT v3 training pipeline.

Includes sync-aware breathing schedules and physics-informed
learning rate adaptation.
"""

import torch
import math
from typing import Optional, Dict, Any, List
from torch.optim.lr_scheduler import _LRScheduler


class SyncTriggeredBreathingScheduler(_LRScheduler):
    """
    Learning rate scheduler with sync-order-triggered breathing cycles.
    
    Reduces learning rate when sync order is unstable (too high or too low)
    and increases when sync order is in target range.
    """
    
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 500,
        base_lr: float = 1e-3,
        min_lr: float = 1e-6,
        sync_target: float = 0.618,
        sync_threshold_low: float = 0.4,
        sync_threshold_high: float = 0.7,
        breathing_duration: int = 10,
        pressure_reduction: float = 0.5,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.sync_target = sync_target
        self.sync_threshold_low = sync_threshold_low
        self.sync_threshold_high = sync_threshold_high
        self.breathing_duration = breathing_duration
        self.pressure_reduction = pressure_reduction
        
        # Track sync order history
        self.sync_history = []
        self.in_breathing_phase = False
        self.breathing_start = 0
        
        super().__init__(optimizer, last_epoch=last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.", UserWarning
            )
        
        step = self.last_epoch
        
        # Warmup phase
        if step < self.warmup_steps:
            return [self.base_lr * (step / self.warmup_steps) for _ in self.optimizer.param_groups]
        
        # Check if we should trigger breathing
        if self.sync_history:
            recent_sync = sum(self.sync_history[-10:]) / len(self.sync_history[-10:])
            
            # Trigger breathing if sync is unstable
            should_breathe = (
                recent_sync < self.sync_threshold_low or 
                recent_sync > self.sync_threshold_high
            )
            
            if should_breathe and not self.in_breathing_phase:
                self.in_breathing_phase = True
                self.breathing_start = step
            
            # End breathing phase
            if self.in_breathing_phase and step - self.breathing_start >= self.breathing_duration:
                self.in_breathing_phase = False
        
        # Compute base learning rate (cosine decay)
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        # Cosine decay with golden ratio modulation
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        golden_mod = 1.0 + 0.1 * math.sin(2 * math.pi * 1.618 * progress)
        
        base_lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay * golden_mod
        
        # Apply breathing reduction
        if self.in_breathing_phase:
            base_lr *= self.pressure_reduction
        
        return [base_lr for _ in self.optimizer.param_groups]
    
    def update_sync_order(self, sync_order: float):
        """Update sync order history."""
        self.sync_history.append(sync_order)
        if len(self.sync_history) > 100:
            self.sync_history.pop(0)
    
    def is_breathing(self) -> bool:
        """Check if currently in breathing phase."""
        return self.in_breathing_phase


class EnergyBasedScheduler(_LRScheduler):
    """
    Learning rate scheduler based on energy stability.
    
    Reduces learning rate when energy becomes unstable,
    increases when energy is stable.
    """
    
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 500,
        base_lr: float = 1e-3,
        min_lr: float = 1e-6,
        energy_threshold: float = 0.1,
        adaptation_rate: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.energy_threshold = energy_threshold
        self.adaptation_rate = adaptation_rate
        
        # Track energy history
        self.energy_history = []
        self.current_multiplier = 1.0
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step = self.last_epoch
        
        # Warmup phase
        if step < self.warmup_steps:
            return [self.base_lr * (step / self.warmup_steps) for _ in self.optimizer.param_groups]
        
        # Check energy stability
        if len(self.energy_history) >= 10:
            recent_energy = sum(self.energy_history[-10:]) / len(self.energy_history[-10:])
            
            if recent_energy > self.energy_threshold:
                # Energy unstable - reduce learning rate
                self.current_multiplier = max(0.1, self.current_multiplier * (1 - self.adaptation_rate))
            else:
                # Energy stable - slowly increase learning rate
                self.current_multiplier = min(2.0, self.current_multiplier * (1 + self.adaptation_rate * 0.1))
        
        # Cosine decay
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
        
        # Apply energy-based multiplier
        lr *= self.current_multiplier
        
        return [lr for _ in self.optimizer.param_groups]
    
    def update_energy_stability(self, energy_stability: float):
        """Update energy stability history."""
        self.energy_history.append(energy_stability)
        if len(self.energy_history) > 100:
            self.energy_history.pop(0)


class GoldenRatioScheduler(_LRScheduler):
    """
    Learning rate scheduler with golden ratio modulation.
    
    Uses φ-based cycles to create natural harmonic patterns
    in the learning rate schedule.
    """
    
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 500,
        base_lr: float = 1e-3,
        min_lr: float = 1e-6,
        phi: float = 1.618034,
        cycles: int = 3,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.phi = phi
        self.cycles = cycles
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate with golden ratio modulation."""
        step = self.last_epoch
        
        # Warmup phase
        if step < self.warmup_steps:
            return [self.base_lr * (step / self.warmup_steps) for _ in self.optimizer.param_groups]
        
        # Main training phase
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        # Base cosine decay
        base_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        # Golden ratio modulation
        cycle_progress = progress * self.cycles
        golden_mod = 1.0 + 0.2 * math.sin(2 * math.pi * self.phi * cycle_progress)
        
        # Combine
        lr = self.min_lr + (self.base_lr - self.min_lr) * base_decay * golden_mod
        
        return [lr for _ in self.optimizer.param_groups]


class CurriculumScheduler(_LRScheduler):
    """
    Learning rate scheduler for multi-scale curriculum learning.
    
    Progressively increases model capacity by adding frequency bands
    and adjusting learning rates accordingly.
    """
    
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 500,
        base_lr: float = 1e-3,
        min_lr: float = 1e-6,
        start_bands: int = 2,
        max_bands: int = 6,
        expansion_schedule: str = "exponential",
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.start_bands = start_bands
        self.max_bands = max_bands
        self.expansion_schedule = expansion_schedule
        
        # Track current number of bands
        self.current_bands = start_bands
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate based on curriculum progress."""
        step = self.last_epoch
        
        # Warmup phase
        if step < self.warmup_steps:
            return [self.base_lr * (step / self.warmup_steps) for _ in self.optimizer.param_groups]
        
        # Compute number of active bands
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        if self.expansion_schedule == "linear":
            num_bands = self.start_bands + int(progress * (self.max_bands - self.start_bands))
        elif self.expansion_schedule == "exponential":
            alpha = math.log(self.max_bands / self.start_bands)
            num_bands = int(self.start_bands * math.exp(alpha * progress))
        else:  # golden_ratio
            phi = 1.618034
            num_bands = min(self.max_bands, self.start_bands + int(progress * phi * (self.max_bands - self.start_bands)))
        
        self.current_bands = max(self.start_bands, min(self.max_bands, num_bands))
        
        # Adjust learning rate based on number of bands
        # More bands = smaller learning rate (more complex model)
        band_factor = self.start_bands / self.current_bands
        
        # Cosine decay
        lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        # Apply band factor
        lr *= band_factor
        
        return [lr for _ in self.optimizer.param_groups]
    
    def get_current_bands(self) -> int:
        """Get current number of active frequency bands."""
        return self.current_bands


class PhaseSpaceScheduler(_LRScheduler):
    """
    Learning rate scheduler based on phase space exploration.
    
    Adapts learning rate based on phase space coverage and
    synchronization patterns.
    """
    
    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps: int = 500,
        base_lr: float = 1e-3,
        min_lr: float = 1e-6,
        exploration_factor: float = 2.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.exploration_factor = exploration_factor
        
        # Track phase space metrics
        self.phase_coverage = 0.0
        self.sync_diversity = 0.0
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate based on phase space exploration."""
        step = self.last_epoch
        
        # Warmup phase
        if step < self.warmup_steps:
            return [self.base_lr * (step / self.warmup_steps) for _ in self.optimizer.param_groups]
        
        # Main training phase
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        progress = min(progress, 1.0)
        
        # Base cosine decay
        base_lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        # Exploration bonus based on phase space coverage
        if self.phase_coverage < 0.5:  # Low coverage - encourage exploration
            exploration_bonus = self.exploration_factor
        else:  # Good coverage - focus on exploitation
            exploration_bonus = 1.0
        
        lr = base_lr * exploration_bonus
        
        return [lr for _ in self.optimizer.param_groups]
    
    def update_phase_metrics(self, phase_coverage: float, sync_diversity: float):
        """Update phase space metrics."""
        self.phase_coverage = phase_coverage
        self.sync_diversity = sync_diversity


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Any,
    scheduler_type: str = "SyncTriggeredBreathing"
) -> _LRScheduler:
    """
    Factory function to create scheduler based on configuration.
    
    Args:
        optimizer: The optimizer to schedule
        config: Training configuration
        scheduler_type: Type of scheduler to create
        
    Returns:
        Configured learning rate scheduler
    """
    common_args = {
        'optimizer': optimizer,
        'total_steps': config.total_steps,
        'warmup_steps': config.warmup_steps,
        'base_lr': config.learning_rate,
        'min_lr': config.min_learning_rate,
    }
    
    if scheduler_type == "SyncTriggeredBreathing":
        # Filter config to only valid SyncTriggeredBreathingScheduler params
        valid_params = {'sync_threshold_low', 'sync_threshold_high', 'breathing_duration', 'pressure_reduction'}
        filtered_kwargs = {k: v for k, v in config.novel_methods.adaptive_breathing.__dict__.items() 
                          if k in valid_params}
        return SyncTriggeredBreathingScheduler(
            **common_args,
            sync_target=config.sync_target,
            **filtered_kwargs
        )
    elif scheduler_type == "EnergyBased":
        return EnergyBasedScheduler(**common_args)
    elif scheduler_type == "GoldenRatio":
        return GoldenRatioScheduler(**common_args)
    elif scheduler_type == "Curriculum":
        return CurriculumScheduler(
            **common_args,
            **config.novel_methods.curriculum.__dict__
        )
    elif scheduler_type == "PhaseSpace":
        return PhaseSpaceScheduler(**common_args)
    else:
        # Default cosine scheduler
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.total_steps - config.warmup_steps,
            eta_min=config.min_learning_rate
        )