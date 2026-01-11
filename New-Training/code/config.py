"""
Configuration system for HVT v3 training pipeline.

This module defines all hyperparameters, method toggles, and presets
for the novel training approaches designed specifically for oscillator-based architectures.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch


@dataclass
class SyncAwareLossConfig:
    """Configuration for sync-aware loss functions."""
    lambda_sync: float = 0.1  # Weight for sync order regularization
    lambda_diversity: float = 0.05  # Weight for diversity across bands
    target_sync: float = 0.618  # Golden ratio complement
    sync_margin: float = 0.1  # Margin for sync order constraints
    use_adaptive_sync: bool = True  # Adapt sync target during training


@dataclass
class PhysicsInformedOptimizerConfig:
    """Configuration for physics-informed optimization."""
    optimizer_type: str = "SymplecticAdam"  # SymplecticAdam, HamiltonianSGD, or standard
    symplectic_coef: float = 0.01  # Coefficient for symplectic updates
    energy_preservation: bool = True  # Preserve oscillator energy
    phase_momentum: float = 0.9  # Momentum for phase updates
    use_natural_gradients: bool = False  # Use natural gradients for oscillator manifold
    damping_adaptation: bool = True  # Adapt damping based on sync order


@dataclass
class AdaptiveBreathingConfig:
    """Configuration for adaptive breathing schedules."""
    breathing_type: str = "sync_triggered"  # fixed, sync_triggered, or energy_based
    base_interval: int = 100  # Base breathing interval
    sync_threshold_low: float = 0.4  # Low sync threshold for breathing
    sync_threshold_high: float = 0.7  # High sync threshold for breathing
    energy_threshold: float = 0.1  # Energy stability threshold
    breathing_duration: int = 10  # Duration of breathing phase
    pressure_reduction: float = 0.5  # Classification weight reduction during breathing


@dataclass
class MultiScaleCurriculumConfig:
    """Configuration for multi-scale curriculum learning."""
    curriculum_type: str = "frequency_expansion"  # frequency_expansion, band_progression, or gabor_scale
    start_bands: int = 2  # Starting number of frequency bands
    max_bands: int = 6  # Maximum number of frequency bands
    expansion_schedule: str = "exponential"  # linear, exponential, or golden_ratio
    expansion_rate: float = 0.1  # Rate of band expansion (per epoch)
    gabor_scale_progression: List[float] = field(default_factory=lambda: [0.5, 0.7, 1.0, 1.3, 1.6])
    freeze_previous_bands: bool = True  # Freeze previously learned bands


@dataclass
class GeometricConsistencyConfig:
    """Configuration for geometric consistency regularization."""
    lambda_geometric: float = 0.1  # Weight for geometric consistency loss
    consistency_type: str = "se3_composition"  # se3_composition, phase_smoothness, or manifold


@dataclass
class NovelMethodConfig:
    """Configuration for all novel training methods."""
    # Method toggles
    use_sync_aware_loss: bool = True
    use_physics_informed_optimizer: bool = True
    use_adaptive_breathing: bool = True
    use_multi_scale_curriculum: bool = True
    use_geometric_consistency: bool = True
    
    # Method configurations
    sync_aware_loss: SyncAwareLossConfig = field(default_factory=SyncAwareLossConfig)
    physics_optimizer: PhysicsInformedOptimizerConfig = field(default_factory=PhysicsInformedOptimizerConfig)
    adaptive_breathing: AdaptiveBreathingConfig = field(default_factory=AdaptiveBreathingConfig)
    curriculum: MultiScaleCurriculumConfig = field(default_factory=MultiScaleCurriculumConfig)
    geometric_consistency: GeometricConsistencyConfig = field(default_factory=GeometricConsistencyConfig)


@dataclass
class TrainingConfig:
    """Main training configuration."""
    # Dataset
    dataset_name: str = "cifar10"
    dataset_config: Optional[str] = None
    image_key: str = "img"
    label_key: str = "label"
    streaming: bool = True
    
    # Model architecture
    num_freq_bands: int = 4
    num_evolution_layers: int = 3
    hidden_dim: int = 64
    num_classes: int = 10
    image_size: int = 32
    patch_size: int = 16
    
    # HVT v2 Model parameters that were missing
    coupling_strength: float = 0.72973525693  # ALPHA_SCALED from hvt_v2.py
    learnable_physics: bool = True
    num_routing_heads: int = 4
    use_phase_routing: bool = True
    use_spectral_norm: bool = False
    
    # Training parameters
    total_steps: int = 10000
    warmup_steps: int = 500
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Harmonic-specific parameters
    base_omega: float = 0.2618  # PHI / TAU
    sync_target: float = 0.618
    oscillator_warmup_weight: float = 0.1
    
    # Novel methods
    novel_methods: NovelMethodConfig = field(default_factory=NovelMethodConfig)
    
    # Hardware
    device: str = "cpu"
    num_workers: int = 0
    pin_memory: bool = False
    mixed_precision: bool = False
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 500
    log_every: int = 25
    keep_best_n: int = 3
    
    # Validation
    val_every: int = 1000
    val_steps: int = 100
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "hvt_v3_training"
    experiment_name: Optional[str] = None
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        def convert(obj):
            if hasattr(obj, '__dict__'):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            else:
                return obj
        return convert(self)


# Preset configurations
PRESETS = {
    "baseline": {
        "description": "Baseline HVT training without novel methods",
        "novel_methods": {
            "use_sync_aware_loss": False,
            "use_physics_informed_optimizer": False,
            "use_adaptive_breathing": False,
            "use_multi_scale_curriculum": False,
            "use_geometric_consistency": False
        }
    },
    "full_novel": {
        "description": "All novel methods enabled",
        "novel_methods": {
            "use_sync_aware_loss": True,
            "use_physics_informed_optimizer": True,
            "use_adaptive_breathing": True,
            "use_multi_scale_curriculum": True,
            "use_geometric_consistency": True
        }
    },
    "fast": {
        "description": "Fast training with reduced complexity",
        "total_steps": 5000,
        "warmup_steps": 250,
        "batch_size": 64,
        "novel_methods": {
            "use_sync_aware_loss": True,
            "use_physics_informed_optimizer": False,
            "use_adaptive_breathing": True,
            "use_multi_scale_curriculum": False,
            "use_geometric_consistency": False
        }
    },
    "max_quality": {
        "description": "Maximum quality training with all optimizations",
        "total_steps": 20000,
        "warmup_steps": 1000,
        "batch_size": 16,
        "novel_methods": {
            "use_sync_aware_loss": True,
            "use_physics_informed_optimizer": True,
            "use_adaptive_breathing": True,
            "use_multi_scale_curriculum": True,
            "use_geometric_consistency": True
        },
        "sync_aware_loss": {
            "lambda_sync": 0.2,
            "lambda_diversity": 0.1
        }
    }
}


def get_preset(name: str) -> Dict[str, Any]:
    """Get preset configuration by name."""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return PRESETS[name]


def create_config(preset: str = None, **kwargs) -> TrainingConfig:
    """Create configuration with optional preset and overrides."""
    config = TrainingConfig()
    
    if preset:
        preset_data = get_preset(preset)
        for key, value in preset_data.items():
            if hasattr(config, key):
                if isinstance(value, dict):
                    current = getattr(config, key)
                    for k, v in value.items():
                        if hasattr(current, k):
                            setattr(current, k, v)
                        else:
                            current.__dict__[k] = v
                else:
                    setattr(config, key, value)
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Handle nested attributes
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    break
            else:
                if hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], value)
    
    return config