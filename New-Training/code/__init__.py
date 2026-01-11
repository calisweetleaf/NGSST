"""
HVT v3 Training Pipeline - Optimal training for Harmonic Vision Transformer.

This package provides novel training methods specifically designed for
oscillator-based neural architectures like HVT.
"""

from .config import TrainingConfig, create_config, get_preset
from .trainer import HVTTrainer, TrainingMetrics
from .losses import (
    SyncAwareLoss,
    PhaseCoherenceLoss,
    EnergyStabilityLoss,
    GeometricConsistencyLoss,
    HarmonicLoss
)
from .optimizers import (
    SymplecticAdam,
    HamiltonianSGD,
    OscillatorAdamW,
    AdaptiveCouplingOptimizer,
    get_optimizer
)
from .schedulers import (
    SyncTriggeredBreathingScheduler,
    EnergyBasedScheduler,
    GoldenRatioScheduler,
    CurriculumScheduler,
    PhaseSpaceScheduler,
    get_scheduler
)
from .datasets import (
    StreamingImageDataset,
    MultiDataset,
    FrequencyCurriculumDataset,
    get_datasets,
    get_dataloader
)
from .checkpointing import (
    SafeCheckpointManager,
    OscillatorStateManager,
    create_checkpoint_manager
)

__version__ = "3.0.0"
__author__ = "HVT Development Team"
__description__ = "Optimal training pipeline for Harmonic Vision Transformer"

__all__ = [
    # Configuration
    'TrainingConfig',
    'create_config',
    'get_preset',
    
    # Training
    'HVTTrainer',
    'TrainingMetrics',
    
    # Losses
    'SyncAwareLoss',
    'PhaseCoherenceLoss',
    'EnergyStabilityLoss',
    'GeometricConsistencyLoss',
    'HarmonicLoss',
    
    # Optimizers
    'SymplecticAdam',
    'HamiltonianSGD',
    'OscillatorAdamW',
    'AdaptiveCouplingOptimizer',
    'get_optimizer',
    
    # Schedulers
    'SyncTriggeredBreathingScheduler',
    'EnergyBasedScheduler',
    'GoldenRatioScheduler',
    'CurriculumScheduler',
    'PhaseSpaceScheduler',
    'get_scheduler',
    
    # Datasets
    'StreamingImageDataset',
    'MultiDataset',
    'FrequencyCurriculumDataset',
    'get_datasets',
    'get_dataloader',
    
    # Checkpointing
    'SafeCheckpointManager',
    'OscillatorStateManager',
    'create_checkpoint_manager',
]