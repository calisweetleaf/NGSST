"""
RLHF Package for NGSST

Provides reinforcement learning from human feedback for:
- Vision models (classification, generation)
- Language models (text generation)
- Geometric/3D scene composition

Modules:
- vision_rlhf: DPO/GRPO for vision models (HVT integration)
- self_rlhf: Full LLM RLHF pipeline (DPO, PPO, GRPO, SimPO, KTO)
- geometric_rlhf: 3D scene composition environment
"""

from .vision_rlhf import (
    VisionRLHFConfig,
    VisionDPOTrainer,
    VisionGRPOTrainer,
    VisionRewardEnsemble,
    CLIPRewardModel,
    AestheticRewardModel,
    GeometricConsistencyReward,
    VisionPreferenceDataset,
    create_hvt_rlhf_trainer,
)

# Lazy imports for heavy modules
def get_self_rlhf():
    """Get LLM RLHF components (lazy load)."""
    from . import self_rlhf
    return self_rlhf

def get_geometric_rlhf():
    """Get geometric composition environment (lazy load)."""
    from . import geometric_rlhf
    return geometric_rlhf

__all__ = [
    # Vision RLHF
    'VisionRLHFConfig',
    'VisionDPOTrainer',
    'VisionGRPOTrainer',
    'VisionRewardEnsemble',
    'CLIPRewardModel',
    'AestheticRewardModel',
    'GeometricConsistencyReward',
    'VisionPreferenceDataset',
    'create_hvt_rlhf_trainer',
    
    # Lazy loaders
    'get_self_rlhf',
    'get_geometric_rlhf',
]
