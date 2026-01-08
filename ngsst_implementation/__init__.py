"""
Neural Geometric State Space Transformer (NGSST)
A novel vision architecture that models vision as continuous geometric dynamics.
"""

__version__ = "1.0.0"
__author__ = "Vision Modality Research Initiative"

from .models import NGSST, NGSSTConfig
from .modules import (
    NeuralGeometricStateSpace,
    GeometricAttention,
    MultiScaleNeuralImplicitTokenizer,
    PredictiveCodingHead,
    SE3EquivariantConv
)
from .utils import (
    log_SE3,
    hat_operator,
    adaptive_time_constant,
    geometric_consistency_loss
)

__all__ = [
    "NGSST",
    "NGSSTConfig", 
    "NeuralGeometricStateSpace",
    "GeometricAttention",
    "MultiScaleNeuralImplicitTokenizer",
    "PredictiveCodingHead",
    "SE3EquivariantConv",
    "log_SE3",
    "hat_operator",
    "adaptive_time_constant",
    "geometric_consistency_loss"
]