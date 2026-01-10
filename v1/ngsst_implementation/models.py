"""
NGSST Model Implementation.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from .modules import (
 MultiScaleNeuralImplicitTokenizer,
 NeuralGeometricStateSpace,
 GeometricAttention,
 PredictiveCodingHead
)

@dataclass
class NGSSTConfig:
 """Configuration for NGSST model."""

 # Model architecture
 hidden_dim: int = 256
 num_heads: int = 8
 num_layers: int = 12
 num_scales: int = 4

 # Geometric State Space
 state_dim: int = 256
 time_constant_base: float = 1.0

 # Tokenization
 patch_size: int = 16

 # Attention
 window_size: int = 7
 num_global_tokens: int = 4

 # Predictive Coding
 prediction_scales: Tuple[int, ...] = (1, 2, 4, 8)
 uncertainty_weight: float = 0.1
 geometric_weight: float = 0.1

 # Training
 dropout: float = 0.1
 attention_dropout: float = 0.1

 # Task heads
 num_classes: Optional[int] = None # For classification
 detection_head: bool = False
 segmentation_head: bool = False

class NGSST(nn.Module):
 """
 Neural Geometric State Space Transformer.

 A unified architecture for resolution-agnostic vision with continuous
 geometric dynamics.
 """

 def __init__(self, config: NGSSTConfig) -> None:
 super().__init__()
 self.config = config

 # Multi-scale neural implicit tokenization
 self.tokenizer = MultiScaleNeuralImplicitTokenizer(
 hidden_dim=config.hidden_dim,
 num_scales=config.num_scales,
 )

 # Neural Geometric State Space layers
 self.ngss_layers = nn.ModuleList(
 [
 NeuralGeometricStateSpace(
 state_dim=config.hidden_dim,
 input_dim=config.hidden_dim,
 time_constant_base=config.time_constant_base,
 )
 for _ in range(config.num_layers)
 ]
 )

 # Geometric Attention layers
 self.attention_layers = nn.ModuleList(
 [
 GeometricAttention(
 dim=config.hidden_dim,
 num_heads=config.num_heads,
 window_size=config.window_size,
 num_global_tokens=config.num_global_tokens,
 dropout=config.attention_dropout,
 )
 for _ in range(config.num_layers)
 ]
 )

 # Layer norms and MLPs
 self.layer_norms = nn.ModuleList(
 [nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)]
 )

 self.mlps = nn.ModuleList(
 [
 nn.Sequential(
 nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
 nn.GELU(),
 nn.Dropout(config.dropout),
 nn.Linear(4 * config.hidden_dim, config.hidden_dim),
 nn.Dropout(config.dropout),
 )
 for _ in range(config.num_layers)
 ]
 )

 # Predictive coding head (for self-supervised learning)
 self.predictive_head = PredictiveCodingHead(
 dim=config.hidden_dim,
 num_scales=len(config.prediction_scales),
 )

 # Task-specific heads
 if config.num_classes is not None:
 self.classification_head = nn.Linear(
 config.hidden_dim, config.num_classes
 )

 if config.detection_head:
 self.detection_head = nn.Sequential(
 nn.Linear(config.hidden_dim, config.hidden_dim),
 nn.ReLU(),
 nn.Linear(config.hidden_dim, 4 + 1), # bbox + confidence
 )

 # Initialize weights
 self._init_weights()

 def _init_weights(self) -> None:
 """Initialize model weights."""
 for module in self.modules():
 if isinstance(module, nn.Linear):
 nn.init.xavier_uniform_(module.weight)
 if module.bias is not None:
 nn.init.zeros_(module.bias)
 elif isinstance(module, nn.LayerNorm):
 nn.init.ones_(module.weight)
 nn.init.zeros_(module.bias)

 def forward(
 self,
 x: torch.Tensor,
 camera_poses: Optional[torch.Tensor] = None,
 timestamps: Optional[torch.Tensor] = None,
 return_predictions: bool = False,
 return_uncertainty: bool = False,
 **kwargs: Any,
 ) -> Dict[str, Any]:
 """
 Forward pass through NGSST.

 Args:
 x: Input tensor [B, T, H, W, 3] for video or [B, H, W, 3] for image.
 camera_poses: Camera poses [B, T, 4, 4] if available.
 timestamps: Frame timestamps [B, T] if available.
 return_predictions: Whether to return predictive coding outputs.
 return_uncertainty: Whether to return uncertainty estimates.

 Returns:
 Dictionary containing model outputs.
 """
 outputs: Dict[str, Any] = {}

 if x.ndim == 4:
 x = x.unsqueeze(1)

 tokens, token_coords = self.tokenizer(x)

 current_tokens = tokens[0]
 geometric_state = None

 for layer_idx in range(self.config.num_layers):
 if layer_idx < len(self.ngss_layers):
 dt = (
 1.0
 if timestamps is None
 else timestamps[:, 1:] - timestamps[:, :-1]
 )
 geometric_state = self.ngss_layers[layer_idx](
 current_tokens,
 geometric_state,
 camera_poses=camera_poses,
 dt=dt,
 )

 current_tokens = self.attention_layers[layer_idx](
 current_tokens,
 geometric_state=geometric_state,
 camera_poses=camera_poses,
 )

 current_tokens = self.layer_norms[layer_idx](current_tokens)
 current_tokens = current_tokens + self.mlps[layer_idx](
 current_tokens
 )

 if hasattr(self, "classification_head"):
 pooled = current_tokens.mean(dim=[1, 2])
 logits = self.classification_head(pooled)
 outputs["logits"] = logits

 if hasattr(self, "detection_head"):
 detection_out = self.detection_head(current_tokens)
 outputs["detection"] = detection_out

 if return_predictions:
 predictions, uncertainties = self.predictive_head(
 current_tokens, geometric_state=geometric_state
 )
 outputs["predictions"] = predictions
 if return_uncertainty:
 outputs["uncertainties"] = uncertainties

 if geometric_state is not None:
 outputs["geometric_state"] = geometric_state

 return outputs

 def get_geometric_embeddings(self, x: torch.Tensor) -> torch.Tensor:
 """
 Extract geometric embeddings for analysis.

 Args:
 x: Input tensor [B, H, W, 3]

 Returns:
 Geometric embeddings [B, N, D]
 """
 with torch.no_grad():
 outputs = self.forward(x)
 return outputs.get("geometric_state", outputs.get("tokens"))

 def predict_next_frame(
 self,
 x: torch.Tensor,
 camera_poses: Optional[torch.Tensor] = None,
 delta_t: float = 1.0,
 ) -> Tuple[torch.Tensor, torch.Tensor]:
 """
 Predict next frame using predictive coding.

 Args:
 x: Current frame [B, H, W, 3]
 camera_poses: Camera poses [B, 4, 4]
 delta_t: Time step

 Returns:
 Predicted frame and uncertainty.
 """
 outputs = self.forward(
 x,
 camera_poses=camera_poses,
 return_predictions=True,
 return_uncertainty=True,
 )

 predictions = outputs["predictions"]
 uncertainties = outputs["uncertainties"]

 return predictions[0], uncertainties[0]

class NGSSTForClassification(NGSST):
 """NGSST for image classification tasks."""

 def __init__(self, config: NGSSTConfig) -> None:
 if config.num_classes is None:
 raise ValueError("num_classes must be specified for classification")
 super().__init__(config)

 def forward(
 self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
 ) -> Dict[str, Any]:
 outputs = super().forward(x)
 logits = outputs["logits"]

 result: Dict[str, Any] = {"logits": logits}

 if labels is not None:
 loss = nn.CrossEntropyLoss()(logits, labels)
 result["loss"] = loss

 return result

class NGSSTForDetection(NGSST):
 """NGSST for object detection tasks."""

 def __init__(self, config: NGSSTConfig) -> None:
 config.detection_head = True
 super().__init__(config)

 def forward(
 self,
 x: torch.Tensor,
 targets: Optional[Dict[str, torch.Tensor]] = None,
 ) -> Dict[str, Any]:
 outputs = super().forward(x)
 detection_out = outputs["detection"]

 boxes = detection_out[..., :4] # [B, T, N, 4]
 confidences = detection_out[..., 4] # [B, T, N]

 result: Dict[str, Any] = {"boxes": boxes, "confidences": confidences}

 if targets is not None:
 loss = self._compute_detection_loss(boxes, confidences, targets)
 result["loss"] = loss

 return result

 def _compute_detection_loss(
 self,
 boxes: torch.Tensor,
 confidences: torch.Tensor,
 targets: Dict[str, torch.Tensor],
 ) -> torch.Tensor:
 """Compute detection loss."""
 gt_boxes = targets["boxes"]
 gt_labels = targets["labels"]

 box_loss = nn.SmoothL1Loss()(boxes, gt_boxes)
 label_loss = nn.BCEWithLogitsLoss()(confidences, gt_labels.float())

 return box_loss + label_loss
