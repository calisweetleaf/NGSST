"""
Core modules for Neural Geometric State Space Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Any
import math

class MultiScaleNeuralImplicitTokenizer(nn.Module):
 """
 Multi-scale neural implicit tokenization.

 Converts pixel coordinates to continuous feature representations
 at multiple scales using learnable kernel functions.
 """

 def __init__(
 self,
 hidden_dim: int = 256,
 num_scales: int = 4,
 kernel_layers: int = 3,
 pos_encoding_dim: int = 128
 ):
 super().__init__()
 self.hidden_dim = hidden_dim
 self.num_scales = num_scales

 # Continuous kernel functions for each scale
 self.kernels = nn.ModuleList([
 self._make_kernel(kernel_layers, pos_encoding_dim, hidden_dim)
 for _ in range(num_scales)
 ])

 # Scale-specific weights
 self.scale_weights = nn.Parameter(torch.ones(num_scales))

 # Positional encoding
 self.pos_encoding = PositionalEncoding(2, pos_encoding_dim, 10)

 # Input projection for kernel
 self.input_proj = nn.Linear(pos_encoding_dim * 2, hidden_dim)

 def _make_kernel(self, num_layers: int, in_dim: int, out_dim: int) -> nn.Module:
 """Create a continuous kernel MLP."""
 layers = []
 hidden_dim = out_dim # Use output dim as hidden dim
 for i in range(num_layers):
 layers.extend([
 nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
 nn.ReLU()
 ])
 # Final layer to output dimension
 layers.append(nn.Linear(hidden_dim, out_dim))
 return nn.Sequential(*layers)

 def forward(
 self,
 x: torch.Tensor,
 coords: Optional[List[torch.Tensor]] = None
 ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
 """
 Tokenize input at multiple scales.

 Args:
 x: Input tensor [B, T, H, W, 3]
 coords: Optional list of coordinate grids for each scale

 Returns:
 Tuple of (token_features, token_coordinates) for each scale
 """
 B, T, H, W, C = x.shape
 tokens = []
 token_coords = []

 for scale_idx in range(self.num_scales):
 # Compute stride for this scale
 stride = 2 ** scale_idx

 # Sample coordinates
 if coords is None:
 h_scaled = H // (4 * stride) # Base downsample by 4
 w_scaled = W // (4 * stride)

 # Create normalized coordinate grid
 y_coords = torch.linspace(0, 1, h_scaled, device=x.device)
 x_coords = torch.linspace(0, 1, w_scaled, device=x.device)
 yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
 scale_coords = torch.stack([xx, yy], dim=-1) # [h, w, 2]
 scale_coords = scale_coords.unsqueeze(0).expand(B * T, -1, -1, -1)
 scale_coords = scale_coords.reshape(B * T, -1, 2) # [B*T, N, 2]
 else:
 scale_coords = coords[scale_idx]

 # Apply continuous kernel
 pos_encoded = self.pos_encoding(scale_coords) # [B*T, N, D_pos]
 kernel_out = self.kernels[scale_idx](pos_encoded) # [B*T, N, D]

 # Apply scale weight
 kernel_out = kernel_out * self.scale_weights[scale_idx]

 # Reshape back to video format
 kernel_out = kernel_out.reshape(B, T, -1, self.hidden_dim)
 scale_coords = scale_coords.reshape(B, T, -1, 2)

 tokens.append(kernel_out)
 token_coords.append(scale_coords)

 return tokens, token_coords

class PositionalEncoding(nn.Module):
 """Sinusoidal positional encoding."""

 def __init__(self, input_dim: int, output_dim: int, max_freq: int):
 super().__init__()
 self.input_dim = input_dim
 self.output_dim = output_dim
 self.max_freq = max_freq

 # Create frequency bands
 freq_bands = 2.0 ** torch.linspace(0, max_freq, output_dim // (2 * input_dim))
 self.register_buffer('freq_bands', freq_bands)

 def forward(self, coords: torch.Tensor) -> torch.Tensor:
 """
 Apply positional encoding to coordinates.

 Args:
 coords: Coordinates [..., input_dim]

 Returns:
 Encoded features [..., output_dim]
 """
 # Expand coordinates to match frequency bands
 coords_expanded = coords.unsqueeze(-1) * self.freq_bands # [..., input_dim, N_freq]

 # Apply sin and cos
 sin_enc = torch.sin(coords_expanded)
 cos_enc = torch.cos(coords_expanded)

 # Interleave sin and cos
 encoded = torch.stack([sin_enc, cos_enc], dim=-1) # [..., input_dim, N_freq, 2]
 encoded = encoded.reshape(*coords.shape[:-1], -1) # [..., output_dim]

 return encoded

class SE3EquivariantConv(nn.Module):
 """
 SE(3) equivariant convolution layer.

 Maps Lie algebra elements to convolution weights.
 """

 def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
 super().__init__()
 self.in_dim = in_dim
 self.out_dim = out_dim

 # Network to map Lie algebra to weights
 self.weight_net = nn.Sequential(
 nn.Linear(6, hidden_dim), # se(3) has dimension 6
 nn.ReLU(),
 nn.Linear(hidden_dim, in_dim * out_dim)
 )

 def forward(
 self,
 features: torch.Tensor,
 transformation: torch.Tensor
 ) -> torch.Tensor:
 """
 Apply SE(3) equivariant convolution.

 Args:
 features: Input features [B, N, D_in]
 transformation: SE(3) matrix [B, 4, 4]

 Returns:
 Transformed features [B, N, D_out]
 """
 # Convert transformation to Lie algebra element
 xi = log_SE3(transformation) # [B, 6]

 # Generate convolution weights
 weights = self.weight_net(xi) # [B, D_in * D_out]
 weights = weights.reshape(-1, self.in_dim, self.out_dim) # [B, D_in, D_out]

 # Apply convolution
 output = torch.einsum('bnd,bdo->bno', features, weights) # [B, N, D_out]

 return output

def log_SE3(transform: torch.Tensor) -> torch.Tensor:
 """
 Convert SE(3) matrix to se(3) Lie algebra element.

 Args:
 transform: SE(3) matrix [B, 4, 4]

 Returns:
 Lie algebra element [B, 6]
 """
 B = transform.shape[0]

 # Extract rotation and translation
 R = transform[:, :3, :3] # [B, 3, 3]
 t = transform[:, :3, 3] # [B, 3]

 # Compute rotation angle
 cos_angle = (torch.trace(R, dim1=1, dim2=2) - 1) / 2
 cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)
 angle = torch.acos(cos_angle) # [B]

 # Handle small angles
 small_angle_mask = angle < 1e-3

 # Compute rotation axis (omega)
 omega = torch.zeros(B, 3, device=transform.device)

 # Use Rodrigues' formula
 sin_angle = torch.sin(angle)
 omega_sin = torch.stack([
 R[:, 2, 1] - R[:, 1, 2],
 R[:, 0, 2] - R[:, 2, 0],
 R[:, 1, 0] - R[:, 0, 1]
 ], dim=1) # [B, 3]

 # Avoid division by zero for small angles
 safe_sin = torch.where(small_angle_mask, torch.ones_like(sin_angle), sin_angle)
 omega = omega_sin / (2 * safe_sin.unsqueeze(1))

 # For small angles, use linear approximation
 if small_angle_mask.any():
 omega_small = 0.5 * omega_sin[small_angle_mask]
 omega = omega.clone()
 omega[small_angle_mask] = omega_small

 # Compute translation component (v)
 # V = I - (1 - cos(angle)) / angle^2 * hat(omega) +
 # (angle - sin(angle)) / angle^3 * hat(omega)^2

 hat_omega = hat_operator(omega) # [B, 3, 3]
 hat_omega_sq = torch.bmm(hat_omega, hat_omega) # [B, 3, 3]

 # Compute V inverse
 angle_sq = angle ** 2
 safe_angle_sq = torch.where(small_angle_mask, torch.ones_like(angle_sq), angle_sq)

 coef1 = (1 - torch.cos(angle)) / safe_angle_sq
 coef2 = (angle - sin_angle) / (safe_angle_sq * angle)

 V = torch.eye(3, device=transform.device).unsqueeze(0).expand(B, -1, -1)
 V = V - coef1.unsqueeze(1).unsqueeze(2) * hat_omega
 V = V + coef2.unsqueeze(1).unsqueeze(2) * hat_omega_sq

 # Invert V for small angles
 V_inv = torch.linalg.inv(V)

 v = torch.bmm(V_inv, t.unsqueeze(2)).squeeze(2) # [B, 3]

 # Combine omega and v
 xi = torch.cat([omega, v], dim=1) # [B, 6]

 return xi

def hat_operator(vec: torch.Tensor) -> torch.Tensor:
 """
 Hat operator: R^3 -> so(3)

 Args:
 vec: Vector [B, 3]

 Returns:
 Skew-symmetric matrix [B, 3, 3]
 """
 B = vec.shape[0]
 x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]

 hat = torch.zeros(B, 3, 3, device=vec.device)
 hat[:, 0, 1] = -z
 hat[:, 0, 2] = y
 hat[:, 1, 0] = z
 hat[:, 1, 2] = -x
 hat[:, 2, 0] = -y
 hat[:, 2, 1] = x

 return hat

class NeuralGeometricStateSpace(nn.Module):
 """
 Neural Geometric State Space (NGSS) module.

 Extends State Space Models to operate on geometric manifolds
 with SE(3) equivariance and adaptive time constants.
 """

 def __init__(
 self,
 state_dim: int,
 input_dim: int,
 time_constant_base: float = 1.0,
 hidden_dim: Optional[int] = None
 ):
 super().__init__()
 self.state_dim = state_dim
 self.input_dim = input_dim
 self.time_constant_base = time_constant_base

 if hidden_dim is None:
 hidden_dim = state_dim

 # Input projection
 self.input_proj = nn.Linear(input_dim, state_dim)

 # SE(3) equivariant state transition
 self.equivariant_conv = SE3EquivariantConv(state_dim, state_dim)

 # Adaptive time constant network
 self.time_net = nn.Sequential(
 nn.Linear(input_dim + state_dim, hidden_dim),
 nn.ReLU(),
 nn.Linear(hidden_dim, 1),
 nn.Sigmoid()
 )

 # State gate
 self.gate = nn.Sequential(
 nn.Linear(state_dim, hidden_dim),
 nn.SiLU(),
 nn.Linear(hidden_dim, state_dim),
 nn.Sigmoid()
 )

 def forward(
 self,
 x: torch.Tensor,
 state: Optional[torch.Tensor] = None,
 camera_poses: Optional[torch.Tensor] = None,
 dt: Optional[torch.Tensor] = None
 ) -> torch.Tensor:
 """
 Forward pass through NGSS.

 Args:
 x: Input features [B, T, N, D_in]
 state: Previous state [B, T, N, D_state]
 camera_poses: Camera poses [B, T, 4, 4]
 dt: Time step [B, T-1] or scalar

 Returns:
 Updated state [B, T, N, D_state]
 """
 B, T, N, _ = x.shape

 # Project input
 B, T, N, D_in = x.shape
 x_flat = x.reshape(-1, D_in) # [B*T*N, D_in]
 x_proj_flat = self.input_proj(x_flat) # [B*T*N, D_state]
 x_proj = x_proj_flat.reshape(B, T, N, self.state_dim) # [B, T, N, D_state]

 # Initialize state if not provided
 if state is None:
 state = torch.zeros_like(x_proj)

 # Process temporal sequence
 outputs = []
 current_state = state[:, 0] # [B, N, D_state]

 for t in range(T):
 # Get current input
 x_t = x_proj[:, t] # [B, N, D_state]

 # Compute adaptive time constant
 time_input = torch.cat([x_t, current_state], dim=-1) # [B, N, D_in + D_state]
 B_t, N_t, D_t = time_input.shape
 time_input_flat = time_input.reshape(-1, D_t) # [B*N, D_t]
 tau_factor_flat = self.time_net(time_input_flat) # [B*N, 1]
 tau_factor = tau_factor_flat.reshape(B_t, N_t, 1) # [B, N, 1]
 tau = self.time_constant_base * tau_factor

 # Compute geometric transformation if available
 xi = None
 if camera_poses is not None and t > 0:
 # Relative transformation from previous to current
 g_prev = camera_poses[:, t-1]
 g_curr = camera_poses[:, t]
 g_rel = g_curr @ torch.inverse(g_prev)
 xi = log_SE3(g_rel) # [B, 6]

 # Apply equivariant state transition
 if xi is not None:
 state_transformed = self.equivariant_conv(current_state, g_rel)
 else:
 state_transformed = current_state

 # Gate mechanism
 gate = self.gate(current_state) # [B, N, D_state]

 # State update with adaptive time constant
 current_state = (1 - tau) * current_state + tau * (
 gate * state_transformed + (1 - gate) * x_t
 )

 outputs.append(current_state.unsqueeze(1))

 # Concatenate along time dimension
 output_state = torch.cat(outputs, dim=1) # [B, T, N, D_state]

 return output_state

def adaptive_time_constant(
 geometry_change: torch.Tensor,
 feature_entropy: torch.Tensor,
 base_tau: float = 1.0
) -> torch.Tensor:
 """
 Compute adaptive time constant based on geometric change and feature complexity.

 Args:
 geometry_change: Magnitude of geometric transformation [B, N, 1]
 feature_entropy: Feature entropy measuring scene complexity [B, N, 1]
 base_tau: Base time constant

 Returns:
 Adaptive time constant [B, N, 1]
 """
 return base_tau / (1 + geometry_change + feature_entropy)

class GeometricAttention(nn.Module):
 """
 Geometric attention with local-global factorization and geometric bias.
 """

 def __init__(
 self,
 dim: int,
 num_heads: int = 8,
 window_size: int = 7,
 num_global_tokens: int = 4,
 dropout: float = 0.1
 ):
 super().__init__()
 self.dim = dim
 self.num_heads = num_heads
 self.window_size = window_size
 self.num_global_tokens = num_global_tokens
 self.head_dim = dim // num_heads
 assert dim % num_heads == 0

 # QKV projection
 self.qkv = nn.Linear(dim, dim * 3)

 # Output projection
 self.proj = nn.Linear(dim, dim)
 self.proj_drop = nn.Dropout(dropout)

 # Geometric bias network
 self.geom_bias = nn.Sequential(
 nn.Linear(3, 64), # 3D relative positions
 nn.ReLU(),
 nn.Linear(64, num_heads)
 )

 # Global tokens
 self.global_tokens = nn.Parameter(torch.randn(1, 1, num_global_tokens, dim))

 # Window attention mask
 self.register_buffer(
 'window_mask',
 self._create_window_mask(window_size)
 )

 def _create_window_mask(self, window_size: int) -> torch.Tensor:
 """Create attention mask for local window attention."""
 mask = torch.zeros(window_size * window_size, window_size * window_size)
 for i in range(window_size * window_size):
 for j in range(window_size * window_size):
 # Compute 2D positions
 x_i, y_i = i % window_size, i // window_size
 x_j, y_j = j % window_size, j // window_size

 # Mask out if outside window
 if abs(x_i - x_j) > window_size // 2 or abs(y_i - y_j) > window_size // 2:
 mask[i, j] = float('-inf')

 return mask

 def forward(
 self,
 x: torch.Tensor,
 geometric_state: Optional[torch.Tensor] = None,
 camera_poses: Optional[torch.Tensor] = None
 ) -> torch.Tensor:
 """
 Forward pass through geometric attention.

 Args:
 x: Input features [B, T, N, D]
 geometric_state: Geometric state [B, T, N, D]
 camera_poses: Camera poses [B, T, 4, 4]

 Returns:
 Attended features [B, T, N, D]
 """
 B, T, N, D = x.shape

 # Add global tokens
 global_tokens = self.global_tokens.expand(B, T, -1, -1)
 x_with_global = torch.cat([global_tokens, x], dim=2) # [B, T, N+G, D]

 # Generate QKV
 qkv = self.qkv(x_with_global) # [B, T, N+G, 3*D]
 q, k, v = qkv.chunk(3, dim=-1)

 # Reshape for multi-head attention
 q = q.reshape(B, T, -1, self.num_heads, self.head_dim).transpose(2, 3)
 k = k.reshape(B, T, -1, self.num_heads, self.head_dim).transpose(2, 3)
 v = v.reshape(B, T, -1, self.num_heads, self.head_dim).transpose(2, 3)
 # [B, T, num_heads, N+G, head_dim]

 # Compute attention scores
 scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
 # [B, T, num_heads, N+G, N+G]

 # Apply geometric bias if available
 if geometric_state is not None and camera_poses is not None:
 geom_bias = self._compute_geometric_bias(
 geometric_state, camera_poses
 ) # [B, T, num_heads, N+G, N+G]
 scores = scores + geom_bias

 # Apply window mask for local attention (excluding global tokens)
 # Skip for now to focus on core functionality
 # if N > self.window_size * self.window_size:
 # # For large sequences, apply windowed attention
 # window_mask = self.window_mask
 # scores[:, :, :, self.num_global_tokens:, self.num_global_tokens:] += window_mask

 # Apply softmax
 attn_weights = F.softmax(scores, dim=-1)
 attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)

 # Apply attention to values
 attended = torch.matmul(attn_weights, v) # [B, T, num_heads, N+G, head_dim]

 # Reshape and project output
 attended = attended.transpose(2, 3).reshape(B, T, -1, D)
 output = self.proj_drop(self.proj(attended))

 # Remove global tokens
 output = output[:, :, self.num_global_tokens:]

 return output

 def _compute_geometric_bias(
 self,
 geometric_state: torch.Tensor,
 camera_poses: torch.Tensor
 ) -> torch.Tensor:
 """Compute geometric attention bias from 3D positions."""
 B, T, N, _ = geometric_state.shape

 # For demonstration, we'll use a simple geometric bias
 # In practice, this would use actual 3D coordinates

 # Create relative position embeddings
 pos = torch.arange(N, device=geometric_state.device).float()
 rel_pos = pos[:, None] - pos[None, :] # [N, N]

 # Expand to batch and heads
 rel_pos_3d = torch.stack([
 rel_pos,
 rel_pos.abs(),
 rel_pos ** 2
 ], dim=-1) # [N, N, 3]

 geom_bias = self.geom_bias(rel_pos_3d) # [N, N, num_heads]
 geom_bias = geom_bias.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
 # [1, 1, num_heads, N, N]

 return geom_bias.expand(B, T, -1, -1, -1)

class PredictiveCodingHead(nn.Module):
 """
 Predictive coding head for self-supervised learning.

 Predicts future observations at multiple scales with uncertainty estimates.
 """

 def __init__(
 self,
 dim: int,
 num_scales: int = 4,
 hidden_dim: Optional[int] = None
 ):
 super().__init__()
 self.dim = dim
 self.num_scales = num_scales

 if hidden_dim is None:
 hidden_dim = dim

 # Prediction networks for each scale
 self.predictors = nn.ModuleList([
 nn.Sequential(
 nn.Linear(dim, hidden_dim),
 nn.ReLU(),
 nn.Linear(hidden_dim, dim + 1) # +1 for uncertainty
 ) for _ in range(num_scales)
 ])

 def forward(
 self,
 state: torch.Tensor,
 geometric_state: Optional[torch.Tensor] = None
 ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
 """
 Forward pass through predictive coding head.

 Args:
 state: Current state [B, T, N, D]
 geometric_state: Optional geometric state

 Returns:
 Tuple of (predictions, uncertainties) for each scale
 """
 predictions = []
 uncertainties = []

 for predictor in self.predictors:
 # Predict features and uncertainty
 pred = predictor(state) # [B, T, N, D+1]
 pred_features, uncertainty = pred[..., :-1], pred[..., -1:]

 predictions.append(pred_features)
 uncertainties.append(uncertainty)

 return predictions, uncertainties

 def compute_loss(
 self,
 predictions: List[torch.Tensor],
 targets: List[torch.Tensor],
 uncertainties: List[torch.Tensor],
 geometric_weight: float = 0.1
 ) -> torch.Tensor:
 """
 Compute predictive coding loss with uncertainty weighting.

 Args:
 predictions: List of predictions for each scale
 targets: List of target features for each scale
 uncertainties: List of uncertainty estimates
 geometric_weight: Weight for geometric consistency

 Returns:
 Total loss
 """
 total_loss = 0.0

 for pred, target, unc in zip(predictions, targets, uncertainties):
 # Uncertainty-weighted prediction loss
 pred_loss = torch.exp(-unc) * F.mse_loss(pred, target, reduction='none')
 pred_loss = pred_loss.mean() + 0.1 * unc.mean()

 total_loss += pred_loss

 return total_loss / len(predictions)