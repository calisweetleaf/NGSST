"""
Utility functions for NGSST
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

def log_SE3(transform: torch.Tensor) -> torch.Tensor:
 """
 Convert SE(3) matrix to se(3) Lie algebra element.

 This is a numerically stable implementation that handles edge cases
 like small rotations and identity transformations.

 Args:
 transform: SE(3) matrix [B, 4, 4]

 Returns:
 Lie algebra element [B, 6] (omega, v)
 """
 B = transform.shape[0]

 # Extract rotation and translation
 R = transform[:, :3, :3] # [B, 3, 3]
 t = transform[:, :3, 3] # [B, 3]

 # Compute rotation angle using trace
 cos_angle = (torch.trace(R, dim1=1, dim2=2) - 1) / 2
 cos_angle = torch.clamp(cos_angle, -1 + 1e-7, 1 - 1e-7)
 angle = torch.acos(cos_angle) # [B]

 # Handle small angles for numerical stability
 small_angle_mask = angle < 1e-3

 # Compute rotation axis (omega) using Rodrigues' formula
 omega = torch.zeros(B, 3, device=transform.device)

 # Standard case: angle > epsilon
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
 # For SE(3), we need to compute V^{-1} where
 # V = I - (1 - cos(angle)) / angle^2 * hat(omega) +
 # (angle - sin(angle)) / angle^3 * hat(omega)^2

 hat_omega = hat_operator(omega) # [B, 3, 3]
 hat_omega_sq = torch.bmm(hat_omega, hat_omega) # [B, 3, 3]

 # Compute coefficients with numerical stability
 angle_sq = angle ** 2
 safe_angle_sq = torch.where(small_angle_mask, torch.ones_like(angle_sq), angle_sq)

 # Use Taylor series expansion for small angles
 if small_angle_mask.any():
 # For small angles: (1 - cos(a)) / a^2 ≈ 1/2 - a^2/24
 coef1 = torch.where(
 small_angle_mask,
 0.5 - angle_sq / 24,
 (1 - torch.cos(angle)) / safe_angle_sq
 )

 # For small angles: (a - sin(a)) / a^3 ≈ 1/6 - a^2/120
 coef2 = torch.where(
 small_angle_mask,
 1/6 - angle_sq / 120,
 (angle - sin_angle) / (safe_angle_sq * angle)
 )
 else:
 coef1 = (1 - torch.cos(angle)) / safe_angle_sq
 coef2 = (angle - sin_angle) / (safe_angle_sq * angle)

 # Compute V matrix
 V = torch.eye(3, device=transform.device).unsqueeze(0).expand(B, -1, -1)
 V = V - coef1.unsqueeze(1).unsqueeze(2) * hat_omega
 V = V + coef2.unsqueeze(1).unsqueeze(2) * hat_omega_sq

 # Invert V (V is always invertible for valid SE(3) matrices)
 V_inv = torch.linalg.inv(V)

 # Compute translation component in Lie algebra
 v = torch.bmm(V_inv, t.unsqueeze(2)).squeeze(2) # [B, 3]

 # Combine omega and v to get full se(3) element
 xi = torch.cat([omega, v], dim=1) # [B, 6]

 return xi

def hat_operator(vec: torch.Tensor) -> torch.Tensor:
 """
 Hat operator: R^3 -> so(3)

 Maps a 3D vector to the space of skew-symmetric matrices.

 Args:
 vec: Vector [B, 3]

 Returns:
 Skew-symmetric matrix [B, 3, 3]
 """
 B = vec.shape[0]
 x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]

 hat = torch.zeros(B, 3, 3, device=vec.device, dtype=vec.dtype)
 hat[:, 0, 1] = -z
 hat[:, 0, 2] = y
 hat[:, 1, 0] = z
 hat[:, 1, 2] = -x
 hat[:, 2, 0] = -y
 hat[:, 2, 1] = x

 return hat

def adaptive_time_constant(
 geometry_change: torch.Tensor,
 feature_entropy: torch.Tensor,
 base_tau: float = 1.0,
 min_tau: float = 0.1,
 max_tau: float = 10.0
) -> torch.Tensor:
 """
 Compute adaptive time constant based on geometric change and feature complexity.

 The time constant adapts to scene dynamics:
 - Small change & low entropy -> large tau (slow adaptation)
 - Large change or high entropy -> small tau (fast adaptation)

 Args:
 geometry_change: Magnitude of geometric transformation [B, N, 1]
 feature_entropy: Feature entropy measuring scene complexity [B, N, 1]
 base_tau: Base time constant
 min_tau: Minimum time constant (for numerical stability)
 max_tau: Maximum time constant

 Returns:
 Adaptive time constant [B, N, 1]
 """
 # Compute adaptation factor
 adaptation_factor = 1.0 + geometry_change + feature_entropy

 # Compute time constant
 tau = base_tau / adaptation_factor

 # Clamp to valid range
 tau = torch.clamp(tau, min_tau, max_tau)

 return tau

def geometric_consistency_loss(
 predictions: torch.Tensor,
 targets: torch.Tensor,
 transformations: torch.Tensor,
 weight: float = 1.0
) -> torch.Tensor:
 """
 Compute geometric consistency loss.

 Enforces that predictions transform consistently with camera motion.

 Args:
 predictions: Predicted features [B, N, D]
 targets: Target features [B, N, D]
 transformations: Camera transformations [B, 4, 4]
 weight: Loss weight

 Returns:
 Geometric consistency loss
 """
 B, N, D = predictions.shape

 # For demonstration, we'll use a simple consistency loss
 # In practice, this would apply the transformation to the predictions

 # Compute relative transformation
 if transformations.shape[1] == 2:
 # Transformations are [g_t, g_{t+1}]
 g_t = transformations[:, 0] # [B, 4, 4]
 g_t_plus_1 = transformations[:, 1] # [B, 4, 4]

 # Relative transformation
 g_rel = g_t_plus_1 @ torch.inverse(g_t) # [B, 4, 4]
 else:
 g_rel = transformations

 # Simple consistency: predictions should be close to targets after transformation
 # This is a simplified version - full implementation would properly transform features
 consistency_error = F.mse_loss(predictions, targets)

 return weight * consistency_error

def compute_feature_entropy(features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
 """
 Compute entropy of features as a measure of scene complexity.

 Args:
 features: Feature tensor [B, N, D]
 eps: Small constant for numerical stability

 Returns:
 Entropy values [B, N, 1]
 """
 # Normalize features
 features_norm = F.normalize(features, dim=-1)

 # Compute variance across feature dimensions
 variance = torch.var(features_norm, dim=-1, keepdim=True) # [B, N, 1]

 # Entropy is proportional to variance for Gaussian-like distributions
 entropy = 0.5 * torch.log(2 * math.pi * math.e * (variance + eps))

 return entropy

def compute_geometry_change(
 curr_poses: torch.Tensor,
 prev_poses: Optional[torch.Tensor] = None
) -> torch.Tensor:
 """
 Compute magnitude of geometric transformation between poses.

 Args:
 curr_poses: Current camera poses [B, 4, 4]
 prev_poses: Previous camera poses [B, 4, 4]

 Returns:
 Geometry change magnitude [B, 1]
 """
 if prev_poses is None:
 # Assume identity transformation as previous pose
 prev_poses = torch.eye(4, device=curr_poses.device).unsqueeze(0).expand_as(curr_poses)

 # Compute relative transformation
 rel_transform = curr_poses @ torch.inverse(prev_poses) # [B, 4, 4]

 # Extract rotation and translation components
 rel_rotation = rel_transform[:, :3, :3] # [B, 3, 3]
 rel_translation = rel_transform[:, :3, 3] # [B, 3]

 # Compute rotation angle
 trace = torch.trace(rel_rotation, dim1=1, dim2=2)
 rotation_angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))

 # Compute translation magnitude
 translation_magnitude = torch.norm(rel_translation, dim=1)

 # Combined geometry change
 geometry_change = rotation_angle + translation_magnitude

 return geometry_change.unsqueeze(1) # [B, 1]

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
 """
 Partition input into windows for windowed attention.

 Args:
 x: Input tensor [B, H, W, D]
 window_size: Window size

 Returns:
 Windowed tensor and original shape
 """
 B, H, W, D = x.shape

 # Pad to multiple of window size
 pad_h = (window_size - H % window_size) % window_size
 pad_w = (window_size - W % window_size) % window_size

 if pad_h > 0 or pad_w > 0:
 x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
 H_padded, W_padded = x.shape[1], x.shape[2]
 else:
 H_padded, W_padded = H, W

 # Reshape into windows
 x = x.reshape(B, H_padded // window_size, window_size, W_padded // window_size, window_size, D)
 x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
 x = x.reshape(B, -1, window_size * window_size, D)

 return x, (H, W)

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
 """
 Reverse window partitioning.

 Args:
 windows: Windowed tensor [B, num_windows, window_size*window_size, D]
 window_size: Window size
 H: Original height
 W: Original width

 Returns:
 Original tensor [B, H, W, D]
 """
 B = windows.shape[0]

 # Compute padded dimensions
 pad_h = (window_size - H % window_size) % window_size
 pad_w = (window_size - W % window_size) % window_size
 H_padded = H + pad_h
 W_padded = W + pad_w

 # Reshape windows back to spatial grid
 num_win_h = H_padded // window_size
 num_win_w = W_padded // window_size

 x = windows.reshape(B, num_win_h, num_win_w, window_size, window_size, -1)
 x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
 x = x.reshape(B, H_padded, W_padded, -1)

 # Remove padding if present
 if pad_h > 0 or pad_w > 0:
 x = x[:, :H, :W, :]

 return x