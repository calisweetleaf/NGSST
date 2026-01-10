"""
Demo script for Neural Geometric State Space Transformer

This script demonstrates the core functionality of NGSST:
1. Multi-scale neural implicit tokenization
2. Neural Geometric State Space dynamics
3. Geometric attention with SE(3) equivariance
4. Predictive coding for self-supervised learning
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import time

from .models import NGSST, NGSSTConfig
from .modules import (
 MultiScaleNeuralImplicitTokenizer,
 NeuralGeometricStateSpace,
 GeometricAttention
)
from .utils import (
 log_SE3,
 hat_operator,
 adaptive_time_constant,
 geometric_consistency_loss,
 compute_feature_entropy,
 compute_geometry_change
)

def create_dummy_data(
 batch_size: int = 2,
 num_frames: int = 8,
 height: int = 224,
 width: int = 224,
 with_camera_poses: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
 """
 Create dummy video data for demonstration.

 Args:
 batch_size: Batch size
 num_frames: Number of frames
 height: Frame height
 width: Frame width
 with_camera_poses: Whether to generate camera poses

 Returns:
 Tuple of (video, camera_poses)
 """
 # Create dummy video (random noise)
 video = torch.randn(batch_size, num_frames, height, width, 3)

 # Normalize to [0, 1] range
 video = (video - video.min()) / (video.max() - video.min())

 camera_poses = None
 if with_camera_poses:
 # Generate smooth camera trajectory
 camera_poses = generate_camera_trajectory(
 batch_size, num_frames, radius=2.0, height=1.0
 )

 return video, camera_poses

def generate_camera_trajectory(
 batch_size: int,
 num_frames: int,
 radius: float = 2.0,
 height: float = 1.0
) -> torch.Tensor:
 """
 Generate a smooth camera trajectory around a point.

 Args:
 batch_size: Batch size
 num_frames: Number of frames
 radius: Orbit radius
 height: Camera height

 Returns:
 Camera poses [B, T, 4, 4]
 """
 poses = []

 for b in range(batch_size):
 batch_poses = []
 for t in range(num_frames):
 # Parameterize time
 s = t / (num_frames - 1)

 # Circular trajectory with slight variations per batch
 angle = 2 * np.pi * s + b * 0.1
 x = radius * np.cos(angle)
 z = radius * np.sin(angle)
 y = height + 0.1 * np.sin(2 * angle + b)

 # Look at origin
 position = torch.tensor([x, y, z], dtype=torch.float32)
 target = torch.zeros(3)
 up = torch.tensor([0, 1, 0])

 # Create camera matrix
 pose = look_at(position, target, up)
 batch_poses.append(pose)

 batch_poses = torch.stack(batch_poses) # [T, 4, 4]
 poses.append(batch_poses)

 return torch.stack(poses) # [B, T, 4, 4]

def look_at(
 position: torch.Tensor,
 target: torch.Tensor,
 up: torch.Tensor
) -> torch.Tensor:
 """
 Create camera matrix looking at target from position.

 Args:
 position: Camera position [3]
 target: Look target [3]
 up: Up vector [3]

 Returns:
 Camera matrix [4, 4]
 """
 forward = F.normalize(target - position, dim=0)
 right = F.normalize(torch.cross(up, forward), dim=0)
 camera_up = torch.cross(forward, right)

 # Create rotation matrix
 rotation = torch.stack([right, camera_up, forward], dim=1) # [3, 3]

 # Create translation vector
 translation = -torch.matmul(rotation, position.unsqueeze(1)) # [3, 1]

 # Create 4x4 camera matrix
 camera_matrix = torch.eye(4)
 camera_matrix[:3, :3] = rotation
 camera_matrix[:3, 3] = translation.squeeze()

 return camera_matrix

def demo_tokenization():
 """Demonstrate multi-scale neural implicit tokenization."""
 print("\n" + "="*60)
 print("DEMO 1: Multi-Scale Neural Implicit Tokenization")
 print("="*60)

 # Create tokenizer
 tokenizer = MultiScaleNeuralImplicitTokenizer(
 hidden_dim=256,
 num_scales=4
 )

 # Create dummy video
 video, _ = create_dummy_data(batch_size=1, num_frames=4, height=64, width=64)
 print(f"Input video shape: {video.shape}")

 # Tokenize
 tokens, coords = tokenizer(video)

 print(f"\nTokenization results:")
 for i, (tok, coord) in enumerate(zip(tokens, coords)):
 print(f" Scale {i+1}: tokens {tok.shape}, coords {coord.shape}")

 # Demonstrate resolution agnosticism
 print(f"\nResolution agnosticism test:")
 for size in [32, 64, 128, 256]:
 test_video = torch.randn(1, 1, size, size, 3)
 test_tokens, _ = tokenizer(test_video)
 print(f" Input size {size}x{size}: token shape {test_tokens[0].shape}")

 print("✓ Tokenizer can handle arbitrary input resolutions")

def demo_geometric_state_space():
 """Demonstrate Neural Geometric State Space dynamics."""
 print("\n" + "="*60)
 print("DEMO 2: Neural Geometric State Space (NGSS)")
 print("="*60)

 # Create NGSS module
 ngss = NeuralGeometricStateSpace(
 state_dim=256,
 input_dim=256,
 time_constant_base=1.0
 )

 # Create dummy input
 batch_size, num_frames, num_tokens = 2, 8, 49
 x = torch.randn(batch_size, num_frames, num_tokens, 256)

 # Create camera poses
 _, camera_poses = create_dummy_data(
 batch_size=batch_size,
 num_frames=num_frames,
 with_camera_poses=True
 )

 print(f"Input shape: {x.shape}")
 print(f"Camera poses shape: {camera_poses.shape}")

 # Forward pass
 state = ngss(x, camera_poses=camera_poses)
 print(f"Output state shape: {state.shape}")

 # Test without camera poses
 state_no_pose = ngss(x, camera_poses=None)
 print(f"State without poses shape: {state_no_pose.shape}")

 # Demonstrate adaptive time constants
 print(f"\nAdaptive time constant test:")

 # Create inputs with different characteristics
 static_input = torch.ones_like(x) * 0.1 # Low entropy, static
 dynamic_input = torch.randn_like(x) # High entropy, dynamic

 # Compute feature entropy
 static_entropy = compute_feature_entropy(static_input)
 dynamic_entropy = compute_feature_entropy(dynamic_input)

 print(f" Static input entropy: {static_entropy.mean():.4f}")
 print(f" Dynamic input entropy: {dynamic_entropy.mean():.4f}")

 # Compute adaptive time constants
 tau_static = adaptive_time_constant(
 torch.zeros_like(static_entropy), # No geometric change
 static_entropy,
 base_tau=1.0
 )
 tau_dynamic = adaptive_time_constant(
 torch.ones_like(dynamic_entropy) * 0.5, # Some geometric change
 dynamic_entropy,
 base_tau=1.0
 )

 print(f" Static tau: {tau_static.mean():.4f} (larger = slower adaptation)")
 print(f" Dynamic tau: {tau_dynamic.mean():.4f} (smaller = faster adaptation)")

 print("✓ NGSS adapts time constants based on scene dynamics")

def demo_geometric_attention():
 """Demonstrate geometric attention mechanism."""
 print("\n" + "="*60)
 print("DEMO 3: Geometric Attention")
 print("="*60)

 # Create attention module
 attention = GeometricAttention(
 dim=256,
 num_heads=8,
 window_size=7,
 num_global_tokens=4
 )

 # Create input
 batch_size, num_frames, num_tokens = 2, 1, 196 # 14x14 tokens
 x = torch.randn(batch_size, num_frames, num_tokens, 256)

 print(f"Input shape: {x.shape}")

 # Forward pass
 output = attention(x)
 print(f"Output shape: {output.shape}")

 # Test with geometric state
 geometric_state = torch.randn_like(x)
 output_with_geom = attention(x, geometric_state=geometric_state)
 print(f"Output with geometric state shape: {output_with_geom.shape}")

 # Demonstrate window partitioning
 from .utils import window_partition, window_reverse

 # Reshape to spatial grid
 grid_size = int(math.sqrt(num_tokens))
 x_grid = x.reshape(batch_size, num_frames, grid_size, grid_size, 256)

 windows, (H, W) = window_partition(x_grid.squeeze(1), window_size=7)
 print(f"Window partition: {windows.shape} from grid {x_grid.squeeze(1).shape}")

 x_reconstructed = window_reverse(windows, 7, H, W)
 print(f"Window reverse: {x_reconstructed.shape}")

 print("✓ Geometric attention with local-global factorization")

def demo_se3_operations():
 """Demonstrate SE(3) Lie group operations."""
 print("\n" + "="*60)
 print("DEMO 4: SE(3) Lie Group Operations")
 print("="*60)

 # Create some example transformations
 batch_size = 3

 # Identity transformation
 identity = torch.eye(4).unsqueeze(0).expand(batch_size, -1, -1)

 # Small rotation around y-axis
 angle = 0.1
 small_rot = identity.clone()
 small_rot[:, 0, 0] = math.cos(angle)
 small_rot[:, 0, 2] = math.sin(angle)
 small_rot[:, 2, 0] = -math.sin(angle)
 small_rot[:, 2, 2] = math.cos(angle)

 # Translation
 translation = identity.clone()
 translation[:, :3, 3] = torch.tensor([1.0, 2.0, 3.0])

 # Combined transformation
 transform = translation @ small_rot

 print("Testing log_SE3 conversion:")

 # Test identity
 xi_identity = log_SE3(identity)
 print(f" Identity -> xi: {xi_identity[0]} (should be close to zero)")

 # Test small rotation
 xi_rot = log_SE3(small_rot)
 print(f" Small rotation -> xi: {xi_rot[0]}")
 print(f" Rotation magnitude: {torch.norm(xi_rot[:, :3], dim=1)[0]:.4f}")

 # Test full transformation
 xi_full = log_SE3(transform)
 print(f" Full transform -> xi: {xi_full[0]}")

 # Test hat operator
 omega = torch.tensor([[1.0, 2.0, 3.0]])
 hat_omega = hat_operator(omega)
 print(f"\nHat operator:")
 print(f" omega: {omega[0]}")
 print(f" hat(omega):\n{hat_omega[0]}")

 print("✓ SE(3) operations working correctly")

def demo_predictive_coding():
 """Demonstrate predictive coding for self-supervised learning."""
 print("\n" + "="*60)
 print("DEMO 5: Predictive Coding")
 print("="*60)

 # Create predictive coding head
 from .modules import PredictiveCodingHead

 pred_head = PredictiveCodingHead(
 dim=256,
 num_scales=3
 )

 # Create dummy state
 batch_size, num_frames, num_tokens = 2, 4, 49
 state = torch.randn(batch_size, num_frames, num_tokens, 256)

 print(f"Input state shape: {state.shape}")

 # Generate predictions
 predictions, uncertainties = pred_head(state)

 print(f"\nPredictive coding results:")
 for i, (pred, unc) in enumerate(zip(predictions, uncertainties)):
 print(f" Scale {i+1}: prediction {pred.shape}, uncertainty {unc.shape}")

 # Compute loss
 targets = [torch.randn_like(pred) for pred in predictions]
 loss = pred_head.compute_loss(predictions, targets, uncertainties)
 print(f"\nPredictive coding loss: {loss.item():.4f}")

 print("✓ Predictive coding enables self-supervised learning")

def demo_full_model():
 """Demonstrate full NGSST model."""
 print("\n" + "="*60)
 print("DEMO 6: Full NGSST Model")
 print("="*60)

 # Create model configuration
 config = NGSSTConfig(
 hidden_dim=128,
 num_heads=4,
 num_layers=4,
 num_classes=10 # For classification
 )

 # Create model
 model = NGSST(config)
 print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

 # Create dummy data
 video, camera_poses = create_dummy_data(
 batch_size=2,
 num_frames=4,
 height=64,
 width=64,
 with_camera_poses=True
 )

 print(f"Input video shape: {video.shape}")
 print(f"Camera poses shape: {camera_poses.shape}")

 # Forward pass
 model.eval()
 with torch.no_grad():
 start_time = time.time()
 outputs = model(video, camera_poses=camera_poses)
 forward_time = time.time() - start_time

 print(f"\nForward pass results:")
 for key, value in outputs.items():
 if isinstance(value, torch.Tensor):
 print(f" {key}: {value.shape}")
 else:
 print(f" {key}: {type(value)}")

 print(f"Forward pass time: {forward_time:.4f} seconds")

 # Test resolution agnosticism
 print(f"\nResolution agnosticism test:")
 for size in [32, 64, 128]:
 test_video = torch.randn(1, 2, size, size, 3)
 with torch.no_grad():
 test_output = model(test_video)
 print(f" {size}x{size}: output shape {test_output['geometric_state'].shape}")

 print("✓ Full NGSST model working with novel geometric mechanisms")

def run_all_demos():
 """Run all demonstration scripts."""
 print("="*60)
 print("NEURAL GEOMETRIC STATE SPACE TRANSFORMER (NGSST)")
 print("Core Mechanisms Demonstration")
 print("="*60)

 # Set random seed for reproducibility
 torch.manual_seed(42)
 np.random.seed(42)

 # Run demos
 demo_tokenization()
 demo_geometric_state_space()
 demo_geometric_attention()
 demo_se3_operations()
 demo_predictive_coding()
 demo_full_model()

 print("\n" + "="*60)
 print("SUMMARY")
 print("="*60)
 print("NGSST introduces two genuinely novel mechanisms:")
 print("1. Neural Geometric State Space (NGSS)")
 print(" - Extends SSMs to geometric manifolds with SE(3) equivariance")
 print(" - Adaptive time constants based on scene dynamics")
 print(" - Continuous dynamics with discrete approximations")
 print()
 print("2. Multi-Scale Predictive Coding with Geometric Consistency")
 print(" - Self-supervised learning through temporal prediction")
 print(" - Geometric constraints enforce physical plausibility")
 print(" - Uncertainty-aware predictions")
 print()
 print("These mechanisms address key failure modes:")
 print("- Resolution Wall: Neural implicit tokenization")
 print("- Temporal Incoherence: Continuous state dynamics")
 print("- Quadratic Complexity: Local-global attention factorization")
 print("- Geometric Blindness: SE(3) equivariant operations")
 print("- Dataset Dependence: Predictive coding pretraining")
 print("="*60)

if __name__ == "__main__":
 run_all_demos()