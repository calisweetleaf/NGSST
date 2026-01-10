"""
Core modules for Neural Geometric State Space Transformer.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleNeuralImplicitTokenizer(nn.Module):
    """Multi-scale neural implicit tokenization."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_scales: int = 4,
        kernel_layers: int = 3,
        pos_encoding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales

        self.kernels = nn.ModuleList(
            [
                self._make_kernel(kernel_layers, pos_encoding_dim, hidden_dim)
                for _ in range(num_scales)
            ]
        )

        self.scale_weights = nn.Parameter(torch.ones(num_scales))
        self.pos_encoding = PositionalEncoding(2, pos_encoding_dim, 10)
        self.input_proj = nn.Linear(pos_encoding_dim * 2, hidden_dim)

    def _make_kernel(self, num_layers: int, in_dim: int, out_dim: int) -> nn.Module:
        layers: List[nn.Module] = []
        hidden_dim = out_dim
        for i in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                    nn.ReLU(),
                ]
            )
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, coords: Optional[List[torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if x.ndim != 5:
            raise ValueError("Tokenizer expects input shape [B, T, H, W, C]")
        batch, timesteps, height, width, _ = x.shape
        tokens: List[torch.Tensor] = []
        token_coords: List[torch.Tensor] = []

        for scale_idx in range(self.num_scales):
            stride = 2 ** scale_idx
            if coords is None:
                h_scaled = max(1, height // (4 * stride))
                w_scaled = max(1, width // (4 * stride))
                y_coords = torch.linspace(0, 1, h_scaled, device=x.device)
                x_coords = torch.linspace(0, 1, w_scaled, device=x.device)
                yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
                scale_coords = torch.stack([xx, yy], dim=-1)
                scale_coords = scale_coords.unsqueeze(0).expand(batch * timesteps, -1, -1, -1)
                scale_coords = scale_coords.reshape(batch * timesteps, -1, 2)
            else:
                scale_coords = coords[scale_idx]

            pos_encoded = self.pos_encoding(scale_coords)
            kernel_out = self.kernels[scale_idx](pos_encoded)
            kernel_out = kernel_out * self.scale_weights[scale_idx]
            kernel_out = kernel_out.reshape(batch, timesteps, -1, self.hidden_dim)
            scale_coords = scale_coords.reshape(batch, timesteps, -1, 2)

            tokens.append(kernel_out)
            token_coords.append(scale_coords)

        return tokens, token_coords


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, input_dim: int, output_dim: int, max_freq: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        freq_bands = 2.0 ** torch.linspace(0, max_freq, output_dim // (2 * input_dim))
        self.register_buffer("freq_bands", freq_bands)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords_expanded = coords.unsqueeze(-1) * self.freq_bands
        sin_enc = torch.sin(coords_expanded)
        cos_enc = torch.cos(coords_expanded)
        encoded = torch.stack([sin_enc, cos_enc], dim=-1)
        encoded = encoded.reshape(*coords.shape[:-1], -1)
        return encoded


def log_SE3(transform: torch.Tensor) -> torch.Tensor:
    """Convert SE(3) matrix to se(3) Lie algebra element."""

    if transform.ndim != 3 or transform.shape[-2:] != (4, 4):
        raise ValueError("transform must have shape [B, 4, 4]")

    batch = transform.shape[0]
    rotation = transform[:, :3, :3]
    translation = transform[:, :3, 3]

    cos_angle = (torch.diagonal(rotation, dim1=1, dim2=2).sum(dim=1) - 1.0) / 2.0
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_angle)
    small_angle_mask = angle < 1e-3

    sin_angle = torch.sin(angle)
    omega_sin = torch.stack(
        [
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ],
        dim=1,
    )
    safe_sin = torch.where(small_angle_mask, torch.ones_like(sin_angle), sin_angle)
    omega = omega_sin / (2 * safe_sin.unsqueeze(1))
    if small_angle_mask.any():
        omega_small = 0.5 * omega_sin[small_angle_mask]
        omega = omega.clone()
        omega[small_angle_mask] = omega_small

    hat_omega = hat_operator(omega)
    hat_omega_sq = torch.bmm(hat_omega, hat_omega)

    angle_sq = torch.where(small_angle_mask, torch.ones_like(angle), angle) ** 2
    coef1 = (1 - torch.cos(angle)) / angle_sq
    coef2 = (angle - sin_angle) / (angle_sq * torch.where(small_angle_mask, torch.ones_like(angle), angle))

    ident = torch.eye(3, device=transform.device).unsqueeze(0).expand(batch, -1, -1)
    V = ident - coef1.unsqueeze(1).unsqueeze(2) * hat_omega + coef2.unsqueeze(1).unsqueeze(2) * hat_omega_sq
    V_inv = torch.linalg.inv(V)
    v = torch.bmm(V_inv, translation.unsqueeze(2)).squeeze(2)
    xi = torch.cat([omega, v], dim=1)
    return xi


def hat_operator(vec: torch.Tensor) -> torch.Tensor:
    if vec.ndim != 2 or vec.shape[1] != 3:
        raise ValueError("vec must have shape [B, 3]")
    batch = vec.shape[0]
    x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
    hat = torch.zeros(batch, 3, 3, device=vec.device, dtype=vec.dtype)
    hat[:, 0, 1] = -z
    hat[:, 0, 2] = y
    hat[:, 1, 0] = z
    hat[:, 1, 2] = -x
    hat[:, 2, 0] = -y
    hat[:, 2, 1] = x
    return hat


class SE3EquivariantConv(nn.Module):
    """SE(3) equivariant convolution layer."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight_net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim * out_dim),
        )

    def forward(self, features: torch.Tensor, transformation: torch.Tensor) -> torch.Tensor:
        xi = log_SE3(transformation)
        weights = self.weight_net(xi).reshape(-1, self.in_dim, self.out_dim)
        output = torch.einsum("bnd,bdo->bno", features, weights)
        return output


class NeuralGeometricStateSpace(nn.Module):
    """Neural Geometric State Space (NGSS) module."""

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        time_constant_base: float = 1.0,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.time_constant_base = time_constant_base
        hidden_dim = hidden_dim or state_dim

        self.input_proj = nn.Linear(input_dim, state_dim)
        self.equivariant_conv = SE3EquivariantConv(state_dim, state_dim)
        self.time_net = nn.Sequential(
            nn.Linear(input_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.gate = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        camera_poses: Optional[torch.Tensor] = None,
        dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("NGSS expects input shape [B, T, N, D]")
        batch, timesteps, tokens, input_dim = x.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {input_dim}")

        x_proj = self.input_proj(x)
        if state is None:
            state = torch.zeros_like(x_proj)

        outputs: List[torch.Tensor] = []
        current_state = state[:, 0]

        for t in range(timesteps):
            x_t = x_proj[:, t]
            time_input = torch.cat([x_t, current_state], dim=-1)
            tau = self.time_constant_base * self.time_net(time_input)

            xi = None
            if camera_poses is not None and t > 0:
                g_prev = camera_poses[:, t - 1]
                g_curr = camera_poses[:, t]
                xi = g_curr @ torch.inverse(g_prev)

            if xi is not None:
                state_transformed = self.equivariant_conv(current_state, xi)
            else:
                state_transformed = current_state

            gate = self.gate(current_state)
            current_state = (1 - tau) * current_state + tau * (gate * state_transformed + (1 - gate) * x_t)
            outputs.append(current_state.unsqueeze(1))

        return torch.cat(outputs, dim=1)


def adaptive_time_constant(
    geometry_change: torch.Tensor, feature_entropy: torch.Tensor, base_tau: float = 1.0
) -> torch.Tensor:
    return base_tau / (1 + geometry_change + feature_entropy)


class GeometricAttention(nn.Module):
    """Geometric attention with local-global factorization and geometric bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: int = 7,
        num_global_tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.num_global_tokens = num_global_tokens
        self.head_dim = dim // num_heads
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.geom_bias = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, num_heads),
        )
        self.global_tokens = nn.Parameter(torch.randn(1, 1, num_global_tokens, dim))

    def forward(
        self,
        x: torch.Tensor,
        geometric_state: Optional[torch.Tensor] = None,
        camera_poses: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, timesteps, tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {dim}")

        global_tokens = self.global_tokens.expand(batch, timesteps, -1, -1)
        x_with_global = torch.cat([global_tokens, x], dim=2)
        qkv = self.qkv(x_with_global)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(batch, timesteps, -1, self.num_heads, self.head_dim).transpose(2, 3)
        k = k.reshape(batch, timesteps, -1, self.num_heads, self.head_dim).transpose(2, 3)
        v = v.reshape(batch, timesteps, -1, self.num_heads, self.head_dim).transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if geometric_state is not None and camera_poses is not None:
            geom_bias = self._compute_geometric_bias(geometric_state)
            scores = scores + geom_bias

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=0.1, training=self.training)
        attended = torch.matmul(attn_weights, v)
        attended = attended.transpose(2, 3).reshape(batch, timesteps, -1, dim)
        output = self.proj_drop(self.proj(attended))
        output = output[:, :, self.num_global_tokens :]
        return output

    def _compute_geometric_bias(self, geometric_state: torch.Tensor) -> torch.Tensor:
        batch, timesteps, tokens, _ = geometric_state.shape
        pos = torch.arange(tokens, device=geometric_state.device).float()
        rel_pos = pos[:, None] - pos[None, :]
        rel_pos_3d = torch.stack([rel_pos, rel_pos.abs(), rel_pos ** 2], dim=-1)
        geom_bias = self.geom_bias(rel_pos_3d)
        geom_bias = geom_bias.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        return geom_bias.expand(batch, timesteps, -1, -1, -1)


class PredictiveCodingHead(nn.Module):
    """Predictive coding head for self-supervised learning."""

    def __init__(self, dim: int, num_scales: int = 4, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        hidden_dim = hidden_dim or dim
        self.predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, dim + 1),
                )
                for _ in range(num_scales)
            ]
        )

    def forward(
        self, state: torch.Tensor, geometric_state: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        predictions: List[torch.Tensor] = []
        uncertainties: List[torch.Tensor] = []
        for predictor in self.predictors:
            pred = predictor(state)
            pred_features, uncertainty = pred[..., :-1], pred[..., -1:]
            predictions.append(pred_features)
            uncertainties.append(uncertainty)
        return predictions, uncertainties

    def compute_loss(
        self,
        predictions: List[torch.Tensor],
        targets: List[torch.Tensor],
        uncertainties: List[torch.Tensor],
        geometric_weight: float = 0.1,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=predictions[0].device)
        for pred, target, unc in zip(predictions, targets, uncertainties):
            pred_loss = torch.exp(-unc) * F.mse_loss(pred, target, reduction="none")
            pred_loss = pred_loss.mean() + 0.1 * unc.mean()
            total_loss = total_loss + pred_loss
        return total_loss / len(predictions)


__all__ = [
    "MultiScaleNeuralImplicitTokenizer",
    "PositionalEncoding",
    "log_SE3",
    "hat_operator",
    "SE3EquivariantConv",
    "NeuralGeometricStateSpace",
    "adaptive_time_constant",
    "GeometricAttention",
    "PredictiveCodingHead",
]
