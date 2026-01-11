"""
Novel loss functions for HVT v3 training pipeline.

These loss functions are specifically designed for oscillator-based architectures,
incorporating synchronization order, phase coherence, and physical constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional, Tuple


class SyncAwareLoss(nn.Module):
    """
    Loss function that balances task performance with synchronization quality.
    
    L_total = L_task + λ_sync·|R - R_target|² + λ_div·Var(R_bands)
    
    Where:
    - R = mean sync order across all bands
    - R_target = φ - 1 ≈ 0.618 (golden ratio complement)
    - Var(R_bands) = variance of sync order across frequency bands
    
    The diversity term prevents all bands from synchronizing identically
    (which would reduce multi-scale expressiveness).
    """
    
    def __init__(
        self, 
        lambda_sync: float = 0.1,
        lambda_diversity: float = 0.05,
        target_sync: float = 0.618,
        sync_margin: float = 0.1,
        use_adaptive_sync: bool = True
    ):
        super().__init__()
        self.lambda_sync = lambda_sync
        self.lambda_diversity = lambda_diversity
        self.target_sync = target_sync
        self.sync_margin = sync_margin
        self.use_adaptive_sync = use_adaptive_sync
        
        # Track sync history for adaptive target
        self.register_buffer('sync_history', torch.zeros(100))
        self.register_buffer('history_idx', torch.tensor(0))
        
    def forward(
        self, 
        logits: torch.Tensor,
        labels: torch.Tensor,
        sync_order: torch.Tensor,
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute sync-aware loss.
        
        Args:
            logits: Model predictions [B, num_classes]
            labels: Ground truth labels [B]
            sync_order: Sync order per band [B, num_bands]
            return_components: Whether to return individual loss components
            
        Returns:
            Dictionary with total loss and components
        """
        # Task loss (classification)
        L_task = F.cross_entropy(logits, labels)
        
        # Sync order statistics
        mean_sync = sync_order.mean(dim=-1)  # [B]
        sync_var = sync_order.var(dim=-1)    # [B]
        
        # Adaptive sync target based on training progress
        if self.use_adaptive_sync and self.history_idx > 10:
            recent_sync = self.sync_history[:self.history_idx].mean()
            # Gradually increase target as model learns
            adaptive_target = self.target_sync + 0.1 * torch.sigmoid(recent_sync - self.target_sync)
            target = adaptive_target
        else:
            target = self.target_sync
        
        # Sync order loss (encourage target sync)
        sync_error = mean_sync - target
        L_sync = torch.clamp(sync_error**2 - self.sync_margin**2, min=0.0).mean()
        
        # Diversity loss (prevent uniform sync across bands)
        # We want some variance but not too much
        target_var = 0.1  # Target variance
        var_error = sync_var - target_var
        L_div = torch.clamp(var_error**2 - 0.01, min=0.0).mean()
        
        # Combine losses
        L_total = L_task + self.lambda_sync * L_sync + self.lambda_diversity * L_div
        
        # Update sync history
        if self.training:
            self.sync_history[self.history_idx % 100] = mean_sync.detach().mean()
            self.history_idx += 1
        
        if return_components:
            return {
                'loss': L_total,
                'task_loss': L_task,
                'sync_loss': L_sync,
                'diversity_loss': L_div,
                'mean_sync_order': mean_sync.mean(),
                'sync_variance': sync_var.mean(),
            }
        return {'loss': L_total}


class PhaseCoherenceLoss(nn.Module):
    """
    Loss function based on phase coherence between oscillators.
    
    Encourages meaningful phase relationships while preventing
    phase collapse or chaotic behavior.
    """
    
    def __init__(
        self,
        lambda_coherence: float = 0.1,
        lambda_smoothness: float = 0.01,
        coherence_threshold: float = 0.3
    ):
        super().__init__()
        self.lambda_coherence = lambda_coherence
        self.lambda_smoothness = lambda_smoothness
        self.coherence_threshold = coherence_threshold
    
    def forward(
        self,
        phase: torch.Tensor,  # [B, N, K]
        amplitude: torch.Tensor,  # [B, N, K]
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute phase coherence loss.
        
        Args:
            phase: Oscillator phases [B, N, K]
            amplitude: Oscillator amplitudes [B, N, K]
            
        Returns:
            Dictionary with loss and components
        """
        B, N, K = phase.shape
        
        # Compute pairwise phase differences
        phase_i = phase.unsqueeze(2)  # [B, N, 1, K]
        phase_j = phase.unsqueeze(1)  # [B, 1, N, K]
        phase_diff = phase_i - phase_j  # [B, N, N, K]
        
        # Compute phase coherence (cosine similarity)
        cos_sim = torch.cos(phase_diff)  # [B, N, N, K]
        
        # Weight by amplitudes
        amp_i = amplitude.unsqueeze(2)  # [B, N, 1, K]
        amp_j = amplitude.unsqueeze(1)  # [B, 1, N, K]
        amp_weight = torch.sqrt(amp_i * amp_j + 1e-8)  # [B, N, N, K]
        
        # Weighted coherence
        weighted_coherence = cos_sim * amp_weight  # [B, N, N, K]
        
        # Average over bands
        coherence = weighted_coherence.mean(dim=-1)  # [B, N, N]
        
        # Encourage moderate coherence (not too high, not too low)
        target_coherence = self.coherence_threshold
        coherence_error = coherence - target_coherence
        L_coherence = torch.clamp(coherence_error**2 - 0.01, min=0.0).mean()
        
        # Phase smoothness (prevent rapid phase changes)
        if phase.requires_grad:
            phase_grad = torch.gradient(phase, dim=1)[0]  # Spatial gradient
            L_smoothness = (phase_grad**2).mean()
        else:
            L_smoothness = torch.tensor(0.0, device=phase.device)
        
        L_total = self.lambda_coherence * L_coherence + self.lambda_smoothness * L_smoothness
        
        if return_components:
            return {
                'loss': L_total,
                'coherence_loss': L_coherence,
                'smoothness_loss': L_smoothness,
                'mean_coherence': coherence.mean(),
            }
        return {'loss': L_total}


class EnergyStabilityLoss(nn.Module):
    """
    Loss function for maintaining energy stability in oscillator dynamics.
    
    Prevents runaway energy growth or collapse while allowing
    meaningful energy variations for learning.
    """
    
    def __init__(
        self,
        lambda_energy: float = 0.05,
        energy_target: float = 1.0,
        stability_threshold: float = 0.1
    ):
        super().__init__()
        self.lambda_energy = lambda_energy
        self.energy_target = energy_target
        self.stability_threshold = stability_threshold
        
        # Track energy history
        self.register_buffer('energy_history', torch.zeros(100))
        self.register_buffer('energy_idx', torch.tensor(0))
    
    def forward(
        self,
        amplitude: torch.Tensor,  # [B, N, K]
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute energy stability loss.
        
        Args:
            amplitude: Oscillator amplitudes [B, N, K]
            
        Returns:
            Dictionary with loss and components
        """
        # Compute energy (squared amplitude)
        energy = (amplitude**2).mean(dim=[1, 2])  # [B]
        
        # Energy stability (coefficient of variation)
        energy_mean = energy.mean()
        energy_std = energy.std()
        stability = energy_std / (energy_mean + 1e-8)
        
        # Encourage stability
        L_stability = torch.clamp(stability - self.stability_threshold, min=0.0)
        
        # Encourage reasonable energy level
        energy_error = energy_mean - self.energy_target
        L_energy = energy_error**2
        
        L_total = self.lambda_energy * (L_stability + L_energy)
        
        # Update energy history
        if self.training:
            self.energy_history[self.energy_idx % 100] = energy_mean.detach()
            self.energy_idx += 1
        
        if return_components:
            return {
                'loss': L_total,
                'stability_loss': L_stability,
                'energy_loss': L_energy,
                'energy_mean': energy_mean,
                'energy_stability': stability,
            }
        return {'loss': L_total}


class GeometricConsistencyLoss(nn.Module):
    """
    Loss function for enforcing geometric consistency in SE(3) motion encoding.
    
    Ensures that predicted transformations are self-consistent and
    respect the Lie group structure of SE(3).
    """
    
    def __init__(
        self,
        lambda_geometric: float = 0.1,
        consistency_type: str = "se3_composition"
    ):
        super().__init__()
        self.lambda_geometric = lambda_geometric
        self.consistency_type = consistency_type
    
    def forward(
        self,
        xi: torch.Tensor,  # [B, 6] Lie algebra elements
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute geometric consistency loss.
        
        Args:
            xi: SE(3) Lie algebra elements [B, 6]
            
        Returns:
            Dictionary with loss and components
        """
        if self.consistency_type == "se3_composition":
            return self._se3_composition_loss(xi, return_components)
        elif self.consistency_type == "manifold":
            return self._manifold_loss(xi, return_components)
        else:
            return {'loss': torch.tensor(0.0, device=xi.device)}
    
    def _se3_composition_loss(
        self, 
        xi: torch.Tensor, 
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Enforce SE(3) composition consistency."""
        B = xi.shape[0]
        
        # Split into rotation and translation
        omega = xi[:, :3]  # Rotation part
        v = xi[:, 3:]      # Translation part
        
        # Enforce reasonable magnitude for rotation
        rot_magnitude = torch.norm(omega, dim=-1)
        L_rotation = torch.clamp(rot_magnitude - 1.0, min=0.0).mean()
        
        # Enforce reasonable magnitude for translation
        trans_magnitude = torch.norm(v, dim=-1)
        L_translation = torch.clamp(trans_magnitude - 1.0, min=0.0).mean()
        
        L_total = self.lambda_geometric * (L_rotation + L_translation)
        
        if return_components:
            return {
                'loss': L_total,
                'rotation_loss': L_rotation,
                'translation_loss': L_translation,
                'mean_rotation': rot_magnitude.mean(),
                'mean_translation': trans_magnitude.mean(),
            }
        return {'loss': L_total}
    
    def _manifold_loss(
        self, 
        xi: torch.Tensor, 
        return_components: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Enforce manifold constraints."""
        # For SE(3), the Lie algebra should be well-behaved
        # This is a simplified version - full implementation would use
        # the exponential map and manifold operations
        
        # Encourage smoothness in the Lie algebra
        xi_smoothness = (xi**2).mean()
        
        L_total = self.lambda_geometric * xi_smoothness
        
        if return_components:
            return {
                'loss': L_total,
                'manifold_loss': xi_smoothness,
                'xi_norm': torch.norm(xi, dim=-1).mean(),
            }
        return {'loss': L_total}


class HarmonicLoss(nn.Module):
    """
    Combined loss function for HVT training.
    
    Integrates all novel loss components with appropriate weighting.
    """
    
    def __init__(
        self,
        config: Optional[Any] = None,
        reconstruction_weight: float = 0.5,
        classification_weight: float = 1.0,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.classification_weight = classification_weight
        
        # Initialize component losses
        if config and config.novel_methods.use_sync_aware_loss:
            self.sync_loss = SyncAwareLoss(**config.novel_methods.sync_aware_loss.__dict__)
        else:
            self.sync_loss = None
            
        if config and config.novel_methods.use_geometric_consistency:
            self.geometric_loss = GeometricConsistencyLoss(**config.novel_methods.geometric_consistency.__dict__)
        else:
            self.geometric_loss = None
            
        self.phase_loss = PhaseCoherenceLoss()
        self.energy_loss = EnergyStabilityLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        sync_order: Optional[torch.Tensor] = None,
        phase_evolution: Optional[torch.Tensor] = None,
        amplitude_evolution: Optional[torch.Tensor] = None,
        xi_motion: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined harmonic loss.
        
        Args:
            predictions: Dictionary with model outputs
            targets: Target values (for reconstruction)
            sync_order: Sync order tensor
            phase_evolution: Phase evolution over time
            amplitude_evolution: Amplitude evolution over time
            xi_motion: SE(3) motion parameters
            labels: Classification labels
            
        Returns:
            Dictionary with total loss and components
        """
        total_loss = 0.0
        loss_components = {}
        
        # Reconstruction loss (hvt_v2 returns 'output', but we also check 'reconstruction' for flexibility)
        recon_key = 'reconstruction' if 'reconstruction' in predictions else 'output' if 'output' in predictions else None
        if recon_key is not None:
            L_recon = F.mse_loss(predictions[recon_key], targets)
            total_loss += self.reconstruction_weight * L_recon
            loss_components['reconstruction_loss'] = L_recon
        
        # Classification loss
        if 'logits' in predictions and labels is not None:
            L_cls = F.cross_entropy(predictions['logits'], labels)
            total_loss += self.classification_weight * L_cls
            loss_components['classification_loss'] = L_cls
        
        # Sync-aware loss
        if self.sync_loss is not None and 'logits' in predictions and labels is not None and sync_order is not None:
            sync_result = self.sync_loss(
                predictions['logits'], labels, sync_order, return_components=True
            )
            total_loss += sync_result['loss']
            for key, value in sync_result.items():
                if key != 'loss':
                    loss_components[key] = value
        
        # Phase coherence loss
        if phase_evolution is not None and amplitude_evolution is not None:
            # Handle multi-layer evolution: [B, num_layers, N, num_bands]
            if phase_evolution.ndim == 4:
                B, L, N, K = phase_evolution.shape
                # Apply loss per layer and average
                total_phase_loss = 0.0
                for layer_idx in range(L):
                    layer_phase = phase_evolution[:, layer_idx]  # [B, N, K]
                    layer_amp = amplitude_evolution[:, layer_idx]  # [B, N, K]
                    layer_result = self.phase_loss(layer_phase, layer_amp, return_components=True)
                    total_phase_loss += layer_result['loss']
                phase_result = {'loss': total_phase_loss / L}
            else:
                # Single layer: [B, N, K]
                phase_result = self.phase_loss(phase_evolution, amplitude_evolution, return_components=True)
            
            total_loss += phase_result['loss']
            for key, value in phase_result.items():
                if key != 'loss':
                    loss_components[key] = value
        
        # Energy stability loss
        if amplitude_evolution is not None:
            # Handle multi-layer evolution: [B, num_layers, N, num_bands]
            if amplitude_evolution.ndim == 4:
                B, L, N, K = amplitude_evolution.shape
                # Apply loss per layer and average
                total_energy_loss = 0.0
                for layer_idx in range(L):
                    layer_amp = amplitude_evolution[:, layer_idx]  # [B, N, K]
                    layer_result = self.energy_loss(layer_amp, return_components=True)
                    total_energy_loss += layer_result['loss']
                energy_result = {'loss': total_energy_loss / L}
            else:
                # Single layer: [B, N, K]
                energy_result = self.energy_loss(amplitude_evolution, return_components=True)
            
            total_loss += energy_result['loss']
            for key, value in energy_result.items():
                if key != 'loss':
                    loss_components[key] = value
        
        # Geometric consistency loss
        if self.geometric_loss is not None and xi_motion is not None:
            geom_result = self.geometric_loss(xi_motion, return_components=True)
            total_loss += geom_result['loss']
            for key, value in geom_result.items():
                if key != 'loss':
                    loss_components[key] = value
        
        loss_components['total'] = total_loss
        return loss_components