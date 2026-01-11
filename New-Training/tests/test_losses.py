"""
Unit tests for novel loss functions.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from hvt_training_v3.losses import (
    SyncAwareLoss,
    PhaseCoherenceLoss,
    EnergyStabilityLoss,
    GeometricConsistencyLoss,
    HarmonicLoss
)


class TestSyncAwareLoss:
    """Test sync-aware loss function."""
    
    def test_initialization(self):
        """Test loss initialization."""
        loss = SyncAwareLoss(
            lambda_sync=0.1,
            lambda_diversity=0.05,
            target_sync=0.618
        )
        
        assert loss.lambda_sync == 0.1
        assert loss.lambda_diversity == 0.05
        assert loss.target_sync == 0.618
    
    def test_forward_pass(self):
        """Test forward pass computation."""
        loss = SyncAwareLoss()
        
        # Create dummy inputs
        batch_size, num_classes = 16, 10
        logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))
        sync_order = torch.rand(batch_size, 4)  # 4 frequency bands
        
        # Compute loss
        result = loss(logits, labels, sync_order, return_components=True)
        
        assert 'loss' in result
        assert 'task_loss' in result
        assert 'sync_loss' in result
        assert 'diversity_loss' in result
        assert 'mean_sync_order' in result
        
        # Check loss values are reasonable
        assert result['loss'].item() > 0
        assert result['task_loss'] >= 0
        assert result['sync_loss'] >= 0
        assert 0 <= result['mean_sync_order'] <= 1
    
    def test_adaptive_sync_target(self):
        """Test adaptive sync target functionality."""
        loss = SyncAwareLoss(use_adaptive_sync=True)
        
        # Simulate training history
        for i in range(20):
            logits = torch.randn(4, 10)
            labels = torch.randint(0, 10, (4,))
            sync_order = torch.full((4, 4), 0.6 + 0.1 * np.sin(i * 0.1))
            
            result = loss(logits, labels, sync_order)
            assert 'loss' in result
    
    def test_diversity_loss(self):
        """Test diversity loss computation."""
        loss = SyncAwareLoss(lambda_diversity=0.1)
        
        # Create sync order with high variance (should increase loss)
        sync_order = torch.tensor([[0.1, 0.9, 0.2, 0.8]])
        logits = torch.randn(1, 10)
        labels = torch.randint(0, 10, (1,))
        
        result = loss(logits, labels, sync_order, return_components=True)
        
        # High diversity should increase total loss
        assert result['diversity_loss'] > 0


class TestPhaseCoherenceLoss:
    """Test phase coherence loss function."""
    
    def test_initialization(self):
        """Test loss initialization."""
        loss = PhaseCoherenceLoss(
            lambda_coherence=0.1,
            lambda_smoothness=0.01
        )
        
        assert loss.lambda_coherence == 0.1
        assert loss.lambda_smoothness == 0.01
    
    def test_forward_pass(self):
        """Test forward pass computation."""
        loss = PhaseCoherenceLoss()
        
        # Create dummy phase and amplitude tensors
        batch_size, num_oscillators, num_bands = 8, 16, 4
        phase = torch.randn(batch_size, num_oscillators, num_bands)
        amplitude = torch.rand(batch_size, num_oscillators, num_bands)
        
        result = loss(phase, amplitude, return_components=True)
        
        assert 'loss' in result
        assert 'coherence_loss' in result
        assert 'smoothness_loss' in result
        assert 'mean_coherence' in result
        
        assert result['loss'].item() >= 0
        assert 0 <= result['mean_coherence'] <= 1
    
    def test_phase_coherence_computation(self):
        """Test phase coherence calculation."""
        loss = PhaseCoherenceLoss()
        
        # Create perfectly coherent phases
        phase = torch.zeros(1, 10, 1)  # All phases equal
        amplitude = torch.ones(1, 10, 1)
        
        result = loss(phase, amplitude, return_components=True)
        
        # Perfect coherence should result in high coherence value
        assert result['mean_coherence'] > 0.9
    
    def test_phase_smoothness(self):
        """Test phase smoothness regularization."""
        loss = PhaseCoherenceLoss(lambda_smoothness=0.1)
        
        # Create rapidly changing phases
        phase = torch.randn(1, 10, 1).requires_grad_(True)
        amplitude = torch.ones(1, 10, 1)
        
        result = loss(phase, amplitude, return_components=True)
        
        # Should compute smoothness loss
        assert 'smoothness_loss' in result


class TestEnergyStabilityLoss:
    """Test energy stability loss function."""
    
    def test_initialization(self):
        """Test loss initialization."""
        loss = EnergyStabilityLoss(
            lambda_energy=0.05,
            energy_target=1.0,
            stability_threshold=0.1
        )
        
        assert loss.lambda_energy == 0.05
        assert loss.energy_target == 1.0
        assert loss.stability_threshold == 0.1
    
    def test_forward_pass(self):
        """Test forward pass computation."""
        loss = EnergyStabilityLoss()
        
        # Create amplitude tensor
        amplitude = torch.rand(8, 16, 4) + 0.5  # Range [0.5, 1.5]
        
        result = loss(amplitude, return_components=True)
        
        assert 'loss' in result
        assert 'stability_loss' in result
        assert 'energy_loss' in result
        assert 'energy_mean' in result
        assert 'energy_stability' in result
        
        assert result['loss'].item() >= 0
        assert result['energy_mean'] > 0
        assert result['energy_stability'] >= 0
    
    def test_energy_stability_computation(self):
        """Test energy stability calculation."""
        loss = EnergyStabilityLoss(stability_threshold=0.1)
        
        # Create stable energy (low variance)
        amplitude = torch.ones(4, 10, 2) + 0.1 * torch.randn(4, 10, 2)
        
        result = loss(amplitude, return_components=True)
        
        # Should have low stability value
        assert result['energy_stability'] < 0.1
    
    def test_energy_tracking(self):
        """Test energy history tracking."""
        loss = EnergyStabilityLoss()
        
        # Simulate multiple forward passes
        for i in range(10):
            amplitude = torch.rand(2, 8, 2) * (1 + 0.1 * i)
            result = loss(amplitude)
        
        # History should be tracked
        assert len(loss.energy_history) == 10


class TestGeometricConsistencyLoss:
    """Test geometric consistency loss function."""
    
    def test_initialization(self):
        """Test loss initialization."""
        loss = GeometricConsistencyLoss(
            lambda_geometric=0.1,
            consistency_type="se3_composition"
        )
        
        assert loss.lambda_geometric == 0.1
        assert loss.consistency_type == "se3_composition"
    
    def test_se3_computation(self):
        """Test SE(3) composition consistency."""
        loss = GeometricConsistencyLoss(consistency_type="se3_composition")
        
        # Create SE(3) parameters
        batch_size = 4
        xi = torch.randn(batch_size, 6)  # [ω, v] parameters
        
        result = loss(xi, return_components=True)
        
        assert 'loss' in result
        assert 'rotation_loss' in result
        assert 'translation_loss' in result
        assert 'mean_rotation' in result
        assert 'mean_translation' in result
    
    def test_manifold_consistency(self):
        """Test manifold consistency loss."""
        loss = GeometricConsistencyLoss(consistency_type="manifold")
        
        xi = torch.randn(2, 6)
        
        result = loss(xi, return_components=True)
        
        assert 'loss' in result
        assert 'manifold_loss' in result
        assert 'xi_norm' in result


class TestHarmonicLoss:
    """Test combined harmonic loss function."""
    
    def test_initialization(self):
        """Test harmonic loss initialization."""
        config = type('Config', (), {
            'novel_methods': type('NovelMethods', (), {
                'use_sync_aware_loss': True,
                'use_geometric_consistency': True,
                'sync_aware_loss': type('SyncConfig', (), {'lambda_sync': 0.1}),
                'geometric_consistency': type('GeomConfig', (), {'lambda_geometric': 0.1})
            })
        })
        
        loss = HarmonicLoss(config)
        
        assert loss.sync_loss is not None
        assert loss.geometric_loss is not None
    
    def test_forward_pass(self):
        """Test combined loss computation."""
        config = type('Config', (), {
            'novel_methods': type('NovelMethods', (), {
                'use_sync_aware_loss': True,
                'use_geometric_consistency': False,
                'sync_aware_loss': type('SyncConfig', (), {'lambda_sync': 0.1}),
                'geometric_consistency': type('GeomConfig', (), {'lambda_geometric': 0.1})
            })
        })
        
        loss = HarmonicLoss(config)
        
        # Create predictions
        predictions = {
            'logits': torch.randn(8, 10),
            'reconstruction': torch.randn(8, 3, 32, 32)
        }
        
        # Create intermediate outputs
        sync_order = torch.rand(8, 4)
        phase_evolution = torch.randn(8, 16, 10, 4)
        amplitude_evolution = torch.rand(8, 16, 10, 4)
        labels = torch.randint(0, 10, (8,))
        
        result = loss(
            predictions=predictions,
            targets=torch.randn(8, 3, 32, 32),
            sync_order=sync_order,
            phase_evolution=phase_evolution,
            amplitude_evolution=amplitude_evolution,
            labels=labels
        )
        
        assert 'total' in result
        assert 'classification_loss' in result
        assert result['total'].item() > 0
    
    def test_loss_weights(self):
        """Test loss weight configuration."""
        config = type('Config', (), {
            'novel_methods': type('NovelMethods', (), {
                'use_sync_aware_loss': False,
                'use_geometric_consistency': False,
                'sync_aware_loss': type('SyncConfig', (), {'lambda_sync': 0.1}),
                'geometric_consistency': type('GeomConfig', (), {'lambda_geometric': 0.1})
            })
        })
        
        loss = HarmonicLoss(
            config,
            reconstruction_weight=0.8,
            classification_weight=1.2
        )
        
        # Test with only reconstruction
        predictions = {'reconstruction': torch.randn(2, 3, 32, 32)}
        result = loss(predictions, targets=torch.randn(2, 3, 32, 32))
        
        assert 'reconstruction_loss' in result
        assert result['total'].item() > 0


if __name__ == "__main__":
    pytest.main([__file__])