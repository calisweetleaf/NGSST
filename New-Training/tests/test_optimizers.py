"""
Unit tests for physics-informed optimizers.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from hvt_training_v3.optimizers import (
    SymplecticAdam,
    HamiltonianSGD,
    OscillatorAdamW,
    AdaptiveCouplingOptimizer,
    get_optimizer
)


class SimpleOscillatorModel(nn.Module):
    """Simple model with oscillator-like parameters for testing."""
    
    def __init__(self):
        super().__init__()
        self.phase = nn.Parameter(torch.randn(10))
        self.amplitude = nn.Parameter(torch.ones(10))
        self.log_damping = nn.Parameter(torch.zeros(10))
        self.weight = nn.Parameter(torch.randn(10, 10))
    
    def forward(self, x):
        return torch.matmul(x, self.weight) * self.amplitude * torch.cos(self.phase)


class TestSymplecticAdam:
    """Test Symplectic Adam optimizer."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        model = SimpleOscillatorModel()
        optimizer = SymplecticAdam(
            model.parameters(),
            lr=0.001,
            symplectic_coef=0.01,
            energy_preservation=True
        )
        
        assert optimizer.param_groups[0]['lr'] == 0.001
        assert optimizer.param_groups[0]['symplectic_coef'] == 0.01
        assert optimizer.param_groups[0]['energy_preservation'] == True
    
    def test_step(self):
        """Test optimizer step."""
        model = SimpleOscillatorModel()
        optimizer = SymplecticAdam(model.parameters(), lr=0.01)
        
        # Forward and backward pass
        x = torch.randn(4, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Store initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.clone()
        
        # Take optimization step
        optimizer.step()
        
        # Check parameters changed
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert not torch.allclose(param, initial_params[name], atol=1e-6)
    
    def test_symplectic_correction(self):
        """Test symplectic correction term."""
        model = SimpleOscillatorModel()
        
        # Test with and without symplectic correction
        optimizer_with = SymplecticAdam(
            model.parameters(),
            lr=0.01,
            symplectic_coef=0.01,
            energy_preservation=True
        )
        
        optimizer_without = SymplecticAdam(
            model.parameters(),
            lr=0.01,
            symplectic_coef=0.0,
            energy_preservation=False
        )
        
        # Compare parameter updates (would need more sophisticated test)
        # For now, just verify they run without error
        x = torch.randn(2, 10)
        
        for optimizer in [optimizer_with, optimizer_without]:
            model.zero_grad()
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
    
    def test_state_dict(self):
        """Test optimizer state saving and loading."""
        model = SimpleOscillatorModel()
        optimizer = SymplecticAdam(model.parameters(), lr=0.01)
        
        # Take some steps to build state
        for i in range(3):
            model.zero_grad()
            x = torch.randn(2, 10)
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        
        # Save state
        state_dict = optimizer.state_dict()
        
        # Create new optimizer and load state
        new_optimizer = SymplecticAdam(model.parameters(), lr=0.01)
        new_optimizer.load_state_dict(state_dict)
        
        # Should have same state
        assert len(new_optimizer.state) == len(optimizer.state)


class TestHamiltonianSGD:
    """Test Hamiltonian SGD optimizer."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        model = SimpleOscillatorModel()
        optimizer = HamiltonianSGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            hamiltonian_coef=0.01
        )
        
        assert optimizer.param_groups[0]['lr'] == 0.01
        assert optimizer.param_groups[0]['momentum'] == 0.9
        assert optimizer.param_groups[0]['hamiltonian_coef'] == 0.01
    
    def test_momentum_update(self):
        """Test momentum-based updates."""
        model = SimpleOscillatorModel()
        optimizer = HamiltonianSGD(model.parameters(), lr=0.01, momentum=0.9)
        
        # Multiple steps to build momentum
        for i in range(5):
            model.zero_grad()
            x = torch.randn(2, 10)
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        
        # Should have momentum buffer
        for param in model.parameters():
            if param.grad is not None:
                state = optimizer.state[param]
                assert 'momentum_buffer' in state
    
    def test_hamiltonian_term(self):
        """Test Hamiltonian correction term."""
        model = SimpleOscillatorModel()
        
        optimizer_with = HamiltonianSGD(
            model.parameters(),
            lr=0.01,
            hamiltonian_coef=0.01
        )
        
        optimizer_without = HamiltonianSGD(
            model.parameters(),
            lr=0.01,
            hamiltonian_coef=0.0
        )
        
        # Both should work without error
        for optimizer in [optimizer_with, optimizer_without]:
            model.zero_grad()
            x = torch.randn(2, 10)
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()


class TestOscillatorAdamW:
    """Test Oscillator AdamW optimizer."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        model = SimpleOscillatorModel()
        optimizer = OscillatorAdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01,
            phase_wrapping=True,
            log_space_params=['log_damping']
        )
        
        assert optimizer.param_groups[0]['lr'] == 0.001
        assert optimizer.param_groups[0]['weight_decay'] == 0.01
    
    def test_phase_wrapping(self):
        """Test phase wrapping functionality."""
        model = SimpleOscillatorModel()
        
        # Set phase outside [-π, π]
        with torch.no_grad():
            model.phase.data = torch.full_like(model.phase.data, 4 * np.pi)
        
        optimizer = OscillatorAdamW(
            model.parameters(),
            lr=0.01,
            phase_wrapping=True
        )
        
        # Take optimization step
        model.zero_grad()
        x = torch.randn(1, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # Phase should be wrapped to [-π, π]
        assert model.phase.data.abs().max() <= np.pi + 1e-6
    
    def test_log_space_constraints(self):
        """Test log space parameter constraints."""
        model = SimpleOscillatorModel()
        
        # Set log_damping to extreme value
        with torch.no_grad():
            model.log_damping.data = torch.full_like(model.log_damping.data, 20.0)
        
        optimizer = OscillatorAdamW(
            model.parameters(),
            lr=0.01,
            log_space_params=['log_damping']
        )
        
        # Take optimization step
        model.zero_grad()
        x = torch.randn(1, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # Should be clamped to reasonable range
        assert model.log_damping.data.abs().max() <= 10.0
    
    def test_weight_decay(self):
        """Test decoupled weight decay."""
        model = SimpleOscillatorModel()
        optimizer = OscillatorAdamW(model.parameters(), lr=0.01, weight_decay=0.1)
        
        # Store initial parameters
        initial_weight = model.weight.clone()
        
        # Take optimization step
        model.zero_grad()
        x = torch.randn(2, 10)
        output = model(x)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # Weight should be smaller due to weight decay
        assert torch.norm(model.weight) < torch.norm(initial_weight)


class TestAdaptiveCouplingOptimizer:
    """Test adaptive coupling optimizer."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        model = SimpleOscillatorModel()
        optimizer = AdaptiveCouplingOptimizer(
            model.parameters(),
            lr=0.01,
            base_coupling=0.1,
            adaptation_rate=0.01,
            sync_target=0.618
        )
        
        assert optimizer.param_groups[0]['lr'] == 0.01
        assert optimizer.param_groups[0]['base_coupling'] == 0.1
        assert optimizer.param_groups[0]['sync_target'] == 0.618
    
    def test_coupling_adaptation(self):
        """Test coupling strength adaptation."""
        model = SimpleOscillatorModel()
        optimizer = AdaptiveCouplingOptimizer(
            model.parameters(),
            lr=0.01,
            adaptation_rate=0.1
        )
        
        # Test with low sync order (should increase coupling)
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Simulate low sync order
        sync_order = torch.tensor(0.2)  # Below threshold
        optimizer.step(sync_order=sync_order)
        
        # Learning rate should increase
        assert optimizer.param_groups[0]['lr'] > initial_lr
        
        # Reset
        optimizer.param_groups[0]['lr'] = 0.01
        
        # Test with high sync order (should decrease coupling)
        sync_order = torch.tensor(0.9)  # Above threshold
        optimizer.step(sync_order=sync_order)
        
        # Learning rate should decrease
        assert optimizer.param_groups[0]['lr'] < 0.01
    
    def test_adaptation_factor_computation(self):
        """Test adaptation factor computation."""
        model = SimpleOscillatorModel()
        optimizer = AdaptiveCouplingOptimizer(
            model.parameters(),
            lr=0.01,
            adaptation_rate=0.1
        )
        
        # Test different sync order values
        test_cases = [
            (0.2, 1.0 + 0.1),  # Low sync -> increase
            (0.9, 1.0 - 0.1),  # High sync -> decrease
            (0.6, 1.0),        # Target sync -> maintain
        ]
        
        for sync_order, expected_factor in test_cases:
            factor = optimizer._compute_adaptation(sync_order)
            assert abs(factor - expected_factor) < 1e-6


class TestOptimizerFactory:
    """Test optimizer factory function."""
    
    def test_get_optimizer_symplectic(self):
        """Test getting SymplecticAdam optimizer."""
        model = SimpleOscillatorModel()
        
        config = type('Config', (), {
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'novel_methods': type('NovelMethods', (), {
                'physics_optimizer': type('PhysicsConfig', (), {
                    'optimizer_type': 'SymplecticAdam',
                    'symplectic_coef': 0.01,
                    'energy_preservation': True
                })
            })
        })
        
        optimizer = get_optimizer(model, config, 'SymplecticAdam')
        
        assert isinstance(optimizer, SymplecticAdam)
        assert optimizer.param_groups[0]['lr'] == 0.001
    
    def test_get_optimizer_hamiltonian(self):
        """Test getting HamiltonianSGD optimizer."""
        model = SimpleOscillatorModel()
        
        config = type('Config', (), {
            'learning_rate': 0.01,
            'weight_decay': 0.01
        })
        
        optimizer = get_optimizer(model, config, 'HamiltonianSGD')
        
        assert isinstance(optimizer, HamiltonianSGD)
    
    def test_get_optimizer_default(self):
        """Test getting default AdamW optimizer."""
        model = SimpleOscillatorModel()
        
        config = type('Config', (), {
            'learning_rate': 0.001,
            'weight_decay': 0.01
        })
        
        optimizer = get_optimizer(model, config, 'UnknownOptimizer')
        
        assert isinstance(optimizer, torch.optim.AdamW)
    
    def test_parameter_groups(self):
        """Test parameter group creation."""
        model = SimpleOscillatorModel()
        
        config = type('Config', (), {
            'learning_rate': 0.001,
            'weight_decay': 0.01
        })
        
        optimizer = get_optimizer(model, config, 'OscillatorAdamW')
        
        # Should have multiple parameter groups
        assert len(optimizer.param_groups) >= 2
        
        # Check that oscillator parameters have special treatment
        found_phase_group = False
        found_log_group = False
        
        for group in optimizer.param_groups:
            if group.get('phase_wrapping', False):
                found_phase_group = True
            if group.get('log_space_params', False):
                found_log_group = True
        
        assert found_phase_group or found_log_group


if __name__ == "__main__":
    pytest.main([__file__])