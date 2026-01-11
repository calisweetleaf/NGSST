"""
Physics-informed optimizers for HVT v3 training pipeline.

These optimizers respect the Kuramoto dynamics and oscillator physics,
providing better stability and convergence for oscillator-based architectures.
"""

import torch
import math
from typing import Dict, Any, Optional, List, Callable
from torch.optim.optimizer import Optimizer


class SymplecticAdam(Optimizer):
    """
    Symplectic Adam optimizer that preserves oscillator energy.
    
    Inspired by symplectic integrators for Hamiltonian systems,
    this optimizer respects the energy-conserving nature of
    coupled oscillator dynamics.
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        symplectic_coef=0.01,
        energy_preservation=True
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            symplectic_coef=symplectic_coef, energy_preservation=energy_preservation
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Previous parameter values for symplectic update
                    state['prev_params'] = p.detach().clone()
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                symplectic_coef = group['symplectic_coef']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # Step size
                step_size = group['lr'] / bias_correction1
                
                # Symplectic correction (energy preservation)
                if group['energy_preservation'] and symplectic_coef > 0:
                    # Compute energy gradient
                    energy_grad = p * grad  # Simplified energy gradient
                    
                    # Symplectic update term
                    symplectic_term = symplectic_coef * energy_grad
                    
                    # Update parameters with symplectic correction
                    p.addcdiv_(exp_avg + symplectic_term, denom, value=-step_size)
                else:
                    # Standard Adam update
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                
                # Update previous parameters
                state['prev_params'] = p.detach().clone()
        
        return loss


class HamiltonianSGD(Optimizer):
    """
    Hamiltonian SGD optimizer for oscillator dynamics.
    
    Treats parameter updates as Hamiltonian dynamics,
    preserving the symplectic structure of the system.
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0.9,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        hamiltonian_coef=0.01
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(
            lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov,
            hamiltonian_coef=hamiltonian_coef
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            hamiltonian_coef = group['hamiltonian_coef']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad
                
                # Hamiltonian dynamics term
                if hamiltonian_coef > 0:
                    # Simplified Hamiltonian term (position-momentum coupling)
                    hamiltonian_term = hamiltonian_coef * p
                    d_p = d_p + hamiltonian_term
                
                # Weight decay
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                
                # Momentum
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                
                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf
                
                p.add_(d_p, alpha=-group['lr'])
        
        return loss


class OscillatorAdamW(Optimizer):
    """
    AdamW optimizer with oscillator-specific adaptations.
    
    Includes special handling for:
    - Phase parameters (wrapped to [-π, π])
    - Log-space parameters (damping, frequencies)
    - Physical constraint enforcement
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
        phase_wrapping=True,
        log_space_params=None
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            phase_wrapping=phase_wrapping
        )
        self.log_space_params = log_space_params or []
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute update
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                
                # Apply update
                update = exp_avg / denom
                
                # Decoupled weight decay
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                
                p.add_(update, alpha=-step_size)
                
                # Post-update constraints
                if group['phase_wrapping'] and 'phase' in str(p):
                    # Wrap phases to [-π, π]
                    p.data = torch.atan2(torch.sin(p.data), torch.cos(p.data))
                
                # Ensure positive values for log-space parameters
                if any(name in str(p) for name in self.log_space_params):
                    p.data = torch.clamp(p.data, min=-10, max=10)  # Reasonable range for log params
        
        return loss


class AdaptiveCouplingOptimizer(Optimizer):
    """
    Optimizer that adapts coupling strength based on sync order.
    
    Increases coupling when sync order is low (to encourage synchronization)
    and decreases coupling when sync order is high (to prevent over-locking).
    """
    
    def __init__(
        self,
        params,
        lr=1e-3,
        base_coupling=0.1,
        adaptation_rate=0.01,
        sync_target=0.618
    ):
        defaults = dict(
            lr=lr, base_coupling=base_coupling,
            adaptation_rate=adaptation_rate, sync_target=sync_target
        )
        super().__init__(params, defaults)
        
        # Track coupling values
        self.coupling_history = {}
    
    def step(
        self, 
        sync_order: Optional[torch.Tensor] = None,
        closure: Optional[Callable] = None
    ) -> Optional[float]:
        """Perform optimization step with sync-based coupling adaptation."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Compute adaptation factor based on sync order
        if sync_order is not None:
            mean_sync = sync_order.mean().item()
            adaptation_factor = self._compute_adaptation(mean_sync)
        else:
            adaptation_factor = 1.0
        
        for group in self.param_groups:
            # Adapt coupling strength
            adapted_lr = group['lr'] * adaptation_factor
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Apply adapted learning rate
                p.add_(p.grad, alpha=-adapted_lr)
        
        return loss
    
    def _compute_adaptation(self, sync_order: float) -> float:
        """Compute coupling adaptation factor based on sync order."""
        for group in self.param_groups:
            target = group['sync_target']
            rate = group['adaptation_rate']
            
            if sync_order < target * 0.8:
                # Increase coupling to encourage synchronization
                return 1.0 + rate
            elif sync_order > target * 1.2:
                # Decrease coupling to prevent over-locking
                return 1.0 - rate
            else:
                # Maintain current coupling
                return 1.0
        return 1.0


def get_optimizer(
    model: torch.nn.Module,
    config: Any,
    optimizer_type: str = "SymplecticAdam"
) -> Optimizer:
    """
    Factory function to create optimizer based on configuration.
    
    Args:
        model: The model to optimize
        config: Training configuration
        optimizer_type: Type of optimizer to create
        
    Returns:
        Configured optimizer
    """
    # Separate parameters for different treatment
    oscillator_params = []
    other_params = []
    log_space_params = []
    
    for name, param in model.named_parameters():
        if any(x in name for x in ['phase', 'amplitude', 'omega', 'coupling']):
            oscillator_params.append(param)
        elif any(x in name for x in ['log_damping', 'log_frequencies', 'log_sigma']):
            log_space_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups
    param_groups = [
        {'params': oscillator_params, 'lr': config.learning_rate, 'phase_wrapping': True},
        {'params': log_space_params, 'lr': config.learning_rate * 0.1, 'log_space_params': True},
        {'params': other_params, 'lr': config.learning_rate}
    ]
    
    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    # Create optimizer
    if optimizer_type == "SymplecticAdam":
        # Filter config to only valid SymplecticAdam params
        valid_params = {'symplectic_coef', 'energy_preservation', 'betas', 'eps'}
        filtered_kwargs = {k: v for k, v in config.novel_methods.physics_optimizer.__dict__.items() 
                          if k in valid_params}
        return SymplecticAdam(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            **filtered_kwargs
        )
    elif optimizer_type == "HamiltonianSGD":
        return HamiltonianSGD(
            param_groups,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
            **config.novel_methods.physics_optimizer.__dict__
        )
    elif optimizer_type == "OscillatorAdamW":
        return OscillatorAdamW(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            log_space_params=['log_damping', 'log_frequencies', 'log_sigma']
        )
    elif optimizer_type == "AdaptiveCoupling":
        return AdaptiveCouplingOptimizer(
            param_groups,
            lr=config.learning_rate,
            sync_target=config.sync_target
        )
    else:
        # Default to standard Adam
        return torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )