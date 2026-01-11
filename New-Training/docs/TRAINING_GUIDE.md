# HVT v3 Training Guide

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/hvt-training-v3
cd hvt-training-v3

# Install dependencies
pip install -r requirements.txt

# Optional: Install full feature set
pip install -r requirements.txt datasets wandb tensorboard scipy scikit-learn
```

### Basic Training

```bash
# Train with baseline configuration
python main.py train --preset baseline

# Train with all novel methods enabled
python main.py train --preset experimental

# Train with custom configuration
python main.py train --config configs/my_config.yaml
```

### Resume Training

```bash
# Resume from latest checkpoint
python main.py train --resume checkpoints/latest

# Resume with specific checkpoint
python main.py train --resume checkpoints/checkpoint_20231201_120000
```

### Monitor Training

```bash
# Enable real-time dashboard
python main.py train --preset experimental --dashboard

# Use TensorBoard
python main.py train --preset experimental &
tensorboard --logdir=runs
```

## Configuration

### Configuration Files

HVT v3 uses YAML configuration files for all training parameters. Key configuration sections:

```yaml
# Model architecture
num_freq_bands: 6              # Number of frequency bands
num_evolution_layers: 4        # Number of evolution layers
hidden_dim: 128               # Hidden dimension size

# Training parameters
total_steps: 15000            # Total training steps
batch_size: 32                # Batch size
learning_rate: 3e-4           # Learning rate

# Novel methods
novel_methods:
  use_sync_aware_loss: true   # Enable sync-aware loss
  use_physics_informed_optimizer: true  # Enable physics-informed optimizer
  # ... other method toggles
```

### Preset Configurations

- **baseline**: Standard training without novel methods
- **experimental**: All novel methods enabled
- **fast**: Reduced training time for quick experiments
- **max_quality**: Maximum quality with extended training

### Configuration Overrides

Override any parameter from command line:

```bash
# Override specific parameters
python main.py train --preset experimental \
  --override total_steps=5000 \
  --override batch_size=64 \
  --override novel_methods.sync_aware_loss.lambda_sync=0.2
```

## Novel Training Methods

### 1. Sync-Aware Loss Functions

**Purpose**: Balance task performance with synchronization quality

**Configuration**:
```yaml
novel_methods:
  use_sync_aware_loss: true
  sync_aware_loss:
    lambda_sync: 0.1      # Sync regularization weight
    lambda_diversity: 0.05 # Diversity across bands
    target_sync: 0.618    # Golden ratio target
```

**When to Use**: Always recommended for HVT training

**Monitoring**: Watch sync order in dashboard - should approach 0.618

### 2. Physics-Informed Optimization

**Purpose**: Optimizer that respects Kuramoto dynamics

**Configuration**:
```yaml
novel_methods:
  use_physics_informed_optimizer: true
  physics_optimizer:
    optimizer_type: SymplecticAdam  # or HamiltonianSGD
    symplectic_coef: 0.01          # Energy preservation
```

**When to Use**: Recommended for stability, especially with high learning rates

**Monitoring**: Watch energy stability - should remain stable

### 3. Adaptive Breathing Schedules

**Purpose**: Dynamic learning rate adjustment based on sync order

**Configuration**:
```yaml
novel_methods:
  use_adaptive_breathing: true
  adaptive_breathing:
    sync_threshold_low: 0.4    # Trigger breathing if too desynchronized
    sync_threshold_high: 0.7   # Trigger breathing if over-synchronized
    breathing_duration: 10     # Duration of breathing phase
```

**When to Use**: Helps with exploration and stability

**Monitoring**: Watch for breathing indicator in dashboard

### 4. Multi-Scale Curriculum Learning

**Purpose**: Progressive addition of frequency bands

**Configuration**:
```yaml
novel_methods:
  use_multi_scale_curriculum: true
  curriculum:
    start_bands: 2          # Initial number of bands
    max_bands: 6            # Final number of bands
    expansion_schedule: exponential  # or linear, golden_ratio
```

**When to Use**: Recommended for better feature learning

**Monitoring**: Bands will be added automatically during training

### 5. Geometric Consistency Regularization

**Purpose**: Enforce SE(3) geometric constraints

**Configuration**:
```yaml
novel_methods:
  use_geometric_consistency: true
  geometric_consistency:
    lambda_geometric: 0.1   # Geometric loss weight
    consistency_type: se3_composition
```

**When to Use**: Important for motion understanding tasks

**Monitoring**: Geometric consistency metrics

## Advanced Usage

### Custom Training Loop

```python
from hvt_training_v3 import (
    TrainingConfig, HVTTrainer, create_config,
    SyncAwareLoss, SymplecticAdam
)

# Create configuration
config = create_config(
    preset='experimental',
    total_steps=10000,
    dataset_name='cifar10'
)

# Create trainer
trainer = HVTTrainer(config)

# Train
trainer.train()

# Get results
model = trainer.get_model()
history = trainer.get_training_history()
best_metrics = trainer.get_best_metrics()
```

### Custom Loss Functions

```python
from hvt_training_v3 import HarmonicLoss, SyncAwareLoss

# Create custom loss configuration
loss_config = SyncAwareLossConfig(
    lambda_sync=0.2,
    lambda_diversity=0.1,
    target_sync=0.618
)

# Use in training
loss_fn = HarmonicLoss(config, sync_loss=SyncAwareLoss(**loss_config.__dict__))
```

### Custom Optimizers

```python
from hvt_training_v3 import SymplecticAdam, get_optimizer

# Create model and configuration
model = HarmonicVisionTransformer(...)
config = TrainingConfig(...)

# Get custom optimizer
optimizer = get_optimizer(model, config, optimizer_type='SymplecticAdam')
```

## Monitoring and Analysis

### Real-Time Dashboard

The training dashboard provides live visualization of:

- **Sync Order Evolution**: Current and target synchronization
- **Loss Curves**: Task, sync, and total losses
- **Phase Space**: Real-time phase diagram
- **Energy Stability**: Energy distribution and stability
- **Learning Rate**: Current learning rate schedule
- **Breathing Indicator**: Shows when breathing phases are active

### Key Metrics to Watch

1. **Sync Order**: Should approach 0.618 (golden ratio)
   - Too low (<0.3): System is chaotic
   - Too high (>0.8): System is over-synchronized

2. **Energy Stability**: Should be low and stable
   - High values indicate unstable dynamics

3. **Loss Curves**: Should decrease smoothly
   - Sync loss should stabilize around target

4. **Phase Coherence**: Measures phase organization
   - Higher values indicate better organization

### Post-Training Analysis

```bash
# Analyze training results
python main.py analyze --history-path training_history.json

# Generate comprehensive evaluation
python main.py evaluate --model-path checkpoints/best_model.pt
```

## Troubleshooting

### Common Issues

**1. Sync Order Not Converging**
- Increase `lambda_sync` in sync-aware loss
- Check if learning rate is too high
- Enable adaptive breathing

**2. Energy Instability**
- Enable physics-informed optimizer
- Reduce learning rate
- Check gradient clipping

**3. Slow Convergence**
- Disable some novel methods for baseline comparison
- Increase batch size
- Check data loading performance

**4. Memory Issues**
- Reduce batch size
- Disable mixed precision
- Use gradient checkpointing

### Performance Optimization

**CPU Training** (default):
```yaml
num_workers: 0
pin_memory: false
mixed_precision: false
```

**GPU Training** (if available):
```yaml
device: cuda
num_workers: 4
pin_memory: true
mixed_precision: true
```

### Debugging

Enable verbose logging:
```bash
python main.py train --preset experimental --verbose
```

Check model architecture:
```python
model = HarmonicVisionTransformer(...)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(model)
```

## Best Practices

### For Research

1. **Start with Baseline**: Always compare against baseline (no novel methods)
2. **Ablation Studies**: Test each method independently
3. **Hyperparameter Sweeps**: Use wandb for systematic hyperparameter search
4. **Reproducibility**: Set random seeds and log all hyperparameters

### For Production

1. **Use Experimental Config**: All novel methods enabled for best performance
2. **Extended Training**: Use max_quality preset for final models
3. **Validation Monitoring**: Run validation frequently to detect overfitting
4. **Checkpoint Management**: Keep multiple checkpoints for ensemble methods

### For Exploration

1. **Fast Config**: Use fast preset for quick experiments
2. **Dashboard**: Always use dashboard for real-time monitoring
3. **Novel Methods**: Experiment with different combinations of methods
4. **Analysis**: Generate comprehensive plots and analysis

## Support and Resources

### Documentation

- [Research Synthesis](RESEARCH_SYNTHESIS.md): Historical methods and their adaptation
- [Novel Methods](NOVEL_METHODS.md): Detailed explanation of novel approaches
- [Training Guide](TRAINING_GUIDE.md): This document
- [Results Report](RESULTS_REPORT.md): Validation findings and comparisons

### Examples

See the `examples/` directory for:
- Basic training scripts
- Custom configuration examples
- Analysis notebooks
- Visualization scripts

### Issues and Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the documentation
3. Check existing issues on GitHub
4. Create new issue with detailed description

## Citation

If you use HVT v3 in your research, please cite:

```bibtex
@article{hvt_v3_training,
  title={Optimal Training Methods for Oscillator-Based Vision Transformers},
  author={HVT Development Team},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.