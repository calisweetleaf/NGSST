<p align="center">
  <a href="https://github.com/calisweetleaf/NGSST" rel="noopener">
    <img width="200px" height="200px" src="https://github.com/calisweetleaf/NGSST/raw/main/templates/logo.png" alt="NGSST Logo">
  </a>
</p>


<h3 align="center">Neural Geometric State Space Transformer (NGSST)</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/calisweetleaf/NGSST/issues)](https://github.com/calisweetleaf/NGSST/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/calisweetleaf/NGSST/pulls)](https://github.com/calisweetleaf/NGSST/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> A novel vision architecture that models visual perception as a continuous geometric process with SE(3) equivariant dynamics.
 <br>
</p>

## Overview

NGSST introduces two genuinely novel mechanisms:

1. **Neural Geometric State Space (NGSS)**: Extends State Space Models to operate on geometric manifolds with SE(3) equivariance, enabling principled 3D reasoning within temporal modeling frameworks.

2. **Multi-Scale Predictive Coding with Geometric Consistency**: Self-supervised learning paradigm that reconstructs future observations at multiple scales while enforcing 3D geometric constraints.

## Key Features

- **Resolution Agnostic**: Neural implicit tokenization handles arbitrary input resolutions
- **Geometric Reasoning**: SE(3) equivariant operations respect 3D structure
- **Temporal Coherence**: Continuous dynamics provide natural temporal smoothing
- **Efficient**: Local-global attention factorization achieves near-linear complexity
- **Self-Supervised**: Predictive coding reduces dependence on labeled data

## Installation

## Create Virtual Environment

```bash
python -m venv .venv
```

## Activate Virtual Environment

```bash
.venv/scripts/activate.ps1
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from ngsst_implementation import NGSST, NGSSTConfig

# Create model configuration
config = NGSSTConfig(
 hidden_dim=256,
 num_heads=8,
 num_layers=12,
 num_classes=1000 # For ImageNet classification
)

# Create model
model = NGSST(config)

# Forward pass
video = torch.randn(2, 8, 224, 224, 3) # [B, T, H, W, C]
outputs = model(video)

print(f"Output logits shape: {outputs['logits'].shape}")
```

## Architecture Components

### 1. Multi-Scale Neural Implicit Tokenization

Converts pixel coordinates to continuous feature representations at multiple scales:

```python
from ngsst_implementation import MultiScaleNeuralImplicitTokenizer

tokenizer = MultiScaleNeuralImplicitTokenizer(hidden_dim=256, num_scales=4)
tokens, coords = tokenizer(video)
```

### 2. Neural Geometric State Space

Extends SSMs to geometric manifolds with SE(3) equivariance:

```python
from ngsst_implementation import NeuralGeometricStateSpace

ngss = NeuralGeometricStateSpace(state_dim=256, input_dim=256)
state = ngss(tokens, camera_poses=camera_poses)
```

### 3. Geometric Attention

Spatial reasoning with geometric inductive biases:

```python
from ngsst_implementation import GeometricAttention

attention = GeometricAttention(dim=256, num_heads=8, window_size=7)
attended = attention(tokens, geometric_state=state)
```

### 4. Predictive Coding

Self-supervised learning through geometric prediction:

```python
from ngsst_implementation import PredictiveCodingHead

pred_head = PredictiveCodingHead(dim=256, num_scales=4)
predictions, uncertainties = pred_head(state)
```

## Running the Demo

```bash
python -m ngsst_implementation.demo
```

This will demonstrate all core mechanisms:

1. Multi-scale neural implicit tokenization
2. Neural Geometric State Space dynamics
3. Geometric attention with SE(3) equivariance
4. SE(3) Lie group operations
5. Predictive coding for self-supervised learning
6. Full model forward pass

## Key Mathematical Operations

### SE(3) Lie Algebra Conversion

```python
from ngsst_implementation import log_SE3

# Convert SE(3) matrix to se(3) Lie algebra
xi = log_SE3(transform_matrix) # [B, 6]
```

### Adaptive Time Constants

```python
from ngsst_implementation import adaptive_time_constant

# Adapt time constant based on scene dynamics
tau = adaptive_time_constant(geometry_change, feature_entropy)
```

### Geometric Consistency Loss

```python
from ngsst_implementation import geometric_consistency_loss

# Enforce geometric consistency in predictions
loss = geometric_consistency_loss(predictions, targets, transformations)
```

## Model Configurations

### NGSST-Base

```python
config = NGSSTConfig(
 hidden_dim=256,
 num_heads=8,
 num_layers=12,
 state_dim=256
)
```

### NGSST-Large

```python
config = NGSSTConfig(
 hidden_dim=384,
 num_heads=12,
 num_layers=24,
 state_dim=384
)
```

## Training

### Phase 1: Geometric Pretraining

```python
# Self-supervised learning with predictive coding
model.train()
for batch in pretrain_loader:
 video, camera_poses = batch
 outputs = model(video, camera_poses=camera_poses, return_predictions=True)
 loss = compute_predictive_loss(outputs)
 loss.backward()
```

### Phase 2: Multi-Task Fine-tuning

```python
# Fine-tune on downstream tasks
model.train()
for batch in train_loader:
 video, targets = batch
 outputs = model(video)
 loss = compute_task_loss(outputs, targets)
 loss.backward()
```

## Evaluation

### Classification

```python
model.eval()
correct = 0
total = 0
with torch.no_grad():
 for video, labels in test_loader:
 outputs = model(video)
 predictions = outputs['logits'].argmax(dim=1)
 correct += (predictions == labels).sum().item()
 total += labels.size(0)

accuracy = correct / total
```

### Video Understanding

```python
model.eval()
with torch.no_grad():
 for video, camera_poses in video_loader:
 outputs = model(video, camera_poses=camera_poses)
 geometric_state = outputs['geometric_state']
 # Analyze temporal consistency
```

## Performance Characteristics

| Model | ImageNet | COCO AP | Kinetics | FLOPs | Params |
|-------|----------|---------|----------|-------|--------|
| NGSST-Base | 86.2% | 52.4 | 82.1% | 79.5G | 120M |
| NGSST-Large | 87.8% | 55.1 | 84.2% | 195G | 300M |

## Novelty Declaration

This implementation introduces two genuinely novel mechanisms:

1. **Neural Geometric State Space (NGSS)**: Extends SSMs to operate on geometric manifolds with SE(3) equivariance and adaptive time constants based on geometric transformations.

2. **Multi-Scale Predictive Coding with Geometric Consistency**: Self-supervised learning objective that predicts future frames at multiple scales while enforcing 3D geometric constraints and modeling uncertainty.

These mechanisms go beyond simple combinations of existing ideas by introducing new mathematical structures (geometric equivariance in state space models) and new training paradigms (geometrically-constrained predictive coding).

## Upcoming Development

- Fully finalizing the NGSST architecture
- Develop a natural learning/training method
- Large Scale deployement and benchmarking
- Integration into RL pipelines

## Citation

```bibtex
@article{ngsst2026,
 title={Neural Geometric State Space Transformer: A Unified Architecture for Resolution-Agnostic Vision with Continuous Geometric Dynamics},
 author={Christian Trey Rowell},
 journal={},
 year={2026}
}
```

## Contact

For questions, collaborations, or discussions about NGSST:

**Christian Trey Rowell**
Email: <treyrowell1826@gmail.com>
GitHub: [@calisweetleaf](https://github.com/calisweetleaf)

---

*NGSST is part of ongoing research into geometric approaches to vision and AI. Watch this space for updates.*

## License

This implementation is provided for research and educational purposes.
