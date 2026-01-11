# NGSST Wiki: Harmonic Vision Transformer v2.0

**Release: v1.0.1** | [DOI: 10.5281/zenodo.18203893](https://doi.org/10.5281/zenodo.18203893)

This wiki provides comprehensive documentation for the Neural Geometric State Space Transformer (NGSST), specifically the **Harmonic Vision Transformer (HVT) v2.0**—the first production-ready implementation of oscillator-based vision.

---

## Table of Contents

1. [Introduction](#introduction)
2. [What Changed from v1](#what-changed-from-v1)
3. [Architecture](#architecture)
4. [Training](#training)
5. [New Training Pipeline](#new-training-pipeline)
6. [RLHF and DPO](#rlhf-and-dpo)
7. [API Reference](#api-reference)
8. [Installation](#installation)
9. [Validation](#validation)
10. [Future: R-1 Vision](#future-r-1-vision)
11. [Contributing](#contributing)
12. [Citation](#citation)

---

## Introduction

### From Concept to Production

The original NGSST release (v1) was a conceptual demonstration—a proof that geometric state space ideas could be articulated in code. **HVT v2.0 is different**:

| Aspect | v1 (Demo) | v2 (Production) |
|--------|-----------|-----------------|
| Training | Placeholder loops | Verified CIFAR-10 training |
| RLHF | Not implemented | Full DPO pipeline |
| Validation | Basic forward pass | Comprehensive architecture tests |
| Stability | Experimental | Spectral normalization, RK4, damping |
| Documentation | Conceptual | Implementation-aligned whitepaper |

### Core Thesis

Vision transformers treat attention as the fundamental routing primitive. HVT v2.0 proposes an alternative:

> **Computation IS oscillator evolution. Routing EMERGES from synchronization.**

Instead of computing discrete attention weights through Q/K/V projections and softmax, HVT converts images to oscillator states and evolves them via Kuramoto dynamics. The synchronization order parameter—a physical quantity measuring phase coherence—becomes the routing signal.

---

## What Changed from v1

### Architecture Overhaul

The demo v1 described an "NGSST" with attention mechanisms and SE(3) equivariance. HVT v2.0 replaces this entirely:

- **No Attention**: Transformer attention is gone. Replaced by phase coherence routing.
- **Oscillator Core**: FrequencyOscillatorBank implements Kuramoto dynamics with adaptive coupling.
- **Gabor Tokenization**: FrequencyTokenizer uses learnable Gabor filters for phase/amplitude extraction.
- **Golden Ratio**: Frequency bands use phi^k spacing; sync target is 1/phi ≈ 0.618.

### Training Verification

v1 had no verified training. v2 includes:

- Harmonic learning rate scheduler with golden ratio modulation
- Oscillator warmup and breathing cycles for stability
- Physics-informed loss (sync regularization, phase smoothness, energy conservation)
- Baseline accuracy of 52.3% on CIFAR-10
- Full pipeline improvement to 57.2% (+4.9% absolute)
- Sync order reaching golden ratio target (0.618)

### RLHF Integration

First vision model with Direct Preference Optimization:

- Preference pairs from classification correctness
- Frozen reference model for KL-regularized margin optimization
- Tracked coherence and energy stability during optimization

---

## Architecture

### Pipeline Overview

```
Image [B, 3, H, W]
    │
    ▼
┌─────────────────────────────────┐
│  GABOR FILTER BANK              │
│  Multi-orientation, multi-scale │
│  → Complex responses            │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  FREQUENCY TOKENIZER            │
│  Phase: atan2(Im, Re)           │
│  Amplitude: sqrt(Re² + Im²)     │
│  → [B, N, K] tokens             │
└─────────────────────────────────┘
    │
    ▼
╔═════════════════════════════════╗
║  FREQUENCY OSCILLATOR BANK      ║
║  Kuramoto dynamics:             ║
║  dphi/dt = w + K*sin(dphase) - gamma*phi
║  RK4 integration                ║
║  Adaptive coupling              ║
╚═════════════════════════════════╝
    │
    ▼
┌─────────────────────────────────┐
│  PHASE COHERENCE ROUTER         │
│  R_k = |mean(A*exp(i*phi))|     │
│  Routing via coherence weights  │
└─────────────────────────────────┘
    │
    ▼
Classification Logits + Sync Diagnostics
```

### Key Components

#### FrequencyTokenizer

Converts images to oscillator states using learnable Gabor filters.

```python
from hvt_v2 import FrequencyTokenizer

tokenizer = FrequencyTokenizer(
    num_bands=4,
    hidden_dim=64,
    patch_size=16,
    num_orientations=8
)

# Returns phase [B, N, K] and amplitude [B, N, K]
phase, amplitude = tokenizer(image)
```

#### FrequencyOscillatorBank

The computational core—Kuramoto dynamics with stability controls.

```python
from hvt_v2 import FrequencyOscillatorBank

oscillator = FrequencyOscillatorBank(
    num_bands=4,
    hidden_dim=64,
    num_coupling_scales=3
)

# Evolve oscillator states
new_phase, new_amplitude, diagnostics = oscillator(
    phase, amplitude, dt=0.1, return_diagnostics=True
)

print(f"Sync Order: {diagnostics['sync_order'].mean():.4f}")
```

#### PhaseCoherenceRouter

Replaces attention with coherence-based routing.

```python
from hvt_v2 import PhaseCoherenceRouter

router = PhaseCoherenceRouter(
    num_bands=4,
    hidden_dim=64
)

# Route features based on coherence
routed_features, routing_weights = router(
    phase, amplitude, return_routing_weights=True
)
```

#### SE3MotionEncoder

Maps optical flow to Lie algebra elements for frequency modulation.

```python
from hvt_v2 import SE3MotionEncoder

motion_encoder = SE3MotionEncoder(hidden_dim=64)

# flow: [B, 2, H, W]
xi = motion_encoder(flow)  # [B, 6] Lie algebra element
```

---

## Training

### Harmonic Training Protocol

HVT v2.0 uses a physics-aware training strategy:

1. **Warmup Phase**: Reduced classification pressure, focus on oscillator stabilization
2. **Main Training**: Full loss with harmonic LR schedule
3. **Breathing Cycles**: Periodic relaxation to prevent metastable collapse

### Training Configuration

```python
from train import TrainingConfig

config = TrainingConfig(
    num_freq_bands=4,
    num_evolution_layers=3,
    hidden_dim=64,
    warmup_steps=500,
    breathing_interval=100,
    breathing_duration=10,
    learning_rate=3e-4,
    batch_size=32,
    sync_target=0.618  # Golden ratio complement
)
```

### Loss Function

The HarmonicLoss combines multiple terms:

```
L = lambda_rec * L_reconstruction
  + lambda_sync * L_synchronization
  + lambda_phase * L_phase_smoothness
  + lambda_energy * L_energy_conservation
```

### Running Training

```bash
# Single dataset (CIFAR-10)
python train.py

# Multi-dataset (CIFAR-10 + SVHN alternating)
python train_multi.py
```

---

## New Training Pipeline

**v1.0.1** introduces a unified training system in `run.py` with the full pipeline in `New-Training/code/`.

### Unified Entry Point

```bash
# Default training
python run.py

# Use YAML config
python run.py --config baseline.yaml

# Override parameters
python run.py --steps 15000 --lr 1e-4 --batch-size 64

# Periodic backups
python run.py --config experimental.yaml --backup-epochs 5
```

### Pipeline Components

| Module | Purpose |
|--------|---------|
| `config.py` | YAML-based configuration with presets |
| `trainer.py` | Main training loop with sync monitoring |
| `losses.py` | Sync-aware and frequency-domain losses |
| `optimizers.py` | Physics-informed optimization |
| `schedulers.py` | Adaptive breathing schedules |
| `evaluator.py` | Validation with oscillator diagnostics |
| `checkpointing.py` | Safe checkpoint and resume logic |
| `visualizer.py` | Real-time training dashboards |

### 8 Novel Training Methods

1. **Sync-Aware Loss**: Balances task loss with synchronization quality targeting golden ratio (0.618)
2. **Physics-Informed Optimizer**: Symplectic updates that preserve oscillator energy
3. **Adaptive Breathing Schedules**: LR adjustment triggered by sync instability
4. **Multi-Scale Curriculum**: Progressive frequency band addition
5. **Geometric Consistency**: SE(3) structure enforcement for motion tasks
6. **Phase-Space Curriculum**: Adaptive damping decreasing with training progress
7. **Frequency-Domain Loss**: Multi-scale frequency weighting
8. **Kuramoto Energy Regularization**: Energy-based stability constraints

### Results

| Configuration | Accuracy | Sync Order | Notes |
|---------------|----------|------------|-------|
| Baseline | 52.3% | 0.45 | Standard training |
| Full Pipeline | 57.2% | 0.62 | All methods enabled |

See [New-Training/README.md](New-Training/README.md) for full documentation.

---

## RLHF and DPO

### Vision DPO for Classification

HVT v2.0 adapts Direct Preference Optimization to vision:

**Preference Construction**:

- Chosen: true label
- Rejected: model's incorrect prediction
- Pairs only generated on misclassifications

**DPO Objective**:

```
L_DPO = -log(sigmoid(beta * (delta_policy - delta_reference)))
```

Where:

- delta = log_prob(chosen) - log_prob(rejected)
- Reference model is frozen copy of baseline

### Running RLHF

```bash
# After baseline training completes
python rlhf/run_rlhf.py

# Or use the fixed DPO implementation directly
python rlhf/vision_dpo_fixed.py
```

### Tracking Coherence During DPO

DPO can destabilize oscillator dynamics. Monitor:

- Sync order should remain in [0.5, 0.8] range
- Energy stability should not spike
- Per-band coherence diversity should be maintained

---

## API Reference

### Main Classes

| Class | Purpose |
|-------|---------|
| `HarmonicVisionTransformer` | Full model with all components |
| `FrequencyTokenizer` | Gabor-based image tokenization |
| `FrequencyOscillatorBank` | Kuramoto dynamics core |
| `PhaseCoherenceRouter` | Coherence-based routing |
| `SE3MotionEncoder` | Motion to Lie algebra |
| `GaborFilterBank` | Multi-scale Gabor filters |
| `HarmonicLoss` | Physics-informed loss |

### Key Functions

| Function | Purpose |
|----------|---------|
| `hat_operator(v)` | Vector to skew-symmetric matrix |
| `exp_so3(w)` | Exponential map SO(3) |
| `log_SO3(R)` | Logarithm map SO(3) |

### Physical Constants

| Constant | Value | Role |
|----------|-------|------|
| `PHI` | 1.618034 | Golden ratio |
| `SACRED_RATIO` | PHI / TAU | Base frequency |
| `C_NATURAL` | 299792458.0 | Speed of light (scaling reference) |

---

## Installation

### Requirements

- Python 3.8+
- PyTorch >= 2.0.0
- torchvision
- numpy

### Setup

```bash
git clone https://github.com/calisweetleaf/NGSST.git
cd NGSST

python -m venv .venv
.venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

---

## Validation

### Architecture Validation

```bash
python validation.py
```

This runs:

1. Novelty validation (unique mechanisms)
2. Failure mode analysis
3. Code-paper consistency check
4. Implementation completeness
5. Architectural soundness (gradient flow)

### Quick Model Test

```bash
python hvt_v2.py
```

Runs a demo forward pass with diagnostics.

### Graph Export

```bash
python graph_viz.py
```

Exports computation graphs to `visualizations/computational_graph/`.

---

## Future: R-1 Vision

HVT v2.0 is the stable backbone for what comes next. The R-1 model (NGSST v3) will extend the oscillator substrate to:

- **Reasoning**: Thought via synchronization patterns
- **Multi-modal**: Audio, text, video unified through coherence
- **Text Understanding**: Document encoding via frequency-domain analysis
- **Autonomous Operation**: Self-modifying oscillator configurations

The key insight: if oscillators can route visual information through synchronization, they can route *any* information. Text characters have sharp frequency signatures. Audio has natural temporal structure. The oscillator backbone doesn't care about modality—it cares about phase coherence.

R-1 will be the "4o moment" for this architecture.

---

## Contributing

### Development Workflow

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Run `python validation.py`
5. Submit pull request

### Code Style

- No inline comments unless absolutely necessary
- Docstrings for all public classes/methods
- Type hints throughout
- Location-agnostic imports (use `hvt_v2`, not `v2.harmonic_vision_transformer`)

---

## Citation

```bibtex
@article{hvt2026,
  title={Harmonic Vision Transformer: Oscillator Dynamics on SE(3) Manifolds 
         as the Computational Substrate for Visual Perception},
  author={Christian Trey Rowell},
  year={2026},
  doi={10.5281/zenodo.18203893},
  note={HVT v2.0 Production Release, v1.0.1}
}
```

---

## Contact

**Christian Trey Rowell**  
Email: <treyrowell1826@gmail.com>  
GitHub: [@calisweetleaf](https://github.com/calisweetleaf)

---

*HVT v2.0: Synchronization is not a metaphor. It's the computation.*

**License**: Somnus Sovereign Anti-Exploitation Software License
