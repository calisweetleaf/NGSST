"""
Harmonic Vision Transformer - Architecture Visualization

Generates:
1. ONNX export for Netron visualization
2. Torchviz computational graph (PDF/PNG)
3. Custom layer-by-layer visualization

Run: python -m v2.visualize_architecture
"""

import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hvt_v2 import (
    HarmonicVisionTransformer,
    FrequencyOscillatorBank,
    GaborFilterBank,
    PhaseCoherenceRouter,
    SACRED_RATIO
)


def export_to_onnx(model: nn.Module, save_path: str, input_shape: tuple = (1, 3, 64, 64)):
    """
    Export model to ONNX format for Netron visualization.
    
    Open the .onnx file with Netron: https://netron.app/
    """
    print(f"\n{'='*60}")
    print("ONNX Export for Netron Visualization")
    print('='*60)
    
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    # Simplified forward for ONNX (no intermediates)
    class ONNXWrapper(nn.Module):
        def __init__(self, hvt):
            super().__init__()
            self.hvt = hvt
            
        def forward(self, x):
            out = self.hvt(x, return_intermediates=False, return_routing=False)
            return out['output'], out['logits']
    
    wrapped = ONNXWrapper(model)
    
    try:
        torch.onnx.export(
            wrapped,
            dummy_input,
            save_path,
            input_names=['image'],
            output_names=['reconstruction', 'logits'],
            dynamic_axes={
                'image': {0: 'batch', 2: 'height', 3: 'width'},
                'reconstruction': {0: 'batch'},
                'logits': {0: 'batch'}
            },
            opset_version=14,
            do_constant_folding=True
        )
        print(f"✓ Saved ONNX model to: {save_path}")
        print(f"  Open with Netron: https://netron.app/")
        print(f"  Or install: pip install netron && netron {save_path}")
        return True
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        print("  Note: Some custom ops may not be ONNX-compatible")
        return False


def generate_torchviz_graph(model: nn.Module, save_path: str, input_shape: tuple = (1, 3, 64, 64)):
    """
    Generate computational graph using torchviz.
    
    Requires: pip install torchviz graphviz
    """
    print(f"\n{'='*60}")
    print("Torchviz Computational Graph")
    print('='*60)
    
    try:
        from torchviz import make_dot
    except ImportError:
        print("✗ torchviz not installed. Install with:")
        print("  pip install torchviz")
        print("  Also need graphviz: https://graphviz.org/download/")
        return False
    
    model.eval()
    x = torch.randn(*input_shape, requires_grad=True)
    
    try:
        outputs = model(x)
        
        # Create graph from output
        dot = make_dot(
            outputs['logits'],
            params=dict(model.named_parameters()),
            show_attrs=True,
            show_saved=True
        )
        
        # Customize appearance
        dot.attr(rankdir='TB')  # Top to bottom
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue')
        
        # Save
        dot.render(save_path.replace('.pdf', ''), format='pdf', cleanup=True)
        dot.render(save_path.replace('.pdf', ''), format='png', cleanup=True)
        
        print(f"✓ Saved computational graph:")
        print(f"  PDF: {save_path}")
        print(f"  PNG: {save_path.replace('.pdf', '.png')}")
        return True
        
    except Exception as e:
        print(f"✗ Torchviz failed: {e}")
        return False


def print_architecture_tree(model: nn.Module, max_depth: int = 4):
    """
    Print detailed architecture tree with shapes.
    """
    print(f"\n{'='*60}")
    print("Architecture Tree (Module Hierarchy)")
    print('='*60)
    
    def print_module(module, prefix="", name="", depth=0):
        if depth > max_depth:
            return
            
        # Get module info
        class_name = module.__class__.__name__
        
        # Count parameters
        params = sum(p.numel() for p in module.parameters(recurse=False))
        trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        
        # Format output
        if params > 0:
            param_str = f" [{params:,} params, {trainable:,} trainable]"
        else:
            param_str = ""
        
        # Print with tree structure
        connector = "├── " if prefix else ""
        print(f"{prefix}{connector}{name}: {class_name}{param_str}")
        
        # Recurse into children
        children = list(module.named_children())
        for i, (child_name, child) in enumerate(children):
            is_last = (i == len(children) - 1)
            new_prefix = prefix + ("    " if is_last or not prefix else "│   ")
            print_module(child, new_prefix, child_name, depth + 1)
    
    print_module(model, "", "HarmonicVisionTransformer")


def visualize_gabor_filters(model: nn.Module, save_path: str):
    """
    Visualize the learned Gabor filters.
    """
    print(f"\n{'='*60}")
    print("Gabor Filter Visualization")
    print('='*60)
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("✗ matplotlib not installed: pip install matplotlib")
        return False
    
    # Get Gabor bank from tokenizer
    gabor_bank = model.tokenizer.gabor_bank
    filters = gabor_bank.get_filters().detach().cpu().numpy()
    
    # filters shape: [2*num_filters, K, K] (real and imag pairs)
    num_pairs = filters.shape[0] // 2
    
    # Create subplot grid
    rows = gabor_bank.num_orientations
    cols = gabor_bank.num_scales * 2  # Real and imaginary
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle('Learned Gabor Filters (Real | Imaginary)', fontsize=14)
    
    filter_idx = 0
    for o in range(rows):
        for s in range(gabor_bank.num_scales):
            # Real part
            ax_real = axes[o, s * 2] if rows > 1 else axes[s * 2]
            ax_real.imshow(filters[filter_idx], cmap='RdBu', vmin=-0.3, vmax=0.3)
            ax_real.axis('off')
            if o == 0:
                ax_real.set_title(f'S{s} Re', fontsize=8)
            
            # Imaginary part
            ax_imag = axes[o, s * 2 + 1] if rows > 1 else axes[s * 2 + 1]
            ax_imag.imshow(filters[filter_idx + 1], cmap='RdBu', vmin=-0.3, vmax=0.3)
            ax_imag.axis('off')
            if o == 0:
                ax_imag.set_title(f'S{s} Im', fontsize=8)
            
            filter_idx += 2
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved Gabor filter visualization: {save_path}")
    return True


def visualize_oscillator_evolution(model: nn.Module, save_path: str):
    """
    Visualize oscillator phase evolution over time.
    """
    print(f"\n{'='*60}")
    print("Oscillator Phase Evolution")
    print('='*60)
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("✗ matplotlib not installed")
        return False
    
    # Run model and collect phase evolution
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    
    with torch.no_grad():
        outputs = model(x, return_intermediates=True)
    
    if 'phase_evolution' not in outputs:
        print("✗ No phase evolution data available")
        return False
    
    phase_evo = outputs['phase_evolution'].squeeze(0).cpu().numpy()  # [num_layers, N, K]
    sync_order = outputs['sync_order'].squeeze(0).cpu().numpy()  # [num_layers, K]
    
    num_layers, num_tokens, num_bands = phase_evo.shape
    
    fig, axes = plt.subplots(2, num_layers, figsize=(4 * num_layers, 8))
    fig.suptitle('Oscillator Dynamics Through Layers', fontsize=14)
    
    for layer in range(num_layers):
        # Phase distribution (polar plot approximation as heatmap)
        ax_phase = axes[0, layer] if num_layers > 1 else axes[0]
        phase_data = phase_evo[layer]  # [N, K]
        
        # Show phase as heatmap
        im = ax_phase.imshow(
            np.cos(phase_data),  # Cosine of phase for visualization
            aspect='auto',
            cmap='twilight',
            vmin=-1, vmax=1
        )
        ax_phase.set_title(f'Layer {layer} Phases', fontsize=10)
        ax_phase.set_xlabel('Frequency Band')
        ax_phase.set_ylabel('Token')
        
        # Sync order bar chart
        ax_sync = axes[1, layer] if num_layers > 1 else axes[1]
        bars = ax_sync.bar(range(num_bands), sync_order[layer], color='steelblue')
        ax_sync.axhline(y=0.618, color='gold', linestyle='--', label='φ target')
        ax_sync.set_ylim(0, 1)
        ax_sync.set_title(f'Layer {layer} Sync: {sync_order[layer].mean():.3f}', fontsize=10)
        ax_sync.set_xlabel('Band')
        ax_sync.set_ylabel('R (sync)')
        if layer == num_layers - 1:
            ax_sync.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved oscillator evolution: {save_path}")
    return True


def visualize_routing_patterns(model: nn.Module, save_path: str):
    """
    Visualize phase coherence routing weights.
    """
    print(f"\n{'='*60}")
    print("Phase Coherence Routing Patterns")
    print('='*60)
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("✗ matplotlib not installed")
        return False
    
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    
    with torch.no_grad():
        outputs = model(x, return_routing=True)
    
    if 'routing_weights' not in outputs:
        print("✗ No routing weights available")
        return False
    
    routing_list = outputs['routing_weights']  # List of [B, num_heads, N, N]
    num_layers = len(routing_list)
    
    if num_layers == 0:
        print("✗ Empty routing weights")
        return False
    
    num_heads = routing_list[0].shape[1]
    
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(3 * num_heads, 3 * num_layers))
    fig.suptitle('Phase Coherence Routing Weights\n(NOT Attention - Emergent from Synchronization)', fontsize=12)
    
    for layer_idx, routing in enumerate(routing_list):
        routing = routing[0].cpu().numpy()  # [num_heads, N, N]
        
        for head_idx in range(num_heads):
            if num_layers > 1 and num_heads > 1:
                ax = axes[layer_idx, head_idx]
            elif num_layers > 1:
                ax = axes[layer_idx]
            elif num_heads > 1:
                ax = axes[head_idx]
            else:
                ax = axes
            
            im = ax.imshow(routing[head_idx], cmap='viridis', aspect='auto')
            ax.set_title(f'L{layer_idx} H{head_idx}', fontsize=9)
            
            if head_idx == 0:
                ax.set_ylabel('Query Token')
            if layer_idx == num_layers - 1:
                ax.set_xlabel('Key Token')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved routing patterns: {save_path}")
    return True


def create_layer_diagram(model: nn.Module, save_path: str):
    """
    Create a detailed layer-by-layer diagram as ASCII art for quick reference.
    """
    print(f"\n{'='*60}")
    print("Layer-by-Layer Architecture Diagram")
    print('='*60)
    
    diagram = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    HARMONIC VISION TRANSFORMER v2.0                           ║
║                 Oscillator Dynamics on SE(3) Manifolds                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║   INPUT: [B, 3, H, W]                                                        ║
║      │                                                                        ║
║      ▼                                                                        ║
║   ┌─────────────────────────────────────────────────────────────────────┐    ║
║   │  GABOR FILTER BANK                                                  │    ║
║   │  ├─ Orientations: θ ∈ [0, π) × num_orientations                    │    ║
║   │  ├─ Frequencies: f = f₀ × φˢ  (golden ratio scaling)               │    ║
║   │  └─ Output: Complex responses → Phase + Amplitude                   │    ║
║   └─────────────────────────────────────────────────────────────────────┘    ║
║      │                                                                        ║
║      ▼                                                                        ║
║   ┌─────────────────────────────────────────────────────────────────────┐    ║
║   │  FREQUENCY TOKENIZER                                                │    ║
║   │  ├─ Patches: H/patch × W/patch = N tokens                          │    ║
║   │  ├─ Phase tokens: [B, N, num_bands]                                │    ║
║   │  └─ Amplitude tokens: [B, N, num_bands]                            │    ║
║   └─────────────────────────────────────────────────────────────────────┘    ║
║      │                                                                        ║
║      ▼                                                                        ║
║   ╔═════════════════════════════════════════════════════════════════════╗    ║
║   ║  FREQUENCY OSCILLATOR BANK (× num_evolution_layers)     [THE CORE]  ║    ║
║   ║  ╔═══════════════════════════════════════════════════════════════╗  ║    ║
║   ║  ║  Kuramoto Dynamics:                                          ║  ║    ║
║   ║  ║                                                               ║  ║    ║
║   ║  ║    dφᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ sin(φⱼ - φᵢ) - γφᵢ                  ║  ║    ║
║   ║  ║           ↑       ↑                    ↑                     ║  ║    ║
║   ║  ║       natural  coupling             damping                  ║  ║    ║
║   ║  ║       frequency force                                        ║  ║    ║
║   ║  ╚═══════════════════════════════════════════════════════════════╝  ║    ║
║   ║                                                                     ║    ║
║   ║  Physical Constants (Learnable):                                    ║    ║
║   ║  ├─ C_scaled ≈ 1.95  (speed of light → coupling strength)          ║    ║
║   ║  ├─ G_scaled ≈ -0.47 (gravitational → long-range interactions)     ║    ║
║   ║  └─ α_scaled ≈ 0.73  (fine structure → quantum coupling)           ║    ║
║   ║                                                                     ║    ║
║   ║  SE(3) Motion Modulation:                                           ║    ║
║   ║  └─ ωᵢ = ω₀ × (1 + α||ξ||)  where ξ ∈ se(3)                        ║    ║
║   ║                                                                     ║    ║
║   ║  Sync Order (Kuramoto Order Parameter):                             ║    ║
║   ║  └─ R(t) = |⟨e^(iφ)⟩| ∈ [0, 1]   (target: φ ≈ 0.618)               ║    ║
║   ╚═════════════════════════════════════════════════════════════════════╝    ║
║      │                                                                        ║
║      ▼                                                                        ║
║   ┌─────────────────────────────────────────────────────────────────────┐    ║
║   │  PHASE COHERENCE ROUTER  ⟵ Replaces Attention!                     │    ║
║   │  ├─ Phase Locking Value: PLVᵢⱼ = |⟨e^(i(φᵢ-φⱼ))⟩|                  │    ║
║   │  ├─ Routing: softmax(PLV × amplitude)                              │    ║
║   │  └─ NO Q/K/V projections - routing emerges from dynamics           │    ║
║   └─────────────────────────────────────────────────────────────────────┘    ║
║      │                                                                        ║
║      ▼                                                                        ║
║   ┌─────────────────────────────────────────────────────────────────────┐    ║
║   │  HARMONIC DECODER                                                   │    ║
║   │  ├─ Oscillator states → Feature maps                               │    ║
║   │  ├─ Multi-scale upsampling (×2, ×2, ×2, ×2)                        │    ║
║   │  └─ Final conv → RGB output                                        │    ║
║   └─────────────────────────────────────────────────────────────────────┘    ║
║      │                                                                        ║
║      ▼                                                                        ║
║   OUTPUTS:                                                                    ║
║   ├─ Reconstruction: [B, 3, H, W]                                            ║
║   ├─ Logits: [B, num_classes]                                                ║
║   └─ Sync Order: [B, num_layers, num_bands]                                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

KEY INSIGHT: This is NOT "attention + oscillators"
             The computation IS oscillator evolution.
             Routing EMERGES from synchronization, not learned weights.
"""
    
    print(diagram)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(diagram)
    
    print(f"✓ Saved ASCII diagram: {save_path}")
    return True


def main():
    print("\n" + "="*70)
    print("   HARMONIC VISION TRANSFORMER - ARCHITECTURE VISUALIZATION")
    print("="*70)
    
    # Create output directory
    output_dir = Path(__file__).parent / "visualizations"
    output_dir.mkdir(exist_ok=True)
    
    # Create model
    print("\nInitializing model...")
    model = HarmonicVisionTransformer(
        num_freq_bands=4,
        base_omega=SACRED_RATIO,
        num_evolution_layers=3,
        hidden_dim=64,
        num_classes=10,
        use_phase_routing=True
    )
    model.eval()
    
    # Print stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable: {trainable:,}")
    
    # 1. Architecture tree
    print_architecture_tree(model)
    
    # 2. ASCII diagram
    create_layer_diagram(model, str(output_dir / "architecture_diagram.txt"))
    
    # 3. ONNX for Netron
    export_to_onnx(model, str(output_dir / "harmonic_vit.onnx"))
    
    # 4. Gabor filters
    visualize_gabor_filters(model, str(output_dir / "gabor_filters.png"))
    
    # 5. Oscillator evolution
    visualize_oscillator_evolution(model, str(output_dir / "oscillator_evolution.png"))
    
    # 6. Routing patterns
    visualize_routing_patterns(model, str(output_dir / "routing_patterns.png"))
    
    # 7. Torchviz (optional - needs graphviz)
    generate_torchviz_graph(model, str(output_dir / "computational_graph.pdf"))
    
    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print('='*70)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nTo view in Netron (interactive neural graph):")
    print(f"  1. Go to https://netron.app/")
    print(f"  2. Open: {output_dir / 'harmonic_vit.onnx'}")
    print("\nOr install Netron locally:")
    print("  pip install netron")
    print(f"  netron {output_dir / 'harmonic_vit.onnx'}")


if __name__ == "__main__":
    main()
