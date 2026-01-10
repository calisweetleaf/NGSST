"""
Quick RLHF test on existing HVT checkpoint.

Loads checkpoint, runs DPO with synthetic preferences, evaluates.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy

from hvt_v2 import HarmonicVisionTransformer, SACRED_RATIO
from vision_dpo_fixed import VisionRLHFConfig, VisionDPOTrainer, GeometricConsistencyReward

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except:
    HAS_SAFETENSORS = False


def load_hvt_from_checkpoint(checkpoint_path: str):
    """Load HVT model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Create model with same config as training
    model = HarmonicVisionTransformer(
        num_freq_bands=4,
        base_omega=SACRED_RATIO,
        num_evolution_layers=3,
        hidden_dim=64,
        num_classes=10,
        use_phase_routing=True,
    )
    
    # Load weights
    if checkpoint_path.endswith('.safetensors') and HAS_SAFETENSORS:
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(state_dict)
    print(f"  ✓ Loaded {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model


def create_synthetic_preference_data(num_samples: int = 200):
    """
    Create synthetic preference data for DPO.
    
    For each image:
    - chosen_label = true label (correct)
    - rejected_label = random wrong label (incorrect)
    """
    print(f"\nCreating {num_samples} synthetic preference samples...")
    
    # Random images (CIFAR-10 size)
    images = torch.randn(num_samples, 3, 32, 32)
    
    # True labels
    true_labels = torch.randint(0, 10, (num_samples,))
    
    # Wrong labels (different from true)
    wrong_labels = (true_labels + torch.randint(1, 10, (num_samples,))) % 10
    
    print(f"  ✓ Images: {images.shape}")
    print(f"  ✓ Chosen (correct): {true_labels[:5].tolist()}...")
    print(f"  ✓ Rejected (wrong): {wrong_labels[:5].tolist()}...")
    
    return images, true_labels, wrong_labels


def evaluate_model(model: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor):
    """Evaluate accuracy and sync order."""
    model.eval()
    
    with torch.no_grad():
        outputs = model(images, return_intermediates=True)
        logits = outputs['logits']
        preds = logits.argmax(dim=-1)
        
        accuracy = (preds == labels).float().mean().item()
        sync_order = outputs['sync_order'].mean().item()
        
        # Geometric reward
        geo_reward = GeometricConsistencyReward()
        geo_score = geo_reward(images).mean().item()
    
    return {
        'accuracy': accuracy,
        'sync_order': sync_order,
        'geometric_score': geo_score,
    }


def main():
    print("=" * 60)
    print("VISION RLHF TEST - DPO on HVT Checkpoint")
    print("=" * 60)
    
    # 1. Load checkpoint
    checkpoint_path = "checkpoints/best.safetensors"
    if not Path(checkpoint_path).exists():
        checkpoint_path = "checkpoints/model_step_1000.safetensors"
    if not Path(checkpoint_path).exists():
        print("No checkpoint found! Run training first.")
        return
    
    policy_model = load_hvt_from_checkpoint(checkpoint_path)
    reference_model = copy.deepcopy(policy_model)
    
    # 2. Create preference data
    images, chosen_labels, rejected_labels = create_synthetic_preference_data(200)
    
    # 3. Evaluate BEFORE RLHF
    print("\n" + "=" * 40)
    print("BEFORE DPO:")
    pre_metrics = evaluate_model(policy_model, images[:50], chosen_labels[:50])
    print(f"  Accuracy: {pre_metrics['accuracy']*100:.1f}%")
    print(f"  Sync Order: {pre_metrics['sync_order']:.3f}")
    print(f"  Geometric Score: {pre_metrics['geometric_score']:.3f}")
    
    # 4. Run DPO
    print("\n" + "=" * 40)
    print("RUNNING DPO RLHF...")
    
    config = VisionRLHFConfig(
        learning_rate=1e-5,
        beta=0.1,
        num_epochs=2,
        batch_size=16,
        gradient_clip=1.0,
    )
    
    trainer = VisionDPOTrainer(policy_model, reference_model, config)
    
    # Create batches manually
    num_batches = len(images) // config.batch_size
    
    for epoch in range(config.num_epochs):
        epoch_loss = 0
        epoch_acc = 0
        
        for i in range(num_batches):
            start = i * config.batch_size
            end = start + config.batch_size
            
            batch = {
                'image': images[start:end],
                'chosen_label': chosen_labels[start:end],
                'rejected_label': rejected_labels[start:end],
            }
            
            metrics = trainer.train_step(batch)
            epoch_loss += metrics['loss']
            epoch_acc += metrics['accuracy']
        
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Preference Acc={avg_acc*100:.1f}%")
    
    # 5. Evaluate AFTER RLHF
    print("\n" + "=" * 40)
    print("AFTER DPO:")
    post_metrics = evaluate_model(policy_model, images[:50], chosen_labels[:50])
    print(f"  Accuracy: {post_metrics['accuracy']*100:.1f}%")
    print(f"  Sync Order: {post_metrics['sync_order']:.3f}")
    print(f"  Geometric Score: {post_metrics['geometric_score']:.3f}")
    
    # 6. Compare
    print("\n" + "=" * 40)
    print("COMPARISON:")
    acc_delta = (post_metrics['accuracy'] - pre_metrics['accuracy']) * 100
    sync_delta = post_metrics['sync_order'] - pre_metrics['sync_order']
    geo_delta = post_metrics['geometric_score'] - pre_metrics['geometric_score']
    
    print(f"  Accuracy:  {acc_delta:+.1f}%")
    print(f"  Sync:      {sync_delta:+.4f}")
    print(f"  Geometric: {geo_delta:+.4f}")
    
    # 7. Save RLHF'd model
    output_path = "checkpoints/hvt_rlhf.pt"
    torch.save(policy_model.state_dict(), output_path)
    print(f"\n✓ Saved RLHF model to {output_path}")
    
    print("\n" + "=" * 60)
    print("RLHF TEST COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
