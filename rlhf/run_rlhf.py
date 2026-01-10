"""
═══════════════════════════════════════════════════════════════════════════════
VISION RLHF PIPELINE - Full Training Run
═══════════════════════════════════════════════════════════════════════════════

Complete pipeline for RLHF fine-tuning of HarmonicVisionTransformer:
1. Load trained checkpoint
2. Baseline evaluation on CIFAR-10 test set
3. DPO training with preference data
4. Post-RLHF evaluation

Usage:
    python rlhf/run_rlhf.py --checkpoint checkpoints/best.pt
    python rlhf/run_rlhf.py --checkpoint checkpoints/step_15620.pt --epochs 3

Author: NGSST Research
Date: January 2026
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import argparse
import copy
import time
from typing import Dict, List, Tuple, Optional
import numpy as np

# HVT import
from hvt_v2 import HarmonicVisionTransformer, SACRED_RATIO

# Vision RLHF
from vision_rlhf import (
    VisionRLHFConfig, 
    VisionDPOTrainer, 
    GeometricConsistencyReward,
    AestheticRewardModel,
)

# Safetensors
try:
    from safetensors.torch import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# Dataset
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x


# ═══════════════════════════════════════════════════════════════════════════════
# CIFAR-10 DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset for evaluation and RLHF."""
    
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    def __init__(self, split: str = "train", max_samples: Optional[int] = None):
        if not HAS_DATASETS:
            raise ImportError("pip install datasets")
        
        self.split = split
        ds = load_dataset("uoft-cs/cifar10", split=split)
        
        self.images = []
        self.labels = []
        
        for i, sample in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            self.images.append(sample['img'])
            self.labels.append(sample['label'])
        
        print(f"Loaded {len(self.images)} samples from CIFAR-10 {split}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = self.images[idx]
        label = self.labels[idx]
        
        # Convert PIL to tensor
        img_np = np.array(img).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # [C, H, W]
        
        # Normalize
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        img_t = (img_t - mean) / std
        
        return {
            'image': img_t,
            'label': torch.tensor(label),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# PREFERENCE DATASET FOR DPO
# ═══════════════════════════════════════════════════════════════════════════════

class CIFARPreferenceDataset(Dataset):
    """
    Create preference pairs from CIFAR-10 for DPO.
    
    For each image:
    - chosen_label = true label (correct)
    - rejected_label = random wrong label (incorrect)
    
    This teaches the model to prefer correct classifications.
    """
    
    def __init__(self, base_dataset: CIFAR10Dataset, num_negatives: int = 1):
        self.base = base_dataset
        self.num_negatives = num_negatives
        self.num_classes = 10
    
    def __len__(self) -> int:
        return len(self.base) * self.num_negatives
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_idx = idx % len(self.base)
        sample = self.base[base_idx]
        
        true_label = sample['label'].item()
        
        # Generate random wrong label
        wrong_labels = [i for i in range(self.num_classes) if i != true_label]
        rejected_label = np.random.choice(wrong_labels)
        
        return {
            'image': sample['image'],
            'chosen_label': torch.tensor(true_label),
            'rejected_label': torch.tensor(rejected_label),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: str = "cpu") -> Dict[str, float]:
    """Full evaluation on dataset."""
    model.eval()
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    
    class_correct = [0] * 10
    class_total = [0] * 10
    
    confidences = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(images, return_intermediates=True)
        logits = outputs['logits']
        
        # Loss
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * len(labels)
        
        # Accuracy
        probs = F.softmax(logits, dim=-1)
        conf, preds = probs.max(dim=-1)
        
        correct = (preds == labels)
        total_correct += correct.sum().item()
        total_samples += len(labels)
        
        confidences.extend(conf.cpu().tolist())
        
        # Per-class
        for i in range(len(labels)):
            label = labels[i].item()
            class_total[label] += 1
            if correct[i]:
                class_correct[label] += 1
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples
    avg_conf = np.mean(confidences)
    
    class_acc = {
        CIFAR10Dataset.CLASSES[i]: class_correct[i] / max(class_total[i], 1) 
        for i in range(10)
    }
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'confidence': avg_conf,
        'class_accuracy': class_acc,
        'total_samples': total_samples,
    }


def print_eval_results(results: Dict, title: str = "Evaluation"):
    """Pretty print evaluation results."""
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")
    print(f"  Accuracy:   {results['accuracy']*100:6.2f}%")
    print(f"  Loss:       {results['loss']:6.4f}")
    print(f"  Confidence: {results['confidence']*100:6.2f}%")
    print(f"  Samples:    {results['total_samples']}")
    print(f"{'─' * 60}")
    print("  Per-class accuracy:")
    for cls, acc in results['class_accuracy'].items():
        bar = '█' * int(acc * 20) + '░' * (20 - int(acc * 20))
        print(f"    {cls:12s} │ {bar} │ {acc*100:5.1f}%")
    print(f"{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Vision RLHF Pipeline")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to trained HVT checkpoint")
    parser.add_argument("--epochs", type=int, default=2, help="RLHF epochs")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--train-samples", type=int, default=10000, 
                        help="Training samples for RLHF")
    parser.add_argument("--output", type=str, default="checkpoints/hvt_rlhf.pt",
                        help="Output checkpoint path")
    args = parser.parse_args()
    
    print("═" * 70)
    print("  🔥 VISION RLHF PIPELINE - HarmonicVisionTransformer")
    print("═" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {args.device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LR:         {args.lr}")
    print(f"  Beta:       {args.beta}")
    print("═" * 70)
    
    device = args.device
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Load Model
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📦 Loading HarmonicVisionTransformer...")
    
    # HVT v2.0 architecture (matching training config)
    model = HarmonicVisionTransformer(
        num_freq_bands=4,
        base_omega=SACRED_RATIO,
        coupling_strength=0.1,
        learnable_physics=True,
        patch_size=4,          # For 32x32 images
        hidden_dim=64,         # From training config
        num_routing_heads=4,
        num_evolution_layers=3, # From training config
        num_classes=10,        # CIFAR-10
        use_phase_routing=True,
        use_spectral_norm=False,
    )
    
    checkpoint_path = Path(args.checkpoint)
    
    # Check for safetensors format
    if not checkpoint_path.exists() and checkpoint_path.suffix == '.pt':
        # Try .safetensors instead
        safetensors_path = checkpoint_path.with_suffix('.safetensors')
        if safetensors_path.exists():
            checkpoint_path = safetensors_path
    
    if checkpoint_path.exists():
        if checkpoint_path.suffix == '.safetensors' and HAS_SAFETENSORS:
            state_dict = load_safetensors(str(checkpoint_path))
            # Load with strict=False for minor architecture differences
            result = model.load_state_dict(state_dict, strict=False)
            if result.missing_keys:
                print(f"  ⚠️  Missing keys: {len(result.missing_keys)}")
            if result.unexpected_keys:
                print(f"  ⚠️  Unexpected keys: {len(result.unexpected_keys)}")
            print(f"  ✅ Loaded safetensors checkpoint: {checkpoint_path.name}")
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                step = checkpoint.get('step', 0)
                print(f"  ✅ Loaded checkpoint from step {step}")
            else:
                model.load_state_dict(checkpoint, strict=False)
                print(f"  ✅ Loaded state dict")
    else:
        print(f"  ❌ Checkpoint not found: {checkpoint_path}")
        return
    
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Load Datasets
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📊 Loading CIFAR-10...")
    
    test_dataset = CIFAR10Dataset(split="test")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    train_dataset = CIFAR10Dataset(split="train", max_samples=args.train_samples)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Baseline Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🎯 Baseline Evaluation (pre-RLHF)...")
    baseline_results = evaluate(model, test_loader, device)
    print_eval_results(baseline_results, "BASELINE (Pre-RLHF)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Create Reference Model
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🔧 Creating reference model (frozen copy)...")
    reference_model = copy.deepcopy(model)
    for param in reference_model.parameters():
        param.requires_grad = False
    reference_model.to(device)
    reference_model.eval()
    print("  ✅ Reference model ready")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: DPO Training
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🚀 Starting DPO Training...")
    
    # Create preference dataset
    pref_dataset = CIFARPreferenceDataset(train_dataset, num_negatives=1)
    pref_loader = DataLoader(pref_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Config
    config = VisionRLHFConfig(
        learning_rate=args.lr,
        beta=args.beta,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        gradient_clip=1.0,
    )
    
    # Trainer
    trainer = VisionDPOTrainer(model, reference_model, config)
    
    # Training loop
    print(f"\n  Training on {len(pref_dataset)} preference pairs...")
    print(f"  Batches per epoch: {len(pref_loader)}")
    print()
    
    history = {'loss': [], 'accuracy': [], 'reward_margin': []}
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_margin = 0.0
        n_batches = 0
        
        model.train()
        pbar = tqdm(pref_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            
            epoch_loss += metrics['loss']
            epoch_acc += metrics['accuracy']
            epoch_margin += metrics['reward_margin']
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']*100:.1f}%",
                'margin': f"{metrics['reward_margin']:.3f}",
            })
        
        # Epoch stats
        epoch_loss /= n_batches
        epoch_acc /= n_batches
        epoch_margin /= n_batches
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['reward_margin'].append(epoch_margin)
        
        print(f"\n  Epoch {epoch+1} Complete:")
        print(f"    Loss:   {epoch_loss:.4f}")
        print(f"    DPO Acc: {epoch_acc*100:.1f}%")
        print(f"    Margin: {epoch_margin:.3f}")
        
        # Mid-training eval
        if epoch < args.epochs - 1:
            print("\n  Quick evaluation...")
            mid_results = evaluate(model, test_loader, device)
            print(f"    Test Accuracy: {mid_results['accuracy']*100:.2f}%")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Post-RLHF Evaluation
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🎯 Final Evaluation (post-RLHF)...")
    final_results = evaluate(model, test_loader, device)
    print_eval_results(final_results, "FINAL (Post-RLHF)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 7: Compare Results
    # ─────────────────────────────────────────────────────────────────────────
    print("\n📈 RLHF Impact Analysis")
    print("═" * 60)
    
    baseline_acc = baseline_results['accuracy'] * 100
    final_acc = final_results['accuracy'] * 100
    delta = final_acc - baseline_acc
    
    print(f"  Baseline Accuracy:  {baseline_acc:6.2f}%")
    print(f"  Post-RLHF Accuracy: {final_acc:6.2f}%")
    print(f"  Delta:              {delta:+6.2f}%")
    print()
    
    if delta > 0:
        print(f"  ✅ RLHF IMPROVED accuracy by {delta:.2f}%!")
    elif delta < -1:
        print(f"  ⚠️  RLHF decreased accuracy. Try lower LR or fewer epochs.")
    else:
        print(f"  ➡️  RLHF had minimal effect. Model may need more diverse preferences.")
    
    # Per-class changes
    print("\n  Per-class changes:")
    for cls in CIFAR10Dataset.CLASSES:
        base_c = baseline_results['class_accuracy'][cls] * 100
        final_c = final_results['class_accuracy'][cls] * 100
        delta_c = final_c - base_c
        arrow = "↑" if delta_c > 0 else "↓" if delta_c < 0 else "→"
        print(f"    {cls:12s}: {base_c:5.1f}% → {final_c:5.1f}% ({delta_c:+5.1f}% {arrow})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8: Save
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n💾 Saving RLHF model to {args.output}...")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'rlhf_epochs': args.epochs,
            'rlhf_lr': args.lr,
            'beta': args.beta,
            'baseline_accuracy': baseline_results['accuracy'],
            'final_accuracy': final_results['accuracy'],
        },
        'baseline_results': baseline_results,
        'final_results': final_results,
        'history': history,
    }, output_path)
    
    print(f"  ✅ Saved!")
    
    print("\n" + "═" * 70)
    print("  🎉 RLHF Pipeline Complete!")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
