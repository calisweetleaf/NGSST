"""
Vision DPO for HarmonicVisionTransformer - FIXED VERSION

Proper preference construction for classification:
- Chosen: Images where model predicts correctly
- Rejected: Same images where model predicts incorrectly (from reference model)

This is different from language DPO - we use classification correctness as the preference signal.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Imports
try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("⚠️  safetensors not available, using .pt format")

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class VisionDPODataset(Dataset):
    """
    Dataset for vision DPO training.
    
    Creates preference pairs based on classification correctness:
    - (image, correct_label) = chosen
    - (image, wrong_label) = rejected
    """
    
    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        model_predictions: torch.Tensor,
    ):
        """
        Args:
            images: [N, C, H, W] image tensor
            labels: [N] true labels
            model_predictions: [N] model predictions
        """
        self.images = images
        self.labels = labels
        self.predictions = model_predictions
        
        # Create preference pairs
        self.preference_pairs = []
        
        for i in range(len(images)):
            true_label = labels[i].item()
            pred_label = model_predictions[i].item()
            
            # Only create pairs where model made an error
            if pred_label != true_label:
                self.preference_pairs.append({
                    'image_idx': i,
                    'chosen_label': true_label,      # Correct label
                    'rejected_label': pred_label,    # Wrong prediction
                })
        
        print(f"  Created {len(self.preference_pairs)} preference pairs from {len(images)} samples")
    
    def __len__(self):
        return len(self.preference_pairs)
    
    def __getitem__(self, idx):
        pair = self.preference_pairs[idx]
        image = self.images[pair['image_idx']]
        
        return {
            'image': image,
            'chosen_label': pair['chosen_label'],
            'rejected_label': pair['rejected_label'],
        }


def compute_dpo_loss(
    policy_logits_chosen: torch.Tensor,
    policy_logits_rejected: torch.Tensor,
    ref_logits_chosen: torch.Tensor,
    ref_logits_rejected: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute DPO loss for classification.
    
    Args:
        policy_logits_chosen: [B, num_classes] logits from policy model (current)
        policy_logits_rejected: [B, num_classes] same but for rejected label
        ref_logits_chosen: [B, num_classes] logits from reference model (frozen)
        ref_logits_rejected: [B, num_classes] same but for rejected label
        chosen_labels: [B] correct labels
        rejected_labels: [B] incorrect labels
        beta: DPO temperature
    
    Returns:
        loss: scalar loss
        metrics: dict with accuracy, margin, etc.
    """
    # Log probabilities of chosen vs rejected labels
    policy_log_probs_chosen = F.log_softmax(policy_logits_chosen, dim=-1)
    policy_log_probs_rejected = F.log_softmax(policy_logits_rejected, dim=-1)
    
    ref_log_probs_chosen = F.log_softmax(ref_logits_chosen, dim=-1)
    ref_log_probs_rejected = F.log_softmax(ref_logits_rejected, dim=-1)
    
    # Extract log probs for specific labels
    policy_chosen = policy_log_probs_chosen.gather(1, chosen_labels.unsqueeze(1)).squeeze(1)
    policy_rejected = policy_log_probs_rejected.gather(1, rejected_labels.unsqueeze(1)).squeeze(1)
    
    ref_chosen = ref_log_probs_chosen.gather(1, chosen_labels.unsqueeze(1)).squeeze(1)
    ref_rejected = ref_log_probs_rejected.gather(1, rejected_labels.unsqueeze(1)).squeeze(1)
    
    # DPO loss: -log σ(β * (log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x)))
    policy_logratios = policy_chosen - policy_rejected
    ref_logratios = ref_chosen - ref_rejected
    
    logits = beta * (policy_logratios - ref_logratios)
    loss = -F.logsigmoid(logits).mean()
    
    # Metrics
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen - ref_chosen)
        rejected_rewards = beta * (policy_rejected - ref_rejected)
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        
        # Accuracy: does policy prefer chosen over rejected?
        accuracy = (policy_logratios > 0).float().mean()
    
    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy.item(),
        'reward_margin': reward_margin.item(),
        'chosen_reward': chosen_rewards.mean().item(),
        'rejected_reward': rejected_rewards.mean().item(),
    }
    
    return loss, metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cpu',
) -> Dict:
    """Evaluate classification accuracy."""
    model.eval()
    
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    per_class_correct = torch.zeros(10)
    per_class_total = torch.zeros(10)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            logits = outputs['logits']
            
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=-1)
            
            correct = (preds == labels)
            total_correct += correct.sum().item()
            total_samples += len(labels)
            total_loss += loss.item() * len(labels)
            
            # Per-class accuracy
            for i in range(10):
                mask = labels == i
                if mask.sum() > 0:
                    per_class_correct[i] += correct[mask].sum().item()
                    per_class_total[i] += mask.sum().item()
    
    per_class_acc = per_class_correct / (per_class_total + 1e-8)
    
    return {
        'accuracy': total_correct / total_samples,
        'loss': total_loss / total_samples,
        'per_class_accuracy': per_class_acc.tolist(),
    }


def load_checkpoint_safe(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False,
) -> nn.Module:
    """Load checkpoint with architecture mismatch handling."""
    print(f"  Loading checkpoint: {checkpoint_path}")
    
    if HAS_SAFETENSORS and checkpoint_path.endswith('.safetensors'):
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
    
    # Try to load, handle mismatches
    try:
        model.load_state_dict(state_dict, strict=strict)
        print(f"  ✅ Loaded checkpoint (strict={strict})")
    except RuntimeError as e:
        print(f"  ⚠️  Checkpoint mismatch, attempting partial load...")
        
        # Load only matching keys
        model_dict = model.state_dict()
        matched_keys = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        missing_keys = set(model_dict.keys()) - set(matched_keys.keys())
        unexpected_keys = set(state_dict.keys()) - set(model_dict.keys())
        
        print(f"  Matched: {len(matched_keys)} / {len(model_dict)}")
        print(f"  Missing: {len(missing_keys)}")
        print(f"  Unexpected: {len(unexpected_keys)}")
        
        if len(matched_keys) < len(model_dict) * 0.8:
            raise RuntimeError(f"Too many missing keys ({len(missing_keys)}), checkpoint may be incompatible")
        
        model_dict.update(matched_keys)
        model.load_state_dict(model_dict)
        print(f"  ✅ Loaded {len(matched_keys)} matching parameters")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Vision DPO Training (Fixed)")
    parser.add_argument('--model', type=str, default='hvt_model.pt', help='Path to model file (loads and saves to same file)')
    parser.add_argument('--checkpoint', type=str, default=None, help='[DEPRECATED] Use --model instead')
    parser.add_argument('--output-dir', type=str, default='rlhf_output', help='Output directory for logs')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1, help='DPO temperature')
    parser.add_argument('--num-samples', type=int, default=10000, help='Number of samples for training')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--dataset', type=str, default='uoft-cs/cifar10', help='HuggingFace dataset name')
    parser.add_argument('--dataset-config', type=str, default=None, help='Dataset config name (e.g., cropped_digits for SVHN)')
    parser.add_argument('--image-key', type=str, default='img', help='Key for image in dataset')
    parser.add_argument('--label-key', type=str, default='label', help='Key for label in dataset')
    
    args = parser.parse_args()
    
    # Imports (delayed so --help works without deps)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from hvt_v2 import HarmonicVisionTransformer
    
    # Handle deprecated --checkpoint arg
    model_path = args.model
    if args.checkpoint and not args.model:
        model_path = args.checkpoint
        print("⚠️  --checkpoint is deprecated, use --model instead")
    
    print("=" * 70)
    print("  🔥 VISION DPO TRAINING (FIXED)")
    print("=" * 70)
    print(f"  Model:      {model_path}")
    print(f"  Output:     {args.output_dir}")
    print(f"  Device:     {args.device}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  LR:         {args.lr}")
    print(f"  Beta:       {args.beta}")
    print("=" * 70)
    print()
    
    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load dataset
    print(f"📊 Loading {args.dataset}...")
    if not HAS_DATASETS:
        raise ImportError("datasets library required: pip install datasets")
    
    # Load with config if specified
    if args.dataset_config:
        dataset = load_dataset(args.dataset, args.dataset_config, split="train", streaming=False)
    else:
        dataset = load_dataset(args.dataset, split="train", streaming=False)
    
    # Convert to tensors
    images_list = []
    labels_list = []
    
    for i, sample in enumerate(dataset):
        if i >= args.num_samples:
            break
        
        img = sample[args.image_key]
        if hasattr(img, 'convert'):
            img = img.convert('RGB')
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = img.permute(2, 0, 1)  # HWC -> CHW
        
        # Normalize (generic mean/std)
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        img = (img - mean) / std
        
        images_list.append(img)
        labels_list.append(sample[args.label_key])
    
    images = torch.stack(images_list)
    labels = torch.tensor(labels_list)
    print(f"  Loaded {len(images)} training samples")
    
    # Load test set for evaluation
    if args.dataset_config:
        test_dataset = load_dataset(args.dataset, args.dataset_config, split="test", streaming=False)
    else:
        test_dataset = load_dataset(args.dataset, split="test", streaming=False)
    test_images_list = []
    test_labels_list = []
    
    for sample in test_dataset:
        img = sample[args.image_key]
        if hasattr(img, 'convert'):
            img = img.convert('RGB')
        img = torch.from_numpy(np.array(img)).float() / 255.0
        img = img.permute(2, 0, 1)
        
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        img = (img - mean) / std
        
        test_images_list.append(img)
        test_labels_list.append(sample[args.label_key])
    
    test_images = torch.stack(test_images_list)
    test_labels = torch.tensor(test_labels_list)
    print(f"  Loaded {len(test_images)} test samples")
    print()
    
    # Load model
    print("📦 Loading HarmonicVisionTransformer...")
    model = HarmonicVisionTransformer(
        num_freq_bands=4,
        num_evolution_layers=3,
        hidden_dim=64,
        num_classes=10,
        patch_size=4,  # From training config
    ).to(args.device)
    
    load_checkpoint_safe(model, model_path, strict=False)
    print()
    
    # Baseline evaluation
    print("🎯 Baseline Evaluation (pre-DPO)...")
    test_loader = DataLoader(
        list(zip(test_images, test_labels)),
        batch_size=64,
        shuffle=False,
        collate_fn=lambda batch: {'image': torch.stack([x[0] for x in batch]), 'label': torch.tensor([x[1] for x in batch])}
    )
    
    baseline_metrics = evaluate_model(model, test_loader, args.device)
    
    print()
    print("=" * 60)
    print("  BASELINE (Pre-DPO)")
    print("=" * 60)
    print(f"  Accuracy:  {baseline_metrics['accuracy']*100:.2f}%")
    print(f"  Loss:      {baseline_metrics['loss']:.4f}")
    print("  " + "─" * 56)
    print("  Per-class accuracy:")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    for i, (name, acc) in enumerate(zip(class_names, baseline_metrics['per_class_accuracy'])):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"    {name:12} │ {bar} │ {acc*100:>5.1f}%")
    print("=" * 60)
    print()
    
    # Get initial predictions for preference pair creation
    print("🔍 Generating initial predictions for preference pairs...")
    model.eval()
    with torch.no_grad():
        all_preds = []
        for i in range(0, len(images), 64):
            batch = images[i:i+64].to(args.device)
            outputs = model(batch)
            preds = outputs['logits'].argmax(dim=-1)
            all_preds.append(preds.cpu())
        initial_preds = torch.cat(all_preds)
    print()
    
    # Create reference model (frozen)
    print("🔧 Creating reference model (frozen copy)...")
    reference_model = HarmonicVisionTransformer(
        num_freq_bands=4,
        num_evolution_layers=3,
        hidden_dim=64,
        num_classes=10,
        patch_size=4,  # From training config
    ).to(args.device)
    reference_model.load_state_dict(model.state_dict())
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    print("  ✅ Reference model ready")
    print()
    
    # Create DPO dataset
    dpo_dataset = VisionDPODataset(images, labels, initial_preds)
    
    if len(dpo_dataset) == 0:
        print("❌ No preference pairs created - model is already perfect!")
        return
    
    dpo_loader = DataLoader(
        dpo_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Training loop
    print("🚀 Starting DPO Training...")
    print()
    print(f"  Training on {len(dpo_dataset)} preference pairs...")
    print(f"  Batches per epoch: {len(dpo_loader)}")
    print()
    
    for epoch in range(args.epochs):
        model.train()
        
        epoch_metrics = {
            'loss': [],
            'accuracy': [],
            'reward_margin': [],
            'chosen_reward': [],
            'rejected_reward': [],
        }
        
        pbar = tqdm(dpo_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            images_batch = batch['image'].to(args.device)
            chosen_labels = torch.tensor(batch['chosen_label']).to(args.device)
            rejected_labels = torch.tensor(batch['rejected_label']).to(args.device)
            
            # Forward pass - policy model
            policy_outputs = model(images_batch)
            policy_logits = policy_outputs['logits']
            
            # Forward pass - reference model
            with torch.no_grad():
                ref_outputs = reference_model(images_batch)
                ref_logits = ref_outputs['logits']
            
            # Compute DPO loss
            loss, metrics = compute_dpo_loss(
                policy_logits, policy_logits,  # Same logits for both
                ref_logits, ref_logits,        # Same logits for both
                chosen_labels, rejected_labels,
                beta=args.beta,
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track metrics
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'acc': f"{metrics['accuracy']*100:.1f}%",
                'margin': f"{metrics['reward_margin']:.3f}",
            })
        
        # Epoch summary
        print()
        print(f"  Epoch {epoch+1} Complete:")
        print(f"    Loss:        {np.mean(epoch_metrics['loss']):.4f}")
        print(f"    DPO Acc:     {np.mean(epoch_metrics['accuracy'])*100:.1f}%")
        print(f"    Reward Margin: {np.mean(epoch_metrics['reward_margin']):.3f}")
        
        # Quick evaluation
        print("  Quick evaluation...")
        eval_metrics = evaluate_model(model, test_loader, args.device)
        print(f"    Test Accuracy: {eval_metrics['accuracy']*100:.2f}%")
        print()
        
        # Save directly back to the model file
        torch.save(model.state_dict(), model_path)
        print(f"  💾 Saved to {model_path}")
    
    # Final evaluation
    print("🎯 Final Evaluation (post-DPO)...")
    final_metrics = evaluate_model(model, test_loader, args.device)
    
    print()
    print("=" * 60)
    print("  FINAL RESULTS (Post-DPO)")
    print("=" * 60)
    print(f"  Accuracy:  {final_metrics['accuracy']*100:.2f}%")
    print(f"  Improvement: {(final_metrics['accuracy'] - baseline_metrics['accuracy'])*100:+.2f}%")
    print(f"  Loss:      {final_metrics['loss']:.4f}")
    print("  " + "─" * 56)
    print("  Per-class accuracy:")
    for i, (name, acc) in enumerate(zip(class_names, final_metrics['per_class_accuracy'])):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        baseline_acc = baseline_metrics['per_class_accuracy'][i]
        delta = (acc - baseline_acc) * 100
        delta_str = f"({delta:+.1f}%)" if abs(delta) > 0.1 else ""
        print(f"    {name:12} │ {bar} │ {acc*100:>5.1f}% {delta_str}")
    print("=" * 60)
    print()
    print("✅ DPO Training Complete!")
    print(f"💾 Model saved to: {model_path}")


if __name__ == "__main__":
    main()
