"""
Evaluate HVT models on real CIFAR-10 test set.

Compares: Original checkpoint vs RLHF-finetuned model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

from hvt_v2 import HarmonicVisionTransformer, SACRED_RATIO

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except:
    HAS_SAFETENSORS = False


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def load_hvt(checkpoint_path: str) -> HarmonicVisionTransformer:
    """Load HVT from checkpoint."""
    model = HarmonicVisionTransformer(
        num_freq_bands=4,
        base_omega=SACRED_RATIO,
        num_evolution_layers=3,
        hidden_dim=64,
        num_classes=10,
        use_phase_routing=True,
    )
    
    if checkpoint_path.endswith('.safetensors') and HAS_SAFETENSORS:
        state_dict = load_file(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_batch(batch, mean, std):
    """Convert PIL images to normalized tensors."""
    images = []
    for img in batch['img']:
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensor = (tensor - mean.view(3, 1, 1)) / std.view(3, 1, 1)
        images.append(tensor)
    
    return torch.stack(images), torch.tensor(batch['label'])


def evaluate_model(model: torch.nn.Module, test_dataset, num_samples: int = 2000):
    """Evaluate on CIFAR-10 test set."""
    model.eval()
    
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2470, 0.2435, 0.2616])
    
    correct = 0
    total = 0
    sync_orders = []
    per_class_correct = {i: 0 for i in range(10)}
    per_class_total = {i: 0 for i in range(10)}
    
    batch_size = 32
    batch_imgs = []
    batch_labels = []
    
    pbar = tqdm(total=num_samples, desc="Evaluating")
    
    for i, sample in enumerate(test_dataset):
        if i >= num_samples:
            break
        
        batch_imgs.append(sample['img'])
        batch_labels.append(sample['label'])
        
        if len(batch_imgs) == batch_size or i == num_samples - 1:
            images, labels = preprocess_batch(
                {'img': batch_imgs, 'label': batch_labels},
                mean, std
            )
            
            with torch.no_grad():
                outputs = model(images)
                logits = outputs['logits']
                preds = logits.argmax(dim=-1)
                sync_orders.append(outputs['sync_order'].mean().item())
                
                for pred, label in zip(preds, labels):
                    total += 1
                    per_class_total[label.item()] += 1
                    if pred.item() == label.item():
                        correct += 1
                        per_class_correct[label.item()] += 1
            
            pbar.update(len(batch_imgs))
            batch_imgs = []
            batch_labels = []
    
    pbar.close()
    
    accuracy = correct / total if total > 0 else 0
    avg_sync = np.mean(sync_orders)
    
    per_class_acc = {}
    for i in range(10):
        if per_class_total[i] > 0:
            per_class_acc[CIFAR10_CLASSES[i]] = per_class_correct[i] / per_class_total[i]
        else:
            per_class_acc[CIFAR10_CLASSES[i]] = 0.0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'sync_order': avg_sync,
        'per_class': per_class_acc,
    }


def main():
    print("=" * 70)
    print("CIFAR-10 EVALUATION: Original vs RLHF Model")
    print("=" * 70)
    
    print("\nLoading CIFAR-10 test set...")
    test_dataset = load_dataset("uoft-cs/cifar10", split="test", streaming=True)
    
    num_samples = 2000
    
    print("\n" + "─" * 70)
    print("ORIGINAL CHECKPOINT (best.safetensors)")
    print("─" * 70)
    
    original_path = "checkpoints_svhn/best.safetensors"
    if not Path(original_path).exists():
        original_path = "checkpoints_svhn/model_step_7810.safetensors"
    
    if Path(original_path).exists():
        original_model = load_hvt(original_path)
        print(f"Loaded: {original_path}")
        
        test_dataset = load_dataset("uoft-cs/cifar10", split="test", streaming=True)
        original_results = evaluate_model(original_model, test_dataset, num_samples)
        
        print(f"\n  Overall Accuracy: {original_results['accuracy']*100:.2f}%")
        print(f"  Correct: {original_results['correct']}/{original_results['total']}")
        print(f"  Sync Order: {original_results['sync_order']:.4f}")
        print("\n  Per-Class Accuracy:")
        for cls, acc in original_results['per_class'].items():
            bar = "█" * int(acc * 20)
            print(f"    {cls:12s}: {acc*100:5.1f}% {bar}")
    else:
        print("  ⚠ No original checkpoint found")
        original_results = None
    
    print("\n" + "─" * 70)
    print("RLHF MODEL (hvt_rlhf.pt)")
    print("─" * 70)
    
    rlhf_path = "checkpoints/hvt_rlhf.pt"
    
    if Path(rlhf_path).exists():
        rlhf_model = load_hvt(rlhf_path)
        print(f"Loaded: {rlhf_path}")
        
        test_dataset = load_dataset("uoft-cs/cifar10", split="test", streaming=True)
        rlhf_results = evaluate_model(rlhf_model, test_dataset, num_samples)
        
        print(f"\n  Overall Accuracy: {rlhf_results['accuracy']*100:.2f}%")
        print(f"  Correct: {rlhf_results['correct']}/{rlhf_results['total']}")
        print(f"  Sync Order: {rlhf_results['sync_order']:.4f}")
        print("\n  Per-Class Accuracy:")
        for cls, acc in rlhf_results['per_class'].items():
            bar = "█" * int(acc * 20)
            print(f"    {cls:12s}: {acc*100:5.1f}% {bar}")
    else:
        print("  ⚠ No RLHF checkpoint found")
        rlhf_results = None
    
    if original_results and rlhf_results:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        
        acc_delta = (rlhf_results['accuracy'] - original_results['accuracy']) * 100
        sync_delta = rlhf_results['sync_order'] - original_results['sync_order']
        
        print(f"\n  Accuracy Change: {acc_delta:+.2f}%")
        print(f"  Sync Order Change: {sync_delta:+.4f}")
        
        print("\n  Per-Class Changes:")
        for cls in CIFAR10_CLASSES:
            orig = original_results['per_class'][cls] * 100
            rlhf = rlhf_results['per_class'][cls] * 100
            delta = rlhf - orig
            symbol = "↑" if delta > 0 else "↓" if delta < 0 else "="
            print(f"    {cls:12s}: {orig:5.1f}% → {rlhf:5.1f}% ({delta:+5.1f}% {symbol})")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
