"""
Multi-Dataset Training Script
Trains on CIFAR-10 and SVHN alternating to prevent catastrophic forgetting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
from pathlib import Path
import sys
import time

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from hvt_v2 import HarmonicVisionTransformer

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


class MultiDatasetLoader:
    """Alternates between CIFAR-10 and SVHN datasets."""
    
    def __init__(self, batch_size=32, samples_per_dataset=5000):
        if not HAS_DATASETS:
            raise ImportError("datasets library required")
        
        self.batch_size = batch_size
        self.samples_per_dataset = samples_per_dataset
        
        # Normalization
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        
        print("📊 Loading datasets...")
        
        # Load CIFAR-10
        print("  Loading CIFAR-10...")
        cifar = load_dataset("uoft-cs/cifar10", split="train", streaming=False)
        self.cifar_data = self._process_dataset(cifar, 'img', samples_per_dataset)
        print(f"    {len(self.cifar_data[0])} samples")
        
        # Load SVHN
        print("  Loading SVHN...")
        svhn = load_dataset("ufldl-stanford/svhn", "cropped_digits", split="train", streaming=False)
        self.svhn_data = self._process_dataset(svhn, 'image', samples_per_dataset)
        print(f"    {len(self.svhn_data[0])} samples")
        
        # Load test sets
        print("  Loading test sets...")
        cifar_test = load_dataset("uoft-cs/cifar10", split="test", streaming=False)
        self.cifar_test = self._process_dataset(cifar_test, 'img', 10000)
        
        svhn_test = load_dataset("ufldl-stanford/svhn", "cropped_digits", split="test", streaming=False)
        self.svhn_test = self._process_dataset(svhn_test, 'image', 10000)
        print()
    
    def _process_dataset(self, dataset, img_key, max_samples):
        images = []
        labels = []
        
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            
            img = sample[img_key]
            if hasattr(img, 'convert'):
                img = img.convert('RGB')
            img = torch.from_numpy(np.array(img)).float() / 255.0
            img = img.permute(2, 0, 1)
            img = (img - self.mean) / self.std
            
            images.append(img)
            labels.append(sample['label'])
        
        return torch.stack(images), torch.tensor(labels)
    
    def get_mixed_batches(self, steps_per_dataset=100):
        """Yield batches alternating between datasets."""
        cifar_imgs, cifar_labels = self.cifar_data
        svhn_imgs, svhn_labels = self.svhn_data
        
        batch_size = self.batch_size
        
        while True:
            # CIFAR-10 batches
            for _ in range(steps_per_dataset):
                idx = torch.randperm(len(cifar_imgs))[:batch_size]
                yield cifar_imgs[idx], cifar_labels[idx], 'cifar'
            
            # SVHN batches
            for _ in range(steps_per_dataset):
                idx = torch.randperm(len(svhn_imgs))[:batch_size]
                yield svhn_imgs[idx], svhn_labels[idx], 'svhn'


def evaluate(model, images, labels, device, batch_size=64):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i+batch_size].to(device)
            batch_labels = labels[i:i+batch_size].to(device)
            
            outputs = model(batch_imgs)
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output'))
            else:
                logits = outputs
            
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)
    
    model.train()
    return correct / total if total > 0 else 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='hvt_model.pt')
    parser.add_argument('--steps', type=int, default=2000, help='Total training steps')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alternate-every', type=int, default=50, help='Switch dataset every N steps')
    parser.add_argument('--samples', type=int, default=10000, help='Samples per dataset')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    print("=" * 70)
    print("  🔀 MULTI-DATASET TRAINING")
    print("  CIFAR-10 + SVHN Alternating")
    print("=" * 70)
    print(f"  Model: {args.model}")
    print(f"  Steps: {args.steps}")
    print(f"  Alternate every: {args.alternate_every} steps")
    print(f"  LR: {args.lr}")
    print("=" * 70)
    print()
    
    # Load data
    loader = MultiDatasetLoader(
        batch_size=args.batch_size,
        samples_per_dataset=args.samples
    )
    
    # Create model
    model = HarmonicVisionTransformer(
        num_freq_bands=4,
        num_evolution_layers=3,
        hidden_dim=64,
        num_classes=10,
        patch_size=4,
    ).to(args.device)
    
    # Load weights
    model_path = Path(args.model)
    if model_path.exists():
        sd = torch.load(model_path, map_location='cpu')
        model.load_state_dict(sd, strict=False)
        print(f"📦 Loaded {args.model}")
    else:
        print(f"🆕 Starting fresh")
    
    # Initial evaluation
    print("\n📊 Initial Evaluation:")
    cifar_acc = evaluate(model, loader.cifar_test[0], loader.cifar_test[1], args.device)
    svhn_acc = evaluate(model, loader.svhn_test[0], loader.svhn_test[1], args.device)
    print(f"  CIFAR-10: {cifar_acc*100:.2f}%")
    print(f"  SVHN:     {svhn_acc*100:.2f}%")
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    print("🚀 Training...")
    batch_gen = loader.get_mixed_batches(steps_per_dataset=args.alternate_every)
    
    model.train()
    start_time = time.time()
    running_loss = 0
    running_correct = 0
    running_total = 0
    current_dataset = None
    
    for step in range(1, args.steps + 1):
        imgs, labels, dataset = next(batch_gen)
        imgs = imgs.to(args.device)
        labels = labels.to(args.device)
        
        if dataset != current_dataset:
            current_dataset = dataset
        
        optimizer.zero_grad()
        outputs = model(imgs)
        
        if isinstance(outputs, dict):
            logits = outputs.get('logits', outputs.get('output'))
        else:
            logits = outputs
        
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        preds = logits.argmax(dim=-1)
        running_correct += (preds == labels).sum().item()
        running_total += len(labels)
        
        # Log
        if step % 50 == 0:
            avg_loss = running_loss / 50
            avg_acc = running_correct / running_total * 100
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            
            ds_emoji = "🖼️" if current_dataset == 'cifar' else "🔢"
            print(f"  Step {step:4d} │ {ds_emoji} {current_dataset:5s} │ Loss: {avg_loss:.4f} │ Acc: {avg_acc:5.1f}% │ {steps_per_sec:.1f} steps/s")
            
            running_loss = 0
            running_correct = 0
            running_total = 0
        
        # Evaluate periodically
        if step % 500 == 0:
            print("  ─" * 30)
            cifar_acc = evaluate(model, loader.cifar_test[0], loader.cifar_test[1], args.device)
            svhn_acc = evaluate(model, loader.svhn_test[0], loader.svhn_test[1], args.device)
            print(f"  📊 Eval │ CIFAR-10: {cifar_acc*100:.1f}% │ SVHN: {svhn_acc*100:.1f}%")
            print("  ─" * 30)
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    cifar_acc = evaluate(model, loader.cifar_test[0], loader.cifar_test[1], args.device)
    svhn_acc = evaluate(model, loader.svhn_test[0], loader.svhn_test[1], args.device)
    print(f"  CIFAR-10: {cifar_acc*100:.2f}%")
    print(f"  SVHN:     {svhn_acc*100:.2f}%")
    print(f"  Combined: {(cifar_acc + svhn_acc) / 2 * 100:.2f}%")
    print("=" * 60)
    
    # Save
    torch.save(model.state_dict(), args.model)
    print(f"\n💾 Saved to {args.model}")


if __name__ == "__main__":
    main()
