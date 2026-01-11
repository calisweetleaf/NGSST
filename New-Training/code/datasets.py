"""
Dataset handling for HVT v3 training pipeline.

Includes streaming datasets, multi-dataset support, and
specialized data loading for oscillator-based training.
"""

import torch
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Dict, Any, Optional, Tuple, Iterator, List, Callable
import numpy as np
from pathlib import Path
import json
from PIL import Image
import warnings

# Optional imports
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import torchvision.transforms as transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class StreamingImageDataset(IterableDataset):
    """
    Memory-efficient streaming dataset wrapper.
    
    Streams data from HuggingFace datasets without loading to RAM.
    Applies minimal preprocessing on-the-fly.
    """
    
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_key: str = "img",
        label_key: str = "label",
        image_size: int = 32,
        streaming: bool = True,
        dataset_config: Optional[str] = None,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.image_key = image_key
        self.label_key = label_key
        self.image_size = image_size
        self.streaming = streaming
        self.dataset_config = dataset_config
        self.transform = transform
        self.max_samples = max_samples
        
        # Normalization parameters (CIFAR-10 defaults)
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        self.std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install with: pip install datasets")
    
    def _load_dataset(self):
        """Lazy load the dataset."""
        if self.dataset_config:
            return load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.split,
                streaming=self.streaming,
            )
        return load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=self.streaming,
        )
    
    def _process_sample(self, sample: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
        """Process a single sample."""
        try:
            # Get image
            img = sample[self.image_key]
            
            # Convert PIL to tensor if needed
            if hasattr(img, 'convert'):
                img = img.convert('RGB')
                img = torch.from_numpy(np.array(img)).float() / 255.0
                img = img.permute(2, 0, 1)  # HWC -> CHW
            elif isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
                if img.max() > 1.0:
                    img = img / 255.0
                if img.dim() == 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1)
            
            # Resize if needed
            if img.shape[-1] != self.image_size or img.shape[-2] != self.image_size:
                img = F.interpolate(
                    img.unsqueeze(0), 
                    size=(self.image_size, self.image_size),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            
            # Normalize
            img = (img - self.mean) / self.std
            
            # Apply custom transform
            if self.transform:
                img = self.transform(img)
            
            # Get label
            label = sample[self.label_key]
            if isinstance(label, (list, np.ndarray)):
                label = label[0]
            
            return img, int(label)
            
        except Exception as e:
            # Return dummy sample on error
            dummy_img = torch.randn(3, self.image_size, self.image_size)
            dummy_label = 0
            return dummy_img, dummy_label
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        dataset = self._load_dataset()
        sample_count = 0
        
        for sample in dataset:
            if self.max_samples and sample_count >= self.max_samples:
                break
            
            yield self._process_sample(sample)
            sample_count += 1
    
    def __len__(self) -> int:
        """Return length if max_samples is specified, otherwise unknown."""
        if self.max_samples:
            return self.max_samples
        else:
            return float('inf')


class MultiDataset(IterableDataset):
    """
    Multi-dataset wrapper that alternates between datasets.
    
    Useful for training on multiple domains simultaneously
    to test robustness and prevent catastrophic forgetting.
    """
    
    def __init__(
        self,
        datasets: List[StreamingImageDataset],
        sampling_strategy: str = "alternating",  # "alternating", "proportional", "round_robin"
        weights: Optional[List[float]] = None,
    ):
        self.datasets = datasets
        self.sampling_strategy = sampling_strategy
        self.weights = weights or [1.0] * len(datasets)
        
        if len(self.weights) != len(self.datasets):
            raise ValueError("Number of weights must match number of datasets")
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int, int]]:
        """Iterate over datasets according to sampling strategy."""
        if self.sampling_strategy == "alternating":
            yield from self._alternating_iter()
        elif self.sampling_strategy == "proportional":
            yield from self._proportional_iter()
        elif self.sampling_strategy == "round_robin":
            yield from self._round_robin_iter()
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")
    
    def _alternating_iter(self) -> Iterator[Tuple[torch.Tensor, int, int]]:
        """Alternate between datasets."""
        iterators = [iter(dataset) for dataset in self.datasets]
        
        while True:
            for i, iterator in enumerate(iterators):
                try:
                    img, label = next(iterator)
                    yield img, label, i  # Include dataset index
                except StopIteration:
                    # Restart exhausted iterator
                    iterators[i] = iter(self.datasets[i])
                    img, label = next(iterators[i])
                    yield img, label, i
    
    def _proportional_iter(self) -> Iterator[Tuple[torch.Tensor, int, int]]:
        """Sample proportionally to weights."""
        import random
        
        iterators = [iter(dataset) for dataset in self.datasets]
        
        while True:
            # Choose dataset based on weights
            dataset_idx = random.choices(range(len(self.datasets)), weights=self.weights)[0]
            
            try:
                img, label = next(iterators[dataset_idx])
                yield img, label, dataset_idx
            except StopIteration:
                # Restart exhausted iterator
                iterators[dataset_idx] = iter(self.datasets[dataset_idx])
                img, label = next(iterators[dataset_idx])
                yield img, label, dataset_idx
    
    def _round_robin_iter(self) -> Iterator[Tuple[torch.Tensor, int, int]]:
        """Round-robin through datasets."""
        iterators = [iter(dataset) for dataset in self.datasets]
        current_idx = 0
        
        while True:
            try:
                img, label = next(iterators[current_idx])
                yield img, label, current_idx
                current_idx = (current_idx + 1) % len(self.datasets)
            except StopIteration:
                # Restart exhausted iterator
                iterators[current_idx] = iter(self.datasets[current_idx])
                img, label = next(iterators[current_idx])
                yield img, label, current_idx


class FrequencyCurriculumDataset(Dataset):
    """
    Dataset wrapper for multi-scale curriculum learning.
    
    Progressively adds frequency bands during training.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        curriculum_config: Dict[str, Any],
        transform: Optional[Callable] = None,
    ):
        self.base_dataset = base_dataset
        self.config = curriculum_config
        self.transform = transform
        
        self.start_bands = curriculum_config.get('start_bands', 2)
        self.max_bands = curriculum_config.get('max_bands', 6)
        self.expansion_schedule = curriculum_config.get('expansion_schedule', 'exponential')
        self.freeze_previous = curriculum_config.get('freeze_previous_bands', True)
        
        # Current number of active bands
        self.current_bands = self.start_bands
        
        # Track which bands are frozen
        self.frozen_bands = set()
    
    def set_epoch(self, epoch: int, total_epochs: int):
        """Update curriculum based on current epoch."""
        progress = epoch / total_epochs
        
        if self.expansion_schedule == "linear":
            num_bands = self.start_bands + int(progress * (self.max_bands - self.start_bands))
        elif self.expansion_schedule == "exponential":
            alpha = math.log(self.max_bands / self.start_bands)
            num_bands = int(self.start_bands * math.exp(alpha * progress))
        else:  # golden_ratio
            phi = 1.618034
            num_bands = min(self.max_bands, self.start_bands + int(progress * phi * (self.max_bands - self.start_bands)))
        
        # Update current bands
        old_bands = self.current_bands
        self.current_bands = max(self.start_bands, min(self.max_bands, num_bands))
        
        # Freeze previously learned bands if requested
        if self.freeze_previous and self.current_bands > old_bands:
            for band in range(old_bands):
                self.frozen_bands.add(band)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """Get sample with curriculum information."""
        img, label = self.base_dataset[idx]
        
        # Apply frequency-specific transforms
        if hasattr(self.transform, 'set_active_bands'):
            self.transform.set_active_bands(self.current_bands)
        
        # Only apply transform if img is NOT already a tensor (avoid double ToTensor)
        if self.transform and not isinstance(img, torch.Tensor):
            img = self.transform(img)
        elif isinstance(img, torch.Tensor):
            # Already a tensor, just ensure float and normalized if needed
            if img.dtype != torch.float32:
                img = img.float()
        
        # Return curriculum info
        curriculum_info = {
            'current_bands': self.current_bands,
            'frozen_bands': list(self.frozen_bands),
            'curriculum_progress': (self.current_bands - self.start_bands) / (self.max_bands - self.start_bands)
        }
        
        return img, label, curriculum_info
    
    def __len__(self) -> int:
        return len(self.base_dataset)


class OscillatorValidationDataset(Dataset):
    """
    Validation dataset with oscillator-specific metrics.
    
    Computes and stores sync order, energy, and other oscillator
    metrics for validation purposes.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        model: Optional[torch.nn.Module] = None,
        compute_metrics: bool = True,
    ):
        self.base_dataset = base_dataset
        self.model = model
        self.compute_metrics = compute_metrics
        
        # Store pre-computed metrics if model is provided
        self.metrics_cache = {}
        
        if model is not None:
            self._precompute_metrics()
    
    def _precompute_metrics(self):
        """Pre-compute oscillator metrics for all samples."""
        self.model.eval()
        with torch.no_grad():
            for i in range(len(self.base_dataset)):
                img, label = self.base_dataset[i]
                img = img.unsqueeze(0)  # Add batch dimension
                
                if hasattr(self.model, 'compute_oscillator_metrics'):
                    metrics = self.model.compute_oscillator_metrics(img)
                    self.metrics_cache[i] = metrics
                else:
                    self.metrics_cache[i] = {}
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """Get sample with validation metrics."""
        img, label = self.base_dataset[idx]
        
        metrics = self.metrics_cache.get(idx, {})
        
        return img, label, metrics
    
    def __len__(self) -> int:
        return len(self.base_dataset)


def get_datasets(config: Any) -> Tuple[Dataset, Dataset]:
    """
    Factory function to create datasets based on configuration.
    
    Args:
        config: Training configuration
        
    Returns:
        Training and validation datasets
    """
    if config.dataset_name == "cifar10" or config.dataset_name == "cifar100":
        if HAS_TORCHVISION:
            import torchvision.datasets as datasets
            
            # Standard CIFAR (10 or 100)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            
            if config.dataset_name == "cifar10":
                train_cls = datasets.CIFAR10
                val_cls = datasets.CIFAR10
            else:
                train_cls = datasets.CIFAR100
                val_cls = datasets.CIFAR100
                # Ensure num_classes matches
                try:
                    config.num_classes = 100
                except Exception:
                    pass

            train_dataset = train_cls(
                root='./data', train=True, download=True, transform=transform
            )
            val_dataset = val_cls(
                root='./data', train=False, download=True, transform=transform
            )
            
            # Wrap with curriculum if enabled
            if config.novel_methods.use_multi_scale_curriculum:
                train_dataset = FrequencyCurriculumDataset(
                    train_dataset, 
                    config.novel_methods.curriculum.__dict__,
                    transform=transform
                )
            
            return train_dataset, val_dataset
        else:
            raise ImportError("torchvision required for CIFAR datasets")
    
    elif config.dataset_name == "multi":
        # Multi-dataset training
        datasets_list = []
        
        # CIFAR-10
        if HAS_TORCHVISION:
            import torchvision.datasets as tv_datasets
            
            cifar_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
            
            cifar_dataset = tv_datasets.CIFAR10(
                root='./data', train=True, download=True, transform=cifar_transform
            )
            datasets_list.append(cifar_dataset)
        
        # SVHN
        if HAS_TORCHVISION:
            svhn_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])
            
            svhn_dataset = tv_datasets.SVHN(
                root='./data', split='train', download=True, transform=svhn_transform
            )
            datasets_list.append(svhn_dataset)
        
        # Wrap with streaming datasets
        streaming_datasets = []
        for dataset in datasets_list:
            streaming_dataset = StreamingImageDataset(
                dataset_name="dummy",  # Will be overridden
                streaming=False,  # Already loaded
                transform=None  # Already applied
            )
            # Override the _load_dataset method
            streaming_dataset._load_dataset = lambda d=dataset: iter(d)
            streaming_datasets.append(streaming_dataset)
        
        # Create multi-dataset
        multi_dataset = MultiDataset(streaming_datasets, sampling_strategy="alternating")
        
        # Create validation datasets
        val_datasets = []
        if HAS_TORCHVISION:
            cifar_val = tv_datasets.CIFAR10(
                root='./data', train=False, download=True, transform=cifar_transform
            )
            svhn_val = tv_datasets.SVHN(
                root='./data', split='test', download=True, transform=svhn_transform
            )
            val_datasets = [cifar_val, svhn_val]
        
        return multi_dataset, val_datasets
    
    else:
        # Generic streaming dataset
        train_dataset = StreamingImageDataset(
            dataset_name=config.dataset_name,
            split="train",
            image_key=config.image_key,
            label_key=config.label_key,
            image_size=config.image_size,
            streaming=config.streaming,
            dataset_config=config.dataset_config,
        )
        
        val_dataset = StreamingImageDataset(
            dataset_name=config.dataset_name,
            split="test" if config.dataset_name != "svhn" else "test",
            image_key=config.image_key,
            label_key=config.label_key,
            image_size=config.image_size,
            streaming=config.streaming,
            dataset_config=config.dataset_config,
            max_samples=10000,  # Limit validation set size
        )
        
        return train_dataset, val_dataset


def get_dataloader(
    dataset: Dataset,
    config: Any,
    is_training: bool = True,
    collate_fn: Optional[Callable] = None,
) -> DataLoader:
    """
    Create DataLoader for given dataset.
    
    Args:
        dataset: The dataset to load
        config: Training configuration
        is_training: Whether this is for training
        collate_fn: Custom collate function
        
    Returns:
        Configured DataLoader
    """
    def default_collate(batch):
        """Default collate function that handles curriculum info."""
        if len(batch[0]) == 3:  # Has curriculum info
            images, labels, info = zip(*batch)
            images = torch.stack(images)
            labels = torch.tensor(labels)
            return images, labels, info
        else:  # Standard batch
            images, labels = zip(*batch)
            images = torch.stack(images)
            labels = torch.tensor(labels)
            return images, labels
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_training and not hasattr(dataset, '__iter__'),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn or default_collate,
        drop_last=is_training,
    )