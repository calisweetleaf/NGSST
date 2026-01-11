#!/usr/bin/env python
"""
HVT Training - Single Entry Point

Usage:
    python run.py                          # Use default config
    python run.py --config baseline.yaml   # Use specific config
    python run.py --backup-epochs 5        # Save backup every 5 epochs
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for hvt_v2 import
sys.path.insert(0, str(Path(__file__).parent.parent))
# Ensure New-Training code package is discoverable when running from repo root
sys.path.insert(0, str(Path(__file__).parent / 'New-Training'))

# Robust import helper for local 'code' package (prefer local files over stdlib 'code')
def _import_local_code_member(member_path: str, attr: str):
    import importlib.util
    import types
    import sys as _sys

    # Prefer loading the local file from New-Training/code or ./code
    candidate_paths = [
        Path(__file__).parent / "New-Training" / "code" / (member_path.split('.', 1)[1] + ".py"),
        Path(__file__).parent / "code" / (member_path.split('.', 1)[1] + ".py"),
    ]

    for p in candidate_paths:
        if p.exists():
            # Ensure a package module 'code' exists so relative imports inside the file work
            code_pkg_name = 'code'
            code_pkg = types.ModuleType(code_pkg_name)
            code_pkg.__path__ = [str(p.parent)]
            _sys.modules[code_pkg_name] = code_pkg

            submod_name = member_path
            spec = importlib.util.spec_from_file_location(submod_name, str(p))
            mod = importlib.util.module_from_spec(spec)
            _sys.modules[submod_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception as e:
                raise ImportError(f"Failed to load local module {submod_name} from {p}: {e}")
            return getattr(mod, attr)

    # Fall back to normal import (may pick up installed package)
    try:
        module = __import__(member_path, fromlist=[attr])
        return getattr(module, attr)
    except Exception as e:
        raise ImportError(f"Could not import {member_path}.{attr}: {e}")

HVTProductionTrainer = _import_local_code_member("code.train", "HVTProductionTrainer")
train_hvt = _import_local_code_member("code.train", "train_hvt")
TrainingConfig = _import_local_code_member("code.config", "TrainingConfig")


def main():
    parser = argparse.ArgumentParser(
        description="HVT Production Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                           # Train with defaults
  python run.py --config baseline.yaml    # Use baseline config
  python run.py --config experimental.yaml --backup-epochs 5
  python run.py --steps 1000 --lr 1e-4    # Override params
        """
    )
    
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--backup-epochs', type=int, default=0,
        help='Save backup every N epochs (0 = disabled)'
    )
    parser.add_argument(
        '--steps', type=int, default=None,
        help='Override total training steps'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        help='Override learning rate'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Override batch size'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cpu/cuda)'
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config_path = Path(__file__).parent / 'configs' / args.config
        if not config_path.exists():
            config_path = Path(args.config)
        config = TrainingConfig.from_yaml(str(config_path))
    else:
        config = TrainingConfig()
    
    # Apply overrides
    if args.steps:
        config.total_steps = args.steps
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    
    print("=" * 60)
    print("HVT Production Training")
    print("=" * 60)
    print(f"Config: {args.config or 'default'}")
    print(f"Steps: {config.total_steps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print(f"Backup epochs: {args.backup_epochs or 'disabled'}")
    print("=" * 60)
    
    # Create trainer and run
    trainer = HVTProductionTrainer(config)
    history = trainer.train(backup_every_n_epochs=args.backup_epochs)
    
    print("\n✅ Training complete!")
    print(f"Best accuracy: {trainer.best_accuracy*100:.1f}%")
    print(f"Model saved to: {Path(config.checkpoint_dir) / 'hvt_model.pt'}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
