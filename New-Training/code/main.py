"""
Main entry point for HVT v3 training pipeline.

Provides command-line interface for training with novel methods,
validation, and analysis.
"""

import argparse
import sys
import logging
from pathlib import Path
import json
import yaml
from typing import Dict, Any, Optional

from .config import TrainingConfig, get_preset
from .trainer import HVTTrainer
from .evaluator import run_comprehensive_evaluation
from .visualizer import TrainingDashboard, plot_sync_evolution, plot_training_curves
from .checkpointing import create_checkpoint_manager

# Import HVT model - handle both package and standalone usage
_parent = Path(__file__).resolve().parent.parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
from hvt_v2 import HarmonicVisionTransformer


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('hvt_training.log'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: Optional[str], preset: Optional[str], overrides: Dict[str, Any]) -> TrainingConfig:
    """Load configuration from file, preset, or create default."""
    if config_path:
        # Load from YAML file
        config = TrainingConfig.from_yaml(config_path)
    elif preset:
        # Use preset configuration
        preset_data = get_preset(preset)
        config = create_config(preset=preset, **overrides)
    else:
        # Default configuration
        config = TrainingConfig()
    
    # Apply overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # Handle nested attributes
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    break
            else:
                if hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], value)
    
    return config


def train_model(args):
    """Train HVT model with specified configuration."""
    print("🌊 Starting HVT v3 Training Pipeline")
    print("=" * 50)
    
    # Load configuration
    config = load_config(args.config, args.preset, args.override)
    
    print(f"Configuration:")
    print(f"  Dataset: {config.dataset_name}")
    print(f"  Total Steps: {config.total_steps:,}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Device: {config.device}")
    print(f"  Mixed Precision: {config.mixed_precision}")
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Create trainer
    trainer = HVTTrainer(config)
    
    # Create checkpoint manager
    checkpoint_manager = create_checkpoint_manager(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Create dashboard if requested
    dashboard = None
    if args.dashboard:
        print("\nStarting training dashboard...")
        dashboard = TrainingDashboard()
        dashboard.show()
    
    try:
        # Start training
        print("\n🚀 Starting training...")
        trainer.train(resume_from=args.resume)
        
        # Final evaluation
        print("\n📊 Running final evaluation...")
        model = trainer.get_model()
        
        # Save final model
        final_path = Path(config.checkpoint_dir) / "final_model.pt"
        checkpoint_manager.save_model(model, str(final_path))
        
        # Generate final plots
        history = trainer.get_training_history()
        if history:
            plot_training_curves(history, str(Path(config.checkpoint_dir) / "training_curves.png"))
            
            sync_history = [h['sync_order'] for h in history]
            plot_sync_evolution(sync_history, str(Path(config.checkpoint_dir) / "sync_evolution.png"))
        
        print(f"\n✅ Training completed!")
        print(f"   Final model saved to: {final_path}")
        print(f"   Training history: {len(history)} steps")
        
        # Print best metrics
        best_metrics = trainer.get_best_metrics()
        if best_metrics:
            print(f"\n🏆 Best validation metrics:")
            for dataset, metrics in best_metrics.items():
                print(f"   {dataset}: Acc={metrics.get('val_accuracy', 0)*100:.1f}%, "
                      f"Loss={metrics.get('val_loss', 0):.4f}, "
                      f"Sync={metrics.get('val_sync_order', 0):.3f}")
        
        return trainer
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        
        # Save checkpoint before exiting
        print("Saving checkpoint...")
        trainer.save_checkpoint(is_final=True)
        
        if dashboard:
            dashboard.stop_live_updates()
            dashboard.save_snapshot("interrupted_dashboard.png")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        logging.exception("Training failed")
        
        if dashboard:
            dashboard.stop_live_updates()
        
        raise


def evaluate_model(args):
    """Evaluate trained HVT model."""
    print("📊 HVT Model Evaluation")
    print("=" * 30)
    
    # Load configuration
    config = load_config(args.config, args.preset, {})
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Load model
    if not args.model_path:
        print("❌ Error: Please specify model path with --model-path")
        sys.exit(1)
    
    print(f"Loading model from: {args.model_path}")
    
    # Create model
    model = HarmonicVisionTransformer(
        image_size=config.image_size,
        num_classes=config.num_classes,
        num_freq_bands=config.num_freq_bands,
        base_omega=config.base_omega,
        num_evolution_layers=config.num_evolution_layers,
        hidden_dim=config.hidden_dim,
    )
    
    # Load checkpoint
    checkpoint_manager = create_checkpoint_manager(config)
    try:
        metadata = checkpoint_manager.load_model(model, args.model_path)
        print(f"Model loaded successfully")
        print(f"Metadata: {metadata}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Run evaluation
    print("\nRunning comprehensive evaluation...")
    
    # Load test dataset (simplified - in practice would use proper test set)
    from .datasets import get_datasets
    _, test_dataset = get_datasets(config)
    
    output_dir = args.output_dir or "results/evaluation"
    
    summary = run_comprehensive_evaluation(
        model_path=args.model_path,
        config=config,
        test_dataset=test_dataset,
        output_dir=output_dir
    )
    
    print(f"\n✅ Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"\nSummary:")
    print(f"  Final Accuracy: {summary['summary']['final_accuracy']*100:.1f}%")
    print(f"  Best Sync Order: {summary['summary']['best_sync_order']:.3f}")
    print(f"  Energy Stability: {summary['summary']['energy_stability']:.3f}")
    print(f"  Samples/sec: {summary['summary']['samples_per_second']:.1f}")
    
    return summary


def analyze_training(args):
    """Analyze training results and generate reports."""
    print("📈 Training Analysis")
    print("=" * 25)
    
    # Load training history
    if not args.history_path:
        print("❌ Error: Please specify training history path")
        sys.exit(1)
    
    try:
        with open(args.history_path, 'r') as f:
            history = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load history: {e}")
        sys.exit(1)
    
    print(f"Loaded training history: {len(history)} steps")
    
    # Generate analysis plots
    from .visualizer import MetricsPlotter
    
    plotter = MetricsPlotter(output_dir=args.output_dir)
    
    # Training curves
    plotter.plot_training_curves(history, save_path=f"{args.output_dir}/analysis_curves.png")
    
    # Sync evolution
    sync_history = [h['sync_order'] for h in history]
    plot_sync_evolution(sync_history, save_path=f"{args.output_dir}/analysis_sync.png")
    
    # Generate summary statistics
    stats = {
        'total_steps': len(history),
        'final_loss': history[-1]['loss'],
        'final_accuracy': history[-1]['accuracy'],
        'final_sync_order': history[-1]['sync_order'],
        'best_accuracy': max(h['accuracy'] for h in history),
        'best_sync_order': max(h['sync_order'] for h in history),
        'avg_sync_order': sum(h['sync_order'] for h in history) / len(history),
        'sync_stability': np.std([h['sync_order'] for h in history]),
    }
    
    print(f"\n📊 Analysis Results:")
    print(f"  Total Steps: {stats['total_steps']}")
    print(f"  Final Loss: {stats['final_loss']:.4f}")
    print(f"  Final Accuracy: {stats['final_accuracy']*100:.1f}%")
    print(f"  Best Accuracy: {stats['best_accuracy']*100:.1f}%")
    print(f"  Final Sync Order: {stats['final_sync_order']:.3f}")
    print(f"  Best Sync Order: {stats['best_sync_order']:.3f}")
    print(f"  Avg Sync Order: {stats['avg_sync_order']:.3f}")
    print(f"  Sync Stability: {stats['sync_stability']:.3f}")
    
    # Save stats
    with open(f"{args.output_dir}/analysis_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    
    return stats


def create_config_template(args):
    """Create configuration template file."""
    config = TrainingConfig()
    
    # Convert to YAML
    config_dict = config.to_dict()
    
    output_path = args.output or "config_template.yaml"
    
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"✅ Configuration template created: {output_path}")
    print(f"\nTo use this configuration:")
    print(f"  python main.py train --config {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HVT v3 Training Pipeline - Oscillator-based Vision Transformer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  python main.py train
  
  # Train with preset configuration
  python main.py train --preset full_novel
  
  # Train with custom config and overrides
  python main.py train --config my_config.yaml --override total_steps=5000
  
  # Resume training from checkpoint
  python main.py train --resume checkpoints/latest
  
  # Evaluate trained model
  python main.py evaluate --model-path checkpoints/best_model.pt
  
  # Analyze training results
  python main.py analyze --history-path training_history.json
  
  # Create config template
  python main.py create-config --output my_config.yaml
        """
    )
    
    # Common arguments
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--preset', type=str, choices=['baseline', 'full_novel', 'fast', 'max_quality'],
                       help='Use preset configuration')
    parser.add_argument('--override', nargs='*', help='Override configuration parameters (key=value)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train HVT model')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint directory')
    train_parser.add_argument('--dashboard', action='store_true', help='Show training dashboard')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze training results')
    analyze_parser.add_argument('--history-path', type=str, required=True, help='Path to training history')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create configuration template')
    config_parser.add_argument('--output', type=str, help='Output file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Parse override parameters
    if args.override:
        override_dict = {}
        for override in args.override:
            if '=' in override:
                key, value = override.split('=', 1)
                # Try to convert value to appropriate type
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                override_dict[key] = value
        args.override = override_dict
    else:
        args.override = {}
    
    # Execute command
    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'analyze':
        analyze_training(args)
    elif args.command == 'create-config':
        create_config_template(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()