#!/usr/bin/env python3
"""
Structured Alzheimer's Training Pipeline
=======================================

Command-line interface for training structured Alzheimer's disease prediction models
with multi-seed support, ensemble methods, and comprehensive evaluation metrics.

Usage:
    python scripts/train_alzheimers_structured.py --data-path data/alzheimer_dataset.csv
    python scripts/train_alzheimers_structured.py --data-path data/alzheimer_dataset.csv --config configs/structured_alz_ensemble.yaml --ensemble
    python scripts/train_alzheimers_structured.py --data-path data/alzheimer_dataset.csv --override-seeds 42 1337 2025 --epochs 100
"""

import argparse
import yaml
import json
import os
import time
import logging
import sys
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Any

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from aimedres.training.structured_alz_trainer import StructuredAlzTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('TrainAlzheimersStructured')


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Structured Alzheimer Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default config
  python scripts/train_alzheimers_structured.py --data-path data/alzheimer_dataset.csv
  
  # Training with ensemble models
  python scripts/train_alzheimers_structured.py --data-path data/alzheimer_dataset.csv --ensemble
  
  # Custom configuration and seeds
  python scripts/train_alzheimers_structured.py \\
    --data-path data/alzheimer_dataset.csv \\
    --config src/duetmind_adaptive/training/configs/structured_alz_ensemble.yaml \\
    --override-seeds 42 1337 2025 \\
    --epochs 100 \\
    --batch-size 32
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data-path', 
        required=True,
        help='Path to the Alzheimer dataset (CSV or parquet)'
    )
    
    # Configuration
    parser.add_argument(
        '--config', 
        default='src/duetmind_adaptive/training/configs/structured_alz_baseline.yaml',
        help='Path to configuration YAML file (default: baseline config)'
    )
    
    # Data configuration
    parser.add_argument(
        '--target-column',
        help='Name of target column (auto-detected if not specified)'
    )
    
    # Training overrides
    parser.add_argument(
        '--epochs', 
        type=int,
        help='Number of training epochs (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int,
        help='Batch size for training (overrides config)'
    )
    
    parser.add_argument(
        '--patience', 
        type=int,
        help='Early stopping patience (overrides config)'
    )
    
    parser.add_argument(
        '--override-seeds', 
        nargs='*', 
        type=int,
        help='Random seeds for multi-seed training (overrides config)'
    )
    
    # Model configuration  
    parser.add_argument(
        '--ensemble', 
        action='store_true',
        help='Enable ensemble training (overrides config)'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        help='Output directory for artifacts (overrides config)'
    )
    
    # Utility options
    parser.add_argument(
        '--save-test-split',
        action='store_true', 
        help='Save test split for later evaluation'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration and exit without training'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate configuration file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a dictionary")
    
    logger.info(f"Loaded configuration profile: {config.get('profile', 'unknown')}")
    return config


def apply_overrides(config: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Apply command-line overrides to configuration
    
    Args:
        config: Base configuration dictionary
        args: Parsed command-line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Create a copy to avoid modifying original
    config = config.copy()
    
    # Apply training parameter overrides
    for param in ['epochs', 'batch_size', 'patience']:
        value = getattr(args, param)
        if value is not None:
            config[param] = value
            logger.info(f"Override {param}: {value}")
    
    # Apply seed overrides
    if args.override_seeds:
        config['seeds'] = args.override_seeds
        logger.info(f"Override seeds: {args.override_seeds}")
    
    # Apply ensemble override
    if args.ensemble:
        config['ensemble'] = True
        logger.info("Override ensemble: enabled")
    
    # Apply output directory override
    if args.output_dir:
        config['output_dir'] = args.output_dir
        logger.info(f"Override output_dir: {args.output_dir}")
    
    # Add data path to config
    config['data_path'] = args.data_path
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['seeds', 'models', 'output_dir']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    if not config['seeds']:
        raise ValueError("At least one seed must be specified")
    
    if not config['models']:
        raise ValueError("At least one model must be specified")
    
    # Validate data path
    data_path = Path(config.get('data_path', ''))
    if not data_path.exists():
        raise ValueError(f"Data path does not exist: {data_path}")
    
    logger.info("Configuration validation passed")


def run_multi_seed_training(config: Dict[str, Any], target_column: str = None) -> List[Dict[str, Any]]:
    """
    Execute training across multiple seeds
    
    Args:
        config: Training configuration
        target_column: Optional target column name
        
    Returns:
        List of training results for each seed
    """
    data_path = Path(config['data_path'])
    output_root = Path(config.get('output_dir', 'metrics/structured_alz'))
    output_root.mkdir(parents=True, exist_ok=True)
    
    seeds = config['seeds']
    logger.info(f"Starting multi-seed training with seeds: {seeds}")
    
    aggregate_results = []
    
    for i, seed in enumerate(seeds):
        logger.info(f"Training with seed {seed} ({i+1}/{len(seeds)})")
        
        # Create seed-specific output directory
        run_dir = output_root / f"runs/seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Initialize trainer
            trainer = StructuredAlzTrainer(
                config=config,
                seed=seed,
                output_dir=run_dir,
                target_column=target_column
            )
            
            # Run training
            start_time = time.time()
            run_metrics = trainer.run_full_training(data_path)
            training_time = time.time() - start_time
            
            # Add timing information
            run_metrics['total_training_time'] = training_time
            run_metrics['seed'] = seed
            
            # Save run metrics
            with open(run_dir / 'run_metrics.json', 'w') as f:
                json.dump(run_metrics, f, indent=2, default=str)
            
            aggregate_results.append(run_metrics)
            
            primary_metric = config.get('metric_primary', 'macro_f1')
            best_score_key = f'best_val_{primary_metric}'
            logger.info(f"Completed seed {seed}: "
                       f"best_model={run_metrics.get('best_model', 'N/A')}, "
                       f"best_score={run_metrics.get(best_score_key, 0):.4f}, "
                       f"time={training_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Failed to train with seed {seed}: {e}")
            # Store error information
            error_result = {
                'seed': seed,
                'error': str(e),
                'total_training_time': 0
            }
            aggregate_results.append(error_result)
    
    return aggregate_results


def aggregate_results(results: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate results across multiple seeds
    
    Args:
        results: List of training results from each seed
        config: Training configuration
        
    Returns:
        Aggregated statistics dictionary
    """
    logger.info("Aggregating multi-seed results")
    
    primary_metric = config.get('metric_primary', 'macro_f1')
    
    def collect_metric(metric_name: str) -> List[float]:
        """Collect metric values from all successful runs"""
        values = []
        for result in results:
            if 'error' not in result and metric_name in result:
                values.append(result[metric_name])
        return values
    
    # Collect key metrics
    metrics_to_aggregate = [
        f'best_val_{primary_metric}',
        'best_val_accuracy',
        'total_training_time'
    ]
    
    summary = {
        'total_seeds': len(results),
        'successful_seeds': len([r for r in results if 'error' not in r]),
        'failed_seeds': len([r for r in results if 'error' in r]),
        'config_profile': config.get('profile', 'unknown'),
        'primary_metric': primary_metric
    }
    
    # Calculate statistics for each metric
    for metric in metrics_to_aggregate:
        values = collect_metric(metric)
        
        if values:
            summary[metric] = {
                'mean': float(mean(values)),
                'std': float(pstdev(values)) if len(values) > 1 else 0.0,
                'min': float(min(values)),
                'max': float(max(values)),
                'count': len(values),
                'values': values
            }
        else:
            summary[metric] = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0,
                'values': []
            }
    
    # Add model performance breakdown
    model_performance = {}
    for result in results:
        if 'error' not in result and 'best_model' in result:
            model_name = result['best_model']
            if model_name not in model_performance:
                model_performance[model_name] = 0
            model_performance[model_name] += 1
    
    summary['best_model_frequency'] = model_performance
    
    # Add timing information
    total_time = sum(collect_metric('total_training_time'))
    summary['total_pipeline_time'] = total_time
    
    return summary


def save_aggregate_results(summary: Dict[str, Any], results: List[Dict[str, Any]], 
                          output_dir: Path) -> None:
    """
    Save aggregated results to files
    
    Args:
        summary: Aggregated summary statistics  
        results: Individual seed results
        output_dir: Output directory
    """
    # Save JSON summary
    summary_path = output_dir / 'aggregate_metrics.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save detailed results
    detailed_path = output_dir / 'detailed_results.json'
    with open(detailed_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create markdown summary table
    primary_metric = summary['primary_metric']
    metric_key = f'best_val_{primary_metric}'
    
    markdown_content = f"""# Structured Alzheimer's Training Results

## Summary

- **Profile**: {summary['config_profile']}
- **Seeds**: {summary['total_seeds']} ({summary['successful_seeds']} successful, {summary['failed_seeds']} failed)
- **Primary Metric**: {primary_metric}
- **Total Training Time**: {summary['total_pipeline_time']:.1f} seconds

## Performance Metrics

| Metric | Mean | Std | Min | Max | Count |
|--------|------|-----|-----|-----|-------|
"""
    
    for metric_name, stats in summary.items():
        if isinstance(stats, dict) and 'mean' in stats:
            clean_name = metric_name.replace('_', ' ').title()
            markdown_content += f"| {clean_name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |\n"
    
    markdown_content += f"""
## Best Model Frequency

"""
    
    for model_name, count in summary.get('best_model_frequency', {}).items():
        markdown_content += f"- **{model_name}**: {count} times\n"
    
    # Save markdown
    markdown_path = output_dir / 'aggregate_summary.md'
    with open(markdown_path, 'w') as f:
        f.write(markdown_content)
    
    logger.info(f"Aggregate results saved to {output_dir}")


def main():
    """Main training pipeline execution"""
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('StructuredAlzTrainer').setLevel(logging.DEBUG)
    
    try:
        # Load and validate configuration
        config = load_config(args.config)
        config = apply_overrides(config, args)
        validate_config(config)
        
        # Show configuration and exit if dry run
        if args.dry_run:
            print("=== Configuration (Dry Run) ===")
            print(yaml.dump(config, default_flow_style=False))
            print("=== Training would use the above configuration ===")
            return True
        
        # Log configuration summary
        logger.info(f"Starting training with {len(config['seeds'])} seeds: {config['seeds']}")
        logger.info(f"Models: {config.get('models', [])}")
        logger.info(f"Ensemble: {config.get('ensemble', False)}")
        logger.info(f"Output: {config['output_dir']}")
        
        # Execute multi-seed training
        start_time = time.time()
        results = run_multi_seed_training(config, args.target_column)
        
        # Aggregate results
        summary = aggregate_results(results, config)
        
        # Save results
        output_dir = Path(config['output_dir'])
        save_aggregate_results(summary, results, output_dir)
        
        # Print final summary
        total_time = time.time() - start_time
        primary_metric = config.get('metric_primary', 'macro_f1')
        
        print("\n" + "="*60)
        print("🏁 TRAINING PIPELINE COMPLETED")
        print("="*60)
        print(f"📊 Seeds: {summary['successful_seeds']}/{summary['total_seeds']} successful")
        print(f"⏱️  Total Time: {total_time:.1f} seconds")
        
        if summary['successful_seeds'] > 0:
            metric_stats = summary.get(f'best_val_{primary_metric}', {})
            print(f"🎯 {primary_metric.replace('_', ' ').title()}: {metric_stats.get('mean', 0):.4f} ± {metric_stats.get('std', 0):.4f}")
            
            best_models = summary.get('best_model_frequency', {})
            if best_models:
                best_model = max(best_models.items(), key=lambda x: x[1])
                print(f"🏆 Best Model: {best_model[0]} ({best_model[1]}/{summary['successful_seeds']} times)")
        
        print(f"📁 Results saved to: {output_dir}")
        print("="*60)
        
        return summary['successful_seeds'] > 0
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)