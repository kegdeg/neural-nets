#!/usr/bin/env python3
"""
CLI tool for training and evaluating neural network models on Reuters dataset.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def _import_dependencies():
    """Lazy import of dependencies - only when actually needed."""
    import tensorflow as tf
    import numpy as np
    from neural_nets.data_processing import load_reuters_data
    from neural_nets.models import (
        build_baseline_model,
        build_small_model,
        build_bigger_model,
        build_bigger_model2,
        build_regularised_model,
        build_regularised_model2,
        build_custom_model
    )
    from neural_nets.training import train_model, create_early_stopping
    from neural_nets.evaluation import (
        plot_metrics,
        print_metrics,
        analyse_overfitting,
        print_model_comparison_table
    )
    from neural_nets.hyperparameter_tuning import create_tuner
    
    return {
        'tf': tf,
        'np': np,
        'load_reuters_data': load_reuters_data,
        'build_baseline_model': build_baseline_model,
        'build_small_model': build_small_model,
        'build_bigger_model': build_bigger_model,
        'build_bigger_model2': build_bigger_model2,
        'build_regularised_model': build_regularised_model,
        'build_regularised_model2': build_regularised_model2,
        'build_custom_model': build_custom_model,
        'train_model': train_model,
        'create_early_stopping': create_early_stopping,
        'plot_metrics': plot_metrics,
        'print_metrics': print_metrics,
        'analyse_overfitting': analyse_overfitting,
        'print_model_comparison_table': print_model_comparison_table,
        'create_tuner': create_tuner,
    }


def set_seed(deps, seed=42):
    """Set random seed for reproducibility."""
    deps['tf'].random.set_seed(seed)
    deps['np'].random.seed(seed)


def train_single_model(args):
    """Train a single model."""
    deps = _import_dependencies()
    
    print(f"\n{'='*60}")
    print(f"Training {args.model} model")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    x_train, _, x_test, _, one_hot_train_labels, one_hot_test_labels = deps['load_reuters_data']()
    print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")
    
    # Build model
    model_builders = {
        'baseline': deps['build_baseline_model'],
        'small': deps['build_small_model'],
        'bigger': deps['build_bigger_model'],
        'bigger2': deps['build_bigger_model2'],
        'regularised': deps['build_regularised_model'],
        'regularised2': deps['build_regularised_model2'],
    }
    
    if args.model not in model_builders:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {', '.join(model_builders.keys())}")
        return
    
    model = model_builders[args.model]()
    
    # Create callbacks
    callbacks = []
    if args.early_stopping:
        callbacks.append(deps['create_early_stopping'](
            monitor='val_f1_score',
            patience=args.patience,
            min_delta=args.min_delta
        ))
    
    # Train model
    history = deps['train_model'](
        model,
        x_train,
        one_hot_train_labels,
        validation_split=args.validation_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks if callbacks else None,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating on test set...")
    test_metrics = model.evaluate(x_test, one_hot_test_labels, verbose=0)
    deps['print_metrics'](args.model, test_metrics)
    
    # Analyze overfitting
    deps['analyse_overfitting'](history, args.model)
    
    # Plot metrics
    if args.plot:
        deps['plot_metrics'](history, args.model, save_path=args.save_plot)
    
    # Save model
    if args.save_model:
        model_path = args.save_model if args.save_model != 'auto' else f'models/{args.model}.keras'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"\nModel saved to {model_path}")


def train_all_models(args):
    """Train all available models and compare them."""
    deps = _import_dependencies()
    
    print(f"\n{'='*60}")
    print("Training all models for comparison")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    x_train, _, x_test, _, one_hot_train_labels, one_hot_test_labels = deps['load_reuters_data']()
    
    # Define models to train
    models_to_train = {
        'Baseline': deps['build_baseline_model'],
        'Small': deps['build_small_model'],
        'Bigger': deps['build_bigger_model'],
        'Bigger2': deps['build_bigger_model2'],
        'Regularised': deps['build_regularised_model'],
        'Regularised2': deps['build_regularised_model2'],
    }
    
    trained_models = {}
    results = {}
    
    # Train each model
    for name, builder in models_to_train.items():
        print(f"\n{'='*60}")
        print(f"Training {name} model")
        print(f"{'='*60}")
        
        model = builder()
        
        callbacks = []
        if args.early_stopping:
            callbacks.append(deps['create_early_stopping'](
                monitor='val_f1_score',
                patience=args.patience
            ))
        
        history = deps['train_model'](
            model,
            x_train,
            one_hot_train_labels,
            validation_split=args.validation_split,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks if callbacks else None,
            verbose=1
        )
        
        # Evaluate
        test_metrics = model.evaluate(x_test, one_hot_test_labels, verbose=0)
        results[name] = test_metrics
        trained_models[name] = model
        
        deps['print_metrics'](name, test_metrics)
        deps['analyse_overfitting'](history, name)
        
        # Save model if requested
        if args.save_models:
            model_path = f'models/{name.lower()}.keras'
            os.makedirs('models', exist_ok=True)
            model.save(model_path)
            print(f"Model saved to {model_path}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    deps['print_model_comparison_table'](results)


def hyperparameter_tuning(args):
    """Perform hyperparameter tuning."""
    deps = _import_dependencies()
    
    print(f"\n{'='*60}")
    print("Hyperparameter Tuning")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    x_train, _, x_test, _, one_hot_train_labels, one_hot_test_labels = deps['load_reuters_data']()
    
    # Create tuner
    tuner = deps['create_tuner'](
        directory=args.tuning_dir,
        project_name=args.project_name,
        max_epochs=args.max_epochs,
        factor=args.factor
    )
    
    # Create early stopping
    early_stopping = deps['create_early_stopping'](
        monitor='val_f1_score',
        patience=args.patience,
        min_delta=args.min_delta
    )
    
    # Search
    print("\nStarting hyperparameter search...")
    tuner.search(
        x_train,
        one_hot_train_labels,
        validation_split=args.validation_split,
        epochs=args.max_epochs,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters()[0]
    print("\nBest Hyperparameters:")
    for key, value in best_hps.values.items():
        if not key.startswith('tuner/'):
            print(f"  {key}: {value}")
    
    # Build and train best model
    print("\nTraining best model...")
    best_model = tuner.hypermodel.build(best_hps)
    best_model.compile(
        optimizer=deps['tf'].keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate')),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision', 'recall',
                 deps['tf'].keras.metrics.F1Score(average='weighted', threshold=None, name='f1_score')]
    )
    
    history = best_model.fit(
        x_train,
        one_hot_train_labels,
        epochs=args.max_epochs,
        validation_split=args.validation_split,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    test_metrics = best_model.evaluate(x_test, one_hot_test_labels, verbose=0)
    deps['print_metrics']("Best Model", test_metrics)
    deps['analyse_overfitting'](history, "Best Model")
    
    if args.plot:
        deps['plot_metrics'](history, "Best Model", save_path=args.save_plot)
    
    if args.save_model:
        model_path = args.save_model if args.save_model != 'auto' else 'models/best_model.keras'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        best_model.save(model_path)
        print(f"\nBest model saved to {model_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Neural Networks for Reuters News Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a single model
  python main.py train --model small --epochs 20

  # Train all models and compare
  python main.py train-all --epochs 20 --save-models

  # Perform hyperparameter tuning
  python main.py tune --max-epochs 50 --tuning-dir tuning_results
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a single model')
    train_parser.add_argument('--model', type=str, required=True,
                             choices=['baseline', 'small', 'bigger', 'bigger2', 'regularised', 'regularised2'],
                             help='Model to train')
    train_parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    train_parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split')
    train_parser.add_argument('--early-stopping', action='store_true', help='Use early stopping')
    train_parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    train_parser.add_argument('--min-delta', type=float, default=1e-4, help='Early stopping min delta')
    train_parser.add_argument('--plot', action='store_true', help='Plot training metrics')
    train_parser.add_argument('--save-plot', type=str, help='Path to save plot')
    train_parser.add_argument('--save-model', type=str, nargs='?', const='auto',
                             help='Save model (use "auto" for automatic path)')
    
    # Train-all command
    train_all_parser = subparsers.add_parser('train-all', help='Train all models and compare')
    train_all_parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    train_all_parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    train_all_parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split')
    train_all_parser.add_argument('--early-stopping', action='store_true', help='Use early stopping')
    train_all_parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    train_all_parser.add_argument('--min-delta', type=float, default=1e-4, help='Early stopping min delta')
    train_all_parser.add_argument('--save-models', action='store_true', help='Save all trained models')
    
    # Tune command
    tune_parser = subparsers.add_parser('tune', help='Perform hyperparameter tuning')
    tune_parser.add_argument('--max-epochs', type=int, default=50, help='Maximum epochs per trial')
    tune_parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split')
    tune_parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    tune_parser.add_argument('--min-delta', type=float, default=1e-4, help='Early stopping min delta')
    tune_parser.add_argument('--tuning-dir', type=str, default='tuning', help='Directory for tuning results')
    tune_parser.add_argument('--project-name', type=str, default='reuters_tuning', help='Project name')
    tune_parser.add_argument('--factor', type=int, default=3, help='Hyperband reduction factor')
    tune_parser.add_argument('--plot', action='store_true', help='Plot training metrics')
    tune_parser.add_argument('--save-plot', type=str, help='Path to save plot')
    tune_parser.add_argument('--save-model', type=str, nargs='?', const='auto',
                             help='Save best model (use "auto" for automatic path)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Set random seed (imports dependencies)
    deps = _import_dependencies()
    set_seed(deps, 42)
    
    # Execute command
    if args.command == 'train':
        train_single_model(args)
    elif args.command == 'train-all':
        train_all_models(args)
    elif args.command == 'tune':
        hyperparameter_tuning(args)


if __name__ == '__main__':
    main()

