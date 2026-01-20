"""
Main Training Script for Hybrid CNN-BiLSTM-Attention Model

Complete CLI for training sentiment analysis models with:
- Configuration via YAML files or command-line arguments
- Automatic data loading and preprocessing
- Model initialization and training
- Comprehensive evaluation and checkpointing
- TensorBoard logging

Usage:
    # Basic training with config file
    python train_hybrid_dl.py --config config/hybrid_dl_config.yaml

    # Override config with CLI arguments
    python train_hybrid_dl.py --config config/hybrid_dl_config.yaml --batch_size 64 --max_epochs 100

    # Train from scratch with minimal args
    python train_hybrid_dl.py --train_csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv

    # Resume training from checkpoint
    python train_hybrid_dl.py --config config/hybrid_dl_config.yaml --resume models/checkpoint.pt

Author: Thesis Project
Version: 1.0.0
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from pathlib import Path
import sys
from datetime import datetime
import json

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from architectures.hybrid_cnn_bilstm import HybridCNNBiLSTM
from architectures.embeddings import EmbeddingManager
from data.preprocessing import Vocabulary
from data.data_loaders import load_from_csv, create_stratified_split_loaders, CSVSentimentDataset, create_data_loaders
from training.trainer import HybridDLTrainer
from training.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, ProgressLogger, TrainingHistory
from training.evaluator import evaluate_model


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Hybrid CNN-BiLSTM-Attention Sentiment Analysis Model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Config file
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file (recommended)')

    # Data paths
    parser.add_argument('--train_csv', type=str, default=None,
                       help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default=None,
                       help='Path to validation CSV file')
    parser.add_argument('--test_csv', type=str, default=None,
                       help='Path to test CSV file')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Name of text column in CSV')
    parser.add_argument('--label_column', type=str, default='label',
                       help='Name of label column in CSV')

    # Vocabulary
    parser.add_argument('--vocab_path', type=str, default=None,
                       help='Path to existing vocabulary pickle')
    parser.add_argument('--vocab_max_size', type=int, default=20000,
                       help='Maximum vocabulary size')
    parser.add_argument('--vocab_min_freq', type=int, default=2,
                       help='Minimum word frequency')

    # Model architecture
    parser.add_argument('--embedding_dim', type=int, default=300,
                       help='Embedding dimension')
    parser.add_argument('--max_len', type=int, default=200,
                       help='Maximum sequence length')
    parser.add_argument('--num_classes', type=int, default=3,
                       help='Number of classes')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=50,
                       help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--gradient_clip', type=float, default=5.0,
                       help='Gradient clipping norm (0 = no clipping)')

    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=7,
                       help='Early stopping patience')
    parser.add_argument('--early_stopping_monitor', type=str, default='val_f1_macro',
                       help='Metric to monitor for early stopping')

    # Paths
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='Output directory for models and logs')
    parser.add_argument('--tensorboard_dir', type=str, default='./runs',
                       help='TensorBoard log directory')

    # Embeddings
    parser.add_argument('--glove_path', type=str, default=None,
                       help='Path to GloVe embeddings file')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu', 'mps'],
                       help='Device to train on')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Experiment name
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: timestamp)')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def merge_config_and_args(config: dict, args: argparse.Namespace) -> dict:
    """
    Merge YAML config and CLI arguments, with CLI taking precedence

    Args:
        config: Config dict from YAML
        args: Parsed CLI arguments

    Returns:
        Merged configuration dict
    """
    # Start with config file
    merged = config.copy() if config else {}

    # Override with CLI args (only if not None and not default)
    parser = argparse.ArgumentParser()
    parse_args()  # Get defaults

    for key, value in vars(args).items():
        if value is not None:
            # Navigate nested config structure
            if key == 'batch_size':
                merged.setdefault('training', {})['batch_size'] = value
            elif key == 'max_epochs':
                merged.setdefault('training', {})['max_epochs'] = value
            elif key == 'learning_rate':
                merged.setdefault('training', {})['learning_rate'] = value
            elif key == 'weight_decay':
                merged.setdefault('training', {})['weight_decay'] = value
            elif key == 'gradient_clip':
                merged.setdefault('training', {})['gradient_clip'] = value
            elif key == 'embedding_dim':
                merged.setdefault('model', {})['embedding_dim'] = value
            elif key == 'max_len':
                merged.setdefault('data', {})['max_len'] = value
            elif key == 'vocab_max_size':
                merged.setdefault('data', {})['vocab_max_size'] = value
            elif key == 'vocab_min_freq':
                merged.setdefault('data', {})['vocab_min_freq'] = value
            elif key == 'glove_path':
                merged.setdefault('embeddings', {})['path'] = value
            else:
                merged[key] = value

    return merged


def main():
    """Main training pipeline"""

    # Parse arguments
    args = parse_args()

    # Load config file if provided
    if args.config:
        print(f"📄 Loading config from: {args.config}")
        config = load_config(args.config)
    else:
        config = {}

    # Merge config and CLI args
    config = merge_config_and_args(config, args)

    # Set random seed
    set_seed(config.get('seed', 42))

    # Create experiment name
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        exp_name = f"hybrid_dl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create output directory
    output_dir = Path(config.get('output_dir', './output')) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print(f"EXPERIMENT: {exp_name}")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"TensorBoard logs: {config.get('tensorboard_dir', './runs')}")
    print("="*80 + "\n")

    # Save config to output directory
    config_save_path = output_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[OK] Configuration saved to: {config_save_path}\n")

    # ====================
    # 1. LOAD DATA
    # ====================
    print("="*80)
    print("STEP 1: LOADING DATA")
    print("="*80 + "\n")

    data_config = config.get('data', {})
    train_csv = config.get('train_csv', args.train_csv)
    val_csv = config.get('val_csv', args.val_csv)
    test_csv = config.get('test_csv', args.test_csv)

    if not train_csv:
        raise ValueError("--train_csv must be provided")

    # Load data and build vocabulary
    train_loader, val_loader, test_loader, vocab = load_from_csv(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        vocab_path=config.get('vocab_path', args.vocab_path),
        text_column=config.get('text_column', 'text'),
        label_column=config.get('label_column', 'label'),
        max_len=data_config.get('max_len', 200),
        batch_size=config.get('training', {}).get('batch_size', 32),
        vocab_max_size=data_config.get('vocab_max_size', 20000),
        vocab_min_freq=data_config.get('vocab_min_freq', 2),
        save_vocab_path=str(output_dir / 'vocab.pkl')
    )

    # ====================
    # 2. BUILD MODEL
    # ====================
    print("\n" + "="*80)
    print("STEP 2: BUILDING MODEL")
    print("="*80 + "\n")

    model_config = config.get('model', {})
    vocab_size = len(vocab)

    model = HybridCNNBiLSTM(
        vocab_size=vocab_size,
        embedding_dim=model_config.get('embedding_dim', 300),
        num_classes=config.get('num_classes', 3),
        cnn_filter_sizes=model_config.get('cnn', {}).get('filter_sizes', [3, 4, 5]),
        cnn_num_filters=model_config.get('cnn', {}).get('num_filters', 128),
        lstm_hidden_size=model_config.get('bilstm', {}).get('hidden_size', 128),
        lstm_num_layers=model_config.get('bilstm', {}).get('num_layers', 2),
        attention_num_heads=model_config.get('attention', {}).get('num_heads', 4),
        classifier_hidden_sizes=model_config.get('classifier', {}).get('hidden_sizes', [256, 128]),
        dropout_cnn=model_config.get('cnn', {}).get('dropout', 0.3),
        dropout_lstm=model_config.get('bilstm', {}).get('dropout', 0.3),
        dropout_attention=model_config.get('attention', {}).get('dropout', 0.1),
        dropout_classifier=model_config.get('classifier', {}).get('dropout', [0.5, 0.4])
    )

    print(f"[OK] Model created")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    # Load GloVe embeddings if provided
    embeddings_config = config.get('embeddings', {})
    glove_path = embeddings_config.get('path', args.glove_path)

    if glove_path and Path(glove_path).exists():
        print(f"Loading GloVe embeddings from: {glove_path}")
        embedding_manager = EmbeddingManager()

        embeddings_dict = embedding_manager.load_glove_embeddings(
            glove_path,
            embedding_dim=embeddings_config.get('dim', 300)
        )

        embedding_matrix = embedding_manager.create_embedding_matrix(
            vocab.word2idx,
            embeddings_dict,
            embedding_dim=embeddings_config.get('dim', 300)
        )

        # Load into model
        model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))

        # Freeze embeddings if specified
        if not embeddings_config.get('trainable', True):
            model.embedding.weight.requires_grad = False
            print("   Embeddings frozen (not trainable)")
        else:
            print("   Embeddings trainable")

        coverage = embedding_manager.compute_coverage(vocab.word2idx, embeddings_dict)
        print(f"[OK] GloVe embeddings loaded")
        print(f"   Vocabulary coverage: {coverage:.2%}\n")

    # ====================
    # 3. SETUP TRAINING
    # ====================
    print("="*80)
    print("STEP 3: SETTING UP TRAINING")
    print("="*80 + "\n")

    training_config = config.get('training', {})

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.get('learning_rate', 0.001),
        weight_decay=training_config.get('weight_decay', 0.0001)
    )

    print(f"[OK] Optimizer: Adam (lr={training_config.get('learning_rate', 0.001)}, "
          f"weight_decay={training_config.get('weight_decay', 0.0001)})")

    # Learning rate scheduler
    scheduler = None
    lr_config = training_config.get('lr_scheduler', {})

    if lr_config.get('type') == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=lr_config.get('factor', 0.5),
            patience=lr_config.get('patience', 3),
            verbose=True
        )
        print(f"[OK] LR Scheduler: ReduceLROnPlateau (patience={lr_config.get('patience', 3)})")

    elif lr_config.get('type') == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=lr_config.get('step_size', 10),
            gamma=lr_config.get('gamma', 0.1)
        )
        print(f"[OK] LR Scheduler: StepLR (step_size={lr_config.get('step_size', 10)})")

    elif lr_config.get('type') == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=training_config.get('max_epochs', 50)
        )
        print(f"[OK] LR Scheduler: CosineAnnealingLR")

    # Callbacks
    callbacks = []

    # Progress logger
    callbacks.append(ProgressLogger(print_freq=config.get('log_frequency', 10)))

    # Early stopping
    es_config = training_config.get('early_stopping', {})
    if es_config.get('enabled', True):
        callbacks.append(EarlyStopping(
            monitor=config.get('early_stopping_monitor', 'val_f1_macro'),
            patience=es_config.get('patience', 7),
            mode='max',
            verbose=True
        ))
        print(f"[OK] Early Stopping (patience={es_config.get('patience', 7)})")

    # Model checkpoint
    checkpoint_path = output_dir / 'checkpoints' / 'hybrid_epoch{epoch:02d}_f1{val_f1_macro:.4f}.pt'
    callbacks.append(ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor='val_f1_macro',
        mode='max',
        save_best_only=True,
        verbose=True
    ))
    print(f"[OK] Model Checkpoint (saving best model)")

    # Learning rate scheduler callback
    if scheduler is not None:
        callbacks.append(LearningRateScheduler(
            scheduler,
            monitor='val_f1_macro' if isinstance(scheduler, ReduceLROnPlateau) else None,
            verbose=True
        ))

    # Training history
    callbacks.append(TrainingHistory(
        save_path=str(output_dir / 'training_history.json')
    ))

    print()

    # ====================
    # 4. CREATE TRAINER
    # ====================
    trainer = HybridDLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.get('device', 'auto'),
        max_epochs=training_config.get('max_epochs', 50),
        gradient_clip=training_config.get('gradient_clip', 5.0),
        tensorboard_log_dir=config.get('tensorboard_dir', './runs'),
        callbacks=callbacks,
        config=config
    )

    # Resume from checkpoint if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ====================
    # 5. TRAIN
    # ====================
    print("="*80)
    print("STEP 4: TRAINING")
    print("="*80 + "\n")

    trainer.train()

    # ====================
    # 6. EVALUATE ON TEST SET
    # ====================
    if test_loader is not None:
        print("\n" + "="*80)
        print("STEP 5: FINAL EVALUATION ON TEST SET")
        print("="*80 + "\n")

        test_metrics = trainer.evaluate_on_test(test_loader, verbose=True)

        # Save test results
        test_results_path = output_dir / 'test_results.json'
        with open(test_results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)

        print(f"\n[OK] Test results saved to: {test_results_path}")

    # ====================
    # 7. SAVE FINAL MODEL
    # ====================
    final_model_path = output_dir / 'final_model.pt'
    trainer.save_checkpoint(
        str(final_model_path),
        epoch=trainer.current_epoch,
        metrics=test_metrics if test_loader else {}
    )

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n[OUTPUT] All outputs saved to: {output_dir}")
    print(f"   - Configuration: config.yaml")
    print(f"   - Vocabulary: vocab.pkl")
    print(f"   - Best model: checkpoints/")
    print(f"   - Final model: final_model.pt")
    print(f"   - Training history: training_history.json")
    if test_loader:
        print(f"   - Test results: test_results.json")
    print(f"\n[INFO] View training curves:")
    print(f"   tensorboard --logdir={config.get('tensorboard_dir', './runs')}")
    print()


if __name__ == "__main__":
    main()
