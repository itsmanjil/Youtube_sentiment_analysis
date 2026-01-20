"""
Resume training from epoch 3 checkpoint with optimized configuration
Continues training for epochs 4-7 to complete the full training run
"""

import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

# Add paths
sys.path.append(str(Path(__file__).parent / 'research'))

from research.architectures.hybrid_cnn_bilstm import HybridCNNBiLSTM
from research.data.data_loaders import load_from_csv
from research.training.trainer import HybridDLTrainer
from research.training.callbacks import EarlyStopping, ModelCheckpoint, TrainingHistory

def set_seed(seed):
    """Set random seeds for reproducibility"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    print("="*80)
    print("RESUMING TRAINING FROM EPOCH 3")
    print("="*80)

    # Paths
    config_path = Path('output/thesis_full_gpu_optimized/config.yaml')
    checkpoint_path = Path('output/thesis_full_gpu_optimized/checkpoints/hybrid_epoch03_f10.6880.pt')
    vocab_path = Path('output/thesis_full_gpu_optimized/vocab.pkl')
    output_dir = Path('output/thesis_full_gpu_optimized')

    # Load config
    print(f"\n[1/5] Loading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seed
    set_seed(config.get('seed', 42))

    # Training config
    training_config = config.get('training', {})
    batch_size = training_config.get('batch_size', 192)
    max_epochs = training_config.get('max_epochs', 7)
    learning_rate = training_config.get('learning_rate', 0.002)

    print(f"   Batch size: {batch_size}")
    print(f"   Max epochs: {max_epochs}")
    print(f"   Learning rate: {learning_rate}")

    # Load data
    print(f"\n[2/5] Loading data...")
    train_loader, val_loader, test_loader, vocab = load_from_csv(
        train_csv=config.get('train_csv'),
        val_csv=config.get('val_csv'),
        test_csv=config.get('test_csv'),
        text_column=config.get('text_column', 'text'),
        label_column=config.get('label_column', 'label'),
        max_len=config.get('data', {}).get('max_len', 200),
        batch_size=batch_size,
        vocab_path=str(vocab_path)
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Initialize model
    print(f"\n[3/5] Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")

    model = HybridCNNBiLSTM(
        vocab_size=len(vocab.word2idx),
        embedding_dim=config.get('model', {}).get('embedding_dim', 300),
        num_classes=config.get('num_classes', 3)
    )

    # Load checkpoint
    print(f"\n[4/5] Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint.get('epoch', 3)
    print(f"   Loaded model from epoch {start_epoch}")
    print(f"   Will resume training from epoch {start_epoch + 1}")

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=training_config.get('weight_decay', 0.0001)
    )

    # Don't load optimizer state to avoid device mismatch issues
    # The optimizer will start fresh with the loaded learning rate
    print(f"   Using fresh optimizer state (avoiding device mismatch)")

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # Fresh scheduler state
    print(f"   Using fresh scheduler state")

    # Setup callbacks
    callbacks = []

    # Early stopping
    callbacks.append(EarlyStopping(
        monitor='val_f1_macro',
        patience=7,
        mode='max',
        verbose=True
    ))

    # Model checkpoint
    checkpoint_filepath = str(output_dir / 'checkpoints' / 'hybrid_epoch{epoch:02d}_f1{val_f1_macro:.4f}.pt')
    callbacks.append(ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_f1_macro',
        mode='max',
        save_best_only=True,
        verbose=True
    ))

    # Training history
    callbacks.append(TrainingHistory(
        save_path=str(output_dir / 'training_history.json')
    ))

    # Create trainer
    print(f"\n[5/5] Creating trainer...")
    trainer = HybridDLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_epochs=max_epochs,
        gradient_clip=training_config.get('gradient_clip', 5.0),
        callbacks=callbacks,
        config=config
    )

    # Set current epoch to resume from
    trainer.current_epoch = start_epoch

    # Load training history to continue tracking
    history_path = output_dir / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            history_data = json.load(f)
            # The TrainingHistory callback will load this automatically
            print(f"   Loaded existing training history ({len(history_data.get('history', {}).get('train_accuracy', []))} epochs)")

    print("\n" + "="*80)
    print(f"STARTING TRAINING FROM EPOCH {start_epoch + 1} TO {max_epochs}")
    print("="*80 + "\n")

    # Train
    trainer.train()

    # Evaluate on test set
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80 + "\n")

    test_metrics = trainer.evaluate_on_test(test_loader, verbose=True)

    # Save test results
    test_results_path = output_dir / 'test_results.json'
    with open(test_results_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n[OK] Test results saved to: {test_results_path}")

    # Save final model
    final_model_path = output_dir / 'final_model.pt'
    trainer.save_checkpoint(
        str(final_model_path),
        epoch=trainer.current_epoch,
        metrics=test_metrics
    )

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nAll outputs saved to: {output_dir}")

if __name__ == '__main__':
    main()