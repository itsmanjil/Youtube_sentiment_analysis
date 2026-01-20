"""
Training Infrastructure Test Script

Tests the complete deep learning training infrastructure:
- Data loading and preprocessing
- Model initialization
- Training loop
- Evaluation metrics
- Checkpointing

Usage:
    python test_training_infrastructure.py

Author: Master's Thesis
Version: 1.0
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'research'))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np


def test_imports():
    """Test 1: Verify all imports work"""
    print("\n" + "="*80)
    print("TEST 1: IMPORTS")
    print("="*80)

    try:
        # Core PyTorch
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")

        # Research modules
        from research.architectures.hybrid_cnn_bilstm import HybridCNNBiLSTM
        print("[OK] HybridCNNBiLSTM architecture")

        from research.architectures.embeddings import EmbeddingManager
        print("[OK] EmbeddingManager")

        from research.data.preprocessing import Vocabulary
        print("[OK] Vocabulary")

        from research.data.dataset import SentimentDataset, CSVSentimentDataset
        print("[OK] Dataset classes")

        from research.data.data_loaders import create_data_loaders
        print("[OK] DataLoader utilities")

        from research.training.trainer import HybridDLTrainer
        print("[OK] Trainer")

        from research.training.callbacks import (
            EarlyStopping, ModelCheckpoint, ProgressLogger
        )
        print("[OK] Callbacks")

        from research.training.evaluator import evaluate_model
        print("[OK] Evaluator")

        return True

    except ImportError as e:
        print(f"[FAIL] Import error: {e}")
        return False


def test_vocabulary():
    """Test 2: Vocabulary building"""
    print("\n" + "="*80)
    print("TEST 2: VOCABULARY")
    print("="*80)

    try:
        from research.data.preprocessing import Vocabulary

        # Sample texts
        texts = [
            "This is a great movie!",
            "I love this product",
            "Terrible experience, very disappointed",
            "Amazing quality and fast shipping",
            "Not worth the price"
        ]

        # Build vocabulary
        vocab = Vocabulary(max_size=1000, min_freq=1)
        vocab.build_from_texts(texts)

        print(f"[OK] Vocabulary built")
        print(f"   Size: {len(vocab)}")
        print(f"   Sample tokens: {list(vocab.word2idx.keys())[:10]}")

        # Test encoding
        encoded = vocab.encode("This is amazing")
        print(f"   Encoded 'This is amazing': {encoded}")

        decoded = vocab.decode(encoded)
        print(f"   Decoded: {decoded}")

        return True

    except Exception as e:
        print(f"[FAIL] Vocabulary test failed: {e}")
        return False


def test_dataset():
    """Test 3: Dataset and DataLoader"""
    print("\n" + "="*80)
    print("TEST 3: DATASET & DATALOADER")
    print("="*80)

    try:
        from research.data.preprocessing import Vocabulary
        from research.data.dataset import SentimentDataset
        from research.data.data_loaders import create_data_loaders

        # Create dummy data
        texts = [f"Sample comment number {i} with sentiment" for i in range(100)]
        labels = [i % 3 for i in range(100)]  # 0, 1, 2

        # Build vocab
        vocab = Vocabulary(max_size=1000, min_freq=1)
        vocab.build_from_texts(texts)

        # Create dataset
        dataset = SentimentDataset(texts, labels, vocab, max_len=50)
        print(f"[OK] Dataset created: {len(dataset)} samples")

        # Test __getitem__
        sample = dataset[0]
        print(f"   Sample keys: {sample.keys()}")
        print(f"   Input IDs shape: {sample['input_ids'].shape}")
        print(f"   Label: {sample['label']}")

        # Create loaders
        from torch.utils.data import random_split
        train_size = 70
        val_size = 15
        test_size = 15

        train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
        train_loader, val_loader, test_loader = create_data_loaders(
            train_ds, val_ds, test_ds, batch_size=16
        )

        print(f"[OK] DataLoaders created")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")

        # Test batch
        batch = next(iter(train_loader))
        print(f"   Batch input_ids shape: {batch['input_ids'].shape}")
        print(f"   Batch labels shape: {batch['labels'].shape}")

        return True

    except Exception as e:
        print(f"[FAIL] Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test 4: Model architecture"""
    print("\n" + "="*80)
    print("TEST 4: MODEL ARCHITECTURE")
    print("="*80)

    try:
        from research.architectures.hybrid_cnn_bilstm import HybridCNNBiLSTM

        # Create model
        model = HybridCNNBiLSTM(
            vocab_size=1000,
            embedding_dim=100,
            num_classes=3,
            cnn_filter_sizes=[3, 4, 5],
            cnn_num_filters=64,
            lstm_hidden_size=64,
            lstm_num_layers=2,
            attention_num_heads=4,
            classifier_hidden_sizes=[128, 64]
        )

        print(f"[OK] Model created")
        print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Test forward pass
        batch_size = 8
        seq_len = 50
        dummy_input = torch.randint(0, 1000, (batch_size, seq_len))

        model.eval()
        with torch.no_grad():
            logits, attention_weights = model(dummy_input)

        print(f"   [OK] Forward pass successful")
        print(f"      Input shape: {dummy_input.shape}")
        print(f"      Output logits shape: {logits.shape}")
        print(f"      Attention keys: {list(attention_weights.keys())}")

        return True

    except Exception as e:
        print(f"[FAIL] Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer():
    """Test 5: Training loop"""
    print("\n" + "="*80)
    print("TEST 5: TRAINING LOOP")
    print("="*80)

    try:
        from research.architectures.hybrid_cnn_bilstm import HybridCNNBiLSTM
        from research.data.preprocessing import Vocabulary
        from research.data.dataset import SentimentDataset
        from research.data.data_loaders import create_data_loaders
        from research.training.trainer import HybridDLTrainer
        from research.training.callbacks import ProgressLogger

        # Create small dataset
        texts = [f"Sample text {i}" for i in range(50)]
        labels = [i % 3 for i in range(50)]

        vocab = Vocabulary(max_size=100, min_freq=1)
        vocab.build_from_texts(texts)

        dataset = SentimentDataset(texts, labels, vocab, max_len=20)

        # Split
        from torch.utils.data import random_split
        train_ds, val_ds = random_split(dataset, [40, 10])

        train_loader, val_loader, _ = create_data_loaders(
            train_ds, val_ds, None, batch_size=8
        )

        # Create model
        model = HybridCNNBiLSTM(
            vocab_size=len(vocab),
            embedding_dim=50,
            num_classes=3,
            cnn_filter_sizes=[3],
            cnn_num_filters=32,
            lstm_hidden_size=32,
            lstm_num_layers=1,
            attention_num_heads=2,
            classifier_hidden_sizes=[32]
        )

        # Create trainer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        trainer = HybridDLTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device='cpu',
            max_epochs=2,  # Only 2 epochs for testing
            callbacks=[ProgressLogger(print_freq=1)]
        )

        print("[OK] Trainer created")

        # Train for 2 epochs
        print("\n>> Training for 2 epochs (test mode)...")
        trainer.train()

        print("\n[OK] Training completed successfully")
        print(f"   Final epoch: {trainer.current_epoch}")
        print(f"   Training history: {len(trainer.training_history)} epochs")

        return True

    except Exception as e:
        print(f"[FAIL] Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpointing():
    """Test 6: Model checkpointing"""
    print("\n" + "="*80)
    print("TEST 6: CHECKPOINTING")
    print("="*80)

    try:
        from research.architectures.hybrid_cnn_bilstm import HybridCNNBiLSTM

        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        checkpoint_path = temp_dir / 'test_checkpoint.pt'

        # Create model
        model = HybridCNNBiLSTM(
            vocab_size=100,
            embedding_dim=50,
            num_classes=3
        )

        optimizer = torch.optim.Adam(model.parameters())

        # Save checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': 5,
            'metrics': {'accuracy': 0.85}
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"[OK] Checkpoint saved to: {checkpoint_path}")

        # Load checkpoint
        loaded = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(loaded['model_state_dict'])
        optimizer.load_state_dict(loaded['optimizer_state_dict'])

        print(f"[OK] Checkpoint loaded successfully")
        print(f"   Epoch: {loaded['epoch']}")
        print(f"   Metrics: {loaded['metrics']}")

        # Cleanup
        shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        print(f"[FAIL] Checkpointing test failed: {e}")
        return False


def test_csv_loading():
    """Test 7: CSV data loading"""
    print("\n" + "="*80)
    print("TEST 7: CSV DATA LOADING")
    print("="*80)

    try:
        from research.data.data_loaders import load_from_csv

        # Create temporary CSV files
        temp_dir = Path(tempfile.mkdtemp())

        train_data = pd.DataFrame({
            'text': [f'Training sample {i}' for i in range(50)],
            'label': [i % 3 for i in range(50)]
        })

        val_data = pd.DataFrame({
            'text': [f'Validation sample {i}' for i in range(15)],
            'label': [i % 3 for i in range(15)]
        })

        train_csv = temp_dir / 'train.csv'
        val_csv = temp_dir / 'val.csv'

        train_data.to_csv(train_csv, index=False)
        val_data.to_csv(val_csv, index=False)

        print(f"[OK] Created test CSV files")

        # Load with utility function
        train_loader, val_loader, _, vocab = load_from_csv(
            train_csv=str(train_csv),
            val_csv=str(val_csv),
            batch_size=16,
            vocab_max_size=1000
        )

        print(f"[OK] CSV loading successful")
        print(f"   Vocabulary size: {len(vocab)}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")

        # Cleanup
        shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        print(f"[FAIL] CSV loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TRAINING INFRASTRUCTURE TEST SUITE")
    print("="*80)

    tests = [
        ("Imports", test_imports),
        ("Vocabulary", test_vocabulary),
        ("Dataset & DataLoader", test_dataset),
        ("Model Architecture", test_model),
        ("Training Loop", test_trainer),
        ("Checkpointing", test_checkpointing),
        ("CSV Loading", test_csv_loading),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n[FAIL] {test_name} crashed: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}  {test_name}")

    total = len(results)
    passed = sum(results.values())

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! Training infrastructure is ready.")
        print("\nNext steps:")
        print("1. Prepare training data:")
        print("   python prepare_youtube_training_data.py --video_list videos.txt")
        print("\n2. Train the model:")
        print("   python research/train_hybrid_dl.py --config research/config/hybrid_dl_config.yaml")
    else:
        print("\nWARNING: Some tests failed. Review errors above.")


if __name__ == "__main__":
    main()
