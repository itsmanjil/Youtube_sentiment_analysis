"""
DataLoader Factory and Utilities for Training

Provides convenient functions to create train/validation/test DataLoaders
with appropriate settings for sentiment analysis training.
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split, SubsetRandomSampler
import numpy as np
from typing import Tuple, Optional, List
from pathlib import Path
import pickle

from .dataset import SentimentDataset, CSVSentimentDataset, InMemoryDataset
from .preprocessing import Vocabulary


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = True,
    collate_fn = None
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create train, validation, and test DataLoaders with standard settings

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset (optional)
        test_dataset: Test dataset (optional)
        batch_size: Batch size for all loaders
        num_workers: Number of data loading workers (0 for Windows compatibility)
        pin_memory: Pin memory for faster GPU transfer
        collate_fn: Custom collate function (uses dataset's collate_fn if None)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        val_loader and test_loader can be None if datasets not provided

    Example:
        >>> train_loader, val_loader, test_loader = create_data_loaders(
        ...     train_dataset,
        ...     val_dataset,
        ...     test_dataset,
        ...     batch_size=32
        ... )
    """

    # Use dataset's collate function if not provided
    if collate_fn is None:
        if hasattr(train_dataset, 'collate_fn'):
            collate_fn = train_dataset.collate_fn
        else:
            collate_fn = SentimentDataset.collate_fn

    # Train loader (with shuffling)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False  # Keep all samples
    )

    # Validation loader (no shuffling)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=collate_fn,
            drop_last=False
        )

    # Test loader (no shuffling)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            collate_fn=collate_fn,
            drop_last=False
        )

    return train_loader, val_loader, test_loader


def create_stratified_split_loaders(
    dataset: Dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    batch_size: int = 32,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create stratified train/val/test splits from a single dataset

    Maintains class balance across splits using stratified sampling.

    Args:
        dataset: Complete dataset to split
        train_ratio: Proportion for training (default: 0.7)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for test (default: 0.15)
        batch_size: Batch size for all loaders
        random_seed: Random seed for reproducibility
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        >>> train_loader, val_loader, test_loader = create_stratified_split_loaders(
        ...     full_dataset,
        ...     train_ratio=0.7,
        ...     val_ratio=0.15,
        ...     test_ratio=0.15
        ... )
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Extract labels from dataset
    if hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # Fallback: iterate through dataset
        labels = np.array([dataset[i]['label'].item() for i in range(len(dataset))])

    # Get unique classes
    unique_classes = np.unique(labels)

    train_indices = []
    val_indices = []
    test_indices = []

    np.random.seed(random_seed)

    # Stratified split for each class
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)

        n_total = len(cls_indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_indices.extend(cls_indices[:n_train])
        val_indices.extend(cls_indices[n_train:n_train + n_val])
        test_indices.extend(cls_indices[n_train + n_val:])

    # Shuffle indices
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Determine collate function
    collate_fn = getattr(dataset, 'collate_fn', SentimentDataset.collate_fn)

    # Create loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    print("[OK] Created stratified splits:")
    print(f"   Train: {len(train_indices)} samples ({train_ratio*100:.1f}%)")
    print(f"   Val:   {len(val_indices)} samples ({val_ratio*100:.1f}%)")
    print(f"   Test:  {len(test_indices)} samples ({test_ratio*100:.1f}%)")

    return train_loader, val_loader, test_loader


def create_k_fold_loaders(
    dataset: Dataset,
    n_folds: int = 10,
    fold_index: int = 0,
    batch_size: int = 32,
    random_seed: int = 42,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation loaders for k-fold cross-validation

    Useful for thesis-level experiments requiring k-fold CV.

    Args:
        dataset: Complete dataset
        n_folds: Number of folds (default: 10)
        fold_index: Which fold to use as validation (0 to n_folds-1)
        batch_size: Batch size
        random_seed: Random seed
        num_workers: Number of workers

    Returns:
        Tuple of (train_loader, val_loader) for the specified fold

    Example:
        >>> # Train on all 10 folds
        >>> for fold in range(10):
        ...     train_loader, val_loader = create_k_fold_loaders(
        ...         dataset, n_folds=10, fold_index=fold
        ...     )
        ...     # Train model with this fold
    """
    from sklearn.model_selection import StratifiedKFold

    assert 0 <= fold_index < n_folds, f"fold_index must be in [0, {n_folds-1}]"

    # Extract labels
    if hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i]['label'].item() for i in range(len(dataset))])

    # Create stratified k-fold splits
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    # Get train/val indices for this fold
    splits = list(skf.split(np.arange(len(dataset)), labels))
    train_indices, val_indices = splits[fold_index]

    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Determine collate function
    collate_fn = getattr(dataset, 'collate_fn', SentimentDataset.collate_fn)

    # Create loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn
    )

    print(f"[OK] Created fold {fold_index+1}/{n_folds}:")
    print(f"   Train: {len(train_indices)} samples")
    print(f"   Val:   {len(val_indices)} samples")

    return train_loader, val_loader


def load_from_csv(
    train_csv: str,
    val_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    vocab_path: Optional[str] = None,
    text_column: str = 'text',
    label_column: str = 'label',
    max_len: int = 200,
    batch_size: int = 32,
    build_vocab_from_train: bool = True,
    vocab_max_size: int = 20000,
    vocab_min_freq: int = 2,
    save_vocab_path: Optional[str] = None
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader], Vocabulary]:
    """
    Complete pipeline: Load CSV files -> Build vocabulary -> Create DataLoaders

    One-stop function for loading data from CSV files.

    Args:
        train_csv: Path to training CSV
        val_csv: Path to validation CSV (optional)
        test_csv: Path to test CSV (optional)
        vocab_path: Path to existing vocabulary pickle (optional)
        text_column: Name of text column in CSV
        label_column: Name of label column in CSV
        max_len: Maximum sequence length
        batch_size: Batch size
        build_vocab_from_train: Build vocab from training data if vocab_path is None
        vocab_max_size: Maximum vocabulary size
        vocab_min_freq: Minimum word frequency
        save_vocab_path: Path to save vocabulary pickle (optional)

    Returns:
        Tuple of (train_loader, val_loader, test_loader, vocabulary)

    Example:
        >>> train_loader, val_loader, test_loader, vocab = load_from_csv(
        ...     train_csv='data/train.csv',
        ...     val_csv='data/val.csv',
        ...     test_csv='data/test.csv',
        ...     save_vocab_path='models/vocab.pkl'
        ... )
    """
    import pandas as pd

    print("\n" + "="*80)
    print("Loading Data from CSV Files")
    print("="*80 + "\n")

    # Load or build vocabulary
    if vocab_path is not None and Path(vocab_path).exists():
        print(f"Loading vocabulary from {vocab_path}...")
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        print(f"[OK] Loaded vocabulary with {len(vocab)} tokens\n")

    elif build_vocab_from_train:
        print(f"Building vocabulary from {train_csv}...")
        df_train = pd.read_csv(train_csv)
        texts = df_train[text_column].astype(str).tolist()

        vocab = Vocabulary(max_size=vocab_max_size, min_freq=vocab_min_freq)
        vocab.build_from_texts(texts)
        print(f"[OK] Built vocabulary with {len(vocab)} tokens\n")

        # Save if requested
        if save_vocab_path:
            Path(save_vocab_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_vocab_path, 'wb') as f:
                pickle.dump(vocab, f)
            print(f"[OK] Saved vocabulary to {save_vocab_path}\n")

    else:
        raise ValueError("Either vocab_path or build_vocab_from_train must be provided")

    # Create datasets
    print("Creating datasets...")
    train_dataset = CSVSentimentDataset(
        train_csv, vocab, text_column, label_column, max_len
    )

    val_dataset = None
    if val_csv:
        val_dataset = CSVSentimentDataset(
            val_csv, vocab, text_column, label_column, max_len
        )

    test_dataset = None
    if test_csv:
        test_dataset = CSVSentimentDataset(
            test_csv, vocab, text_column, label_column, max_len
        )

    print()

    # Create loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size
    )

    print("\n[OK] DataLoaders created:")
    print(f"   Train batches: {len(train_loader)}")
    if val_loader:
        print(f"   Val batches:   {len(val_loader)}")
    if test_loader:
        print(f"   Test batches:  {len(test_loader)}")

    return train_loader, val_loader, test_loader, vocab


# Example and testing
if __name__ == "__main__":
    """
    Test data loader utilities

    Usage:
        cd backend/research
        python -m data.data_loaders
    """
    print("Testing DataLoader utilities...\n")

    from .dataset import SentimentDataset
    from .preprocessing import Vocabulary

    # Mock data
    texts = [
        f"Sample text number {i} with different content" for i in range(100)
    ]
    labels = [i % 3 for i in range(100)]  # Balanced classes: 0, 1, 2

    # Create vocabulary and dataset
    vocab = Vocabulary(max_size=1000, min_freq=1)
    vocab.build_from_texts(texts)

    dataset = SentimentDataset(texts, labels, vocab, max_len=50)

    print("1. Testing create_data_loaders...")
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    from torch.utils.data import random_split
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader, val_loader, test_loader = create_data_loaders(
        train_ds, val_ds, test_ds, batch_size=16
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}\n")

    print("2. Testing create_stratified_split_loaders...")
    train_loader, val_loader, test_loader = create_stratified_split_loaders(
        dataset, batch_size=16
    )
    print()

    print("3. Testing create_k_fold_loaders...")
    train_loader, val_loader = create_k_fold_loaders(
        dataset, n_folds=5, fold_index=0, batch_size=16
    )
    print()

    print("[OK] All DataLoader tests passed!")
