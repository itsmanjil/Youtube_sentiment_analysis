"""
PyTorch Dataset Classes for Sentiment Analysis Training

Provides flexible dataset classes for training the hybrid CNN-BiLSTM-Attention model.
Supports in-memory datasets, CSV files, and integration with existing Django models.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import pickle
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from .preprocessing import tokenize, Vocabulary


class SentimentDataset(Dataset):
    """
    PyTorch Dataset for sentiment analysis with text and labels

    Supports:
    - Automatic tokenization and encoding
    - Dynamic padding
    - Label mapping (text labels to indices)
    - Optional max length truncation

    Args:
        texts: List of text samples
        labels: List of labels (can be strings or integers)
        vocab: Vocabulary object for encoding
        max_len: Maximum sequence length (default: 200)
        label_map: Optional mapping from label names to indices
                   Default: {'negative': 0, 'neutral': 1, 'positive': 2}

    Example:
        >>> vocab = Vocabulary()
        >>> vocab.build_from_texts(train_texts)
        >>> dataset = SentimentDataset(train_texts, train_labels, vocab)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        texts: List[str],
        labels: Union[List[str], List[int]],
        vocab: Vocabulary,
        max_len: int = 200,
        label_map: Optional[Dict[str, int]] = None
    ):
        self.texts = texts
        self.vocab = vocab
        self.max_len = max_len

        # Default label mapping for sentiment analysis
        if label_map is None:
            self.label_map = {
                'negative': 0,
                'neutral': 1,
                'positive': 2,
                0: 0,
                1: 1,
                2: 2
            }
        else:
            self.label_map = label_map

        # Convert labels to indices
        self.labels = [self._map_label(label) for label in labels]

        # Pre-encode all texts for faster training
        self.encoded_texts = []
        self.lengths = []

        for text in texts:
            indices = vocab.encode(text)
            # Truncate if needed
            if len(indices) > max_len:
                indices = indices[:max_len]

            self.encoded_texts.append(indices)
            self.lengths.append(len(indices))

    def _map_label(self, label: Union[str, int]) -> int:
        """Map label to integer index"""
        if isinstance(label, int):
            return label

        # Handle string labels (case-insensitive)
        label_lower = str(label).lower()
        if label_lower in self.label_map:
            return self.label_map[label_lower]

        # Try exact match
        if label in self.label_map:
            return self.label_map[label]

        raise ValueError(f"Unknown label: {label}. Valid labels: {list(self.label_map.keys())}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
            - input_ids: Tensor of token indices [seq_len]
            - label: Tensor with label index [1]
            - length: Actual sequence length before padding
            - text: Original text (for debugging)
        """
        return {
            'input_ids': torch.tensor(self.encoded_texts[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'length': self.lengths[idx],
            'text': self.texts[idx]
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching with dynamic padding

        Pads sequences to the maximum length in the batch (not global max_len)
        for computational efficiency.

        Args:
            batch: List of samples from __getitem__

        Returns:
            Batched tensors with dynamic padding
        """
        # Find max length in this batch
        max_len_batch = max(item['length'] for item in batch)

        batch_size = len(batch)

        # Initialize padded tensor
        input_ids = torch.zeros(batch_size, max_len_batch, dtype=torch.long)
        labels = torch.zeros(batch_size, dtype=torch.long)
        lengths = []
        texts = []

        for i, item in enumerate(batch):
            seq_len = item['length']
            input_ids[i, :seq_len] = item['input_ids']
            labels[i] = item['label']
            lengths.append(seq_len)
            texts.append(item['text'])

        return {
            'input_ids': input_ids,
            'labels': labels,
            'lengths': torch.tensor(lengths, dtype=torch.long),
            'texts': texts
        }


class CSVSentimentDataset(SentimentDataset):
    """
    Load sentiment dataset from CSV file

    CSV Format:
        text,label
        "This is great!",positive
        "This is terrible",negative
        "This is okay",neutral

    Args:
        csv_path: Path to CSV file
        vocab: Vocabulary object (or path to vocab pickle)
        text_column: Name of text column (default: 'text')
        label_column: Name of label column (default: 'label')
        max_len: Maximum sequence length
        label_map: Optional label mapping

    Example:
        >>> dataset = CSVSentimentDataset(
        ...     'data/train.csv',
        ...     vocab='models/vocab.pkl',
        ...     text_column='comment',
        ...     label_column='sentiment'
        ... )
    """

    def __init__(
        self,
        csv_path: str,
        vocab: Union[Vocabulary, str],
        text_column: str = 'text',
        label_column: str = 'label',
        max_len: int = 200,
        label_map: Optional[Dict[str, int]] = None
    ):
        # Load CSV
        df = pd.read_csv(csv_path)

        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in CSV. "
                           f"Available columns: {df.columns.tolist()}")

        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV. "
                           f"Available columns: {df.columns.tolist()}")

        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].tolist()

        # Load vocabulary if path provided
        if isinstance(vocab, str):
            with open(vocab, 'rb') as f:
                vocab = pickle.load(f)

        super().__init__(texts, labels, vocab, max_len, label_map)

        print(f"[OK] Loaded {len(self)} samples from {csv_path}")
        print(f"   Label distribution: {pd.Series(self.labels).value_counts().to_dict()}")


class DjangoSentimentDataset(SentimentDataset):
    """
    Load sentiment dataset from Django models (YouTubeComment)

    Integrates with existing Django backend to load comments from database.

    Args:
        queryset: Django QuerySet of YouTubeComment objects
        vocab: Vocabulary object
        text_field: Field name for text (default: 'text')
        label_field: Field name for label (default: 'sentiment')
        max_len: Maximum sequence length
        label_map: Optional label mapping

    Example:
        >>> from app.models import YouTubeComment
        >>> comments = YouTubeComment.objects.filter(sentiment__isnull=False)
        >>> dataset = DjangoSentimentDataset(
        ...     queryset=comments,
        ...     vocab=vocab,
        ...     text_field='text',
        ...     label_field='sentiment'
        ... )
    """

    def __init__(
        self,
        queryset,
        vocab: Vocabulary,
        text_field: str = 'text',
        label_field: str = 'sentiment',
        max_len: int = 200,
        label_map: Optional[Dict[str, int]] = None
    ):
        # Extract texts and labels from queryset
        texts = []
        labels = []

        for obj in queryset:
            text = getattr(obj, text_field, None)
            label = getattr(obj, label_field, None)

            if text is not None and label is not None:
                texts.append(str(text))
                labels.append(label)

        if not texts:
            raise ValueError("No valid samples found in queryset. "
                           f"Check that '{text_field}' and '{label_field}' fields exist.")

        super().__init__(texts, labels, vocab, max_len, label_map)

        print(f"[OK] Loaded {len(self)} samples from Django queryset")
        print(f"   Label distribution: {pd.Series(self.labels).value_counts().to_dict()}")


class InMemoryDataset(Dataset):
    """
    Simple in-memory dataset for pre-encoded data

    Use this when data is already tokenized and encoded, for maximum speed.

    Args:
        input_ids: List or array of encoded sequences
        labels: List or array of labels
        lengths: Optional list of sequence lengths

    Example:
        >>> input_ids = [vocab.encode(text) for text in texts]
        >>> dataset = InMemoryDataset(input_ids, labels)
    """

    def __init__(
        self,
        input_ids: Union[List[List[int]], np.ndarray],
        labels: Union[List[int], np.ndarray],
        lengths: Optional[Union[List[int], np.ndarray]] = None
    ):
        self.input_ids = input_ids
        self.labels = labels

        if lengths is None:
            self.lengths = [len(seq) for seq in input_ids]
        else:
            self.lengths = lengths

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'length': self.lengths[idx]
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Dynamic padding collate function"""
        max_len_batch = max(item['length'] for item in batch)
        batch_size = len(batch)

        input_ids = torch.zeros(batch_size, max_len_batch, dtype=torch.long)
        labels = torch.zeros(batch_size, dtype=torch.long)
        lengths = []

        for i, item in enumerate(batch):
            seq_len = item['length']
            input_ids[i, :seq_len] = item['input_ids']
            labels[i] = item['label']
            lengths.append(seq_len)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'lengths': torch.tensor(lengths, dtype=torch.long)
        }


# Example and testing
if __name__ == "__main__":
    """
    Test dataset classes with mock data

    Usage:
        cd backend/research
        python -m data.dataset
    """
    print("Testing SentimentDataset classes...\n")

    # Mock data
    texts = [
        "This movie is amazing!",
        "Terrible experience, would not recommend.",
        "It's okay, nothing special.",
        "Absolutely loved it! Best movie ever!",
        "Waste of time and money."
    ]

    labels = ['positive', 'negative', 'neutral', 'positive', 'negative']

    # Create vocabulary
    print("1. Building vocabulary...")
    vocab = Vocabulary(max_size=1000, min_freq=1)
    vocab.build_from_texts(texts)
    print(f"   Vocabulary size: {len(vocab)}")
    print(f"   Sample words: {list(vocab.word2idx.keys())[:10]}\n")

    # Create dataset
    print("2. Creating SentimentDataset...")
    dataset = SentimentDataset(texts, labels, vocab, max_len=50)
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Sample item: {dataset[0]}\n")

    # Test collate function
    print("3. Testing collate function...")
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=SentimentDataset.collate_fn
    )

    batch = next(iter(loader))
    print(f"   Batch keys: {batch.keys()}")
    print(f"   input_ids shape: {batch['input_ids'].shape}")
    print(f"   labels shape: {batch['labels'].shape}")
    print(f"   lengths: {batch['lengths']}")
    print(f"   texts: {batch['texts']}\n")

    print("[OK] All dataset tests passed!")
