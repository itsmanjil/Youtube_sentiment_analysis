"""
IMDB Movie Reviews Dataset Loader

The IMDB dataset contains 50,000 movie reviews for sentiment analysis.
25,000 for training and 25,000 for testing.

Source: https://ai.stanford.edu/~amaas/data/sentiment/

Citation:
    Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011).
    Learning word vectors for sentiment analysis.
    ACL 2011.

Author: [Your Name]
"""

import os
import tarfile
from pathlib import Path
from typing import Tuple, Optional, List
import urllib.request

from ..base import Dataset, DatasetSplit, SentimentLabel


class IMDBDataset(Dataset):
    """
    IMDB Movie Reviews Dataset.

    Contains 50K movie reviews with binary sentiment labels.
    This is a standard benchmark for sentiment analysis.

    Structure:
        - 25,000 training reviews (12,500 pos, 12,500 neg)
        - 25,000 test reviews (12,500 pos, 12,500 neg)

    Note: This is a binary classification dataset (positive/negative).
    No neutral class is available.

    Parameters
    ----------
    data_dir : str
        Directory for data storage
    max_samples : int
        Maximum samples per class to load
    include_unsup : bool
        Whether to include unsupervised reviews
    """

    URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    def __init__(
        self,
        data_dir: str = None,
        max_samples: int = None,
        include_unsup: bool = False,
        download: bool = True
    ):
        super().__init__(
            name="imdb",
            data_dir=data_dir,
            download=download
        )

        self.max_samples = max_samples
        self.include_unsup = include_unsup
        self.n_classes = 2  # Binary: positive/negative

        self.description = (
            "IMDB dataset contains 50,000 movie reviews for binary "
            "sentiment classification (positive/negative)."
        )
        self.citation = (
            "Maas et al. (2011). Learning word vectors for sentiment analysis. ACL 2011."
        )
        self.url = "https://ai.stanford.edu/~amaas/data/sentiment/"

    def _download_data(self) -> None:
        """Download IMDB dataset."""
        os.makedirs(self.data_dir, exist_ok=True)

        tar_path = self.data_dir / "aclImdb_v1.tar.gz"

        if not tar_path.exists():
            print(f"Downloading IMDB dataset...")
            print(f"This may take a while (84MB)...")
            try:
                urllib.request.urlretrieve(self.URL, tar_path)
            except Exception as e:
                print(f"Download failed: {e}")
                print(f"Please download manually from: {self.URL}")
                return

        # Extract
        print("Extracting...")
        with tarfile.open(tar_path, 'r:gz') as tf:
            tf.extractall(self.data_dir)

    def _data_exists(self) -> bool:
        """Check if data files exist."""
        imdb_dir = self.data_dir / "aclImdb"
        return imdb_dir.exists()

    def _load_data(self) -> Tuple[DatasetSplit, Optional[DatasetSplit], DatasetSplit]:
        """Load IMDB data."""
        imdb_dir = self.data_dir / "aclImdb"

        # Load training data
        train_texts, train_labels = self._load_split(imdb_dir / "train")

        # Load test data
        test_texts, test_labels = self._load_split(imdb_dir / "test")

        # Create validation split from training
        val_idx = int(len(train_texts) * 0.9)
        val_texts = train_texts[val_idx:]
        val_labels = train_labels[val_idx:]
        train_texts = train_texts[:val_idx]
        train_labels = train_labels[:val_idx]

        train_split = DatasetSplit(train_texts, train_labels, name='train')
        val_split = DatasetSplit(val_texts, val_labels, name='val')
        test_split = DatasetSplit(test_texts, test_labels, name='test')

        return train_split, val_split, test_split

    def _load_split(self, split_dir: Path) -> Tuple[List[str], List[SentimentLabel]]:
        """Load a single split (train or test)."""
        texts = []
        labels = []

        if not split_dir.exists():
            return texts, labels

        # Load positive reviews
        pos_dir = split_dir / "pos"
        if pos_dir.exists():
            pos_texts = self._load_from_dir(pos_dir)
            texts.extend(pos_texts)
            labels.extend([SentimentLabel.POSITIVE] * len(pos_texts))

        # Load negative reviews
        neg_dir = split_dir / "neg"
        if neg_dir.exists():
            neg_texts = self._load_from_dir(neg_dir)
            texts.extend(neg_texts)
            labels.extend([SentimentLabel.NEGATIVE] * len(neg_texts))

        # Shuffle
        import random
        combined = list(zip(texts, labels))
        random.seed(42)
        random.shuffle(combined)
        texts, labels = zip(*combined) if combined else ([], [])

        return list(texts), list(labels)

    def _load_from_dir(self, directory: Path) -> List[str]:
        """Load all text files from a directory."""
        texts = []
        files = sorted(directory.glob("*.txt"))

        if self.max_samples:
            files = files[:self.max_samples]

        for filepath in files:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                    # Remove HTML tags
                    text = self._clean_html(text)
                    texts.append(text)
            except Exception:
                continue

        return texts

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        import re
        # Remove <br> tags
        text = re.sub(r'<br\s*/?>', ' ', text)
        # Remove other HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()
