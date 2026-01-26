"""
Amazon Reviews Dataset Loader

Amazon product reviews with star ratings that can be mapped to sentiment.

Various subsets available:
- Amazon Review Full (5 classes)
- Amazon Review Polarity (2 classes)

Source: https://huggingface.co/datasets/amazon_polarity

Citation:
    McAuley, J., & Leskovec, J. (2013).
    Hidden factors and hidden topics: understanding rating dimensions with review text.
    RecSys 2013.

Author: [Your Name]
"""

import os
import json
import gzip
from pathlib import Path
from typing import Tuple, Optional, List
import urllib.request

from ..base import Dataset, DatasetSplit, SentimentLabel


class AmazonReviewsDataset(Dataset):
    """
    Amazon Product Reviews Dataset.

    Large-scale product reviews with star ratings.
    Star ratings are mapped to sentiment:
        1-2 stars -> Negative
        3 stars -> Neutral
        4-5 stars -> Positive

    Parameters
    ----------
    data_dir : str
        Directory for data storage
    max_samples : int
        Maximum samples to load per split
    category : str
        Product category (e.g., 'Electronics', 'Books')
    binary : bool
        If True, use binary classification (skip neutral)
    """

    # Sample data URLs (small subsets for testing)
    # Full datasets require Hugging Face or Kaggle download
    SAMPLE_URL = None  # Would need proper hosting

    def __init__(
        self,
        data_dir: str = None,
        max_samples: int = 10000,
        category: str = "all",
        binary: bool = False,
        download: bool = True
    ):
        super().__init__(
            name="amazon_reviews",
            data_dir=data_dir,
            download=download
        )

        self.max_samples = max_samples
        self.category = category
        self.binary = binary
        self.n_classes = 2 if binary else 3

        self.description = (
            "Amazon product reviews with star ratings mapped to sentiment. "
            "One of the largest sentiment datasets available."
        )
        self.citation = (
            "McAuley & Leskovec (2013). Hidden factors and hidden topics. RecSys 2013."
        )
        self.url = "https://nijianmo.github.io/amazon/index.html"

    def _download_data(self) -> None:
        """Download Amazon reviews dataset."""
        os.makedirs(self.data_dir, exist_ok=True)
        print("Amazon Reviews dataset requires manual download.")
        print(f"Please download from: {self.url}")
        print(f"And place in: {self.data_dir}")
        print("\nAlternatively, use Hugging Face datasets:")
        print("  pip install datasets")
        print("  from datasets import load_dataset")
        print("  ds = load_dataset('amazon_polarity')")

    def _data_exists(self) -> bool:
        """Check if data files exist."""
        # Check for various possible file formats
        possible_files = [
            self.data_dir / "train.json",
            self.data_dir / "train.jsonl",
            self.data_dir / "amazon_train.csv",
        ]
        return any(f.exists() for f in possible_files)

    def _load_data(self) -> Tuple[DatasetSplit, Optional[DatasetSplit], DatasetSplit]:
        """Load Amazon reviews data."""
        # Try to load from local files
        train_texts, train_labels = self._try_load_local('train')
        test_texts, test_labels = self._try_load_local('test')

        # If no local data, generate synthetic for testing
        if not train_texts:
            print("No local Amazon data found. Generating synthetic sample data...")
            train_texts, train_labels = self._generate_sample_data(self.max_samples)
            test_texts, test_labels = self._generate_sample_data(self.max_samples // 5)

        # Create validation split
        val_idx = int(len(train_texts) * 0.9)
        val_texts = train_texts[val_idx:]
        val_labels = train_labels[val_idx:]
        train_texts = train_texts[:val_idx]
        train_labels = train_labels[:val_idx]

        return (
            DatasetSplit(train_texts, train_labels, name='train'),
            DatasetSplit(val_texts, val_labels, name='val'),
            DatasetSplit(test_texts, test_labels, name='test')
        )

    def _try_load_local(self, split: str) -> Tuple[List[str], List[SentimentLabel]]:
        """Try to load from local files."""
        texts = []
        labels = []

        # Try JSON format
        json_file = self.data_dir / f"{split}.json"
        if json_file.exists():
            return self._load_json(json_file)

        # Try JSONL format
        jsonl_file = self.data_dir / f"{split}.jsonl"
        if jsonl_file.exists():
            return self._load_jsonl(jsonl_file)

        return texts, labels

    def _load_json(self, filepath: Path) -> Tuple[List[str], List[SentimentLabel]]:
        """Load from JSON file."""
        texts = []
        labels = []

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data[:self.max_samples]:
            text = item.get('text', item.get('reviewText', ''))
            rating = item.get('rating', item.get('overall', 3))

            label = self._rating_to_label(rating)
            if label is None and self.binary:
                continue

            texts.append(text)
            labels.append(label or SentimentLabel.NEUTRAL)

        return texts, labels

    def _load_jsonl(self, filepath: Path) -> Tuple[List[str], List[SentimentLabel]]:
        """Load from JSONL file."""
        texts = []
        labels = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if self.max_samples and i >= self.max_samples:
                    break

                item = json.loads(line)
                text = item.get('text', item.get('reviewText', ''))
                rating = item.get('rating', item.get('overall', 3))

                label = self._rating_to_label(rating)
                if label is None and self.binary:
                    continue

                texts.append(text)
                labels.append(label or SentimentLabel.NEUTRAL)

        return texts, labels

    def _rating_to_label(self, rating: float) -> Optional[SentimentLabel]:
        """Convert star rating to sentiment label."""
        rating = float(rating)
        if rating <= 2:
            return SentimentLabel.NEGATIVE
        elif rating >= 4:
            return SentimentLabel.POSITIVE
        else:  # rating == 3
            return None if self.binary else SentimentLabel.NEUTRAL

    def _generate_sample_data(
        self,
        n_samples: int
    ) -> Tuple[List[str], List[SentimentLabel]]:
        """Generate synthetic sample data for testing."""
        import random
        random.seed(42)

        positive_templates = [
            "Great product! Exactly what I needed. {adj} quality.",
            "Excellent! Works perfectly. {adj} purchase.",
            "Love it! Would definitely recommend. {adj} value.",
            "Amazing quality. Very {adj}. Five stars!",
            "Perfect! {adj} product. Highly recommend.",
        ]

        negative_templates = [
            "Terrible product. Very {adj}. Don't buy.",
            "Waste of money. {adj} quality. Disappointed.",
            "Horrible experience. {adj} customer service.",
            "Broke after one week. Very {adj}.",
            "Not as described. {adj} and misleading.",
        ]

        neutral_templates = [
            "It's okay. Not great, not {adj}.",
            "Average product. Does what it's supposed to do.",
            "Nothing special. Pretty {adj}.",
            "Mediocre. Could be better, could be {adj}.",
            "Acceptable quality. Just {adj} enough.",
        ]

        pos_adj = ["excellent", "fantastic", "wonderful", "superb", "outstanding"]
        neg_adj = ["disappointing", "poor", "awful", "terrible", "bad"]
        neu_adj = ["average", "okay", "decent", "standard", "normal"]

        texts = []
        labels = []

        for _ in range(n_samples):
            r = random.random()
            if r < 0.4:  # 40% positive
                template = random.choice(positive_templates)
                adj = random.choice(pos_adj)
                label = SentimentLabel.POSITIVE
            elif r < 0.8:  # 40% negative
                template = random.choice(negative_templates)
                adj = random.choice(neg_adj)
                label = SentimentLabel.NEGATIVE
            else:  # 20% neutral
                if self.binary:
                    continue
                template = random.choice(neutral_templates)
                adj = random.choice(neu_adj)
                label = SentimentLabel.NEUTRAL

            text = template.format(adj=adj)
            texts.append(text)
            labels.append(label)

        return texts, labels
