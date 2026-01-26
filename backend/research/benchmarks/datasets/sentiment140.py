"""
Sentiment140 Dataset Loader

The Sentiment140 dataset contains 1.6 million tweets annotated for sentiment.
Labels: 0 = negative, 2 = neutral, 4 = positive

Source: http://help.sentiment140.com/for-students

Citation:
    Go, A., Bhayani, R., & Huang, L. (2009).
    Twitter sentiment classification using distant supervision.
    CS224N Project Report, Stanford.

Author: [Your Name]
"""

import os
import csv
from pathlib import Path
from typing import Tuple, Optional, List
import urllib.request
import zipfile

from ..base import Dataset, DatasetSplit, SentimentLabel


class Sentiment140Dataset(Dataset):
    """
    Sentiment140 Twitter Dataset.

    Contains 1.6M tweets with sentiment labels.
    Commonly used for Twitter sentiment analysis evaluation.

    Labels:
        0 -> Negative
        2 -> Neutral (not in original dataset)
        4 -> Positive

    Note: Original dataset only has positive/negative (binary).
    We map to 3-class by treating neutral separately if provided.

    Parameters
    ----------
    data_dir : str
        Directory for data storage
    max_samples : int
        Maximum samples to load (useful for testing)
    binary : bool
        If True, use only positive/negative (ignore neutral)
    """

    URL = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"

    def __init__(
        self,
        data_dir: str = None,
        max_samples: int = None,
        binary: bool = True,
        download: bool = True
    ):
        super().__init__(
            name="sentiment140",
            data_dir=data_dir,
            download=download
        )

        self.max_samples = max_samples
        self.binary = binary
        self.n_classes = 2 if binary else 3

        self.description = (
            "Sentiment140 contains 1.6M tweets labeled for sentiment "
            "using distant supervision (emoticons)."
        )
        self.citation = (
            "Go, A., Bhayani, R., & Huang, L. (2009). "
            "Twitter sentiment classification using distant supervision."
        )
        self.url = "http://help.sentiment140.com/for-students"

    def _download_data(self) -> None:
        """Download Sentiment140 dataset."""
        os.makedirs(self.data_dir, exist_ok=True)

        zip_path = self.data_dir / "sentiment140.zip"

        if not zip_path.exists():
            print(f"Downloading Sentiment140 dataset...")
            print(f"This may take a while (240MB)...")
            try:
                urllib.request.urlretrieve(self.URL, zip_path)
            except Exception as e:
                print(f"Download failed: {e}")
                print("Please download manually from: http://help.sentiment140.com/for-students")
                print(f"And extract to: {self.data_dir}")
                return

        # Extract
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.data_dir)

    def _data_exists(self) -> bool:
        """Check if data files exist."""
        train_file = self.data_dir / "training.1600000.processed.noemoticon.csv"
        test_file = self.data_dir / "testdata.manual.2009.06.14.csv"
        return train_file.exists() or (self.data_dir / "train.csv").exists()

    def _load_data(self) -> Tuple[DatasetSplit, Optional[DatasetSplit], DatasetSplit]:
        """Load Sentiment140 data."""
        # Try different file names
        train_files = [
            self.data_dir / "training.1600000.processed.noemoticon.csv",
            self.data_dir / "train.csv",
        ]
        test_files = [
            self.data_dir / "testdata.manual.2009.06.14.csv",
            self.data_dir / "test.csv",
        ]

        train_file = None
        for f in train_files:
            if f.exists():
                train_file = f
                break

        test_file = None
        for f in test_files:
            if f.exists():
                test_file = f
                break

        # Load training data
        train_texts, train_labels = self._load_csv(train_file, is_train=True)

        # Load test data
        if test_file and test_file.exists():
            test_texts, test_labels = self._load_csv(test_file, is_train=False)
        else:
            # Create test split from train
            split_idx = int(len(train_texts) * 0.9)
            test_texts = train_texts[split_idx:]
            test_labels = train_labels[split_idx:]
            train_texts = train_texts[:split_idx]
            train_labels = train_labels[:split_idx]

        # Create validation split from train
        val_idx = int(len(train_texts) * 0.9)
        val_texts = train_texts[val_idx:]
        val_labels = train_labels[val_idx:]
        train_texts = train_texts[:val_idx]
        train_labels = train_labels[:val_idx]

        train_split = DatasetSplit(train_texts, train_labels, name='train')
        val_split = DatasetSplit(val_texts, val_labels, name='val')
        test_split = DatasetSplit(test_texts, test_labels, name='test')

        return train_split, val_split, test_split

    def _load_csv(
        self,
        filepath: Path,
        is_train: bool = True
    ) -> Tuple[List[str], List[SentimentLabel]]:
        """Load data from CSV file."""
        texts = []
        labels = []

        if filepath is None or not filepath.exists():
            # Return empty if file doesn't exist
            return texts, labels

        # CSV format: polarity, id, date, query, user, text
        # polarity: 0 = negative, 2 = neutral, 4 = positive
        encoding = 'latin-1'  # Sentiment140 uses latin-1

        with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if self.max_samples and i >= self.max_samples:
                    break

                if len(row) < 6:
                    continue

                polarity = int(row[0])
                text = row[5]

                # Map polarity to label
                if polarity == 0:
                    label = SentimentLabel.NEGATIVE
                elif polarity == 4:
                    label = SentimentLabel.POSITIVE
                else:  # polarity == 2
                    if self.binary:
                        continue  # Skip neutral in binary mode
                    label = SentimentLabel.NEUTRAL

                texts.append(text)
                labels.append(label)

        return texts, labels

    def get_sample_texts(self, n: int = 5) -> List[Tuple[str, str]]:
        """Get sample texts for display."""
        samples = []
        for split_name in ['train', 'test']:
            split = getattr(self, split_name)
            for text, label in list(split)[:n]:
                samples.append((text, label.to_string()))
        return samples
