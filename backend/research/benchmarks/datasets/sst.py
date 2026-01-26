"""
Stanford Sentiment Treebank (SST) Dataset Loader

SST is a widely used benchmark for sentiment analysis with
fine-grained sentiment labels at phrase level.

Variants:
- SST-2: Binary (positive/negative)
- SST-5: Fine-grained (very negative, negative, neutral, positive, very positive)

Source: https://nlp.stanford.edu/sentiment/

Citation:
    Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D.,
    Ng, A. Y., & Potts, C. (2013).
    Recursive deep models for semantic compositionality over a sentiment treebank.
    EMNLP 2013.

Author: [Your Name]
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List
import urllib.request
import zipfile

from ..base import Dataset, DatasetSplit, SentimentLabel


class SSTDataset(Dataset):
    """
    Stanford Sentiment Treebank Dataset.

    Sentence-level sentiment classification from movie reviews.
    Standard benchmark for sentiment analysis models.

    Variants:
        SST-2: Binary classification (positive/negative)
        SST-5: 5-way classification (mapped to 3-class)

    Parameters
    ----------
    data_dir : str
        Directory for data storage
    version : str
        'sst2' for binary, 'sst5' for 5-class
    granularity : str
        'sentence' or 'phrase' (phrase includes subphrases)
    """

    SST2_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"

    def __init__(
        self,
        data_dir: str = None,
        version: str = 'sst2',
        granularity: str = 'sentence',
        download: bool = True
    ):
        super().__init__(
            name=f"sst_{version}",
            data_dir=data_dir,
            download=download
        )

        self.version = version
        self.granularity = granularity
        self.n_classes = 2 if version == 'sst2' else 3

        self.description = (
            f"Stanford Sentiment Treebank ({version.upper()}). "
            "Movie review sentences with sentiment labels."
        )
        self.citation = (
            "Socher et al. (2013). Recursive deep models for semantic "
            "compositionality over a sentiment treebank. EMNLP 2013."
        )
        self.url = "https://nlp.stanford.edu/sentiment/"

    def _download_data(self) -> None:
        """Download SST dataset."""
        os.makedirs(self.data_dir, exist_ok=True)

        if self.version == 'sst2':
            zip_path = self.data_dir / "SST-2.zip"
            if not zip_path.exists():
                print("Downloading SST-2 dataset...")
                try:
                    urllib.request.urlretrieve(self.SST2_URL, zip_path)
                except Exception as e:
                    print(f"Download failed: {e}")
                    print("Please download manually or use Hugging Face:")
                    print("  from datasets import load_dataset")
                    print("  ds = load_dataset('sst2')")
                    return

            # Extract
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(self.data_dir)
        else:
            print("SST-5 requires manual download from:")
            print("  https://nlp.stanford.edu/sentiment/")

    def _data_exists(self) -> bool:
        """Check if data files exist."""
        sst_dir = self.data_dir / "SST-2"
        return sst_dir.exists() or (self.data_dir / "train.tsv").exists()

    def _load_data(self) -> Tuple[DatasetSplit, Optional[DatasetSplit], DatasetSplit]:
        """Load SST data."""
        if self.version == 'sst2':
            return self._load_sst2()
        else:
            return self._load_sst5()

    def _load_sst2(self) -> Tuple[DatasetSplit, Optional[DatasetSplit], DatasetSplit]:
        """Load SST-2 binary dataset."""
        sst_dir = self.data_dir / "SST-2"
        if not sst_dir.exists():
            sst_dir = self.data_dir

        # Load train
        train_texts, train_labels = self._load_tsv(sst_dir / "train.tsv")

        # Load dev as validation
        val_texts, val_labels = self._load_tsv(sst_dir / "dev.tsv")

        # SST-2 test set doesn't have labels in GLUE format
        # Use dev as test, create val from train
        if not val_texts:
            val_idx = int(len(train_texts) * 0.9)
            val_texts = train_texts[val_idx:]
            val_labels = train_labels[val_idx:]
            train_texts = train_texts[:val_idx]
            train_labels = train_labels[:val_idx]

        test_texts, test_labels = val_texts, val_labels

        return (
            DatasetSplit(train_texts, train_labels, name='train'),
            DatasetSplit(val_texts, val_labels, name='val'),
            DatasetSplit(test_texts, test_labels, name='test')
        )

    def _load_tsv(self, filepath: Path) -> Tuple[List[str], List[SentimentLabel]]:
        """Load from TSV file."""
        texts = []
        labels = []

        if not filepath.exists():
            return texts, labels

        with open(filepath, 'r', encoding='utf-8') as f:
            # Skip header
            header = f.readline()

            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    text = parts[0]
                    try:
                        label_val = int(parts[1])
                        # SST-2: 0 = negative, 1 = positive
                        label = SentimentLabel.POSITIVE if label_val == 1 else SentimentLabel.NEGATIVE
                        texts.append(text)
                        labels.append(label)
                    except (ValueError, IndexError):
                        continue

        return texts, labels

    def _load_sst5(self) -> Tuple[DatasetSplit, Optional[DatasetSplit], DatasetSplit]:
        """Load SST-5 fine-grained dataset."""
        # SST-5 maps:
        # 0: very negative -> NEGATIVE
        # 1: negative -> NEGATIVE
        # 2: neutral -> NEUTRAL
        # 3: positive -> POSITIVE
        # 4: very positive -> POSITIVE

        # Try to load from local files
        train_texts, train_labels = [], []
        test_texts, test_labels = [], []

        # Check for phrase-level files
        phrases_file = self.data_dir / "dictionary.txt"
        labels_file = self.data_dir / "sentiment_labels.txt"

        if phrases_file.exists() and labels_file.exists():
            train_texts, train_labels = self._load_sst5_files(
                phrases_file, labels_file
            )
        else:
            # Generate sample data for testing
            print("SST-5 data not found. Generating sample data...")
            train_texts, train_labels = self._generate_sample_data(5000)
            test_texts, test_labels = self._generate_sample_data(1000)

        if not test_texts:
            # Split train for test
            split_idx = int(len(train_texts) * 0.8)
            test_texts = train_texts[split_idx:]
            test_labels = train_labels[split_idx:]
            train_texts = train_texts[:split_idx]
            train_labels = train_labels[:split_idx]

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

    def _load_sst5_files(
        self,
        phrases_file: Path,
        labels_file: Path
    ) -> Tuple[List[str], List[SentimentLabel]]:
        """Load SST-5 from original files."""
        # Load phrase dictionary
        phrases = {}
        with open(phrases_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 2:
                    phrase, idx = parts
                    phrases[int(idx)] = phrase

        # Load labels
        texts = []
        labels = []
        with open(labels_file, 'r', encoding='utf-8') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('|')
                if len(parts) == 2:
                    idx, score = int(parts[0]), float(parts[1])
                    if idx in phrases:
                        text = phrases[idx]
                        label = self._score_to_label(score)
                        texts.append(text)
                        labels.append(label)

        return texts, labels

    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert SST-5 score to 3-class label."""
        if score <= 0.4:
            return SentimentLabel.NEGATIVE
        elif score <= 0.6:
            return SentimentLabel.NEUTRAL
        else:
            return SentimentLabel.POSITIVE

    def _generate_sample_data(
        self,
        n_samples: int
    ) -> Tuple[List[str], List[SentimentLabel]]:
        """Generate synthetic movie review data."""
        import random
        random.seed(42)

        positive = [
            "An excellent film with outstanding performances.",
            "Brilliant storytelling and beautiful cinematography.",
            "A masterpiece that will be remembered for generations.",
            "Captivating from start to finish.",
            "One of the best movies I've ever seen.",
        ]

        negative = [
            "A complete waste of time.",
            "Terrible acting and poor script.",
            "Boring and predictable plot.",
            "One of the worst films ever made.",
            "Disappointing on every level.",
        ]

        neutral = [
            "An average film with some good moments.",
            "Neither great nor terrible.",
            "Decent but forgettable.",
            "It has its moments but nothing special.",
            "Okay for a one-time watch.",
        ]

        texts = []
        labels = []

        for _ in range(n_samples):
            r = random.random()
            if r < 0.4:
                texts.append(random.choice(positive))
                labels.append(SentimentLabel.POSITIVE)
            elif r < 0.8:
                texts.append(random.choice(negative))
                labels.append(SentimentLabel.NEGATIVE)
            else:
                texts.append(random.choice(neutral))
                labels.append(SentimentLabel.NEUTRAL)

        return texts, labels
