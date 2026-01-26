"""
Base Classes for Benchmark Datasets

Provides a unified interface for loading and managing
sentiment analysis benchmark datasets.

Author: [Your Name]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Iterator, Any
from enum import Enum
from pathlib import Path
import numpy as np
import json
import hashlib
import os


class SentimentLabel(Enum):
    """Standardized sentiment labels."""
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2

    @classmethod
    def from_string(cls, label: str) -> 'SentimentLabel':
        """Convert string label to SentimentLabel."""
        label_lower = label.lower().strip()
        if label_lower in ['negative', 'neg', '0', 'label_0']:
            return cls.NEGATIVE
        elif label_lower in ['neutral', 'neu', '1', 'label_1']:
            return cls.NEUTRAL
        elif label_lower in ['positive', 'pos', '2', 'label_2']:
            return cls.POSITIVE
        else:
            raise ValueError(f"Unknown label: {label}")

    def to_string(self) -> str:
        """Convert to string label."""
        return ['Negative', 'Neutral', 'Positive'][self.value]


@dataclass
class DatasetSplit:
    """
    A split of a dataset (train, validation, or test).

    Attributes:
        texts: List of text samples
        labels: List of sentiment labels
        metadata: Optional metadata for each sample
        name: Split name (train, val, test)
    """
    texts: List[str]
    labels: List[SentimentLabel]
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    name: str = "unknown"

    def __post_init__(self):
        if len(self.metadata) == 0:
            self.metadata = [{} for _ in self.texts]
        assert len(self.texts) == len(self.labels), "Texts and labels must have same length"

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, SentimentLabel]:
        return self.texts[idx], self.labels[idx]

    def __iter__(self) -> Iterator[Tuple[str, SentimentLabel]]:
        for text, label in zip(self.texts, self.labels):
            yield text, label

    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels."""
        dist = {label.to_string(): 0 for label in SentimentLabel}
        for label in self.labels:
            dist[label.to_string()] += 1
        return dist

    def get_texts_by_label(self, label: SentimentLabel) -> List[str]:
        """Get all texts with a specific label."""
        return [t for t, l in zip(self.texts, self.labels) if l == label]

    def sample(self, n: int, stratified: bool = True, seed: int = 42) -> 'DatasetSplit':
        """
        Sample n items from the split.

        Parameters
        ----------
        n : int
            Number of samples
        stratified : bool
            Whether to maintain label distribution
        seed : int
            Random seed

        Returns
        -------
        DatasetSplit
            Sampled split
        """
        np.random.seed(seed)

        if not stratified:
            indices = np.random.choice(len(self), min(n, len(self)), replace=False)
        else:
            # Stratified sampling
            indices = []
            label_indices = {label: [] for label in SentimentLabel}

            for i, label in enumerate(self.labels):
                label_indices[label].append(i)

            # Calculate samples per class
            total = sum(len(v) for v in label_indices.values())
            for label, idx_list in label_indices.items():
                n_samples = max(1, int(n * len(idx_list) / total))
                if len(idx_list) > 0:
                    sampled = np.random.choice(
                        idx_list,
                        min(n_samples, len(idx_list)),
                        replace=False
                    )
                    indices.extend(sampled)

            np.random.shuffle(indices)
            indices = indices[:n]

        return DatasetSplit(
            texts=[self.texts[i] for i in indices],
            labels=[self.labels[i] for i in indices],
            metadata=[self.metadata[i] for i in indices],
            name=f"{self.name}_sampled_{n}"
        )

    def to_arrays(self) -> Tuple[List[str], np.ndarray]:
        """Convert to arrays for sklearn compatibility."""
        return self.texts, np.array([l.value for l in self.labels])

    def to_string_labels(self) -> Tuple[List[str], List[str]]:
        """Convert to string labels for compatibility."""
        return self.texts, [l.to_string() for l in self.labels]


class Dataset(ABC):
    """
    Abstract base class for sentiment datasets.

    Subclass this to implement specific dataset loaders.
    """

    def __init__(
        self,
        name: str,
        data_dir: str = None,
        download: bool = True,
        cache: bool = True
    ):
        """
        Initialize dataset.

        Parameters
        ----------
        name : str
            Dataset name
        data_dir : str
            Directory for data storage
        download : bool
            Whether to download if not present
        cache : bool
            Whether to cache processed data
        """
        self.name = name
        self.data_dir = Path(data_dir or self._default_data_dir())
        self.download = download
        self.cache = cache

        # Splits
        self._train: Optional[DatasetSplit] = None
        self._val: Optional[DatasetSplit] = None
        self._test: Optional[DatasetSplit] = None

        # Metadata
        self.n_classes = 3  # Default: neg, neu, pos
        self.description = ""
        self.citation = ""
        self.url = ""

    def _default_data_dir(self) -> Path:
        """Get default data directory."""
        return Path(__file__).parent / "data" / self.name

    @abstractmethod
    def _load_data(self) -> Tuple[DatasetSplit, Optional[DatasetSplit], DatasetSplit]:
        """
        Load the dataset splits.

        Returns
        -------
        tuple
            (train_split, val_split, test_split)
            val_split may be None if dataset doesn't have validation set
        """
        pass

    def load(self, force_reload: bool = False) -> None:
        """
        Load the dataset.

        Parameters
        ----------
        force_reload : bool
            Force reload even if cached
        """
        cache_path = self.data_dir / "cache.json"

        # Try loading from cache
        if self.cache and cache_path.exists() and not force_reload:
            try:
                self._load_from_cache(cache_path)
                return
            except Exception:
                pass

        # Download if needed
        if self.download and not self._data_exists():
            self._download_data()

        # Load data
        self._train, self._val, self._test = self._load_data()

        # Save to cache
        if self.cache:
            self._save_to_cache(cache_path)

    def _data_exists(self) -> bool:
        """Check if data files exist."""
        return self.data_dir.exists()

    def _download_data(self) -> None:
        """Download dataset (override in subclass)."""
        os.makedirs(self.data_dir, exist_ok=True)

    def _load_from_cache(self, cache_path: Path) -> None:
        """Load from cache file."""
        with open(cache_path, 'r') as f:
            data = json.load(f)

        self._train = self._dict_to_split(data['train'], 'train')
        if data.get('val'):
            self._val = self._dict_to_split(data['val'], 'val')
        self._test = self._dict_to_split(data['test'], 'test')

    def _save_to_cache(self, cache_path: Path) -> None:
        """Save to cache file."""
        os.makedirs(cache_path.parent, exist_ok=True)

        data = {
            'train': self._split_to_dict(self._train),
            'val': self._split_to_dict(self._val) if self._val else None,
            'test': self._split_to_dict(self._test),
        }

        with open(cache_path, 'w') as f:
            json.dump(data, f)

    def _split_to_dict(self, split: DatasetSplit) -> Dict:
        """Convert split to dictionary for JSON."""
        return {
            'texts': split.texts,
            'labels': [l.value for l in split.labels],
            'name': split.name,
        }

    def _dict_to_split(self, d: Dict, name: str) -> DatasetSplit:
        """Convert dictionary to split."""
        return DatasetSplit(
            texts=d['texts'],
            labels=[SentimentLabel(l) for l in d['labels']],
            name=name
        )

    @property
    def train(self) -> DatasetSplit:
        """Get training split."""
        if self._train is None:
            self.load()
        return self._train

    @property
    def val(self) -> Optional[DatasetSplit]:
        """Get validation split."""
        if self._train is None:
            self.load()
        return self._val

    @property
    def test(self) -> DatasetSplit:
        """Get test split."""
        if self._test is None:
            self.load()
        return self._test

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        info = {
            'name': self.name,
            'description': self.description,
            'n_classes': self.n_classes,
            'url': self.url,
        }

        if self._train:
            info['train_size'] = len(self._train)
            info['train_distribution'] = self._train.get_label_distribution()
        if self._test:
            info['test_size'] = len(self._test)
            info['test_distribution'] = self._test.get_label_distribution()
        if self._val:
            info['val_size'] = len(self._val)

        return info

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class DatasetManager:
    """
    Manager for multiple datasets.

    Provides unified access to all benchmark datasets.

    Example
    -------
    >>> manager = DatasetManager()
    >>> manager.register('sentiment140', Sentiment140Dataset())
    >>> manager.register('imdb', IMDBDataset())
    >>>
    >>> for name, dataset in manager.datasets():
    ...     print(f"{name}: {len(dataset.train)} samples")
    """

    def __init__(self):
        self._datasets: Dict[str, Dataset] = {}

    def register(self, name: str, dataset: Dataset) -> None:
        """Register a dataset."""
        self._datasets[name] = dataset

    def get(self, name: str) -> Dataset:
        """Get a dataset by name."""
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not registered")
        return self._datasets[name]

    def list_datasets(self) -> List[str]:
        """List registered dataset names."""
        return list(self._datasets.keys())

    def datasets(self) -> Iterator[Tuple[str, Dataset]]:
        """Iterate over all datasets."""
        for name, dataset in self._datasets.items():
            yield name, dataset

    def load_all(self) -> None:
        """Load all registered datasets."""
        for name, dataset in self._datasets.items():
            print(f"Loading {name}...")
            dataset.load()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all datasets."""
        return {name: ds.get_info() for name, ds in self._datasets.items()}
