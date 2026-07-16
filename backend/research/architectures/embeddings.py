"""
Embedding Layer Utilities for Pre-trained Embeddings

Handles loading and initialization of pre-trained word embeddings (GloVe, FastText)
for the hybrid sentiment analysis model.

Supports:
- GloVe embeddings (glove.6B.50d, 100d, 200d, 300d)
- FastText embeddings
- Custom embedding initialization strategies
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import pickle


def load_glove_embeddings(glove_path, embedding_dim=300):
    """
    Load GloVe embeddings from file.

    Args:
        glove_path (str): Path to GloVe file (e.g., 'glove.6B.300d.txt')
        embedding_dim (int): Expected embedding dimension

    Returns:
        embeddings_dict: Dictionary mapping word -> embedding vector

    Example:
        >>> embeddings = load_glove_embeddings('embeddings/glove.6B.300d.txt')
        >>> print(f"Loaded {len(embeddings)} word vectors")
    """
    embeddings_dict = {}

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')

            if len(vector) == embedding_dim:
                embeddings_dict[word] = vector

    print(f"Loaded {len(embeddings_dict)} word vectors from {glove_path}")
    return embeddings_dict


def create_embedding_matrix(word2idx, embeddings_dict, embedding_dim=300):
    """
    Create embedding matrix from vocabulary and pre-trained embeddings.

    Args:
        word2idx (dict): Vocabulary mapping word -> index
        embeddings_dict (dict): Pre-trained embeddings word -> vector
        embedding_dim (int): Embedding dimension

    Returns:
        embedding_matrix: NumPy array (vocab_size, embedding_dim)
        coverage_stats: Dict with coverage statistics

    Special tokens:
        - Index 0 (<PAD>): Initialized with zeros
        - Index 1 (<UNK>): Initialized with mean of all vectors
        - Out-of-vocabulary words: Random initialization N(0, 0.01)
    """
    vocab_size = len(word2idx)
    embedding_matrix = np.random.normal(0, 0.01, (vocab_size, embedding_dim)).astype('float32')

    # Initialize <PAD> with zeros
    embedding_matrix[0] = np.zeros(embedding_dim)

    # Collect all pre-trained vectors for <UNK> initialization
    all_vectors = []
    found = 0

    for word, idx in word2idx.items():
        if word in embeddings_dict:
            embedding_matrix[idx] = embeddings_dict[word]
            all_vectors.append(embeddings_dict[word])
            found += 1

    # Initialize <UNK> (index 1) with mean of all vectors
    if all_vectors and 1 < vocab_size:
        embedding_matrix[1] = np.mean(all_vectors, axis=0)

    coverage = found / vocab_size if vocab_size > 0 else 0

    coverage_stats = {
        'vocab_size': vocab_size,
        'found_in_pretrained': found,
        'coverage': coverage,
        'oov_words': vocab_size - found
    }

    print(f"Embedding Matrix Created:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Found in pre-trained: {found} ({coverage:.2%})")
    print(f"  Out-of-vocabulary: {vocab_size - found}")

    return embedding_matrix, coverage_stats


def save_embeddings_cache(embedding_matrix, cache_path):
    """Save embedding matrix to cache for faster loading."""
    np.save(cache_path, embedding_matrix)
    print(f"Saved embedding matrix to {cache_path}")


def load_embeddings_cache(cache_path):
    """Load embedding matrix from cache."""
    embedding_matrix = np.load(cache_path)
    print(f"Loaded embedding matrix from {cache_path}")
    return embedding_matrix


def initialize_embedding_layer(embedding_layer, embedding_matrix, freeze=False):
    """
    Initialize PyTorch embedding layer with pre-trained embeddings.

    Args:
        embedding_layer (nn.Embedding): PyTorch embedding layer
        embedding_matrix (np.ndarray): Pre-trained embedding matrix
        freeze (bool): Whether to freeze embeddings during training

    Example:
        >>> embedding = nn.Embedding(vocab_size, 300, padding_idx=0)
        >>> initialize_embedding_layer(embedding, embedding_matrix, freeze=False)
    """
    embedding_tensor = torch.FloatTensor(embedding_matrix)
    embedding_layer.weight.data.copy_(embedding_tensor)
    embedding_layer.weight.requires_grad = not freeze

    print(f"Initialized embedding layer: freeze={freeze}")


class EmbeddingManager:
    """
    Manager class for handling embeddings throughout the training pipeline.

    This class provides a unified interface for:
    - Loading pre-trained embeddings
    - Building embedding matrices from vocabulary
    - Caching embeddings for faster loading
    - Initializing PyTorch embedding layers

    Args:
        embedding_path (str): Path to pre-trained embeddings
        embedding_dim (int): Embedding dimension
        cache_dir (str): Directory for caching embeddings

    Example:
        >>> manager = EmbeddingManager(
        ...     embedding_path='embeddings/glove.6B.300d.txt',
        ...     embedding_dim=300,
        ...     cache_dir='models/hybrid_dl'
        ... )
        >>> embedding_matrix = manager.build_embedding_matrix(word2idx)
        >>> manager.initialize_model_embedding(model.embedding)
    """

    def __init__(self, embedding_path, embedding_dim=300, cache_dir=None):
        self.embedding_path = Path(embedding_path)
        self.embedding_dim = embedding_dim
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.embeddings_dict = None
        self.embedding_matrix = None
        self.coverage_stats = None

    def load_embeddings(self, force_reload=False):
        """Load pre-trained embeddings from file."""
        if self.embeddings_dict is None or force_reload:
            print(f"Loading embeddings from {self.embedding_path}...")
            self.embeddings_dict = load_glove_embeddings(
                str(self.embedding_path),
                self.embedding_dim
            )
        return self.embeddings_dict

    def build_embedding_matrix(self, word2idx, use_cache=True):
        """
        Build embedding matrix for the given vocabulary.

        Args:
            word2idx (dict): Vocabulary mapping
            use_cache (bool): Whether to use cached embeddings

        Returns:
            embedding_matrix: NumPy array (vocab_size, embedding_dim)
        """
        # Check for cached embedding matrix
        if use_cache and self.cache_dir:
            cache_path = self.cache_dir / f'embedding_matrix_{len(word2idx)}.npy'
            if cache_path.exists():
                print(f"Loading cached embedding matrix from {cache_path}")
                self.embedding_matrix = load_embeddings_cache(cache_path)
                return self.embedding_matrix

        # Load embeddings if not already loaded
        if self.embeddings_dict is None:
            self.load_embeddings()

        # Create embedding matrix
        self.embedding_matrix, self.coverage_stats = create_embedding_matrix(
            word2idx, self.embeddings_dict, self.embedding_dim
        )

        # Save to cache if enabled
        if use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f'embedding_matrix_{len(word2idx)}.npy'
            save_embeddings_cache(self.embedding_matrix, cache_path)

        return self.embedding_matrix

    def initialize_model_embedding(self, embedding_layer, freeze=False):
        """Initialize a model's embedding layer with the built embedding matrix."""
        if self.embedding_matrix is None:
            raise ValueError("Embedding matrix not built. Call build_embedding_matrix() first.")

        initialize_embedding_layer(embedding_layer, self.embedding_matrix, freeze)

    def get_coverage_stats(self):
        """Get embedding coverage statistics."""
        return self.coverage_stats


if __name__ == "__main__":
    # Unit test for embedding utilities
    print("Testing Embedding Utilities...\n")

    # Create mock vocabulary
    word2idx = {
        '<PAD>': 0,
        '<UNK>': 1,
        'good': 2,
        'bad': 3,
        'movie': 4,
        'xyz123': 5,  # OOV word
    }

    # Create mock pre-trained embeddings
    embeddings_dict = {
        'good': np.random.randn(300).astype('float32'),
        'bad': np.random.randn(300).astype('float32'),
        'movie': np.random.randn(300).astype('float32'),
    }

    # Test embedding matrix creation
    embedding_matrix, stats = create_embedding_matrix(word2idx, embeddings_dict, 300)

    print(f"\nEmbedding matrix shape: {embedding_matrix.shape}")
    print(f"Coverage: {stats['coverage']:.2%}")
    assert embedding_matrix.shape == (len(word2idx), 300)
    assert np.allclose(embedding_matrix[0], np.zeros(300)), "<PAD> should be zeros"
    assert stats['found_in_pretrained'] == 3, "Should find 3 words"

    # Test embedding layer initialization
    embedding_layer = nn.Embedding(len(word2idx), 300, padding_idx=0)
    initialize_embedding_layer(embedding_layer, embedding_matrix, freeze=False)

    print(f"\nEmbedding layer weight shape: {embedding_layer.weight.shape}")
    print(f"Embedding layer requires_grad: {embedding_layer.weight.requires_grad}")
    assert embedding_layer.weight.shape == (len(word2idx), 300)

    print("\n✓ All embedding utility tests passed!")
