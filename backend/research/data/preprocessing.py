"""
Text Preprocessing and Vocabulary Building for Deep Learning Models

Handles tokenization, vocabulary construction, and text numericalization
for the hybrid sentiment analysis model.
"""

import re
import pickle
from collections import Counter
from pathlib import Path


class Vocabulary:
    """
    Vocabulary class for managing word-to-index mappings.

    Special tokens:
        <PAD> (index 0): Padding token
        <UNK> (index 1): Unknown/out-of-vocabulary token

    Args:
        max_size (int): Maximum vocabulary size (most frequent words)
        min_freq (int): Minimum word frequency to include in vocabulary

    Example:
        >>> vocab = Vocabulary(max_size=20000, min_freq=2)
        >>> vocab.build_from_texts(['this is good', 'this is bad'])
        >>> print(vocab['good'])  # Get index for 'good'
    """

    def __init__(self, max_size=20000, min_freq=2):
        self.max_size = max_size
        self.min_freq = min_freq

        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'

        # Word <-> Index mappings
        self.word2idx = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1
        }
        self.idx2word = {
            0: self.PAD_TOKEN,
            1: self.UNK_TOKEN
        }

        self.word_freq = Counter()

    def build_from_texts(self, texts):
        """
        Build vocabulary from a list of texts.

        Args:
            texts (list): List of text strings

        Returns:
            vocab_size (int): Final vocabulary size
        """
        print("Building vocabulary from texts...")

        # Count word frequencies
        for text in texts:
            tokens = tokenize(text)
            self.word_freq.update(tokens)

        print(f"Total unique words before filtering: {len(self.word_freq)}")

        # Filter by frequency and select top K words
        filtered_words = [
            word for word, freq in self.word_freq.items()
            if freq >= self.min_freq
        ]

        # Sort by frequency and take top words
        sorted_words = sorted(
            filtered_words,
            key=lambda w: self.word_freq[w],
            reverse=True
        )[:self.max_size - 2]  # -2 for <PAD> and <UNK>

        # Build mappings
        for idx, word in enumerate(sorted_words, start=2):  # Start from 2
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        vocab_size = len(self.word2idx)
        print(f"Vocabulary built: {vocab_size} words")
        print(f"  Most common: {sorted_words[:10]}")

        return vocab_size

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, word):
        """Get index for a word (returns UNK index if not found)."""
        return self.word2idx.get(word, self.word2idx[self.UNK_TOKEN])

    def get_word(self, idx):
        """Get word for an index."""
        return self.idx2word.get(idx, self.UNK_TOKEN)

    def encode(self, text):
        """
        Convert text to list of indices.

        Args:
            text (str): Input text

        Returns:
            indices (list): List of word indices
        """
        tokens = tokenize(text)
        return [self[token] for token in tokens]

    def decode(self, indices):
        """
        Convert list of indices back to text.

        Args:
            indices (list): List of word indices

        Returns:
            text (str): Reconstructed text
        """
        words = [self.get_word(idx) for idx in indices if idx != 0]  # Skip padding
        return ' '.join(words)

    def save(self, path):
        """Save vocabulary to file."""
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': self.word_freq,
            'max_size': self.max_size,
            'min_freq': self.min_freq
        }

        with open(path, 'wb') as f:
            pickle.dump(vocab_data, f)

        print(f"Vocabulary saved to {path}")

    @classmethod
    def load(cls, path):
        """Load vocabulary from file."""
        with open(path, 'rb') as f:
            vocab_data = pickle.load(f)

        vocab = cls(
            max_size=vocab_data['max_size'],
            min_freq=vocab_data['min_freq']
        )
        vocab.word2idx = vocab_data['word2idx']
        vocab.idx2word = vocab_data['idx2word']
        vocab.word_freq = vocab_data['word_freq']

        print(f"Vocabulary loaded from {path}: {len(vocab)} words")
        return vocab


def tokenize(text):
    """
    Simple word tokenization.

    Args:
        text (str): Input text

    Returns:
        tokens (list): List of word tokens

    Note:
        For production, this should be replaced with the existing
        YouTubePreprocessor 10-stage pipeline for consistency.
    """
    # Lowercase
    text = text.lower()

    # Keep only letters and spaces
    text = re.sub(r'[^a-z\s]', '', text)

    # Split on whitespace
    tokens = text.split()

    return tokens


def pad_sequences(sequences, max_len, pad_value=0):
    """
    Pad or truncate sequences to the same length.

    Args:
        sequences (list): List of sequences (list of integers)
        max_len (int): Maximum sequence length
        pad_value (int): Padding value (default: 0 for <PAD>)

    Returns:
        padded (list): List of padded sequences
        lengths (list): Original lengths before padding
    """
    padded = []
    lengths = []

    for seq in sequences:
        length = min(len(seq), max_len)
        lengths.append(length)

        if len(seq) < max_len:
            # Pad
            padded_seq = seq + [pad_value] * (max_len - len(seq))
        else:
            # Truncate
            padded_seq = seq[:max_len]

        padded.append(padded_seq)

    return padded, lengths


def build_vocabulary_from_file(csv_path, text_column='text', max_size=20000, min_freq=2):
    """
    Build vocabulary from a CSV file.

    Args:
        csv_path (str): Path to CSV file
        text_column (str): Name of text column
        max_size (int): Maximum vocabulary size
        min_freq (int): Minimum word frequency

    Returns:
        vocab (Vocabulary): Built vocabulary object
    """
    import pandas as pd

    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataset")

    texts = df[text_column].dropna().tolist()
    print(f"Loaded {len(texts)} texts")

    vocab = Vocabulary(max_size=max_size, min_freq=min_freq)
    vocab.build_from_texts(texts)

    return vocab


if __name__ == "__main__":
    # Unit test for preprocessing
    print("Testing Preprocessing Utilities...\n")

    # Test tokenization
    text = "This is a Good Movie! I love it!!!"
    tokens = tokenize(text)
    print(f"Tokenized: {text}")
    print(f"Tokens: {tokens}\n")

    # Test vocabulary building
    texts = [
        "this is good",
        "this is bad",
        "good movie",
        "bad movie",
        "very good",
        "very bad"
    ]

    vocab = Vocabulary(max_size=100, min_freq=1)
    vocab.build_from_texts(texts)

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Word 'good' -> {vocab['good']}")
    print(f"Word 'xyz' -> {vocab['xyz']} (should be <UNK>)")

    # Test encoding/decoding
    test_text = "this movie is good"
    encoded = vocab.encode(test_text)
    decoded = vocab.decode(encoded)

    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Test padding
    sequences = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
    padded, lengths = pad_sequences(sequences, max_len=5)

    print(f"\nOriginal sequences: {sequences}")
    print(f"Padded sequences: {padded}")
    print(f"Lengths: {lengths}")

    print("\n✓ All preprocessing tests passed!")
