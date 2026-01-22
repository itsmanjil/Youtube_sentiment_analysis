"""
Hybrid CNN-BiLSTM-Attention Sentiment Engine.

This module provides a production wrapper for the research hybrid model.
The actual model architecture is defined in research/architectures/hybrid_cnn_bilstm.py.

Architecture Overview
---------------------
The hybrid model combines:
1. CNN Branch: Captures local n-gram patterns using multiple filter sizes
2. BiLSTM Branch: Captures sequential dependencies bidirectionally
3. Multi-Head Attention: Focuses on relevant parts of the sequence
4. Feature Fusion: Concatenates CNN and BiLSTM-Attention features

Input -> Embedding -> [CNN Branch] ---------+
                  |                          |
                  +-> [BiLSTM -> Attention] -+-> Fusion -> Classifier -> Output

Total Parameters: ~2.5M (trainable)

For training, see: scripts/train/train_hybrid_dl.py
For architecture details, see: research/architectures/hybrid_cnn_bilstm.py
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from src.utils import SENTIMENT_LABELS, normalize_probs
from src.utils.config import get_model_path, Config
from src.sentiment.base import SentimentResult, normalize_label, BaseSentimentEngine


class HybridDLSentimentEngine(BaseSentimentEngine):
    """
    Sentiment analysis using Hybrid CNN-BiLSTM-Attention model.

    This deep learning model combines convolutional and recurrent
    approaches with attention mechanisms for improved performance
    on sentiment analysis tasks.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the trained PyTorch model (.pt file).
        Default: './models/hybrid_dl/hybrid_v1.pt'
    vocab_path : str or Path, optional
        Path to the vocabulary file.
        Default: './models/hybrid_dl/vocab.pkl'
    device : str, optional
        Device to run inference on ('cuda', 'mps', 'cpu', or 'auto').
        Default: 'auto' (automatically selects best available device)

    Attributes
    ----------
    model : HybridCNNBiLSTM
        Loaded PyTorch model.
    vocab : Dict
        Word-to-index mapping for tokenization.
    device : torch.device
        Device for inference.

    Examples
    --------
    >>> engine = HybridDLSentimentEngine()
    >>> result = engine.analyze("This video changed my life!")
    >>> print(result.label, result.score)
    Positive 0.96

    Notes
    -----
    Requires PyTorch. If PyTorch is not installed, this engine
    will raise an ImportError when initialized.

    The model must be trained before use. See:
    - scripts/train/train_hybrid_dl.py for training
    - research/architectures/hybrid_cnn_bilstm.py for architecture
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "./models/hybrid_dl/hybrid_v1.pt",
        vocab_path: Union[str, Path] = "./models/hybrid_dl/vocab.pkl",
        device: str = "auto",
    ):
        # Lazy import PyTorch
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError(
                "HybridDLSentimentEngine requires PyTorch. "
                "Install with: pip install torch"
            )

        self.torch = torch
        self.F = F

        # Resolve paths
        model_path = get_model_path(model_path)
        vocab_path = get_model_path(vocab_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                "Train using: python scripts/train/train_hybrid_dl.py"
            )

        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary not found: {vocab_path}. "
                "The vocabulary file should be created during training."
            )

        # Select device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Load vocabulary
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        # Load model configuration if available
        metadata_path = model_path.parent / "metadata.json"
        self.config = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.config = json.load(f)

        # Load model architecture from research module
        from research.architectures.hybrid_cnn_bilstm import HybridCNNBiLSTM

        # Get model hyperparameters
        vocab_size = len(self.vocab)
        embed_dim = self.config.get("embed_dim", 300)
        hidden_dim = self.config.get("hidden_dim", 128)
        num_classes = self.config.get("num_classes", 3)

        # Initialize model
        self.model = HybridCNNBiLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # Settings
        self.max_len = self.config.get("max_len", 200)
        self.pad_idx = self.vocab.get("<PAD>", 0)
        self.unk_idx = self.vocab.get("<UNK>", 1)

        # Label mapping
        self.idx2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

    def _tokenize(self, text: str) -> List[int]:
        """Convert text to token indices."""
        tokens = text.lower().split()
        indices = [self.vocab.get(token, self.unk_idx) for token in tokens]

        # Truncate or pad
        if len(indices) > self.max_len:
            indices = indices[: self.max_len]
        else:
            indices = indices + [self.pad_idx] * (self.max_len - len(indices))

        return indices

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze the sentiment of a single text.

        Parameters
        ----------
        text : str
            Text to analyze.

        Returns
        -------
        SentimentResult
            Sentiment prediction with probabilities.
        """
        return self.batch_analyze([text])[0]

    def batch_analyze(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze multiple texts efficiently using batching.

        Parameters
        ----------
        texts : List[str]
            List of texts to analyze.

        Returns
        -------
        List[SentimentResult]
            List of sentiment predictions.
        """
        if not texts:
            return []

        # Tokenize all texts
        token_ids = [self._tokenize(text) for text in texts]
        batch_tensor = self.torch.tensor(token_ids, dtype=self.torch.long).to(self.device)

        # Compute sequence lengths (for packing)
        lengths = self.torch.tensor(
            [min(len(text.split()), self.max_len) for text in texts],
            dtype=self.torch.long
        )

        # Inference
        with self.torch.no_grad():
            logits = self.model(batch_tensor, lengths)
            probs = self.F.softmax(logits, dim=-1)

        # Convert to results
        results = []
        probs_np = probs.cpu().numpy()

        for i, prob_row in enumerate(probs_np):
            prob_dict = {
                "Negative": float(prob_row[0]),
                "Neutral": float(prob_row[1]),
                "Positive": float(prob_row[2]),
            }
            prob_dict = normalize_probs(prob_dict)
            label = max(prob_dict, key=prob_dict.get)

            results.append(
                SentimentResult(
                    label=label,
                    score=float(prob_dict[label]),
                    probs=prob_dict,
                    model="hybrid_dl",
                    raw={"logits": logits[i].cpu().tolist()},
                )
            )

        return results

    def get_attention_weights(self, text: str) -> Dict:
        """
        Get attention weights for explainability.

        Parameters
        ----------
        text : str
            Text to analyze.

        Returns
        -------
        Dict
            Attention weights and token information.
        """
        tokens = text.lower().split()
        token_ids = self._tokenize(text)
        batch_tensor = self.torch.tensor([token_ids], dtype=self.torch.long).to(self.device)
        lengths = self.torch.tensor([min(len(tokens), self.max_len)], dtype=self.torch.long)

        with self.torch.no_grad():
            # Get attention weights from model
            _, attention_weights = self.model(batch_tensor, lengths, return_attention=True)

        return {
            "tokens": tokens[: self.max_len],
            "attention_weights": attention_weights[0].cpu().numpy().tolist(),
        }
