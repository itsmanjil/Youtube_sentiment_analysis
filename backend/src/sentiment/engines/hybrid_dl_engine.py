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

For training, see: research/train_hybrid_dl.py
For architecture details, see: research/architectures/hybrid_cnn_bilstm.py
"""

import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from src.utils import normalize_probs
from src.utils.config import get_model_path
from src.utils.runtime_artifacts import load_runtime_artifact_json, verify_artifact_or_raise
from src.sentiment.base import SentimentResult, BaseSentimentEngine


class _VocabularyStub:
    """
    Backward-compatibility stub for vocab.pkl files that were pickled with
    the old `data.preprocessing.Vocabulary` or `research.data.preprocessing.Vocabulary`
    class.  The production engine only needs the `.word2idx` dict.
    """
    def __init__(self):
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}


class _VocabCompatUnpickler(pickle.Unpickler):
    """Redirects legacy Vocabulary class paths to `_VocabularyStub`."""

    _VOCAB_MODULES = {
        "data.preprocessing",
        "research.data.preprocessing",
        "preprocessing",
    }

    def find_class(self, module: str, name: str):
        if module in self._VOCAB_MODULES and name == "Vocabulary":
            return _VocabularyStub
        return super().find_class(module, name)


def _safe_load_vocab(vocab_path: Path):
    """Load vocab.pkl with legacy class-path compatibility."""
    with open(vocab_path, "rb") as f:
        return _VocabCompatUnpickler(f).load()


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
    - research/train_hybrid_dl.py for training
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
        except Exception as exc:
            # In practice, PyTorch can fail to import for reasons other than a
            # missing installation (most commonly a NumPy ABI mismatch).
            raise ImportError(
                "HybridDLSentimentEngine requires a working PyTorch install.\n"
                "If you see an error mentioning NumPy 2.x, install a PyTorch build "
                "compatible with NumPy 2.x or pin NumPy to <2 and reinstall the ML stack.\n"
                f"Original error: {exc}"
            ) from exc

        self.torch = torch
        self.F = F

        # Resolve paths
        model_path = get_model_path(model_path)
        vocab_path = get_model_path(vocab_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. "
                "Train using: python research/train_hybrid_dl.py"
            )

        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary not found: {vocab_path}. "
                "The vocabulary file should be created during training."
            )

        # Verify pinned sha256 *before* deserializing either file — both
        # pickle.load (vocab) and torch.load (model, below) execute arbitrary
        # code as a side effect of deserialization, so checking the hash only
        # after loading can't stop a tampered file from running; it can only
        # notice afterward. See src/sentiment/engines/tfidf_engine.py etc.
        # for the same gate on the classical engines.
        self.artifact_verified = {
            "model": verify_artifact_or_raise(model_path, "hybrid_dl_model"),
            "vocab": verify_artifact_or_raise(vocab_path, "hybrid_dl_vocab"),
        }

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

        # Load vocabulary — uses compat unpickler to handle vocab.pkl files
        # that were pickled with the old `data.preprocessing.Vocabulary` class.
        vocab_obj = _safe_load_vocab(vocab_path)

        word2idx = None
        if isinstance(vocab_obj, dict):
            # Two common formats:
            # 1) {"word2idx": {...}, "idx2word": {...}, ...}
            # 2) {"<PAD>": 0, "<UNK>": 1, ...}
            if isinstance(vocab_obj.get("word2idx"), dict):
                word2idx = vocab_obj["word2idx"]
            else:
                word2idx = vocab_obj
        elif hasattr(vocab_obj, "word2idx"):
            word2idx = getattr(vocab_obj, "word2idx")

        if not isinstance(word2idx, dict) or not word2idx:
            raise RuntimeError(
                "Unsupported vocabulary format. Expected a dict mapping tokens to indices, "
                "or an object with a `.word2idx` dict."
            )

        self.word2idx = {str(key): int(value) for key, value in word2idx.items()}
        self.vocab_size = len(self.word2idx)

        # Load model configuration if available
        metadata_path = model_path.parent / "metadata.json"
        self.metadata: Dict[str, Any] = {}
        self.hyperparams: Dict[str, Any] = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
            if isinstance(self.metadata, dict):
                self.hyperparams = self.metadata.get("hyperparameters", {}) or {}

        # Load model architecture from research module
        from research.architectures.hybrid_cnn_bilstm import HybridCNNBiLSTM

        # Get model hyperparameters
        embedding_dim = int(self.hyperparams.get("embedding_dim", 300))
        num_classes = int(self.hyperparams.get("num_classes", 3))

        # Initialize model
        self.model = HybridCNNBiLSTM(
            vocab_size=self.vocab_size,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            cnn_filter_sizes=self.hyperparams.get("cnn_filter_sizes", [3, 4, 5]),
            cnn_num_filters=int(self.hyperparams.get("cnn_num_filters", 128)),
            lstm_hidden_size=int(self.hyperparams.get("lstm_hidden_size", 128)),
            lstm_num_layers=int(self.hyperparams.get("lstm_num_layers", 2)),
            attention_num_heads=int(self.hyperparams.get("attention_num_heads", 4)),
            classifier_hidden_sizes=self.hyperparams.get("classifier_hidden_sizes", [256, 128]),
            dropout_cnn=float(self.hyperparams.get("dropout_cnn", 0.3)),
            dropout_lstm=float(self.hyperparams.get("dropout_lstm", 0.3)),
            dropout_attention=float(self.hyperparams.get("dropout_attention", 0.1)),
            dropout_classifier=self.hyperparams.get("dropout_classifier", [0.5, 0.4]),
        )

        # Load weights — weights_only=False required for checkpoints saved with
        # older PyTorch / numpy that embed numpy scalars in the state dict.
        # This file is a project-internal artifact so the source is trusted.
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        # Settings
        self.max_len = int(self.hyperparams.get("max_len", 200))
        self.pad_idx = int(self.word2idx.get("<PAD>", 0))
        self.unk_idx = int(self.word2idx.get("<UNK>", 1))

        # Label mapping
        self.idx2label = {0: "Negative", 1: "Neutral", 2: "Positive"}

        # Temperature scaling (load from research results)
        self.temperature = 1.0
        self.calibration_applied = False
        try:
            _ts_data = load_runtime_artifact_json("temperature_scaling") or {}
            for _entry in _ts_data.get("models", []):
                if _entry.get("model") == "hybrid_dl":
                    self.temperature = float(_entry["temperature"])
                    self.calibration_applied = True
                    break
        except Exception:
            pass

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text in a way that matches the training pipeline."""
        if text is None:
            return []
        cleaned = re.sub(r"[^a-zA-Z\\s]", " ", str(text))
        cleaned = " ".join(cleaned.lower().split())
        return cleaned.split() if cleaned else []

    def _encode(self, tokens: List[str]) -> Tuple[List[int], int]:
        """Convert tokens to padded/truncated indices and return (indices, length)."""
        length = min(len(tokens), self.max_len)
        indices = [self.word2idx.get(token, self.unk_idx) for token in tokens[: self.max_len]]

        # Truncate or pad
        if len(indices) < self.max_len:
            indices = indices + [self.pad_idx] * (self.max_len - len(indices))

        return indices, length

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

    # The analyze endpoint allows up to 2000 comments per request (see
    # app/views.py's max_comments bound); running all of them through the
    # model as a single tensor risks exhausting host memory (especially on
    # CPU). Mini-batching keeps peak memory bounded regardless of input size.
    _INFERENCE_BATCH_SIZE = 32

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

        results: List[SentimentResult] = []
        for start in range(0, len(texts), self._INFERENCE_BATCH_SIZE):
            chunk = texts[start : start + self._INFERENCE_BATCH_SIZE]
            results.extend(self._infer_chunk(chunk))
        return results

    def _infer_chunk(self, texts: List[str]) -> List[SentimentResult]:
        """Run inference on a single mini-batch (see _INFERENCE_BATCH_SIZE)."""
        token_lists = [self._tokenize(text) for text in texts]
        encoded = [self._encode(tokens) for tokens in token_lists]
        token_ids = [row[0] for row in encoded]
        lengths_list = [row[1] for row in encoded]
        batch_tensor = self.torch.tensor(token_ids, dtype=self.torch.long).to(self.device)

        # Compute sequence lengths (for packing)
        lengths = self.torch.tensor(lengths_list, dtype=self.torch.long)

        # Inference
        with self.torch.no_grad():
            outputs = self.model(batch_tensor, lengths)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                logits, _attention = outputs
            else:
                logits = outputs
            probs = self.F.softmax(logits / self.temperature, dim=-1)

        # Convert to results
        results = []
        probs_rows = probs.detach().cpu().tolist()
        logits_rows = logits.detach().cpu().tolist()

        for i, prob_row in enumerate(probs_rows):
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
                    raw={"logits": logits_rows[i]},
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
        tokens = self._tokenize(text)
        token_ids, length = self._encode(tokens)
        batch_tensor = self.torch.tensor([token_ids], dtype=self.torch.long).to(self.device)
        lengths = self.torch.tensor([length], dtype=self.torch.long)

        with self.torch.no_grad():
            outputs = self.model(batch_tensor, lengths)
            attention = None
            if isinstance(outputs, tuple) and len(outputs) == 2:
                _logits, attention = outputs

        payload: Dict[str, Any] = {"tokens": tokens[:length]}
        if isinstance(attention, dict):
            pooling = attention.get("attention_pooling")
            if pooling is not None:
                try:
                    payload["attention_pooling"] = pooling[0].detach().cpu().tolist()[:length]
                except Exception:
                    payload["attention_pooling"] = None
        return payload
