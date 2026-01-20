"""
Hybrid Deep Learning Sentiment Engine

Integrates the novel hybrid CNN-BiLSTM-Attention model with the existing
sentiment analysis pipeline, following the established SentimentResult pattern.

Author: Master's Thesis - Computational Intelligence
"""

import sys
from pathlib import Path
import pickle
import json

# Add research directory to path for imports
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / 'research'))

import torch
import torch.nn.functional as F

from .sentiment_engines import SentimentResult, coerce_sentiment_result
from .analysis_utils import normalize_probs
from .youtube_preprocessor import YouTubePreprocessor


def _resolve_model_path(path_value):
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_obj
    return (BASE_DIR / path_obj).resolve()


class HybridDLSentimentEngine:
    """
    Hybrid Deep Learning Sentiment Engine for YouTube comments.

    This engine wraps the hybrid CNN-BiLSTM-Attention PyTorch model and provides
    a unified interface following the existing sentiment engine pattern.

    The engine integrates seamlessly with:
    - Existing YouTubePreprocessor (10-stage pipeline)
    - SentimentResult format
    - Batch processing API
    - GPU/CPU automatic device selection

    Args:
        model_path (str): Path to trained PyTorch model checkpoint (.pt)
        vocab_path (str): Path to vocabulary pickle file (.pkl)
        metadata_path (str): Path to model metadata JSON
        device (str): 'cuda', 'cpu', or 'auto' for automatic selection
        max_len (int): Maximum sequence length for padding/truncation

    Example:
        >>> engine = HybridDLSentimentEngine(
        ...     model_path='./models/hybrid_dl/hybrid_v1.pt',
        ...     vocab_path='./models/hybrid_dl/vocab.pkl',
        ...     device='auto'
        ... )
        >>> result = engine.analyze("This movie is absolutely amazing!")
        >>> print(result.label, result.score)  # "Positive", 0.95
    """

    def __init__(
        self,
        model_path='./models/hybrid_dl/hybrid_v1.pt',
        vocab_path='./models/hybrid_dl/vocab.pkl',
        metadata_path='./models/hybrid_dl/metadata.json',
        device='auto',
        max_len=200
    ):
        # Import here to avoid circular dependencies
        try:
            from research.architectures.hybrid_cnn_bilstm import HybridCNNBiLSTM
            from research.data.preprocessing import Vocabulary
        except ImportError as e:
            raise ImportError(
                f"Failed to import hybrid model components: {e}\n"
                "Ensure research/architectures and research/data are accessible."
            )

        self.model_path = _resolve_model_path(model_path)
        self.vocab_path = _resolve_model_path(vocab_path)
        self.metadata_path = _resolve_model_path(metadata_path)
        self.max_len = max_len

        # Device selection
        self.device = self._get_device(device)
        print(f"HybridDLSentimentEngine using device: {self.device}")

        # Load vocabulary
        if not self.vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found: {self.vocab_path}\n"
                "Train the model first using train_hybrid_dl.py"
            )

        with open(self.vocab_path, 'rb') as f:
            vocab_data = pickle.load(f)

        # Handle both dict format and Vocabulary class object
        if isinstance(vocab_data, dict):
            self.word2idx = vocab_data['word2idx']
            self.idx2word = vocab_data['idx2word']
        else:
            # Vocabulary class object
            self.word2idx = vocab_data.word2idx
            self.idx2word = vocab_data.idx2word
        self.vocab_size = len(self.word2idx)

        print(f"Loaded vocabulary: {self.vocab_size} words")

        # Load metadata
        self.metadata = self._load_metadata()

        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}\n"
                "Train the model first using train_hybrid_dl.py"
            )

        self.model = self._load_model()
        self.model.eval()  # Set to evaluation mode

        # Preprocessor for text cleaning
        self.preprocessor = YouTubePreprocessor()

        # Label mapping
        self.idx2label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        self.label2idx = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

    def _get_device(self, device_str):
        """Select device (GPU/CPU) automatically or based on user preference."""
        if device_str == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')  # Apple Silicon
            else:
                return torch.device('cpu')
        else:
            return torch.device(device_str)

    def _load_metadata(self):
        """Load model metadata (hyperparameters, performance metrics)."""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded model metadata: v{metadata.get('model_version', 'unknown')}")
            return metadata
        else:
            print("Warning: Metadata file not found, using defaults")
            return {}

    def _load_model(self):
        """Load trained PyTorch model from checkpoint."""
        from research.architectures.hybrid_cnn_bilstm import HybridCNNBiLSTM

        # Load checkpoint (weights_only=False for PyTorch 2.6+ compatibility)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Get hyperparameters from checkpoint or metadata
        hyperparams = checkpoint.get('hyperparameters', self.metadata.get('hyperparameters', {}))

        # Create model architecture
        model = HybridCNNBiLSTM(
            vocab_size=self.vocab_size,
            embedding_dim=hyperparams.get('embedding_dim', 300),
            num_classes=3,
            cnn_filter_sizes=hyperparams.get('cnn_filter_sizes', [3, 4, 5]),
            cnn_num_filters=hyperparams.get('cnn_num_filters', 128),
            lstm_hidden_size=hyperparams.get('lstm_hidden_size', 128),
            lstm_num_layers=hyperparams.get('lstm_num_layers', 2),
            attention_num_heads=hyperparams.get('attention_num_heads', 4),
            classifier_hidden_sizes=hyperparams.get('classifier_hidden_sizes', [256, 128]),
            dropout_cnn=hyperparams.get('dropout_cnn', 0.3),
            dropout_lstm=hyperparams.get('dropout_lstm', 0.3),
            dropout_attention=hyperparams.get('dropout_attention', 0.1),
            dropout_classifier=hyperparams.get('dropout_classifier', [0.5, 0.4]),
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        print(f"Loaded model from {self.model_path}")
        return model

    def _preprocess_text(self, text):
        """Preprocess text using existing YouTubePreprocessor."""
        # Use existing preprocessing pipeline
        processed_text, metadata = self.preprocessor.preprocess_youtube_comment(
            text,
            emoji_mode='convert',
            check_spam=False,  # Don't filter during inference
            check_lang=False
        )

        if metadata['filtered']:
            # If filtered, return empty
            return ""

        return processed_text

    def _tokenize_and_encode(self, text):
        """
        Tokenize and encode text to indices.

        Args:
            text (str): Input text

        Returns:
            indices (list): List of word indices
        """
        # Simple tokenization (matches training preprocessing)
        tokens = text.lower().split()

        # Convert to indices
        indices = [self.word2idx.get(token, self.word2idx.get('<UNK>', 1)) for token in tokens]

        return indices

    def _prepare_input(self, text):
        """
        Prepare text for model input.

        Args:
            text (str): Input text

        Returns:
            input_tensor: (1, max_len) tensor
            length: Actual length before padding
        """
        # Preprocess
        processed = self._preprocess_text(text)

        # Encode
        indices = self._tokenize_and_encode(processed)

        # Get actual length
        length = min(len(indices), self.max_len)

        # Pad or truncate
        if len(indices) < self.max_len:
            indices = indices + [0] * (self.max_len - len(indices))  # Pad with <PAD>
        else:
            indices = indices[:self.max_len]  # Truncate

        # Convert to tensor
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)  # (1, max_len)

        return input_tensor, length

    def analyze(self, text):
        """
        Analyze sentiment of a single text.

        Args:
            text (str): Input text

        Returns:
            SentimentResult: Result object with label, score, probs, model name

        Example:
            >>> result = engine.analyze("This is amazing!")
            >>> print(result.label)  # "Positive"
            >>> print(result.probs)  # {"Positive": 0.95, "Neutral": 0.03, "Negative": 0.02}
        """
        # Prepare input
        input_tensor, length = self._prepare_input(text)

        # Model inference
        with torch.no_grad():
            logits, attention_weights = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=-1)  # (1, 3)

        # Get prediction
        probs = probabilities[0].cpu().numpy()  # (3,)
        pred_idx = int(probs.argmax())
        pred_label = self.idx2label[pred_idx]
        pred_score = float(probs[pred_idx])

        # Create probability dictionary
        probs_dict = {
            'Positive': float(probs[2]),
            'Neutral': float(probs[1]),
            'Negative': float(probs[0])
        }

        # Normalize probabilities
        probs_dict = normalize_probs(probs_dict)

        # Create result
        result = SentimentResult(
            label=pred_label,
            score=pred_score,
            probs=probs_dict,
            model="hybrid_dl",
            raw={
                'logits': logits[0].cpu().numpy().tolist(),
                'attention_pooling_weights': attention_weights['attention_pooling'][0].cpu().numpy().tolist()[:50],  # First 50 weights
            }
        )

        return result

    def batch_analyze(self, texts, batch_size=32):
        """
        Analyze sentiment of multiple texts in batches.

        Args:
            texts (list): List of text strings
            batch_size (int): Batch size for inference

        Returns:
            results (list): List of SentimentResult objects

        Example:
            >>> texts = ["Great movie!", "Terrible film", "It was okay"]
            >>> results = engine.batch_analyze(texts)
            >>> print([r.label for r in results])  # ["Positive", "Negative", "Neutral"]
        """
        results = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Prepare batch inputs
            input_tensors = []
            lengths = []

            for text in batch_texts:
                input_tensor, length = self._prepare_input(text)
                input_tensors.append(input_tensor)
                lengths.append(length)

            # Stack into batch
            batch_tensor = torch.cat(input_tensors, dim=0)  # (batch_size, max_len)

            # Batch inference
            with torch.no_grad():
                logits, attention_weights = self.model(batch_tensor)
                probabilities = F.softmax(logits, dim=-1)  # (batch_size, 3)

            # Process each prediction
            probs_batch = probabilities.cpu().numpy()

            for j, probs in enumerate(probs_batch):
                pred_idx = int(probs.argmax())
                pred_label = self.idx2label[pred_idx]
                pred_score = float(probs[pred_idx])

                probs_dict = {
                    'Positive': float(probs[2]),
                    'Neutral': float(probs[1]),
                    'Negative': float(probs[0])
                }

                probs_dict = normalize_probs(probs_dict)

                result = SentimentResult(
                    label=pred_label,
                    score=pred_score,
                    probs=probs_dict,
                    model="hybrid_dl",
                    raw={'logits': logits[j].cpu().numpy().tolist()}
                )

                results.append(result)

        return results


if __name__ == "__main__":
    # Unit test (requires trained model)
    print("HybridDLSentimentEngine Test\n")

    try:
        engine = HybridDLSentimentEngine(
            model_path='./models/hybrid_dl/hybrid_v1.pt',
            vocab_path='./models/hybrid_dl/vocab.pkl',
            device='auto'
        )

        # Test single analysis
        text = "This movie is absolutely amazing! I loved it!"
        result = engine.analyze(text)

        print(f"Text: {text}")
        print(f"Prediction: {result.label}")
        print(f"Confidence: {result.score:.4f}")
        print(f"Probabilities: {result.probs}")

        # Test batch analysis
        texts = [
            "Great movie!",
            "Terrible film, waste of time",
            "It was okay, nothing special"
        ]

        results = engine.batch_analyze(texts)

        print(f"\nBatch Analysis:")
        for text, result in zip(texts, results):
            print(f"  {text[:40]:40s} -> {result.label:10s} ({result.score:.2f})")

        print("\n✓ HybridDLSentimentEngine test passed!")

    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Train the model first using train_hybrid_dl.py")
