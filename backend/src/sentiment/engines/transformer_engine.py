"""
Transformer-based Sentiment Engine (BERT/RoBERTa).

This module provides sentiment analysis using pre-trained transformer models.
BERT (Bidirectional Encoder Representations from Transformers) achieves
state-of-the-art performance on many NLP tasks including sentiment analysis.

Mathematical Foundation
-----------------------
BERT uses bidirectional self-attention to create contextualized embeddings:

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Where Q, K, V are query, key, and value matrices derived from input embeddings.

For classification, the [CLS] token embedding is used:

    h_cls = BERT(input)[0]
    logits = W * h_cls + b
    P(class|input) = softmax(logits)

References
----------
Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding." NAACL-HLT.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from src.utils import SENTIMENT_LABELS, normalize_probs
from src.utils.config import get_model_path, Config
from src.sentiment.base import SentimentResult, BaseSentimentEngine


class TransformerSentimentEngine(BaseSentimentEngine):
    """
    Sentiment analysis using BERT or RoBERTa transformers.

    This engine leverages pre-trained transformer models fine-tuned
    for sentiment classification. It provides state-of-the-art accuracy
    but requires more computational resources than classical models.

    Parameters
    ----------
    model_name_or_path : str, optional
        HuggingFace model name or path to local model.
        Default: 'bert-base-uncased'
    num_labels : int, optional
        Number of sentiment classes.
        Default: 3 (Negative, Neutral, Positive)
    device : str, optional
        Device for inference ('cuda', 'mps', 'cpu', or 'auto').
        Default: 'auto'
    max_length : int, optional
        Maximum sequence length for tokenization.
        Default: 128

    Attributes
    ----------
    model : PreTrainedModel
        Loaded transformer model.
    tokenizer : PreTrainedTokenizer
        Tokenizer for the model.
    device : torch.device
        Device for inference.

    Examples
    --------
    >>> # Use pre-trained BERT
    >>> engine = TransformerSentimentEngine('bert-base-uncased')
    >>> result = engine.analyze("This is the best video ever!")
    >>> print(result.label, result.score)
    Positive 0.97

    >>> # Use fine-tuned model from local path
    >>> engine = TransformerSentimentEngine('./models/transformers/bert')
    >>> result = engine.analyze("Terrible quality, don't watch")
    Negative 0.93

    Notes
    -----
    Requires the transformers library. Install with:
        pip install transformers

    For best results, fine-tune the model on YouTube comment data.
    See: scripts/train/train_transformer.py

    Expected Performance (after fine-tuning):
    - Accuracy: 85-92% (significantly higher than classical models)
    - F1-Macro: 84-91%
    """

    def __init__(
        self,
        model_name_or_path: str = "bert-base-uncased",
        num_labels: int = 3,
        device: str = "auto",
        max_length: int = 128,
    ):
        # Lazy import dependencies
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "TransformerSentimentEngine requires transformers and torch. "
                "Install with: pip install transformers torch"
            )

        self.torch = torch
        self.max_length = max_length
        self.num_labels = num_labels

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

        # Check if model_name_or_path is a local path
        model_path = Path(model_name_or_path)
        if model_path.exists():
            model_name_or_path = str(model_path)

        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=num_labels,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load transformer model '{model_name_or_path}'. "
                f"Error: {e}\n"
                "For a pre-trained model, use a HuggingFace model name like 'bert-base-uncased'. "
                "For a fine-tuned model, provide the path to the saved model directory."
            )

        self.model.to(self.device)
        self.model.eval()

        # Label mapping
        self.idx2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
        self.label2idx = {"Negative": 0, "Neutral": 1, "Positive": 2}

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

        import torch.nn.functional as F

        # Tokenize
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to device
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        # Inference
        with self.torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)

        # Convert to results
        results = []
        probs_np = probs.cpu().numpy()

        for i, prob_row in enumerate(probs_np):
            prob_dict = {
                self.idx2label[j]: float(prob_row[j])
                for j in range(self.num_labels)
            }
            prob_dict = normalize_probs(prob_dict)
            label = max(prob_dict, key=prob_dict.get)

            results.append(
                SentimentResult(
                    label=label,
                    score=float(prob_dict[label]),
                    probs=prob_dict,
                    model="transformer",
                    raw={
                        "model_name": str(self.model.config._name_or_path),
                        "logits": logits[i].cpu().tolist(),
                    },
                )
            )

        return results

    def get_embeddings(self, text: str) -> Dict:
        """
        Get [CLS] token embeddings for a text.

        Useful for explainability and downstream tasks.

        Parameters
        ----------
        text : str
            Text to encode.

        Returns
        -------
        Dict
            Dictionary with 'cls_embedding' and 'tokens'.
        """
        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with self.torch.no_grad():
            outputs = self.model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            cls_embedding = outputs.last_hidden_state[0, 0, :]  # [CLS] token

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        return {
            "cls_embedding": cls_embedding.cpu().numpy().tolist(),
            "tokens": tokens,
        }

    def get_attention_weights(self, text: str, layer: int = -1) -> Dict:
        """
        Get attention weights from a specific layer.

        Parameters
        ----------
        text : str
            Text to analyze.
        layer : int, optional
            Which layer's attention to return (-1 for last layer).
            Default: -1

        Returns
        -------
        Dict
            Dictionary with 'attention_weights' and 'tokens'.
        """
        encodings = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with self.torch.no_grad():
            outputs = self.model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            # outputs.attentions is tuple of (batch, num_heads, seq_len, seq_len)
            attention = outputs.attentions[layer][0]  # (num_heads, seq_len, seq_len)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        return {
            "attention_weights": attention.cpu().numpy().tolist(),
            "tokens": tokens,
            "num_heads": attention.shape[0],
        }
