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

import math
from pathlib import Path
from typing import Dict, List, Optional

from src.utils import normalize_probs
from src.utils.config import Config
from src.utils.calibration import load_temperature_artifact
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

    MODEL_PRESETS = {
        "bert": "bert-base-uncased",
        "transformer": "bert-base-uncased",
        "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "modernbert": "answerdotai/ModernBERT-base",
        "deberta_v3": "microsoft/deberta-v3-base",
        "xlm_v": "facebook/xlm-v-base",
        "mdeberta_v3": "microsoft/mdeberta-v3-base",
    }

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        num_labels: int = 3,
        device: str = "auto",
        max_length: int = 128,
        model_preset: Optional[str] = None,
        calibration_profile: str = "auto",
        temperature_artifact_path: Optional[str] = None,
        temperature: Optional[float] = None,
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
        self.model_preset = self._normalize_preset(model_preset or model_name_or_path)
        self.engine_name = self.model_preset or "transformer"
        self.calibration_profile = str(calibration_profile or "auto").lower().strip()

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

        model_name_or_path = self._resolve_model_name_or_path(model_name_or_path)
        self.model_source = model_name_or_path
        self.model_artifact = Path(model_name_or_path).name if Path(model_name_or_path).exists() else model_name_or_path
        self.temperature = 1.0
        self.temperature_artifact_path = None
        self.calibration_metadata = None
        self.calibration_applied = False

        self._configure_calibration(
            temperature=temperature,
            temperature_artifact_path=temperature_artifact_path,
        )

        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=num_labels,
                use_safetensors=True,
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

    def _normalize_preset(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = str(value).strip().lower().replace("-", "_")
        if normalized in self.MODEL_PRESETS:
            return normalized
        return None

    def _resolve_model_name_or_path(self, model_name_or_path: Optional[str]) -> str:
        explicit_value = str(model_name_or_path).strip() if model_name_or_path else ""

        if explicit_value:
            explicit_path = Path(explicit_value)
            if explicit_path.exists():
                return str(explicit_path)
            explicit_preset = self._normalize_preset(explicit_value)
            if explicit_preset:
                self.model_preset = explicit_preset
            else:
                return explicit_value

        if self.model_preset:
            local_path = Config.get_model_path(self.model_preset)
            if local_path and Path(local_path).exists():
                return str(local_path)
            return self.MODEL_PRESETS[self.model_preset]

        local_bert_path = Config.get_model_path("bert")
        if local_bert_path and Path(local_bert_path).exists():
            self.model_preset = "bert"
            return str(local_bert_path)

        self.model_preset = "bert"
        return self.MODEL_PRESETS["bert"]

    def _resolve_temperature_artifact_path(
        self,
        explicit_path: Optional[str],
    ) -> Optional[Path]:
        if explicit_path:
            path = Path(explicit_path)
            return path if path.exists() else None

        model_path = Path(self.model_source)
        if model_path.exists() and model_path.is_dir():
            candidate = model_path / "temperature_scaling.json"
            if candidate.exists():
                return candidate

        if self.model_preset:
            configured = Config.get_model_path(self.model_preset, "calibration")
            if configured and Path(configured).exists():
                return Path(configured)

        return None

    def _configure_calibration(
        self,
        *,
        temperature: Optional[float],
        temperature_artifact_path: Optional[str],
    ) -> None:
        if self.calibration_profile in {"off", "none", "disabled", "raw"}:
            self.temperature = 1.0
            self.calibration_applied = False
            return

        if temperature is not None:
            self.temperature = max(float(temperature), 1e-6)
            self.calibration_applied = True
            return

        artifact_path = self._resolve_temperature_artifact_path(temperature_artifact_path)
        artifact = load_temperature_artifact(artifact_path)
        if artifact is None:
            self.temperature = 1.0
            self.calibration_applied = False
            return

        self.temperature_artifact_path = str(artifact_path)
        self.calibration_metadata = artifact
        self.temperature = max(float(artifact.get("temperature", 1.0)), 1e-6)
        self.calibration_applied = True

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
            if self.calibration_applied:
                probs = F.softmax(logits / self.temperature, dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)

        # Convert to results
        results = []
        # Avoid `.numpy()` because Torch can be built without a working NumPy bridge
        # (common when the NumPy ABI is incompatible). `.tolist()` is always safe.
        probs_list = probs.detach().cpu().tolist()

        for i, prob_row in enumerate(probs_list):
            prob_dict = {
                self.idx2label[j]: float(prob_row[j])
                for j in range(self.num_labels)
            }
            prob_dict = normalize_probs(prob_dict)
            label = max(prob_dict, key=prob_dict.get)
            sorted_probs = sorted(prob_dict.values(), reverse=True)
            top2_margin = (
                float(sorted_probs[0] - sorted_probs[1])
                if len(sorted_probs) > 1 else float(sorted_probs[0])
            )
            entropy = 0.0
            if self.num_labels > 1:
                entropy = -sum(
                    prob * math.log(max(prob, 1e-12))
                    for prob in prob_dict.values()
                ) / math.log(self.num_labels)

            results.append(
                SentimentResult(
                    label=label,
                    score=float(prob_dict[label]),
                    probs=prob_dict,
                    model=self.engine_name,
                    raw={
                        "model_name": str(self.model.config._name_or_path),
                        "model_preset": self.model_preset,
                        "model_source": self.model_source,
                        "model_artifact": self.model_artifact,
                        "logits": logits[i].cpu().tolist(),
                        "entropy": float(entropy),
                        "top2_margin": float(top2_margin),
                        "calibration_applied": self.calibration_applied,
                        "calibration_profile": self.calibration_profile,
                        "temperature": float(self.temperature),
                        "temperature_artifact_path": self.temperature_artifact_path,
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

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        return {
            "cls_embedding": cls_embedding.detach().cpu().tolist(),
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

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        return {
            "attention_weights": attention.detach().cpu().tolist(),
            "tokens": tokens,
            "num_heads": attention.shape[0],
        }
