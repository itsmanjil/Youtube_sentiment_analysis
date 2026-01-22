"""
Attention Weight Explainer.

This module provides visualization and analysis of attention weights
from attention-based models (Hybrid CNN-BiLSTM-Attention, BERT).

Mathematical Foundation
-----------------------
For self-attention:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

The attention weights alpha = softmax(QK^T / sqrt(d_k)) indicate
which tokens the model focuses on for each position.

For interpretability, we typically:
1. Average attention across heads (multi-head attention)
2. Focus on attention to [CLS] token or from [CLS] to all tokens
3. Use attention rollout for multi-layer models

Caveats:
- Attention weights != feature importance (Jain & Wallace, 2019)
- Attention may capture syntactic rather than semantic patterns
- Consider using gradient-based methods for more accurate attribution

References
----------
Vaswani et al. (2017). Attention Is All You Need.
Jain & Wallace (2019). Attention is not Explanation.
Serrano & Smith (2019). Is Attention Interpretable?
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path


class AttentionExplainer:
    """
    Attention weight visualizer for attention-based models.

    This class extracts and visualizes attention weights from:
    - Hybrid CNN-BiLSTM-Attention model
    - BERT/RoBERTa transformers

    Parameters
    ----------
    model : Any
        Model with attention mechanism.
    tokenizer : Any, optional
        Tokenizer for converting text to tokens.
    model_type : str, optional
        Type of model: 'hybrid', 'bert', or 'auto'.
        Default: 'auto'

    Examples
    --------
    >>> # For Hybrid model
    >>> explainer = AttentionExplainer(hybrid_model, model_type='hybrid')
    >>> explanation = explainer.explain("This video is great!")
    >>> explainer.visualize(explanation, save_path='attention.png')

    >>> # For BERT
    >>> from transformers import BertTokenizer
    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    >>> explainer = AttentionExplainer(bert_model, tokenizer, model_type='bert')
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Optional[Any] = None,
        model_type: str = "auto",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type

        if model_type == "auto":
            self.model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        """Auto-detect model type based on class name."""
        class_name = self.model.__class__.__name__.lower()

        if "bert" in class_name:
            return "bert"
        elif "hybrid" in class_name or "bilstm" in class_name:
            return "hybrid"
        else:
            return "generic"

    def explain(
        self,
        text: str,
        layer: int = -1,
        head: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract attention weights for a text.

        Parameters
        ----------
        text : str
            Text to analyze.
        layer : int, optional
            Which layer's attention to extract (-1 for last).
            Default: -1
        head : int, optional
            Specific attention head to extract.
            If None, averages across all heads.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - text: Original text
            - tokens: Tokenized text
            - attention_weights: Attention matrix or vector
            - token_importance: Aggregated importance per token
        """
        if self.model_type == "bert":
            return self._explain_bert(text, layer, head)
        elif self.model_type == "hybrid":
            return self._explain_hybrid(text)
        else:
            return self._explain_generic(text)

    def _explain_bert(
        self,
        text: str,
        layer: int = -1,
        head: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Extract attention from BERT model."""
        import torch

        if self.tokenizer is None:
            raise ValueError("Tokenizer required for BERT models")

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Get attention weights
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_attentions=True,
            )

        # Extract attention from specified layer
        # Shape: (batch, num_heads, seq_len, seq_len)
        attention = outputs.attentions[layer][0].cpu().numpy()

        if head is not None:
            # Use specific head
            attention_weights = attention[head]
        else:
            # Average across heads
            attention_weights = np.mean(attention, axis=0)

        # Compute token importance (attention from [CLS] to all tokens)
        cls_attention = attention_weights[0, :]  # [CLS] attends to...
        token_importance = cls_attention.tolist()

        return {
            "text": text,
            "tokens": tokens,
            "attention_weights": attention_weights.tolist(),
            "token_importance": token_importance,
            "num_heads": attention.shape[0],
            "layer": layer,
            "head": head,
            "model_type": "bert",
        }

    def _explain_hybrid(self, text: str) -> Dict[str, Any]:
        """Extract attention from Hybrid CNN-BiLSTM-Attention model."""
        import torch

        # Tokenize using simple whitespace
        tokens = text.lower().split()

        # Check if model has get_attention_weights method
        if hasattr(self.model, "get_attention_weights"):
            attention_data = self.model.get_attention_weights(text)
            attention_weights = attention_data.get("attention_weights", [])
            tokens = attention_data.get("tokens", tokens)
        else:
            # Try to extract attention manually
            self.model.eval()

            # This depends on model implementation
            # Assuming model has attention_weights attribute after forward pass
            attention_weights = None

            if hasattr(self.model, "attention_pooling"):
                # Get from attention pooling layer
                attention_weights = getattr(
                    self.model.attention_pooling, "last_weights", None
                )

        if attention_weights is None:
            attention_weights = [1.0 / len(tokens)] * len(tokens)

        # Normalize
        if isinstance(attention_weights, np.ndarray):
            attention_weights = attention_weights.tolist()
        elif hasattr(attention_weights, "cpu"):
            attention_weights = attention_weights.cpu().numpy().tolist()

        # Flatten if needed
        if isinstance(attention_weights, list) and len(attention_weights) > 0:
            if isinstance(attention_weights[0], list):
                attention_weights = attention_weights[0]

        # Ensure same length as tokens
        if len(attention_weights) > len(tokens):
            attention_weights = attention_weights[:len(tokens)]
        elif len(attention_weights) < len(tokens):
            attention_weights = attention_weights + [0] * (len(tokens) - len(attention_weights))

        return {
            "text": text,
            "tokens": tokens,
            "attention_weights": attention_weights,
            "token_importance": attention_weights,
            "model_type": "hybrid",
        }

    def _explain_generic(self, text: str) -> Dict[str, Any]:
        """Fallback for generic models."""
        tokens = text.split()
        uniform_weights = [1.0 / len(tokens)] * len(tokens)

        return {
            "text": text,
            "tokens": tokens,
            "attention_weights": uniform_weights,
            "token_importance": uniform_weights,
            "model_type": "generic",
            "warning": "Could not extract actual attention weights",
        }

    def visualize(
        self,
        explanation: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (12, 4),
    ) -> None:
        """
        Visualize attention weights as a bar plot.

        Parameters
        ----------
        explanation : Dict[str, Any]
            Explanation from explain() method.
        save_path : str or Path, optional
            Path to save visualization.
        figsize : Tuple[int, int], optional
            Figure size.
        """
        import matplotlib.pyplot as plt

        tokens = explanation["tokens"]
        weights = explanation["token_importance"]

        # Truncate if too long
        max_tokens = 50
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            weights = weights[:max_tokens]

        fig, ax = plt.subplots(figsize=figsize)

        x_pos = np.arange(len(tokens))
        colors = plt.cm.Blues(np.array(weights) / max(weights) if max(weights) > 0 else weights)

        bars = ax.bar(x_pos, weights, color=colors, alpha=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Attention Weight")
        ax.set_title(f"Attention Weights ({explanation.get('model_type', 'unknown')} model)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_heatmap(
        self,
        explanation: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 10),
    ) -> None:
        """
        Visualize attention matrix as a heatmap.

        Useful for BERT models where attention is a matrix
        (token-to-token attention).

        Parameters
        ----------
        explanation : Dict[str, Any]
            Explanation with attention_weights as matrix.
        save_path : str or Path, optional
            Path to save visualization.
        figsize : Tuple[int, int], optional
            Figure size.
        """
        import matplotlib.pyplot as plt

        tokens = explanation["tokens"]
        weights = np.array(explanation["attention_weights"])

        if weights.ndim != 2:
            print("Attention weights are not a matrix. Using bar plot instead.")
            return self.visualize(explanation, save_path)

        # Truncate if too large
        max_tokens = 30
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            weights = weights[:max_tokens, :max_tokens]

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(weights, cmap="Blues", aspect="auto")

        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)

        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        ax.set_title("Attention Heatmap")

        plt.colorbar(im, ax=ax, label="Attention Weight")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def visualize_highlighted_text(
        self,
        explanation: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """
        Generate HTML with highlighted text based on attention.

        Parameters
        ----------
        explanation : Dict[str, Any]
            Explanation from explain() method.
        save_path : str or Path, optional
            Path to save HTML file.

        Returns
        -------
        str
            HTML string with highlighted text.
        """
        tokens = explanation["tokens"]
        weights = explanation["token_importance"]

        # Normalize weights to [0, 1]
        max_weight = max(weights) if weights else 1
        min_weight = min(weights) if weights else 0
        weight_range = max_weight - min_weight if max_weight > min_weight else 1

        normalized = [(w - min_weight) / weight_range for w in weights]

        # Generate HTML
        html_parts = [
            "<html><head><style>",
            "body { font-family: Arial, sans-serif; line-height: 1.8; padding: 20px; }",
            ".token { padding: 2px 4px; margin: 1px; border-radius: 3px; }",
            "</style></head><body>",
            f"<h3>Attention Visualization: {explanation.get('model_type', 'unknown')} model</h3>",
            "<p>",
        ]

        for token, weight in zip(tokens, normalized):
            # Color intensity based on weight
            red = int(255 * (1 - weight))
            green = int(255 * (1 - weight * 0.5))
            blue = 255
            color = f"rgb({red}, {green}, {blue})"

            html_parts.append(
                f'<span class="token" style="background-color: {color};" '
                f'title="Weight: {weight:.3f}">{token}</span> '
            )

        html_parts.extend(["</p>", "</body></html>"])
        html = "\n".join(html_parts)

        if save_path:
            with open(save_path, "w") as f:
                f.write(html)

        return html
