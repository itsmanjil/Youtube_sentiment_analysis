"""
BERT-based Sentiment Classifier.

This module implements a BERT-based sentiment classifier following the
standard fine-tuning approach from Devlin et al. (2019).

Architecture
------------
Input Text → BERT Tokenizer → Token IDs
                                   ↓
                              BERT Model
                                   ↓
                          [CLS] Embedding (768-dim)
                                   ↓
                              Dropout (0.1)
                                   ↓
                          Linear Classifier
                                   ↓
                          Softmax → Probabilities

Mathematical Formulation
------------------------
Let x = [x_1, x_2, ..., x_n] be the input token sequence.

1. BERT Encoding:
   H = BERT([CLS], x_1, x_2, ..., x_n, [SEP])
   h_cls = H[0]  # [CLS] token representation

2. Classification:
   logits = W * h_cls + b
   P(y|x) = softmax(logits)

3. Training Loss (Cross-Entropy):
   L = -sum_i y_i * log(P(y_i|x_i))

References
----------
Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019).
BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding. NAACL-HLT.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union


class BERTSentimentClassifier(nn.Module):
    """
    BERT-based sentiment classifier for YouTube comments.

    This model fine-tunes a pre-trained BERT model for 3-class
    sentiment classification (Negative, Neutral, Positive).

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model name or path to local model.
        Default: 'bert-base-uncased'
    num_classes : int, optional
        Number of output classes.
        Default: 3 (Negative, Neutral, Positive)
    dropout : float, optional
        Dropout probability for regularization.
        Default: 0.1
    freeze_bert : bool, optional
        Whether to freeze BERT parameters (feature extraction mode).
        Default: False (fine-tuning mode)

    Attributes
    ----------
    bert : BertModel
        Pre-trained BERT encoder.
    classifier : nn.Linear
        Classification head.
    dropout : nn.Dropout
        Dropout layer.
    config : BertConfig
        BERT configuration.

    Examples
    --------
    >>> from transformers import BertTokenizer
    >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    >>> model = BERTSentimentClassifier()
    >>>
    >>> # Tokenize input
    >>> inputs = tokenizer("This video is amazing!", return_tensors="pt")
    >>>
    >>> # Forward pass
    >>> logits = model(inputs['input_ids'], inputs['attention_mask'])
    >>> probs = torch.softmax(logits, dim=-1)
    >>> predicted_class = probs.argmax(dim=-1)

    Notes
    -----
    Training Tips:
    - Use learning rate 2e-5 to 5e-5 for fine-tuning
    - Train for 3-5 epochs (more can lead to overfitting)
    - Use warmup for first 10% of training steps
    - Gradient clipping at 1.0 is recommended

    For training, see: scripts/train/train_transformer.py
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 3,
        dropout: float = 0.1,
        freeze_bert: bool = False,
    ):
        super().__init__()

        try:
            from transformers import BertModel, BertConfig
        except ImportError:
            raise ImportError(
                "BERTSentimentClassifier requires the transformers library. "
                "Install with: pip install transformers"
            )

        self.model_name = model_name
        self.num_classes = num_classes

        # Load pre-trained BERT
        self.bert = BertModel.from_pretrained(model_name)
        self.config = self.bert.config

        # Optionally freeze BERT parameters
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

        # Initialize classifier weights
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Initialize classifier weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass of the BERT classifier.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape (batch_size, seq_len).
        attention_mask : torch.Tensor
            Attention mask of shape (batch_size, seq_len).
        token_type_ids : torch.Tensor, optional
            Token type IDs for sentence pair tasks.
        output_hidden_states : bool, optional
            Whether to return all hidden states.
        output_attentions : bool, optional
            Whether to return attention weights.

        Returns
        -------
        logits : torch.Tensor
            Classification logits of shape (batch_size, num_classes).
        hidden_states : tuple, optional
            All hidden states if output_hidden_states=True.
        attentions : tuple, optional
            All attention weights if output_attentions=True.
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        # Get [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)

        # Classification
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # (batch_size, num_classes)

        if output_hidden_states or output_attentions:
            return_tuple = (logits,)
            if output_hidden_states:
                return_tuple += (outputs.hidden_states,)
            if output_attentions:
                return_tuple += (outputs.attentions,)
            return return_tuple

        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with probabilities.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape (batch_size, seq_len).
        attention_mask : torch.Tensor
            Attention mask of shape (batch_size, seq_len).

        Returns
        -------
        labels : torch.Tensor
            Predicted class indices of shape (batch_size,).
        probs : torch.Tensor
            Class probabilities of shape (batch_size, num_classes).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            labels = probs.argmax(dim=-1)
        return labels, probs

    def get_cls_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract [CLS] token embeddings.

        Useful for:
        - Feature extraction
        - Similarity computation
        - Visualization (t-SNE, UMAP)
        - Explainability methods

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape (batch_size, seq_len).
        attention_mask : torch.Tensor
            Attention mask of shape (batch_size, seq_len).

        Returns
        -------
        torch.Tensor
            [CLS] embeddings of shape (batch_size, hidden_size).
        """
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return outputs.last_hidden_state[:, 0, :]

    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer: int = -1,
    ) -> torch.Tensor:
        """
        Extract attention weights from a specific layer.

        Useful for explainability and understanding what the model
        focuses on during classification.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape (batch_size, seq_len).
        attention_mask : torch.Tensor
            Attention mask of shape (batch_size, seq_len).
        layer : int, optional
            Which layer's attention to return (-1 for last layer).
            Default: -1

        Returns
        -------
        torch.Tensor
            Attention weights of shape (batch_size, num_heads, seq_len, seq_len).
        """
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            return outputs.attentions[layer]

    def save_pretrained(self, save_directory: str):
        """
        Save the model and configuration.

        Parameters
        ----------
        save_directory : str
            Directory to save the model.
        """
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save BERT model
        self.bert.save_pretrained(save_directory)

        # Save classifier head
        classifier_path = os.path.join(save_directory, "classifier.pt")
        torch.save({
            "classifier_state_dict": self.classifier.state_dict(),
            "dropout_p": self.dropout.p,
            "num_classes": self.num_classes,
        }, classifier_path)

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """
        Load a saved model.

        Parameters
        ----------
        load_directory : str
            Directory containing the saved model.
        **kwargs
            Additional arguments passed to __init__.

        Returns
        -------
        BERTSentimentClassifier
            Loaded model.
        """
        import os

        # Load classifier head configuration
        classifier_path = os.path.join(load_directory, "classifier.pt")
        if os.path.exists(classifier_path):
            classifier_data = torch.load(classifier_path, map_location="cpu")
            num_classes = classifier_data.get("num_classes", 3)
            dropout = classifier_data.get("dropout_p", 0.1)
        else:
            num_classes = kwargs.get("num_classes", 3)
            dropout = kwargs.get("dropout", 0.1)

        # Create model instance
        model = cls(
            model_name=load_directory,
            num_classes=num_classes,
            dropout=dropout,
            **kwargs,
        )

        # Load classifier weights if available
        if os.path.exists(classifier_path):
            model.classifier.load_state_dict(classifier_data["classifier_state_dict"])

        return model


def compute_class_weights(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Uses inverse frequency weighting:
        weight[c] = N / (num_classes * count[c])

    Parameters
    ----------
    labels : torch.Tensor
        Training labels of shape (N,).
    num_classes : int, optional
        Number of classes.
        Default: 3

    Returns
    -------
    torch.Tensor
        Class weights of shape (num_classes,).

    Examples
    --------
    >>> labels = torch.tensor([0, 0, 0, 1, 2, 2])
    >>> weights = compute_class_weights(labels)
    >>> # weights will up-weight class 1 (minority class)
    """
    counts = torch.bincount(labels, minlength=num_classes).float()
    total = counts.sum()
    weights = total / (num_classes * counts.clamp(min=1))
    return weights / weights.sum() * num_classes  # Normalize to sum to num_classes
