"""
Hybrid CNN-BiLSTM-Attention Architecture for Sentiment Analysis

This module implements a novel hybrid deep learning architecture combining:
- CNN: Captures local n-gram patterns (3, 4, 5-word phrases)
- BiLSTM: Models sequential dependencies and context
- Multi-Head Attention: Focuses on sentiment-bearing words
- Feature Fusion: Combines CNN and BiLSTM-Attention representations

This architecture is designed for Master's thesis-level research in
Computational Intelligence for YouTube comment sentiment analysis.

Author: Master's Thesis - Computational Intelligence
Date: 2026-01-14
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention, AttentionPooling


class CNNBranch(nn.Module):
    """
    CNN branch for capturing local n-gram patterns in text.

    Uses multiple filter sizes (3, 4, 5) to capture phrases of different lengths,
    similar to "not good", "very bad movie", "absolutely amazing performance".

    Args:
        embedding_dim (int): Dimension of word embeddings
        filter_sizes (list): List of filter sizes (e.g., [3, 4, 5])
        num_filters (int): Number of filters per size
        dropout (float): Dropout probability

    Input:
        embedded: Tensor of shape (batch_size, seq_len, embedding_dim)

    Output:
        features: Tensor of shape (batch_size, num_filters * len(filter_sizes))
    """

    def __init__(self, embedding_dim, filter_sizes=[3, 4, 5], num_filters=128, dropout=0.3):
        super(CNNBranch, self).__init__()

        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        # Create convolution layers for each filter size
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=fs,
                padding='same'  # Keeps sequence length same
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded):
        """
        Forward pass through CNN branch.

        Args:
            embedded: (batch_size, seq_len, embedding_dim)

        Returns:
            features: (batch_size, num_filters * len(filter_sizes))
        """
        # Conv1d expects (batch, channels, seq_len), so transpose
        x = embedded.transpose(1, 2)  # (batch, embedding_dim, seq_len)

        # Apply convolutions and max pooling for each filter size
        conv_outputs = []
        for conv in self.convs:
            # Convolution + ReLU
            conv_out = F.relu(conv(x))  # (batch, num_filters, seq_len)

            # Global max pooling
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch, num_filters, 1)
            pooled = pooled.squeeze(2)  # (batch, num_filters)

            conv_outputs.append(pooled)

        # Concatenate all filter outputs
        features = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(filter_sizes))
        features = self.dropout(features)

        return features


class BiLSTMBranch(nn.Module):
    """
    Bidirectional LSTM branch for capturing sequential dependencies.

    BiLSTM processes text in both forward and backward directions, allowing
    it to capture context from both past and future words.

    Args:
        embedding_dim (int): Dimension of word embeddings
        hidden_size (int): LSTM hidden state size
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout probability between layers

    Input:
        embedded: Tensor of shape (batch_size, seq_len, embedding_dim)
        lengths: Actual lengths of sequences (for packing)

    Output:
        lstm_out: Tensor of shape (batch_size, seq_len, hidden_size * 2)
    """

    def __init__(self, embedding_dim, hidden_size=128, num_layers=2, dropout=0.3):
        super(BiLSTMBranch, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded, lengths=None):
        """
        Forward pass through BiLSTM branch.

        Args:
            embedded: (batch_size, seq_len, embedding_dim)
            lengths: Actual sequence lengths for packing (optional)

        Returns:
            lstm_out: (batch_size, seq_len, hidden_size * 2)
        """
        # Pack sequences for efficiency (handles variable lengths)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)

        lstm_out = self.dropout(lstm_out)

        return lstm_out


class HybridCNNBiLSTM(nn.Module):
    """
    Hybrid CNN-BiLSTM-Attention model for sentiment analysis.

    This novel architecture combines:
    1. Embedding layer (with pre-trained GloVe/FastText)
    2. Parallel branches:
       - CNN: Local n-gram patterns
       - BiLSTM + Multi-Head Attention: Sequential context with focus
    3. Feature fusion: Concatenate CNN and BiLSTM-Attention features
    4. Classification head: Dense layers with dropout

    Args:
        vocab_size (int): Vocabulary size
        embedding_dim (int): Embedding dimension (e.g., 300 for GloVe)
        num_classes (int): Number of sentiment classes (3: Pos/Neu/Neg)
        cnn_filter_sizes (list): CNN filter sizes
        cnn_num_filters (int): Number of CNN filters per size
        lstm_hidden_size (int): LSTM hidden state size
        lstm_num_layers (int): Number of LSTM layers
        attention_num_heads (int): Number of attention heads
        classifier_hidden_sizes (list): Hidden layer sizes for classifier
        dropout_cnn (float): Dropout for CNN branch
        dropout_lstm (float): Dropout for LSTM branch
        dropout_attention (float): Dropout for attention
        dropout_classifier (list): Dropout rates for classifier layers
        pretrained_embeddings (torch.Tensor): Pre-trained embedding matrix
        freeze_embeddings (bool): Whether to freeze embeddings

    Input:
        input_ids: Token indices (batch_size, seq_len)
        lengths: Actual sequence lengths (optional)

    Output:
        logits: Class logits (batch_size, num_classes)
        attention_weights: Attention weights for interpretability
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=300,
        num_classes=3,
        cnn_filter_sizes=[3, 4, 5],
        cnn_num_filters=128,
        lstm_hidden_size=128,
        lstm_num_layers=2,
        attention_num_heads=4,
        classifier_hidden_sizes=[256, 128],
        dropout_cnn=0.3,
        dropout_lstm=0.3,
        dropout_attention=0.1,
        dropout_classifier=[0.5, 0.4],
        pretrained_embeddings=None,
        freeze_embeddings=False
    ):
        super(HybridCNNBiLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # CNN Branch
        self.cnn_branch = CNNBranch(
            embedding_dim=embedding_dim,
            filter_sizes=cnn_filter_sizes,
            num_filters=cnn_num_filters,
            dropout=dropout_cnn
        )
        cnn_output_dim = cnn_num_filters * len(cnn_filter_sizes)

        # BiLSTM Branch
        self.bilstm_branch = BiLSTMBranch(
            embedding_dim=embedding_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout_lstm
        )
        lstm_output_dim = lstm_hidden_size * 2  # Bidirectional

        # Multi-Head Attention
        self.attention = MultiHeadAttention(
            d_model=lstm_output_dim,
            num_heads=attention_num_heads,
            dropout=dropout_attention
        )

        # Attention Pooling
        self.attention_pooling = AttentionPooling(
            d_model=lstm_output_dim,
            dropout=dropout_attention
        )

        # Feature fusion dimension
        fusion_dim = cnn_output_dim + lstm_output_dim

        # Classification head
        classifier_layers = []
        input_dim = fusion_dim

        for i, hidden_size in enumerate(classifier_hidden_sizes):
            classifier_layers.append(nn.Linear(input_dim, hidden_size))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(dropout_classifier[i] if i < len(dropout_classifier) else 0.3))
            input_dim = hidden_size

        # Output layer
        classifier_layers.append(nn.Linear(input_dim, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, input_ids, lengths=None):
        """
        Forward pass through the hybrid model.

        Args:
            input_ids: Token indices (batch_size, seq_len)
            lengths: Actual sequence lengths (batch_size,)

        Returns:
            logits: Class logits (batch_size, num_classes)
            attention_weights: Dict with attention weights for analysis
        """
        # Embedding
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)

        # CNN Branch
        cnn_features = self.cnn_branch(embedded)  # (batch, cnn_output_dim)

        # BiLSTM Branch
        lstm_out = self.bilstm_branch(embedded, lengths)  # (batch, seq_len, lstm_output_dim)

        # Multi-Head Attention
        attended, mha_weights = self.attention(lstm_out)  # (batch, seq_len, lstm_output_dim)

        # Attention Pooling
        lstm_features, pooling_weights = self.attention_pooling(attended)  # (batch, lstm_output_dim)

        # Feature Fusion
        fused_features = torch.cat([cnn_features, lstm_features], dim=1)  # (batch, fusion_dim)

        # Classification
        logits = self.classifier(fused_features)  # (batch, num_classes)

        # Collect attention weights for interpretability
        attention_weights = {
            'multi_head_attention': mha_weights,  # (batch, num_heads, seq_len, seq_len)
            'attention_pooling': pooling_weights  # (batch, seq_len)
        }

        return logits, attention_weights

    def predict(self, input_ids, lengths=None):
        """
        Predict sentiment classes with probabilities.

        Args:
            input_ids: Token indices (batch_size, seq_len)
            lengths: Actual sequence lengths (optional)

        Returns:
            predictions: Predicted class indices (batch_size,)
            probabilities: Class probabilities (batch_size, num_classes)
        """
        logits, _ = self.forward(input_ids, lengths)
        probabilities = F.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        return predictions, probabilities


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Unit test for hybrid model
    print("Testing Hybrid CNN-BiLSTM-Attention Model...")

    # Model hyperparameters
    vocab_size = 20000
    embedding_dim = 300
    batch_size = 32
    seq_len = 200
    num_classes = 3

    # Create model
    model = HybridCNNBiLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        cnn_filter_sizes=[3, 4, 5],
        cnn_num_filters=128,
        lstm_hidden_size=128,
        lstm_num_layers=2,
        attention_num_heads=4,
        classifier_hidden_sizes=[256, 128],
    )

    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal Parameters: {count_parameters(model):,}")

    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    lengths = torch.randint(50, seq_len, (batch_size,))

    print(f"\nInput shape: {input_ids.shape}")
    logits, attention_weights = model(input_ids, lengths)

    print(f"Output logits shape: {logits.shape}")
    print(f"Multi-head attention weights shape: {attention_weights['multi_head_attention'].shape}")
    print(f"Pooling attention weights shape: {attention_weights['attention_pooling'].shape}")

    # Test prediction
    predictions, probabilities = model.predict(input_ids, lengths)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Probabilities shape: {probabilities.shape}")
    print(f"Sum of probabilities: {probabilities.sum(dim=1)[:5]}")  # Should be ~1.0

    # Verify shapes
    assert logits.shape == (batch_size, num_classes), "Logits shape mismatch"
    assert predictions.shape == (batch_size,), "Predictions shape mismatch"
    assert probabilities.shape == (batch_size, num_classes), "Probabilities shape mismatch"

    print("\n✓ All tests passed!")
    print("Hybrid CNN-BiLSTM-Attention model is ready for training!")
