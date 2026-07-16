"""
Multi-Head Attention Mechanism for Sentiment Analysis

Implements scaled dot-product attention with multiple heads for capturing
different aspects of sentiment in text sequences.

Reference:
    Vaswani et al., "Attention is All You Need", NeurIPS 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism for sentiment analysis.

    This attention module allows the model to jointly attend to information
    from different representation subspaces at different positions.

    Args:
        d_model (int): Dimension of model (BiLSTM output dimension)
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability

    Input:
        x: Tensor of shape (batch_size, seq_len, d_model)
        mask: Optional mask tensor (batch_size, seq_len)

    Output:
        attended: Tensor of shape (batch_size, seq_len, d_model)
        attention_weights: Tensor of shape (batch_size, num_heads, seq_len, seq_len)

    Example:
        >>> attention = MultiHeadAttention(d_model=256, num_heads=4, dropout=0.1)
        >>> x = torch.randn(32, 200, 256)  # (batch, seq_len, d_model)
        >>> output, weights = attention(x)
        >>> print(output.shape)  # torch.Size([32, 200, 256])
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

        Args:
            Q: Query tensor (batch, num_heads, seq_len, d_k)
            K: Key tensor (batch, num_heads, seq_len, d_k)
            V: Value tensor (batch, num_heads, seq_len, d_k)
            mask: Optional mask (batch, 1, 1, seq_len)

        Returns:
            attended: Attention-weighted values (batch, num_heads, seq_len, d_k)
            attention_weights: Attention scores (batch, num_heads, seq_len, seq_len)
        """
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided (for padding tokens)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)

        return attended, attention_weights

    def forward(self, x, mask=None):
        """
        Forward pass through multi-head attention.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional mask (batch_size, seq_len)

        Returns:
            output: Attended output (batch_size, seq_len, d_model)
            attention_weights: Attention weights (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.size()

        # Residual connection
        residual = x

        # Linear projections and reshape for multi-head
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Prepare mask for broadcasting
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)

        # Compute attention
        attended, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.W_o(attended)
        output = self.dropout(output)

        # Residual connection and layer normalization
        output = self.layer_norm(output + residual)

        return output, attention_weights


class AttentionPooling(nn.Module):
    """
    Attention-based pooling to aggregate sequence into fixed-size representation.

    Instead of simple averaging or max pooling, this learns to weight each position
    based on its importance for the final sentiment classification.

    Args:
        d_model (int): Dimension of input features
        dropout (float): Dropout probability

    Input:
        x: Tensor of shape (batch_size, seq_len, d_model)
        mask: Optional mask (batch_size, seq_len)

    Output:
        pooled: Tensor of shape (batch_size, d_model)
        attention_weights: Tensor of shape (batch_size, seq_len)

    Example:
        >>> pooling = AttentionPooling(d_model=256, dropout=0.1)
        >>> x = torch.randn(32, 200, 256)
        >>> pooled, weights = pooling(x)
        >>> print(pooled.shape)  # torch.Size([32, 256])
    """

    def __init__(self, d_model, dropout=0.1):
        super(AttentionPooling, self).__init__()

        self.attention = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Apply attention pooling.

        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Optional mask (batch_size, seq_len)

        Returns:
            pooled: Weighted sum (batch_size, d_model)
            attention_weights: Attention scores (batch_size, seq_len)
        """
        # Compute attention scores for each position
        # (batch, seq_len, d_model) -> (batch, seq_len, 1) -> (batch, seq_len)
        attention_scores = self.attention(x).squeeze(-1)

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum: (batch, seq_len, 1) * (batch, seq_len, d_model) -> (batch, d_model)
        pooled = torch.sum(attention_weights.unsqueeze(-1) * x, dim=1)

        return pooled, attention_weights


if __name__ == "__main__":
    # Unit test for attention modules
    print("Testing Multi-Head Attention...")

    batch_size = 32
    seq_len = 200
    d_model = 256
    num_heads = 4

    # Test MultiHeadAttention
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=0.1)
    x = torch.randn(batch_size, seq_len, d_model)
    output, attention_weights = mha(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    assert output.shape == (batch_size, seq_len, d_model), "Output shape mismatch"
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len), "Attention weights shape mismatch"
    print("✓ MultiHeadAttention test passed\n")

    # Test AttentionPooling
    print("Testing Attention Pooling...")
    pooling = AttentionPooling(d_model=d_model, dropout=0.1)
    pooled, pool_weights = pooling(x)

    print(f"Input shape: {x.shape}")
    print(f"Pooled shape: {pooled.shape}")
    print(f"Pooling weights shape: {pool_weights.shape}")
    assert pooled.shape == (batch_size, d_model), "Pooled shape mismatch"
    assert pool_weights.shape == (batch_size, seq_len), "Pooling weights shape mismatch"
    print("✓ AttentionPooling test passed\n")

    print("All attention module tests passed!")
