import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InductionTransformer(nn.Module):
    """
    Transformer designed for studying induction head formation.

    Features:
    - Accessible attention patterns for analysis
    - Hooks for gradient/Hessian computation
    - Minimal architecture to isolate induction head behavior
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.use_layernorm = use_layernorm

        # Token and position embeddings
        self.token_embed = nn.Embedding(vocab_size + 1, d_model)  # +1 for padding token
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, use_layernorm=use_layernorm
                )
                for _ in range(n_layers)
            ]
        )

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size + 1)

        # Storage for attention patterns (for analysis)
        self.attention_patterns = {}

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

        # Initialize output projection
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input token ids of shape (batch_size, seq_len)
            return_attention: Whether to return attention patterns

        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size + 1)
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Create position indices
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # Embed tokens and positions
        token_emb = self.token_embed(x)
        pos_emb = self.pos_embed(positions)

        # Combine embeddings
        hidden = token_emb + pos_emb

        # Clear attention storage
        self.attention_patterns = {}

        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            hidden, attn = layer(hidden, return_attention=True)

            if return_attention:
                self.attention_patterns[f"layer_{i}"] = attn

        # Project to vocabulary
        logits = self.output_proj(hidden)

        return logits

    def get_attention_patterns(self) -> Dict[str, torch.Tensor]:
        """Return stored attention patterns from last forward pass."""
        return self.attention_patterns


class TransformerLayer(nn.Module):
    """Single transformer layer with accessible internals."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, use_layernorm: bool = True):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, return_weights=True)
        x = x + self.dropout(attn_out)

        if self.use_layernorm:
            x = self.norm1(x)

        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = x + self.dropout(ff_out)

        if self.use_layernorm:
            x = self.norm2(x)

        return x, attn_weights if return_attention else None


class MultiHeadAttention(nn.Module):
    """Multi-head attention with detailed weight access."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Linear projections
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        for module in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Apply causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(scores.device)
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_O(attn_output)

        return output, attn_weights if return_weights else None


class FeedForward(nn.Module):
    """Feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def visualize_attention_patterns(
    model: InductionTransformer,
    sequences: torch.Tensor,
    pattern_info: list,
    layer_idx: int = 0,
    head_idx: int = 0,
    seq_idx: int = 0,
) -> Optional[dict]:
    """
    Visualize attention patterns to identify induction heads.

    Returns dict with attention pattern analysis.
    """
    model.eval()

    with torch.no_grad():
        # Run forward pass
        _ = model(sequences[seq_idx : seq_idx + 1], return_attention=True)

        # Get attention patterns
        attn_patterns = model.get_attention_patterns()
        layer_key = f"layer_{layer_idx}"

        if layer_key in attn_patterns:
            # Shape: (batch=1, n_heads, seq_len, seq_len)
            attn = attn_patterns[layer_key][0, head_idx].cpu().numpy()

            # Analyze induction behavior
            pattern_data = pattern_info[seq_idx]

            induction_scores = []
            for pattern in pattern_data:
                positions = pattern["positions"]
                length = pattern["length"]

                # Check if later occurrences attend to earlier ones
                if len(positions) >= 2:
                    for i in range(1, len(positions)):
                        for j in range(length):
                            if positions[i] + j < attn.shape[0]:
                                # Attention from position in later occurrence
                                # to corresponding position in earlier occurrence
                                score = attn[positions[i] + j, positions[0] + j]
                                induction_scores.append(score)

            return {
                "attention_matrix": attn,
                "induction_scores": induction_scores,
                "mean_induction_score": np.mean(induction_scores) if induction_scores else 0,
                "pattern_info": pattern_data,
            }

    return None
