import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1), :].transpose(0, 1)


class CustomMultiheadAttention(nn.Module):
    """Custom multihead attention that doesn't use efficient attention backend."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear transformation
        output = self.w_o(context)
        return output


class CustomTransformerEncoderLayer(nn.Module):
    """Custom transformer encoder layer with custom attention."""

    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.0, activation="relu"):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, n_heads, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src, src_mask=None):
        # Self attention
        src2 = self.self_attn(src, src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feed forward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class CustomTransformerEncoder(nn.Module):
    """Custom transformer encoder with custom layers."""

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src, mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, mask)
        return output


class ModularAdditionTransformer(nn.Module):
    def __init__(self, p, d_model=128, n_heads=4, n_layers=1, dropout=0.0):
        super().__init__()
        self.p = p
        self.d_model = d_model

        # Token embeddings for inputs
        self.embed = nn.Embedding(p, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        # Custom transformer layers
        encoder_layer = CustomTransformerEncoderLayer(
            d_model=d_model,
            n_heads=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="relu",
        )
        self.transformer = CustomTransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, p)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize embeddings and linear layers
        nn.init.normal_(self.embed.weight, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        # x shape: (batch_size, 2) containing [a, b]
        # Embed inputs
        embedded = self.embed(x)  # (batch_size, 2, d_model)
        embedded = self.pos_enc(embedded)

        # Pass through transformer
        output = self.transformer(embedded)  # (batch_size, 2, d_model)

        # Use the output at the last position for prediction
        final_output = output[:, -1, :]  # (batch_size, d_model)

        # Project to vocabulary size
        logits = self.output_proj(final_output)  # (batch_size, p)

        return logits


class SimpleModularAdditionMLP(nn.Module):
    """Simple MLP for modular addition that supports Hessian computation."""

    def __init__(self, p, hidden_size=128, num_layers=2):
        super().__init__()
        self.p = p

        # Input embedding
        self.input_embed = nn.Embedding(p, hidden_size)

        # MLP layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(hidden_size * 2, hidden_size))  # 2 inputs concatenated
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        self.mlp = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_size, p)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.input_embed.weight, mean=0, std=0.02)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        # x shape: (batch_size, 2) containing [a, b]

        # Embed inputs
        a_emb = self.input_embed(x[:, 0])  # (batch_size, hidden_size)
        b_emb = self.input_embed(x[:, 1])  # (batch_size, hidden_size)

        # Concatenate embeddings
        combined = torch.cat([a_emb, b_emb], dim=1)  # (batch_size, hidden_size * 2)

        # Pass through MLP
        hidden = self.mlp(combined)

        # Output projection
        logits = self.output_proj(hidden)  # (batch_size, p)

        return logits
