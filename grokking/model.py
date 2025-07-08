import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ModularAdditionTransformer(nn.Module):
    def __init__(self, p, d_model=128, n_heads=4, n_layers=1, dropout=0.0):
        super().__init__()
        self.p = p
        self.d_model = d_model

        # Token embeddings for inputs
        self.embed = nn.Embedding(p, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

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