
import torch.nn as nn
import torch

from decimal import Decimal, getcontext

from torch.utils.data import Dataset

def format_using_decimal(value, precision=10):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=1):
        super().__init__()
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # MLP block
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.Tanh(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        #Self-attention with residual connection
        attn_output, _ = self.attention(
            query=x,
            key=x,
            value=x
        )
        x = x + attn_output  # Residual connection
        # MLP with residual connection
        mlp_output = self.mlp(x)
        x = x + mlp_output  # Residual connection
        return x





class hs_longer_attn(nn.Module):
    def __init__(self, num_spins, num_layers, num_heads,embed_dim):
        super().__init__()
        # MLP to embed each real number to a vector of size embed_dim
        self.embedding = nn.Sequential(
            nn.Linear(1, embed_dim ),
            nn.ReLU(),
            nn.Linear(embed_dim , embed_dim)
        )
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=1)
            for _ in range(num_layers)
        ])
        # Final projection to scalar values before summing
        self.final_projection = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x shape: [batch_size, num_spins]
        batch_size, num_spins = x.shape
        # Embed each real number
        x_reshaped = x.view(-1, 1)  # [batch_size * num_spins, 1]
        embeddings = self.embedding(x_reshaped)  # [batch_size * num_spins, embed_dim]
        embeddings = embeddings.view(batch_size, num_spins, -1)  # [batch_size, num_spins, embed_dim]
        # Apply transformer blocks sequentially
        x = embeddings
        for block in self.transformer_blocks:
            x = block(x)
        # Project to scalar values per position
        x = self.final_projection(x)  # Shape: [batch_size, num_spins, 1]
        # Remove the last dimension to get [batch_size, num_spins]
        x = x.squeeze(-1)
        # Sum over all elements for each sample to get energy
        E = x.sum(dim=1, keepdim=True)  # Shape: [batch_size, 1]
        return E


# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]