
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

class CustomAttention(nn.Module):
    def __init__(self):
        super(CustomAttention, self).__init__()
        # Initialize trainable scalar parameters a and b
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        # x shape: [batch_size, feature_dim]
        # batch_size = x.size(0)
        # Reshape x to [batch_size, feature_dim, 1] for matrix operations
        x_col = x.unsqueeze(2)
        # Compute outer product xx^T for each sample in batch
        # Shape: [batch_size, feature_dim, feature_dim]
        outer_product = torch.bmm(x_col, x_col.transpose(1, 2))
        # Scale by parameter a
        scaled_outer = self.a * outer_product
        # Apply tanh
        attention_weights = torch.tanh(scaled_outer)
        # Multiply by x and scale by b
        # Shape after bmm: [batch_size, feature_dim, 1]
        output = torch.bmm(attention_weights, x_col)
        # Final shape: [batch_size, feature_dim]
        return self.b * output.squeeze(2)


class hs_attn(nn.Module):
    def __init__(self, num_spins, num_layers, num_neurons):
        """

                :param num_spins:  Number of spins in the system (input size).
                :param num_layers: Number of attn layers.
                :param num_neurons: Number of dimension
        """
        super(hs_attn,self).__init__()
        self.num_layers = num_layers
        # Initial linear layer to map from num_spins to num_neurons
        self.input_linear = nn.Linear(num_spins, num_neurons)
        # Create attention layers
        self.attention_layers = nn.ModuleList([
            CustomAttention() for _ in range(num_layers)
        ])
        # Create linear layers (one after each attention)
        self.linear_layers = nn.ModuleList([
            nn.Linear(num_neurons, num_neurons) for _ in range(num_layers)
        ])

    def forward(self, S0):
        # S0 shape: [batch_size, num_spins]
        # Initial linear mapping
        x = self.input_linear(S0)
        # Apply attention + linear layers for num_layers times
        for i in range(self.num_layers):
            attention_output = self.attention_layers[i](x)
            # Apply linear layer with residual connection
            x = x + self.linear_layers[i](attention_output)
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