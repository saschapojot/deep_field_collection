import torch
import torch.nn as nn
from itertools import combinations
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

class no_bn_one_residual_block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(no_bn_one_residual_block, self).__init__()
        # Main path with two linear transformations
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(output_dim, output_dim)
        # Shortcut connection (identity or projection)
        self.shortcut = nn.Identity()
        if input_dim != output_dim:
            self.shortcut =nn.Linear(input_dim, output_dim)

    def forward(self, x):
        identity = x
        # Main path
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        # Add the shortcut to the main path
        out += self.shortcut(identity)
        # Final activation
        out = self.tanh(out)
        return out


class hs_resnet(nn.Module):
    def __init__(self, num_spins, num_layers, num_neurons):
        """

        :param num_spins: Number of spins in the system (input size).
        :param num_layers: Number of  layers.
        :param num_neurons: Number of neurons
        """
        super(hs_resnet,self).__init__()
        layers = []
        # Input dimension
        prev_dim = num_spins

        # Stack residual blocks in series
        for _ in range(num_layers):
            layers.append(no_bn_one_residual_block(prev_dim, num_neurons))
            prev_dim = num_neurons
        self.resblock_series = nn.Sequential(*layers)
        print(f"len(layers)={len(layers)}")

    def forward(self, x):
        final_arr = self.resblock_series(x)
        E = final_arr.sum(dim=1, keepdim=True)  # Sum over all elements for each sample
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

