import torch.nn as nn
import torch


from decimal import Decimal, getcontext

from torch.utils.data import Dataset

def format_using_decimal(value, precision=6):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)

class hs_densenet_final_sum(nn.Module):
    def __init__(self, num_spins, num_layers, growth_rate):
        super().__init__()
        self.num_spins = num_spins
        self.num_layers = num_layers
        self.growth_rate = growth_rate

        # Create the nonlinear functions f0, f1, f2, ...
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Calculate input dimension for each layer
            input_dim = num_spins + i * growth_rate

            # Each f_i is Linear -> Tanh -> Linear
            layer = nn.Sequential(
                nn.Linear(input_dim, growth_rate),
                nn.Tanh(),
                nn.Linear(growth_rate, growth_rate)
            )
            self.layers.append(layer)

    def forward(self, S0):
        # S0 shape: [batch, num_spins]
        features = [S0]  # List to store all features
        # Current concatenated state
        S_current = S0
        for i, layer in enumerate(self.layers):
            # F_i = f_i(S_i)
            F_i = layer(S_current)
            features.append(F_i)
            # S_{i+1} = concatenate(S0, F1, F2, ..., F_i)
            S_current = torch.cat(features, dim=1)

        # Final energy: sum all features
        E = S_current.sum(dim=1, keepdim=True)
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