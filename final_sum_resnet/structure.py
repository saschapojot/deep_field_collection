import torch.nn as nn


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


class resnet_final_sum(nn.Module):
    def __init__(self, num_spins, num_layers, num_neurons):
        super().__init__()
        self.num_layers = num_layers

        # Projection layer for first skip connection if dimensions don't match
        if num_neurons != num_spins:
            self.input_projection = nn.Linear(num_spins, num_neurons)
        else:
            self.input_projection = None

        # Effective field layers (F_i)
        self.effective_field_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(num_spins if i == 1 else num_neurons, num_neurons),
                nn.Tanh(),
                nn.Linear(num_neurons, num_neurons)
            ) for i in range(1, num_layers + 1)]
        )

    def forward(self, S0):
        # First residual block
        F1 = self.effective_field_layers[0](S0)

        # Project S0 if needed for dimension matching
        if self.input_projection is not None:
            S0_projected = self.input_projection(S0)
        else:
            S0_projected = S0

        S1 = S0_projected + F1  # Now dimensions match

        # Current state
        S_current = S1

        # Subsequent residual blocks
        for i in range(1, self.num_layers):
            F_i = self.effective_field_layers[i](S_current)
            S_current = S_current + F_i  # Skip connection

        # Final energy as sum of all neurons
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