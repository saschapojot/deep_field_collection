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


class hs_efnn(nn.Module):
    def __init__(self, num_spins, num_layers, num_neurons):
        """

        :param num_spins:  Number of spins in the system (input size).
        :param num_layers: Number of FP layers.
        :param num_neurons: Number of neurons in each FP layer.
        """

        super(hs_efnn,self).__init__()
        self.num_layers = num_layers
        # Effective field layers (F_i)
        self.effective_field_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(num_spins if i == 1 else num_neurons, num_neurons),
                nn.Tanh(),
                # ReciprocalActivation(),
                nn.Linear(num_neurons, num_neurons)
            ) for i in range(1, num_layers + 1)]
        )
        # Quasi-particle layers (S_i)
        self.quasi_particle_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(num_spins, num_neurons),
                nn.Tanh(),
                # ReciprocalActivation(),
                nn.Linear(num_neurons, num_neurons)
            ) for i in range(1, num_layers + 1)]
        )

    def forward(self, S0):
        """

        :param S0 (torch.Tensor): Input spin configurations, shape (batch_size, num_spins).
        :return:   torch.Tensor: Predicted energy, shape (batch_size, 1).
        """
        # Initialize S as the input spin configuration
        S = S0
        for i in range(1, self.num_layers + 1):
            # Compute effective field layer Fi
            Fi = self.effective_field_layers[i - 1](S)
            # Compute quasi-particle layer Si
            Si = self.quasi_particle_layers[i - 1](S0) * Fi
            # Update S for the next layer
            S = Si

        # Output layer to compute energy
        E = S.sum(dim=1, keepdim=True)  # Sum over all elements for each sample
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