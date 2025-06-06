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


class DenseLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseLayer, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)  # First linear layer
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(output_size, output_size)  # Second linear layer

    def forward(self, x):
        out = self.linear1(x)
        out = self.tanh(out)
        out = self.linear2(out)
        return out


class DenseNet_no_bn(nn.Module):
    def __init__(self, num_spins, num_layers, growth_rate):
        super(DenseNet_no_bn,self).__init__()
        self.num_spins = num_spins
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        # Create dense layers
        self.dense_layers = nn.ModuleList()
        for j in range(1,self.num_layers+1):
            dim_in=self.num_spins+(j-1)*self.growth_rate
            dim_out=self.growth_rate
            self.dense_layers.append(DenseLayer(dim_in,dim_out))

    def forward(self, x):
        features = [x]
        for i in range(1, self.num_layers + 1):
            layer = self.dense_layers[i - 1]
            # Concatenate all previous features
            combined = torch.cat(features, dim=1)
            # Apply the current layer
            new_feature = layer(combined)
            # Add the new feature to our collection
            features.append(new_feature)

        E = new_feature.sum(dim=1, keepdim=True)
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
