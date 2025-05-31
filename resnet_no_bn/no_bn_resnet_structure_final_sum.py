import torch.nn as nn


from decimal import Decimal, getcontext

from torch.utils.data import Dataset, DataLoader

def format_using_decimal(value, precision=6):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)



class no_bn_one_residual_block(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dim=None):
        super(no_bn_one_residual_block,self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim

        # Main path with two linear transformations
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        # self.bn2 = nn.BatchNorm1d(output_dim)

        # Shortcut connection (identity or projection)
        self.shortcut = nn.Identity()
        if input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )

    def forward(self, x):
        identity = x
        # Main path
        out = self.linear1(x)
        # out = self.bn1(out)
        out = self.tanh(out)

        out = self.linear2(out)
        # out = self.bn2(out)
        # Add the shortcut to the main path
        out += self.shortcut(identity)
        # Final activation
        out = self.tanh(out)
        return out
class no_bn_resnet_final_sum(nn.Module):
    def __init__(self, num_spins, num_layers, num_neurons):
        super(no_bn_resnet_final_sum,self).__init__()
        layers = []
        # Input dimension
        prev_dim = num_spins

        # Stack residual blocks in series
        for _ in range(num_layers):
            layers.append(no_bn_one_residual_block(prev_dim,num_neurons))
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

