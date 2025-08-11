import torch
import torch.nn as nn
from decimal import Decimal, getcontext
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

t=1

J=16*t
mu=-8.3*t
T=0.1*t
save_interval=25
filter_size=5

def format_using_decimal(value, precision=6):
    # Set the precision higher to ensure correct conversion
    getcontext().prec = precision + 2
    # Convert the float to a Decimal with exact precision
    decimal_value = Decimal(str(value))
    # Normalize to remove trailing zeros
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)



class Phi0Layer(nn.Module):
    def __init__(self, out_channels, kernel_size):
        """
        A modified Phi0Layer where the convolutional matrices (weights) are shared
        across the three input channels (Sigma_x, Sigma_y, Sigma_z).

        Args:
            out_channels (int): Number of output channels per input channel.
            kernel_size (int): Size of the convolution kernel.

        """
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # Padding size to maintain output size
        self.shared_conv_W0 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=0,  # No padding here
            bias=False
        )
        self.shared_conv_W1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=0,  # No padding here
            bias=False
        )

    def forward(self, x):
        """
         Forward pass for the Phi0Layer with periodic boundary conditions (PBC).
        :param x (torch.Tensor): Input tensor of shape (batch_size, 3, N, N).
        :return: torch.Tensor: Output tensor of shape (batch_size, out_channels, N, N).
        """
        # Split input into three individual channels
        x_channels = torch.chunk(x, chunks=3, dim=1)  # [(batch_size, 1, N, N), ...]
        # Apply circular padding to each channel
        padded_channels =[
            F.pad(channel, (self.padding, self.padding, self.padding, self.padding), mode='circular')
            for channel in x_channels
        ]  # [(batch_size, 1, N + 2*padding, N + 2*padding), ...]

        # Apply shared W0 and W1 convolutions to each padded channel
        conv_W0_outputs = [self.shared_conv_W0(channel) for channel in padded_channels]
        conv_W1_outputs = [self.shared_conv_W1(channel) for channel in padded_channels]

        # Perform element-wise multiplication for each channel
        multiplied_outputs = [
            conv_W0 * conv_W1 for conv_W0, conv_W1 in zip(conv_W0_outputs, conv_W1_outputs)
        ]  # [(batch_size, out_channels, N, N), ...]

        # Sum over the 3 input channels
        Phi0 = sum(multiplied_outputs)  # Shape: (batch_size, out_channels, N, N)
        return Phi0


class TLayer(nn.Module):
    def __init__(self, out_channels, kernel_size):
        """
        A layer with the same functionality as Phi0Layer:
               - Applies shared convolutional weights across three input channels.
               - Performs element-wise multiplication between convolution results.
               - Sums the results over the three input channels.
        :param out_channels (int): Number of output channels per input channel.
        :param kernel_size: Size of the convolution kernel.
        """
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # Padding size to maintain output size

        # Set padding=0 because we will handle padding manually
        self.shared_conv_W0 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=0,  # No padding here
            bias=False
        )

        self.shared_conv_W1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            padding=0,  # No padding here
            bias=False
        )

    def forward(self, x):
        """
         Forward pass for the Phi0Layer with periodic boundary conditions (PBC).
        :param x (torch.Tensor): Input tensor of shape (batch_size, 3, N, N).
        :return:  torch.Tensor: Output tensor of shape (batch_size, out_channels, N, N).
        """

        # Split input into three individual channels
        x_channels = torch.chunk(x, chunks=3, dim=1)  # [(batch_size, 1, N, N), ...]

        # Apply circular padding to each channel

        padded_channels = [
            F.pad(channel, (self.padding, self.padding, self.padding, self.padding), mode='circular')
            for channel in x_channels
        ]  # [(batch_size, 1, N + 2*padding, N + 2*padding), ...]

        # Apply shared W0 and W1 convolutions to each padded channel
        conv_W0_outputs = [self.shared_conv_W0(channel) for channel in padded_channels]
        conv_W1_outputs = [self.shared_conv_W1(channel) for channel in padded_channels]

        # Perform element-wise multiplication for each channel
        multiplied_outputs = [
            conv_W0 * conv_W1 for conv_W0, conv_W1 in zip(conv_W0_outputs, conv_W1_outputs)
        ]  # [(batch_size, out_channels, N, N), ...]
        # Sum over the 3 input channels
        T_result = sum(multiplied_outputs)  # Shape: (batch_size, out_channels, N, N)

        return T_result


class NonlinearLayer(nn.Module):
    # Phi0 to F1
    # S1,S2,...,S_{n-1} to F2=f1(S1),F3=f2(S2),...,Fn=f_{n-1}(S_{n-1})
    def __init__(self, in_channels, conv1_out_channels, conv2_out_channels, kernel_size):
        super().__init__()
        self.padding = kernel_size // 2
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv1_out_channels,
            kernel_size=kernel_size,
            padding=0,  # No padding in Conv2d layer since we'll handle it manually
            bias=True
        )
        # Tanh activation
        self.tanh = nn.Tanh()
        # Second convolution
        self.conv2 = nn.Conv2d(
            in_channels=conv1_out_channels,
            out_channels=conv2_out_channels,
            kernel_size=kernel_size,
            padding=0,  # No padding in Conv2d layer since we'll handle it manually
            bias=True
        )

    def forward(self, x):
        #Apply circular padding to the input tensor before convolution
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')
        # First convolution
        x = self.conv1(x)  # Shape: (batch_size, conv1_out_channels, N, N)
        # Tanh activation
        x = self.tanh(x)  # Shape: (batch_size, conv1_out_channels, N, N)
        # Apply circular padding again before the second convolution
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')
        # Second convolution
        x = self.conv2(x)  # Shape: (batch_size, conv2_out_channels, N, N)
        return x


class final_sum_qt_densenet(nn.Module):
    def __init__(self, input_channels, phi0_out_channels, T_out_channels,
                 nonlinear_conv1_out_channels, nonlinear_conv2_out_channels,
                 final_out_channels=1,  # For N x N matrix
                 filter_size=5, stepsAfterInit=1):
        super().__init__()
        self.stepsAfterInit = stepsAfterInit
        # Tanh activation
        self.tanh = nn.Tanh()
        # for densenet, the symmetrization layers are only Phi0 and T1
        # Phi0Layer
        self.phi0_layer = Phi0Layer(out_channels=phi0_out_channels, kernel_size=filter_size)
        # T1  Layer
        self.T1_layer = TLayer(out_channels=T_out_channels, kernel_size=filter_size)
        # NonlinearLayer for Phi0 to F1
        self.nonlinear_layer_Phi0_2_F1 = NonlinearLayer(
            in_channels=phi0_out_channels,
            conv1_out_channels=nonlinear_conv1_out_channels,
            conv2_out_channels=nonlinear_conv2_out_channels,
            kernel_size=filter_size  # Default convolution kernel size

        )
        # f1, ..., f_{n-1}
        self.f_mapping_layers = nn.ModuleList([
            NonlinearLayer(
                in_channels=T_out_channels + nonlinear_conv2_out_channels * (1 + j),  # FIX: Growing concatenation
                conv1_out_channels=nonlinear_conv1_out_channels,
                conv2_out_channels=nonlinear_conv2_out_channels,
                kernel_size=filter_size
            )
            for j in range(stepsAfterInit)
        ])
        # Final mapping layer to N x N matrix
        self.final_mapping_layer = nn.Conv2d(
            in_channels=T_out_channels + nonlinear_conv2_out_channels * (1 + stepsAfterInit),
            # FIX: Final concatenated size
            out_channels=final_out_channels,
            kernel_size=filter_size,
            padding=0,
            bias=True
        )
        self.final_padding = filter_size // 2  # Padding for the final layer

    def initialize_S1(self, x):
        #Step 1: Compute F1 from Phi0Layer and NonlinearLayer
        phi0_output = self.phi0_layer(x)
        F1 = self.nonlinear_layer_Phi0_2_F1(phi0_output)
        # Step 2: Pass input through TLayer and NonlinearLayer
        T1 = self.T1_layer(x)
        # Step 3: Concatenate T1 and F1 along the channel dimension
        S1 = torch.cat([T1, F1], dim=1)  # Shape: (batch_size, T_channels + F1_channels, N, N)
        return T1, F1, S1

    def forward(self, T1,F1,Sn):
        # Sn starts as S1 = [T1, F1] from initialize_S1
        # Start with features list containing T1 and F1
        features = [T1, F1]
        for j in range(0, self.stepsAfterInit):
            # Step 1: Compute F_{n+1} by passing S_n through NonlinearLayer
            Fn_plus_1 = self.f_mapping_layers[j](Sn)
            # Step 2: Append new feature to list
            features.append(Fn_plus_1)
            # Step 3: Concatenate all features to form S_{n+1}
            Sn = torch.cat(features, dim=1)

        # Final mapping with circular padding
        Sn = F.pad(Sn, (self.final_padding, self.final_padding,
                        self.final_padding, self.final_padding), mode='circular')

        final_output = self.final_mapping_layer(Sn)
        E = final_output.view(final_output.size(0), -1).sum(dim=1)  # Sum over all elements for each batch
        return E


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        """
        Custom dataset for supervised learning with `dsnn_qt`.

        Args:
            X (torch.Tensor): Input tensor of shape (num_samples, 3, N, N).
            Y (torch.Tensor): Target tensor of shape (num_samples,).
        """
        self.X = X
        self.Y = Y

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.Y)

    def __getitem__(self, idx):
        """
        Retrieves the input and target at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (input, target) where input is of shape (3, N, N) and target is a scalar.
        """
        return self.X[idx], self.Y[idx]



def count_parameters(model):
    """
    Counts the number of trainable and non-trainable parameters in the model.

    Args:
        model (torch.nn.Module): The model.

    Returns:
        int: Total number of parameters.
        int: Total number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
