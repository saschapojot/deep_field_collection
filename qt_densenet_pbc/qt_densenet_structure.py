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
    # T1 to g1(T1)
    # S1,S2,...,S_{n-1} to F2=f1(S1),F3=f2(S2),...,Fn=f_{n-1}(S_{n-1})
    # q
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
        self.bn_layer = nn.BatchNorm2d(conv1_out_channels)
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
        # Apply circular padding to the input tensor before convolution
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')
        # First convolution
        x = self.conv1(x)  # Shape: (batch_size, conv1_out_channels, N, N)
        # Apply batch normalization
        x = self.bn_layer(x)

        # Tanh activation
        x = self.tanh(x)  # Shape: (batch_size, conv1_out_channels, N, N)
        # Apply circular padding again before the second convolution
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')
        # Second convolution
        x = self.conv2(x)  # Shape: (batch_size, conv2_out_channels, N, N)
        return x

class qt_densenet(nn.Module):
    def __init__(self, input_channels, T0_out_channels,
                 nonlinear_conv1_out_channels, nonlinear_conv2_out_channels,
                 final_out_channels=1,  # For N x N matrix
                 filter_size=5, skip_connection_num=1):
        super().__init__()
        self.skip_connection_num = skip_connection_num
        # Tanh activation
        self.tanh = nn.Tanh()
        #only has T0 layer for densenet
        self.T0_layer=TLayer(out_channels=T0_out_channels,kernel_size=filter_size)

        # NonlinearLayer for T0 and x1
        self.nonlinear_layer_T0_2_x1=NonlinearLayer(
            in_channels=nonlinear_conv2_out_channels,
            conv1_out_channels=nonlinear_conv1_out_channels,
            conv2_out_channels=nonlinear_conv2_out_channels,
            kernel_size=filter_size  # Default convolution kernel size
        )
        self.skip_nonlinear_layers = nn.ModuleList()
        for i in range(0,skip_connection_num):

            # Each layer takes more inputs as we go deeper
            # T0_output + all previous x outputs
            in_channels = T0_out_channels + (i + 1) * nonlinear_conv2_out_channels
            layer = NonlinearLayer(
                in_channels=in_channels,
                conv1_out_channels=nonlinear_conv1_out_channels,
                conv2_out_channels=nonlinear_conv2_out_channels,
                kernel_size=filter_size
            )
            self.skip_nonlinear_layers.append(layer)

        # Final mapping layer to reduce to 1 channel output
        self.final_mapping_layer = NonlinearLayer(
            in_channels=nonlinear_conv2_out_channels,
            conv1_out_channels=nonlinear_conv2_out_channels,
            conv2_out_channels=final_out_channels,  # Map to 1 channel for N x N matrix
            kernel_size=3
        )





    def forward(self,S0):
        # Step 1: Compute x1 from T0Layer and NonlinearLayer
        T0_output = self.T0_layer(S0)
        x1 = self.nonlinear_layer_T0_2_x1(T0_output)
        # Store all feature maps for concatenation
        features = [T0_output, x1]
        # Apply skip connections based on skip_connection_num
        for i in range(self.skip_connection_num):
            # Concatenate all previous features
            concatenated = torch.cat(features, dim=1)
            # Apply nonlinear layer to get the next feature map
            next_feature = self.skip_nonlinear_layers[i](concatenated)

            # Add to our features list
            features.append(next_feature)

        # Step 3: Map the final Sn to N x N matrix
        final_output = self.final_mapping_layer(next_feature)
        final_output = final_output.squeeze(1)  # Remove channel dimension (batch_size, 1, N, N) -> (batch_size, N, N)

        # Step 4: Compute the target scalar E by summing all elements in the N x N matrix
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
