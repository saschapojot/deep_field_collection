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
    # T1 to g1(T1)
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
        # Apply circular padding to the input tensor before convolution
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

class final_sum_qt_efnn(nn.Module):
    def __init__(self, input_channels, phi0_out_channels, T_out_channels,
                 nonlinear_conv1_out_channels, nonlinear_conv2_out_channels,
                 final_out_channels=1,  # For N x N matrix
                 filter_size=5, stepsAfterInit=1):
        super().__init__()
        self.stepsAfterInit = stepsAfterInit
        # Phi0Layer
        self.phi0_layer = Phi0Layer(out_channels=phi0_out_channels, kernel_size=filter_size)
        # T1  Layer
        self.T1_layer = TLayer(out_channels=T_out_channels, kernel_size=filter_size)

        # T layers after init
        self.T_layers_after_init = nn.ModuleList([
            TLayer(out_channels=T_out_channels, kernel_size=filter_size) for _ in
            range(0, stepsAfterInit)
        ])

        # NonlinearLayer for Phi0 and F1
        self.nonlinear_layer_Phi0_2_F1 = NonlinearLayer(
            in_channels=nonlinear_conv2_out_channels,
            conv1_out_channels=nonlinear_conv1_out_channels,
            conv2_out_channels=nonlinear_conv2_out_channels,
            kernel_size=filter_size  # Default convolution kernel size

        )

        # f1, ..., f_{n-1}
        self.f_mapping_layers = nn.ModuleList([
            NonlinearLayer(
                in_channels=nonlinear_conv2_out_channels,
                conv1_out_channels=nonlinear_conv1_out_channels,
                conv2_out_channels=nonlinear_conv2_out_channels,
                kernel_size=filter_size  # Default convolution kernel size

            )
            for _ in range(0, stepsAfterInit)
        ])

        # NonlinearLayer for T1 and S1 (g1)
        self.nonlinear_layer_T1_2_S1 = NonlinearLayer(
            in_channels=T_out_channels,
            conv1_out_channels=nonlinear_conv1_out_channels,
            conv2_out_channels=nonlinear_conv2_out_channels,
            kernel_size=filter_size
        )

        # g2,...,gn
        self.g_mapping_layers = nn.ModuleList([
            NonlinearLayer(
                in_channels=nonlinear_conv2_out_channels,
                conv1_out_channels=nonlinear_conv1_out_channels,
                conv2_out_channels=nonlinear_conv2_out_channels,
                kernel_size=filter_size  # Default convolution kernel size

            )
            for _ in range(0, stepsAfterInit)
        ])

        # Final mapping layer to N x N matrix
        self.final_mapping_layer =nn.Conv2d(
            in_channels=nonlinear_conv2_out_channels,
            out_channels=final_out_channels,
            kernel_size=filter_size,  # Using kernel_size=5 like the rest of the network
            padding=0,  # No padding here since we'll handle it manually
            bias=True
        )
        self.final_padding = filter_size // 2  # Padding for the final layer

    def initialize_S1(self, x):
        # Step 1: Compute F1 from Phi0Layer and NonlinearLayer
        phi0_output = self.phi0_layer(x)
        F1 = self.nonlinear_layer_Phi0_2_F1(phi0_output)
        # Step 2: Pass input through TLayer and NonlinearLayer
        T_output = self.T1_layer(x)
        nonlinear_output = self.nonlinear_layer_T1_2_S1(T_output)
        # Step 3: Compute S1 as pointwise multiplication of F1 and nonlinear_output
        S1 = F1 * nonlinear_output
        return S1

    def forward(self, x, Sn):
        for j in range(0, self.stepsAfterInit):
            # Step 1: Compute F_{n+1} by passing S_n through NonlinearLayer
            Fn_plus_1 = self.f_mapping_layers[j](Sn)
            # Step 2: Pass input through TLayer and NonlinearLayer
            T_output = self.T_layers_after_init[j](x)
            nonlinear_output = self.g_mapping_layers[j](T_output)
            # Step 3: Compute S_{n+1} as pointwise multiplication of Fn_plus_1 and nonlinear_output
            Sn = Fn_plus_1 * nonlinear_output


        # Step 4: Apply circular padding before final convolution
        Sn_padded = F.pad(Sn, (self.final_padding, self.final_padding, self.final_padding, self.final_padding), mode='circular')
        # Map the final Sn to N x N matrix
        final_output = self.final_mapping_layer(Sn_padded)
        E = final_output.view(final_output.size(0), -1).sum(dim=1)  # Sum over all elements for each batch
        # print(f"E.shape={E.shape}")
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
