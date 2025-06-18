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
                symmetrization layer with the same functionality as Phi0Layer:
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
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=1, dropout=0.1):
        super().__init__()
        # Layer normalization before multi-head attention
        self.norm1 = nn.LayerNorm(embed_dim)
        # Multi-head attention
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # Layer normalization before MLP
        self.norm2 = nn.LayerNorm(embed_dim)
        # MLP
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # First block: Norm -> Attention -> Residual
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output  # Skip connection

        # Second block: Norm -> MLP -> Residual
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = x + mlp_output  # Skip connection

        return x


class qt_vit(nn.Module):
    def __init__(self,input_channels,T0_out_channels,filter_size=5, layer_num=1,patch_size=2,embed_dim=70,num_heads=10, dropout=0.1,mlp_ratio=4.0):
        super().__init__()
        self.layer_num=layer_num
        # only has T0 layer for vit
        self.T0_layer = TLayer(out_channels=T0_out_channels, kernel_size=filter_size)
        # Patch to embedding projection
        patch_dim = T0_out_channels * patch_size * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, embed_dim)
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.num_patches = (10 // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Initialize parameters
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(layer_num)
        ])



    def divide_into_patches(self, x, patch_size=2):
        """
        Divide input tensor into patches
        Input: x of shape [batch_size, C, N, N]
        Output: patches of shape [batch_size, num_patches, C*P*P]
        """
        B, C, N, _ = x.shape
        P = patch_size
        assert N % P == 0, f"Lattice size {N} must be divisible by patch size {P}"

        # Number of patches along each dimension
        n_patches = N // P

        # Reshape to separate patches: [B, C, n_patches, P, n_patches, P]
        x = x.reshape(B, C, n_patches, P, n_patches, P)

        # Permute to group dimensions: [B, n_patches, n_patches, C, P, P]
        x = x.permute(0, 2, 4, 1, 3, 5)

        # Reshape to [B, total_patches, flattened_patch_dim]
        patches = x.reshape(B, n_patches * n_patches, C * P * P)

        return patches

    def forward(self, S0):
        B = S0.shape[0]  # Batch size
        # Step 1: Compute T0
        T0_output = self.T0_layer(S0)
        patches=self.divide_into_patches(T0_output)
        # Step 3: Linear projection of flattened patches
        x = self.patch_to_embedding(patches)  # [B, num_patches, embed_dim]
        # Step 4: Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + num_patches, embed_dim]
        # Step 5: Add position embeddings
        x = x + self.pos_embedding

        # Apply dropout
        x = self.dropout(x)

        # Step 6: Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Step 7: Compute energy by summing all elements for each batch

        E = x.view(B, -1).sum(dim=1)  # [B]

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
