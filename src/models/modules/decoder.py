import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrizations

class ResBlock(nn.Module):
    """
    Residual block for HiFi-GAN generator. Implements two parallel dilated convolutions.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: list = [1, 3, 5],
        dropout: float = 0.1
    ):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for dilation in dilations:
            padding = (kernel_size * dilation - dilation) // 2
            self.convs1.append(parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=padding)
            ))
            self.convs2.append(parametrizations.weight_norm(
                nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size-1)//2)
            ))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            residual = x
            x = F.leaky_relu(x, 0.1)
            x = c1(x)
            x = F.leaky_relu(x, 0.1)
            x = self.dropout(x)
            x = c2(x)
            x = x + residual
        return x

class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN generator (decoder) for waveform synthesis.
    Based on Kong et al. (2020).
    """
    def __init__(
        self,
        in_channels: int,
        upsample_rates: list,
        upsample_kernel_sizes: list,
        resblock_kernel_sizes: list,
        resblock_dilation_sizes: list,
        channels: int = 512,
        out_channels: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.pre = parametrizations.weight_norm(
            nn.Conv1d(in_channels, channels, kernel_size=7, padding=3)
        )
        # Upsampling layers and corresponding ResBlock groups
        self.ups = nn.ModuleList()
        self.resblock_groups = nn.ModuleList()  # list of ModuleList per scale

        cur_channels = channels
        for r, k_up in zip(upsample_rates, upsample_kernel_sizes):
            # ConvTranspose to upsample and halve channels
            self.ups.append(parametrizations.weight_norm(
                nn.ConvTranspose1d(cur_channels, cur_channels // 2, k_up, stride=r, padding=(k_up - r) // 2)
            ))
            next_channels = cur_channels // 2

            # Build ResBlocks for this scale
            resblocks_this_scale = nn.ModuleList()
            for k_rb, d_list in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                resblocks_this_scale.append(ResBlock(next_channels, k_rb, d_list, dropout))
            self.resblock_groups.append(resblocks_this_scale)

            # Update channel count for next scale
            cur_channels = next_channels

        # Final conv to waveform
        self.post = parametrizations.weight_norm(
            nn.Conv1d(cur_channels, out_channels, kernel_size=7, padding=3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, T) intermediate latent representation
        Returns:
            waveform: (B, 1, T * prod(upsample_rates)) synthesized audio
        """
        x = self.pre(x)
        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            # Apply the ResBlock group corresponding to this upsampling scale
            resblocks = self.resblock_groups[i]
            for rb in resblocks:
                x = rb(x)
        x = F.leaky_relu(x, 0.1)
        x = self.post(x)
        x = torch.tanh(x)
        return x