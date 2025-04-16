import torch
import torch.nn as nn
from torch import Tensor

class Invertible1x1Conv(nn.Module):
    """
    Invertible 1x1 Convolution for normalizing flow. Performs a learned linear
    rotation at each time step, with tractable log-determinant.
    """
    def __init__(self, channels: int):
        super().__init__()
        # Initialize with a random orthonormal matrix
        w_init = torch.linalg.qr(torch.randn(channels, channels))[0]
        self.weight = nn.Parameter(w_init)

    def forward(self, z: Tensor, reverse: bool = False) -> tuple[Tensor, Tensor]:
        """
        Args:
            z:       Input tensor of shape (B, T, C)
            reverse: Whether to apply the inverse transform.
        Returns:
            z_out:  Transformed tensor of shape (B, T, C)
            logdet: Log-determinant of the Jacobian (per batch element).
        """
        B, T, C = z.size()
        # Compute log-determinant
        sign, logabsdet = torch.slogdet(self.weight)
        logdet = T * logabsdet
        # Choose weight or its inverse
        if reverse:
            weight = torch.inverse(self.weight)
            logdet = -logdet
        else:
            weight = self.weight
        # Apply linear transform: z @ W^T
        z_out = torch.matmul(z, weight.t())  # (B, T, C)
        return z_out, logdet


class ResidualCouplingBlock(nn.Module):
    """
    Affine coupling layer (residual coupling) for normalizing flows.
    Splits channels in half: x = [x1, x2], then transforms x2 conditioned on x1.
    """
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert channels % 2 == 0, "Channels must be even for coupling."
        self.split_len = channels // 2
        # Build a small conv net to predict shift and log_scale
        layers = []
        in_ch = self.split_len
        for _ in range(num_layers - 1):
            layers += [
                nn.Conv1d(in_ch, hidden_channels, kernel_size, padding=kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = hidden_channels
        # Final layer outputs 2 * split_len channels
        layers += [
            nn.Conv1d(hidden_channels, 2 * self.split_len, kernel_size, padding=kernel_size // 2)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor, reverse: bool = False) -> tuple[Tensor, Tensor]:
        """
        Args:
            z:       Input tensor of shape (B, T, C)
            reverse: Whether to invert the coupling.
        Returns:
            z_out:  Output tensor of shape (B, T, C)
            logdet: Log-determinant (sum over time and channels) per batch.
        """
        B, T, C = z.size()
        # Split along channel axis
        z1, z2 = z[:, :, :self.split_len], z[:, :, self.split_len:]
        # Prepare for conv1d: (B, C, T)
        h = z1.transpose(1, 2)  # (B, split_len, T)
        h = self.net(h)         # (B, 2*split_len, T)
        # Back to time-major
        h = h.transpose(1, 2)   # (B, T, 2*split_len)
        shift, log_scale = h.chunk(2, dim=-1)
        # Constrain scale for stability
        log_scale = torch.tanh(log_scale)

        if not reverse:
            z2_out = (z2 + shift) * torch.exp(log_scale)
            # log-det: sum over time and channels
            logdet = torch.sum(log_scale, dim=[1, 2])
        else:
            z2_out = z2 * torch.exp(-log_scale) - shift
            logdet = -torch.sum(log_scale, dim=[1, 2])

        z_out = torch.cat([z1, z2_out], dim=-1)
        return z_out, logdet


class FlowSequential(nn.Module):
    """
    Container for stacking multiple flow layers. Applies or inverts flows in order.
    """
    def __init__(self, flows: nn.ModuleList):
        super().__init__()
        self.flows = flows

    def forward(self, z: Tensor, reverse: bool = False) -> tuple[Tensor, Tensor]:
        """
        Args:
            z:       Input tensor of shape (B, T, C)
            reverse: Whether to apply flows in reverse.
        Returns:
            z_out:  Transformed tensor.
            total_logdet: Sum of log-determinants across all flows.
        """
        total_logdet = z.new_zeros(z.size(0))
        if not reverse:
            for flow in self.flows:
                z, logdet = flow(z, reverse=False)
                total_logdet = total_logdet + logdet
        else:
            for flow in reversed(self.flows):
                z, logdet = flow(z, reverse=True)
                total_logdet = total_logdet + logdet
        return z, total_logdet
