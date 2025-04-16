import torch
import torch.nn as nn

class BNEncoder(nn.Module):
    """
    Bottleneck (BN) Encoder for accent pronunciation information.
    Extracts a frame‑wise latent distribution (mu, logvar) from ASR
    bottleneck features, to be used in the hierarchical CVAE.
    """
    def __init__(
        self,
        bn_dim: int,
        z_dim: int,
        hidden_channels: int = 256,
        kernel_size: int = 3,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Args:
            bn_dim:      Dimensionality of input BN features.
            z_dim:       Latent dimensionality for pronunciation variable z_pr.
            hidden_channels: Number of channels in intermediate Conv1d layers.
            kernel_size: Temporal kernel size for each Conv1d.
            num_layers:  Number of Conv1d→ReLU→LayerNorm→Dropout blocks.
            dropout:     Dropout probability after each block.
        """
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_ch = bn_dim if i == 0 else hidden_channels
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, hidden_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(inplace=True),
                nn.GroupNorm(1, hidden_channels),
                nn.Dropout(dropout),
            ))
        # 1×1 conv to produce mean+logvar
        self.proj = nn.Conv1d(hidden_channels, 2 * z_dim, kernel_size=1)

    def forward(self, bn_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bn_feats: Tensor of shape (batch, time, bn_dim)
        Returns:
            mu:     Tensor of shape (batch, time, z_dim)
            logvar: Tensor of shape (batch, time, z_dim)
        """
        # switch to (batch, channels, time)
        x = bn_feats.transpose(1, 2)
        for block in self.conv_blocks:
            x = block(x)
        stats = self.proj(x)  # (batch, 2*z_dim, time)
        b, cz, t = stats.size()
        z_dim = cz // 2
        # split and restore time-major
        mu     = stats[:, :z_dim, :].transpose(1, 2)
        logvar = stats[:, z_dim:, :].transpose(1, 2)
        return mu, logvar
