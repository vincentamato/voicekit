import torch
import torch.nn as nn

class PosteriorEncoder(nn.Module):
    """
    Posterior Encoder q(z_ac | y) for Accent-VITS.
    Extracts acoustic latent distributions from mel-spectrogram inputs.
    Implements Section 2.4 of Accent-VITS: Conv1d stack → Gaussian posterior over z_ac.
    """
    def __init__(
        self,
        mel_dim: int,
        hidden_channels: int,
        z_ac_dim: int,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Build a stack of Conv1d→ReLU→GroupNorm→Dropout layers
        layers = []
        in_ch = mel_dim
        for _ in range(num_layers):
            layers += [
                nn.Conv1d(in_ch, hidden_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(inplace=True),
                nn.GroupNorm(1, hidden_channels),
                nn.Dropout(dropout),
            ]
            in_ch = hidden_channels
        self.conv_stack = nn.Sequential(*layers)

        # Final projection to 2*z_ac_dim channels for mu and logvar
        self.proj = nn.Conv1d(hidden_channels, 2 * z_ac_dim, kernel_size=1)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mel: Tensor of shape (B, mel_dim, T) - input mel-spectrogram

        Returns:
            mu:    Tensor of shape (B, T, z_ac_dim)      - posterior mean
            logvar:Tensor of shape (B, T, z_ac_dim)      - posterior log-variance
            z_ac:  Tensor of shape (B, T, z_ac_dim)      - sampled latent via reparameterization
        """
        # Apply convolutional stack
        h = self.conv_stack(mel)  # (B, hidden_channels, T)
        stats = self.proj(h)      # (B, 2*z_ac_dim, T)

        # Split channels and transpose to (B, T, z_ac_dim)
        B, C2, T = stats.size()
        z_dim = C2 // 2
        mu     = stats[:, :z_dim, :].transpose(1, 2)
        logvar = stats[:, z_dim:, :].transpose(1, 2)

        # Sample via reparameterization
        eps = torch.randn_like(mu)
        z_ac = mu + torch.exp(0.5 * logvar) * eps

        return mu, logvar, z_ac
