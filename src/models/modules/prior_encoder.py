import torch
import torch.nn as nn

from src.models.modules.flow import Invertible1x1Conv, ResidualCouplingBlock, FlowSequential

class PriorEncoder(nn.Module):
    """
    Prior Encoder: maps pronunciation latent z_pr and speaker ID to
    prior distribution parameters for acoustic latent z_ac, and runs
    z_ac through a sequence of normalizing flows.
    Implements Section 2.3 of Accent‑VITS.
    """
    def __init__(
        self,
        z_pr_dim: int,
        z_ac_dim: int,
        num_speakers: int,
        spk_emb_dim: int = 128,
        hidden_channels: int = 256,
        kernel_size: int = 3,
        num_conv_layers: int = 3,
        dropout: float = 0.1,
        num_flows: int = 4,
    ):
        super().__init__()
        # Speaker embedding
        self.spk_emb = nn.Embedding(num_speakers, spk_emb_dim)

        # Conv stack to map [z_pr; spk_emb] -> hidden
        layers = []
        in_ch = z_pr_dim + spk_emb_dim
        for _ in range(num_conv_layers):
            layers += [
                nn.Conv1d(in_ch, hidden_channels, kernel_size, padding=kernel_size//2),
                nn.ReLU(inplace=True),
                nn.GroupNorm(1, hidden_channels),
                nn.Dropout(dropout),
            ]
            in_ch = hidden_channels
        self.conv_stack = nn.Sequential(*layers)

        # Project to mu and logvar for z_ac prior
        self.proj_mu     = nn.Conv1d(hidden_channels, z_ac_dim,     kernel_size=1)
        self.proj_logvar = nn.Conv1d(hidden_channels, z_ac_dim,     kernel_size=1)

        # Build normalizing flows
        flows = []
        for _ in range(num_flows):
            flows.append(Invertible1x1Conv(z_ac_dim))
            flows.append(ResidualCouplingBlock(z_ac_dim, hidden_channels))
        self.flows = FlowSequential(nn.ModuleList(flows))

    def forward(
        self,
        z_pr: torch.Tensor,
        speaker_ids: torch.LongTensor
    ):
        """
        Args:
            z_pr:        (B, T, z_pr_dim) pronunciation latent
            speaker_ids: (B,)               speaker indices
        Returns:
            mu:         (B, T, z_ac_dim) prior mean
            logvar:     (B, T, z_ac_dim) prior log-variance
            z_ac_flow:  (B, T, z_ac_dim) latent after flows
            logdet:     (B,)               sum of log-determinants
        """
        B, T, _ = z_pr.size()
        # Embed speaker and expand time dimension
        spk = self.spk_emb(speaker_ids)           # (B, spk_emb_dim)
        spk = spk.unsqueeze(1).expand(-1, T, -1)   # (B, T, spk_emb_dim)

        # Concatenate and convert to (B, C, T)
        x = torch.cat([z_pr, spk], dim=-1).transpose(1, 2)

        # Conv processing
        h = self.conv_stack(x)                    # (B, hidden_channels, T)

        # Compute prior params
        mu     = self.proj_mu(h).transpose(1, 2)     # (B, T, z_ac_dim)
        logvar = self.proj_logvar(h).transpose(1, 2) # (B, T, z_ac_dim)

        # Sample z_ac
        eps = torch.randn_like(mu)
        z_ac0 = mu + torch.exp(0.5 * logvar) * eps    # reparameterization

        # Apply normalizing flows
        z_ac_flow, logdet = self.flows(z_ac0, reverse=False)

        return mu, logvar, z_ac_flow, logdet
