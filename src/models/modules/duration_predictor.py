import torch
import torch.nn as nn


class DurationPredictor(nn.Module):
    """
    Predicts the duration (in frames) for each phoneme embedding.
    Based on Section 2.6 of Accent-VITS paper.
    Input is phoneme-level embeddings before Length Regulator.
    Output is predicted log duration for stability.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.5
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Project to a single value per phoneme
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, in_channels)
        Returns:
            log_dur_pred: Tensor of shape (batch, seq_len)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)

        x = self.conv1(x)
        x = x.transpose(1, 2) # Back to (batch, seq_len, channels) for LayerNorm
        x = self.norm1(x)
        x = x.transpose(1, 2) # Back to (batch, channels, seq_len) for Conv
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = x.transpose(1, 2)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Project and reshape
        x = x.transpose(1, 2) # (batch, seq_len, hidden_channels)
        log_dur_pred = self.linear(x).squeeze(-1) # (batch, seq_len)

        return log_dur_pred 