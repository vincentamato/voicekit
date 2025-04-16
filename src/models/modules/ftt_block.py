import torch
import torch.nn as nn

class FFTBlock(nn.Module):
    """
    FFTBlock: a transformer-style block combining self-attention and convolution.
    Used in Accent-VITS pronunciation and BN encoders to model local and global context.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        # 1D convolution for local feature extraction
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, hidden_dim)
        Returns:
            Tensor of same shape after attention+conv processing.
        """
        # Self-attention block
        residual = x
        # nn.MultiheadAttention expects (time, batch, dim)
        x_t = x.transpose(0, 1)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        attn_out = self.dropout1(attn_out)
        x = residual + attn_out.transpose(0, 1)
        x = self.norm1(x)

        # Convolutional block
        residual = x
        # (batch, hidden_dim, time)
        x_c = x.transpose(1, 2)
        x_c = self.conv(x_c)
        x_c = self.relu(x_c)
        x_c = self.dropout2(x_c)
        # back to (batch, time, hidden_dim)
        x = residual + x_c.transpose(1, 2)
        x = self.norm2(x)
        return x
