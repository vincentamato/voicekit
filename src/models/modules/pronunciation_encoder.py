import torch
import torch.nn as nn

from src.models.modules.ftt_block import FFTBlock
from src.models.modules.length_regulator import LengthRegulator
from src.models.modules.duration_predictor import DurationPredictor

class PronunciationEncoder(nn.Module):
    """
    Pronunciation Encoder module p(z_pr | c) for Accent-VITS.
    Encodes accented phoneme sequences into prior distribution parameters (mu, logvar).
    Also includes a Duration Predictor.

    Follows Section 2.1 and 2.6 of Accent-VITS.
    """
    def __init__(
        self,
        n_symbols: int,
        embed_dim: int,
        hidden_dim: int,
        z_dim: int,
        num_fft_blocks_pre: int = 6,
        num_fft_blocks_post: int = 6,
        num_heads: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
        # Duration predictor params
        dp_hidden_channels: int = 256,
        dp_kernel_size: int = 3,
        dp_dropout: float = 0.5,
    ):
        super().__init__()
        # 1) Phoneme embedding + prenet
        self.symbol_emb = nn.Embedding(n_symbols, embed_dim)
        self.prenet = nn.Linear(embed_dim, hidden_dim)

        # 2) Pre-length-regulator FFT blocks
        self.fft_pre = nn.ModuleList([
            FFTBlock(hidden_dim, num_heads, kernel_size, dropout)
            for _ in range(num_fft_blocks_pre)
        ])

        # 3) Duration Predictor (takes pre-LR FFT output)
        self.duration_predictor = DurationPredictor(
            in_channels=hidden_dim,
            hidden_channels=dp_hidden_channels,
            kernel_size=dp_kernel_size,
            dropout=dp_dropout
        )

        # 4) Length Regulator (upsamples phoneme frames to match acoustic frames)
        self.length_regulator = LengthRegulator()

        # 5) Post-LR FFT blocks for context refinement
        self.fft_post = nn.ModuleList([
            FFTBlock(hidden_dim, num_heads, kernel_size, dropout)
            for _ in range(num_fft_blocks_post)
        ])

        # 6) Project to mu and logvar via 1×1 convs
        self.proj_mu     = nn.Conv1d(hidden_dim, z_dim, kernel_size=1)
        self.proj_logvar = nn.Conv1d(hidden_dim, z_dim, kernel_size=1)

    def forward(
        self,
        phonemes: torch.LongTensor,
        durations: torch.LongTensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            phonemes:  (batch, L_eos)   phoneme ID sequence (including EOS)
            durations: (batch, L)       duration (in frames) per phoneme (excluding EOS)
        Returns:
            mu:         (batch, T, z_dim)   prior mean over time frames
            logvar:     (batch, T, z_dim)   prior log-variance over time frames
            log_dur_pred: (batch, L)       predicted log duration per phoneme (excluding EOS)
        """
        # Embed + prenet
        x_emb = self.symbol_emb(phonemes)       # (B, L_eos, embed_dim)
        x = self.prenet(x_emb)                  # (B, L_eos, hidden_dim)

        # Pre-LR FFT
        for block in self.fft_pre:
            x = block(x)                         # (B, L_eos, hidden_dim)

        # Predict duration from pre-LR FFT output (before EOS removal)
        # Detach input to prevent duration predictor gradient from flowing back further?
        # -> Paper doesn't specify, common practice, but let's try without first.
        log_dur_pred_full = self.duration_predictor(x) # (B, L_eos)
        # Slice to match ground truth duration length (remove EOS prediction)
        log_dur_pred = log_dur_pred_full[:, :-1]       # (B, L)

        # Slice embeddings to remove EOS token before LR
        x_no_eos = x[:, :-1, :]                  # (B, L, hidden_dim)
        
        # Length Regulator: upsample using GROUND TRUTH durations during training
        # (Inference would use predicted durations)
        x_upsampled = self.length_regulator(x_no_eos, durations) # (B, T, hidden_dim)

        # Post-LR FFT
        for block in self.fft_post:
            x_upsampled = block(x_upsampled)    # (B, T, hidden_dim)

        # Project: conv1d expects (B, C, T)
        x_t = x_upsampled.transpose(1, 2)       # (B, hidden_dim, T)
        mu     = self.proj_mu(x_t).transpose(1, 2)
        logvar = self.proj_logvar(x_t).transpose(1, 2)

        return mu, logvar, log_dur_pred
