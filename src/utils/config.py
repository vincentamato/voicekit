import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Get project root assuming this file is at <root>/src/utils/config.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

@dataclass
class Config:
    # ─── Dataset settings ─────────────────────────────────────────────────────
    dataset_name: str = "MushanW/GLOBE_V2"
    cache_dir: str = os.path.join(PROJECT_ROOT, "data", "cache", "datasets")

    # ─── Audio settings ──────────────────────────────────────────────────────
    sampling_rate: int = 44100              # Original recording rate (44.1 kHz)
    target_sampling_rate: int = 16000       # Model training rate (16 kHz)
    n_fft: int = 1024                       # FFT window size
    hop_length: int = 200                   # 12.5 ms @ 16 kHz
    win_length: int = 800                   # 50 ms @ 16 kHz
    n_mels: int = 80                        # Mel band count
    mel_fmin: int = 0                       # Mel filter min frequency
    mel_fmax: int = 8000                    # Mel filter max frequency

    # ─── Accent settings ─────────────────────────────────────────────────────
    source_accents: List[str] = field(default_factory=lambda: [
        "United States English", "England English", "Australian English"
    ])
    target_accents: List[str] = field(default_factory=lambda: [
        "United States English", "England English", "Australian English"
    ])

    # ─── ASR for BN feature extraction ─────────────────────────────────────────
    asr_model_name: str = "facebook/hubert-large-ls960-ft"
    bn_layer: int = 9                       # Which hidden layer to extract
    bn_feature_dim: int = 1024              # HuBERT large hidden size

    # ─── Base model dimensions ────────────────────────────────────────────────
    hidden_dim: int = 192                   # Base hidden dimension
    filter_channels: int = 768              # Flow coupling hidden channels
    gin_channels: int = 256                 # Speaker embedding dim
    accent_embed_dim: int = 128             # Accent embedding dim

    # ─── VITS / HiFi-GAN architecture params ─────────────────────────────────
    kernel_size: int = 3
    p_dropout: float = 0.1
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(default_factory=lambda: [[1, 3, 5]] * 3)
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])

    # ─── TOKENIZER & TEXT-TO-PHONEME ─────────────────────────────────────────
    # Symbols = PAD + EOS + ARPABET (39 phones) => total 41 symbols
    n_symbols: int = 41
    embed_dim: int = 192

    # ─── PRONUNCIATION CVAE (p(z_pr|c)) ────────────────────────────────────────
    z_pr_dim: int = 128                      # Latent dim for pronunciation
    pre_fft_blocks: int = 6                  # FFT blocks before LR
    post_fft_blocks: int = 6                 # FFT blocks after LR
    fft_heads: int = 2                       # Multi-head attention heads
    fft_kernel: int = 3                      # FFT conv kernel size
    fft_dropout: float = 0.1                 # FFT dropout rate

    # ─── BOTTLENECK CVAE (q(z_pr|BN)) ────────────────────────────────────────
    bn_hidden_dim: int = 256                 # Conv stack hidden channels
    bn_layers: int = 3                       # Number of conv layers
    bn_kernel: int = 3                       # Conv kernel size
    bn_dropout: float = 0.1                  # Conv dropout rate

    # ─── ACOUSTIC CVAE PRIOR (p(z_ac|z_pr, spk)) ─────────────────────────────
    z_ac_dim: int = 192                      # Acoustic latent dim
    spk_emb_dim: int = 256                   # Speaker embedding dim
    prior_hidden_dim: int = 256              # Prior conv hidden channels
    prior_layers: int = 3                     # Number of prior conv layers
    prior_kernel: int = 3                    # Prior conv kernel size
    prior_dropout: float = 0.1               # Prior dropout rate
    num_flows: int = 4                       # Number of flow steps

    # ─── POSTERIOR ENCODER (q(z_ac|y)) ────────────────────────────────────────
    post_hidden_dim: int = 256               # Posterior conv hidden channels
    post_layers: int = 4                     # Number of posterior conv layers
    post_kernel: int = 3                     # Posterior conv kernel size
    post_dropout: float = 0.1                # Posterior dropout rate

    # ─── HiFi-GAN Decoder settings ────────────────────────────────────────────
    hifigan_channels: int = 512              # Initial generator channels
    hifigan_dropout: float = 0.1             # Generator dropout rate

    # ─── Training settings ────────────────────────────────────────────────────
    batch_size: int = 24
    learning_rate: float = 2e-4
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    lr_decay: float = 0.999
    seed: int = 1234
    max_steps: int = 400000
    num_workers: int = 4

    # ─── Checkpoint & Logging ─────────────────────────────────────────────────
    checkpoint_interval: int = 10000
    eval_interval: int = 1000
    log_interval: int = 100

    # ─── Paths ────────────────────────────────────────────────────────────────
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs")
    checkpoint_dir: str = os.path.join(output_dir, "checkpoints")
    log_dir: str = os.path.join(output_dir, "logs")

    # ─── Loss weights ─────────────────────────────────────────────────────────
    lambda_adv: float = 1.0
    lambda_fm: float = 2.0
    lambda_dur: float = 1.0
    alpha_recon: float = 45.0
    lambda_kl_pron: float = 1.0
    lambda_kl_acoustic: float = 1.0

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'Config':
        return cls(**{k: v for k, v in params.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__annotations__}
