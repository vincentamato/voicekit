import torch
import torch.nn as nn

from src.models.modules.pronunciation_encoder import PronunciationEncoder
from src.models.modules.bn_module import BNEncoder
from src.models.modules.prior_encoder import PriorEncoder
from src.models.modules.posterior_encoder import PosteriorEncoder
from src.models.modules.decoder import HiFiGANGenerator
from src.models.modules.discriminator import HiFiGANDiscriminator

class AccentVITS(nn.Module):
    """
    Accent-VITS end-to-end model for accent transfer TTS.
    Combines PronunciationEncoder, BNEncoder (BN Constraint),
    PriorEncoder (text-to-accent-to-wave), PosteriorEncoder, and
    HiFiGANGenerator (decoder), plus HiFiGANDiscriminator for GAN losses.
    """
    def __init__(self, config):
        super().__init__()
        # Modules
        self.pron_enc = PronunciationEncoder(
            n_symbols=config.n_symbols,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            z_dim=config.z_pr_dim,
            num_fft_blocks_pre=config.pre_fft_blocks,
            num_fft_blocks_post=config.post_fft_blocks,
            num_heads=config.fft_heads,
            kernel_size=config.fft_kernel,
            dropout=config.fft_dropout
        )
        self.bn_enc = BNEncoder(
            bn_dim=config.bn_feature_dim,
            z_dim=config.z_pr_dim,
            hidden_channels=config.bn_hidden_dim,
            kernel_size=config.bn_kernel,
            num_layers=config.bn_layers,
            dropout=config.bn_dropout
        )
        self.prior_enc = PriorEncoder(
            z_pr_dim=config.z_pr_dim,
            z_ac_dim=config.z_ac_dim,
            num_speakers=config.num_speakers,
            spk_emb_dim=config.spk_emb_dim,
            hidden_channels=config.prior_hidden_dim,
            kernel_size=config.prior_kernel,
            num_conv_layers=config.prior_layers,
            dropout=config.prior_dropout,
            num_flows=config.num_flows
        )
        self.post_enc = PosteriorEncoder(
            mel_dim=config.n_mels,
            hidden_channels=config.post_hidden_dim,
            z_ac_dim=config.z_ac_dim,
            num_layers=config.post_layers,
            kernel_size=config.post_kernel,
            dropout=config.post_dropout
        )
        self.decoder = HiFiGANGenerator(
            in_channels=config.z_ac_dim,
            upsample_rates=config.upsample_rates,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            resblock_dilation_sizes=config.resblock_dilation_sizes,
            channels=config.hifigan_channels,
            out_channels=1,
            dropout=config.hifigan_dropout
        )
        self.discriminator = HiFiGANDiscriminator()

    def forward(
        self,
        phonemes: torch.LongTensor,
        durations: torch.LongTensor,
        bn_features: torch.Tensor,
        mel: torch.Tensor,
        speaker_ids: torch.LongTensor
    ):
        """
        Forward pass for training:

        Args:
            phonemes:    (B, L_eos) phoneme ID sequence (incl. EOS)
            durations:   (B, L) duration per phoneme (excl. EOS)
            bn_features: (B, T_bn, bn_dim) ASR bottleneck features
            mel:         (B, n_mels, T) ground-truth mel-spectrogram
            speaker_ids: (B,) speaker indices

        Returns:
            Dict with all latent params, samples, generated waveform, predicted durations,
            and discriminator outputs.
        """
        # 1) Pronunciation prior from text + predicted durations
        mu_pr, logvar_pr, log_dur_pred = self.pron_enc(phonemes, durations)

        # 2) BN posterior q(z_pr | BN)
        mu_bn, logvar_bn = self.bn_enc(bn_features)
        eps_pr = torch.randn_like(mu_bn)
        z_pr = mu_bn + torch.exp(0.5 * logvar_bn) * eps_pr

        # 3) Acoustic posterior q(z_ac | y)
        mu_ac_post, logvar_ac_post, z_ac = self.post_enc(mel)

        # 4) Prior for z_ac: p(z_ac | z_pr, spk)
        # Note: PriorEncoder might need adaptation if z_pr length varies (due to durations)
        # The KL divergence loss already handles this alignment, so maybe okay?
        mu_ac_pr, logvar_ac_pr, z_ac_flow, logdet = self.prior_enc(z_pr, speaker_ids)

        # 5) Decode waveform from flowed z_ac
        x = z_ac_flow.transpose(1, 2)  # (B, C, T)
        y_hat = self.decoder(x)

        # 6) Discriminator outputs for GAN losses
        # NOTE: We don't use the discriminator outputs directly in test.py currently
        # mpd_real, msd_real = self.discriminator(y_hat)

        return {
            'mu_pr': mu_pr,
            'logvar_pr': logvar_pr,
            'mu_bn': mu_bn,
            'logvar_bn': logvar_bn,
            'z_pr': z_pr,
            'mu_ac_post': mu_ac_post,
            'logvar_ac_post': logvar_ac_post,
            'z_ac': z_ac,
            'mu_ac_pr': mu_ac_pr,
            'logvar_ac_pr': logvar_ac_pr,
            'z_ac_flow': z_ac_flow,
            'flow_logdet': logdet,
            'y_hat': y_hat,
            'log_dur_pred': log_dur_pred # Add predicted durations
            # 'mpd_outs': mpd_real,
            # 'msd_outs': msd_real
        }
