import torch
import torch.nn.functional as F
from torch import Tensor
from src.utils.config import Config
from src.utils.audio import mel_spectrogram


def kl_divergence(
    mu_q: Tensor,
    logvar_q: Tensor,
    mu_p: Tensor,
    logvar_p: Tensor
) -> Tensor:
    """
    Compute the KL divergence D_KL[N(mu_q, var_q) || N(mu_p, var_p)]
    across time and latent dimensions, averaged over the batch.

    Handles potential length mismatch in time dimension (dim=1) by aligning
    q (posterior) to p (prior).

    Formula:
        0.5 * [log var_p/var_q + var_q/var_p + (mu_q - mu_p)^2/var_p - 1]
    """
    # Align lengths along time dimension (dim=1)
    t_p = mu_p.size(1)
    t_q = mu_q.size(1)

    if t_q > t_p:
        # Truncate q tensors
        mu_q = mu_q[:, :t_p, :]
        logvar_q = logvar_q[:, :t_p, :]
    elif t_p > t_q:
        # Pad q tensors
        padding = (0, 0, 0, t_p - t_q) # Pad dim 1 (time)
        mu_q = F.pad(mu_q, padding, "constant", 0)
        logvar_q = F.pad(logvar_q, padding, "constant", 0)

    # --- Original KL calculation ---
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = (
        (logvar_p - logvar_q)
        + var_q / var_p
        + (mu_q - mu_p).pow(2) / var_p
        - 1
    ) * 0.5
    # Sum over time and latent dims, then mean over batch
    return kl.sum(dim=-1).sum(dim=-1).mean()


def reconstruction_loss(
    fake_wave: Tensor,
    mel: Tensor,
    config: Config
) -> Tensor:
    """
    L1 loss between mel-spectrogram of generated waveform and ground-truth mel.

    Args:
        fake_wave: Waveform tensor of shape (B, 1, T_hat)
        mel:       Ground-truth mel of shape (B, n_mels, T)
        config:    configuration for mel extraction
    """
    # Convert generated waveform to mel-spectrogram
    mel_hat = mel_spectrogram(fake_wave.squeeze(1), config)

    # Adjust lengths to match before loss calculation
    t_mel = mel.size(2)
    t_mel_hat = mel_hat.size(2)

    if t_mel_hat > t_mel:
        # Truncate generated mel
        mel_hat = mel_hat[:, :, :t_mel]
    elif t_mel > t_mel_hat:
        # Pad generated mel
        padding = (0, t_mel - t_mel_hat) # Pad last dim (time)
        mel_hat = F.pad(mel_hat, padding, "constant", 0)

    return F.l1_loss(mel_hat, mel)


def duration_loss(
    dur_pred: Tensor,
    dur_true: Tensor
) -> Tensor:
    """
    MSE loss between predicted and ground-truth durations.
    """
    return F.mse_loss(dur_pred.float(), dur_true.float())


def discriminator_loss(
    discriminator: torch.nn.Module,
    real_wave: Tensor,
    fake_wave: Tensor
) -> Tensor:
    """
    Least-squares GAN loss for the discriminator:
        L_D = E[(D(real)-1)^2 + D(fake)^2]
    where D outputs lists of feature maps from MPD and MSD.
    """
    mpd_real, msd_real = discriminator(real_wave)
    mpd_fake, msd_fake = discriminator(fake_wave.detach())
    loss = 0.0
    # Multi-Period Discriminator
    for real_feats, fake_feats in zip(mpd_real, mpd_fake):
        for r, f in zip(real_feats, fake_feats):
            loss += torch.mean((r - 1).pow(2)) + torch.mean(f.pow(2))
    # Multi-Scale Discriminator
    for real_feats, fake_feats in zip(msd_real, msd_fake):
        for r, f in zip(real_feats, fake_feats):
            loss += torch.mean((r - 1).pow(2)) + torch.mean(f.pow(2))
    return loss


def generator_adversarial_loss(
    discriminator: torch.nn.Module,
    fake_wave: Tensor
) -> Tensor:
    """
    Least-squares GAN loss for the generator:
        L_G_adv = E[(D(fake)-1)^2]
    """
    mpd_fake, msd_fake = discriminator(fake_wave)
    loss = 0.0
    for fake_feats in mpd_fake:
        for f in fake_feats:
            loss += torch.mean((f - 1).pow(2))
    for fake_feats in msd_fake:
        for f in fake_feats:
            loss += torch.mean((f - 1).pow(2))
    return loss


def feature_matching_loss(
    discriminator: torch.nn.Module,
    real_wave: Tensor,
    fake_wave: Tensor
) -> Tensor:
    """
    Feature matching loss between real and fake feature maps:
        L_fm = E[|D_i(real) - D_i(fake)|] summed over all layers & scales.
    """
    mpd_real, msd_real = discriminator(real_wave)
    mpd_fake, msd_fake = discriminator(fake_wave)
    loss = 0.0
    for real_feats, fake_feats in zip(mpd_real, mpd_fake):
        for r, f in zip(real_feats, fake_feats):
            loss += torch.mean(torch.abs(r - f))
    for real_feats, fake_feats in zip(msd_real, msd_fake):
        for r, f in zip(real_feats, fake_feats):
            loss += torch.mean(torch.abs(r - f))
    return loss


def cvae_loss(
    mu_pr: Tensor,
    logvar_pr: Tensor,
    mu_bn: Tensor,
    logvar_bn: Tensor,
    mu_ac_post: Tensor,
    logvar_ac_post: Tensor,
    mu_ac_pr: Tensor,
    logvar_ac_pr: Tensor,
    recon_l1: Tensor,
    config: Config
) -> Tensor:
    """
    Hierarchical CVAE loss:
        Lcvae = α * L_recon + λ_pr * D_KL(q(z_pr|BN) || p(z_pr|c))
                               + λ_ac * D_KL(q(z_ac|y) || p(z_ac|z_pr,spk))
    """
    # Reconstruction term (mel L1)
    l_recon = config.alpha_recon * recon_l1
    # KL for pronunciation latent
    kl_pr = config.lambda_kl_pron * kl_divergence(mu_bn, logvar_bn, mu_pr, logvar_pr)
    # KL for acoustic latent
    kl_ac = config.lambda_kl_acoustic * kl_divergence(mu_ac_post, logvar_ac_post, mu_ac_pr, logvar_ac_pr)
    return l_recon + kl_pr + kl_ac


def generator_loss(
    discriminator: torch.nn.Module,
    real_wave: Tensor,
    fake_wave: Tensor,
    mu_pr: Tensor,
    logvar_pr: Tensor,
    mu_bn: Tensor,
    logvar_bn: Tensor,
    mu_ac_post: Tensor,
    logvar_ac_post: Tensor,
    mu_ac_pr: Tensor,
    logvar_ac_pr: Tensor,
    dur_pred: Tensor,
    dur_true: Tensor,
    mel: Tensor,
    config: Config
) -> Tensor:
    """
    Combined generator loss:
      L_G = L_adv(G) + λ_fm * L_fm + Lcvae + λ_dur * L_dur
    """
    # Adversarial
    l_adv = generator_adversarial_loss(discriminator, fake_wave)
    # Feature matching
    l_fm = feature_matching_loss(discriminator, real_wave, fake_wave)
    # Reconstruction (mel spectrogram L1)
    l_recon_l1 = reconstruction_loss(fake_wave, mel, config)
    # CVAE (reconstruction + KLs)
    l_cvae = cvae_loss(
        mu_pr, logvar_pr,
        mu_bn, logvar_bn,
        mu_ac_post, logvar_ac_post,
        mu_ac_pr, logvar_ac_pr,
        l_recon_l1,
        config
    )
    # Duration predictor loss
    l_dur = duration_loss(dur_pred, dur_true)
    # Total
    return (
        l_adv
        + config.lambda_fm * l_fm
        + l_cvae
        + config.lambda_dur * l_dur
    )


def discriminator_loss_total(
    discriminator: torch.nn.Module,
    real_wave: Tensor,
    fake_wave: Tensor
) -> Tensor:
    """
    Combined discriminator loss L(D) = L_D (least-squares).
    """
    return discriminator_loss(discriminator, real_wave, fake_wave)