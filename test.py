import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

import torch

from src.data.dataset import GLOBEDataset
from torch.utils.data import DataLoader
from src.utils.config import Config
from src.models.accent_vits import AccentVITS
from src.training.losses import (
    reconstruction_loss,
    kl_divergence,
    duration_loss,
    cvae_loss
)


def main():
    # Load config and dataset
    config = Config()
    # Disable multiple workers for safety here
    config.num_workers = 0
    dataset = GLOBEDataset(split="test", config=config, extract_features=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Update speaker count
    config.num_speakers = len(dataset.speaker_map)

    # Get one sample
    batch = next(iter(dataloader))
    mel_spec     = batch['mel']           # (B, n_mels, T)
    bn_feat      = batch['bn_features']   # (B, T_bn, bn_dim)
    speaker_id   = batch['speaker_id']    # (B,)
    durations    = batch['durations']     # (B, L_ph)
    phoneme_ids  = batch['phoneme_ids']  # (B, L_ph_eos) - Now directly getting IDs
    
    # Instantiate model
    model = AccentVITS(config)
    model.eval()

    # Phoneme IDs are already tensors from the dataset/dataloader

    with torch.no_grad():
        outputs = model(
            phonemes=phoneme_ids, # Use phoneme_ids directly
            durations=durations,
            bn_features=bn_feat,
            mel=mel_spec,
            speaker_ids=speaker_id
        )

    # Extract for losses
    y_hat         = outputs['y_hat']
    mu_pr, lpr    = outputs['mu_pr'], outputs['logvar_pr']
    mu_bn, lbn    = outputs['mu_bn'], outputs['logvar_bn']
    mu_acp, lacp  = outputs['mu_ac_post'], outputs['logvar_ac_post']
    mu_ac, lac    = outputs['mu_ac_pr'], outputs['logvar_ac_pr']
    log_dur_pred  = outputs['log_dur_pred'] # Get predicted log durations

    # Compute losses
    recon = reconstruction_loss(y_hat, mel_spec, config)
    klp   = kl_divergence(mu_bn, lbn, mu_pr, lpr)
    kla   = kl_divergence(mu_acp, lacp, mu_ac, lac)
    # Compute duration loss in log domain for stability
    log_dur_true = torch.log(durations.float().clamp(min=1)) # Clamp min to 1 before log
    durl  = duration_loss(log_dur_pred, log_dur_true) 
    total = cvae_loss(mu_pr, lpr, mu_bn, lbn,
                      mu_acp, lacp, mu_ac, lac,
                      recon, config)

    print(f"Reconstruction loss: {recon.item():.4f}")
    print(f"KL pronunciation: {klp.item():.4f}")
    print(f"KL acoustic: {kla.item():.4f}")
    print(f"Duration loss (log): {durl.item():.4f}") # Use predicted durations
    print(f"CVAE loss: {total.item():.4f}")
    print(f"Generated audio shape: {y_hat.shape}")

if __name__ == "__main__":
    main()
