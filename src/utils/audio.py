import torch
import torchaudio

from src.utils.config import Config

def load_wav(waveform, sr, target_sr=None):
    """Process waveform to target sample rate."""
    target_sr = target_sr or Config().target_sampling_rate
    
    # Convert to mono if needed
    if waveform.dim() > 1 and waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    
    return waveform.squeeze(0)

def mel_spectrogram(waveform, config=None):
    """Convert waveform to mel spectrogram."""
    if config is None:
        config = Config()
        
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.target_sampling_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        f_min=config.mel_fmin,
        f_max=config.mel_fmax,
        power=1.0,  # Using amplitude spectrogram
    )
    mel = mel_transform(waveform)
    # Convert to log scale
    mel = torch.log(torch.clamp(mel, min=1e-5))
    return mel