import torch
from torch.utils.data import DataLoader
from src.data.dataset import GLOBEDataset
from src.utils.config import Config

def collate_fn(batch):
    """
    Custom collate function for variable length sequences.
    """
    # Sort by mel length for packing
    batch.sort(key=lambda x: x['mel'].size(1), reverse=True)
    
    # Get max lengths
    max_mel_len = max([x['mel'].size(1) for x in batch])
    max_bn_len = max([x['bn_features'].size(0) for x in batch])
    
    # Initialize tensors
    mel_padded = torch.zeros(len(batch), batch[0]['mel'].size(0), max_mel_len)
    bn_padded = torch.zeros(len(batch), max_bn_len, batch[0]['bn_features'].size(1))
    
    # Store actual lengths
    mel_lengths = torch.LongTensor(len(batch))
    bn_lengths = torch.LongTensor(len(batch))
    
    # Gather data
    speaker_ids = []
    accent_ids = []
    audio_infos = []
    transcripts = []
    
    # Fill in data
    for i, sample in enumerate(batch):
        mel = sample['mel']
        bn = sample['bn_features']
        
        mel_len = mel.size(1)
        bn_len = bn.size(0)
        
        # Store padded data
        mel_padded[i, :, :mel_len] = mel
        bn_padded[i, :bn_len, :] = bn
        
        # Store lengths
        mel_lengths[i] = mel_len
        bn_lengths[i] = bn_len
        
        # Store metadata
        speaker_ids.append(sample['speaker_id'])
        accent_ids.append(sample['accent_id'])
        audio_infos.append(sample['audio_info'])
        transcripts.append(sample['transcript'])
    
    # Convert to tensors
    speaker_ids = torch.LongTensor(speaker_ids)
    accent_ids = torch.LongTensor(accent_ids)
    
    return {
        'mel': mel_padded,
        'mel_lengths': mel_lengths,
        'bn_features': bn_padded,
        'bn_lengths': bn_lengths,
        'speaker_ids': speaker_ids,
        'accent_ids': accent_ids,
        'audio_infos': audio_infos,
        'transcripts': transcripts
    }

def create_dataloader(split="train", config=None, extract_features=True):
    """Create a dataloader for the GLOBE dataset."""
    config = config or Config()
    
    dataset = GLOBEDataset(
        split=split,
        config=config,
        extract_features=extract_features
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size if split == "train" else 1,
        shuffle=(split == "train"),
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train")
    )
    
    return dataloader