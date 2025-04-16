from src.data.dataloader import create_dataloader
from src.utils.config import Config

def explore_dataset():
    """Explore the GLOBE dataset."""
    config = Config()
    
    print("Creating dataloader...")
    dataloader = create_dataloader(split="test", config=config)
    
    print("Total samples:", len(dataloader.dataset))
    
    # Print example to see what's available
    if len(dataloader.dataset) > 0:
        print("\nExample dataset item keys:")
        example = dataloader.dataset[0]  # First item
        print(list(example.keys()))
        
        print("\nAudio info example:")
        print(example['audio_info'])
    
    # Get a batch
    for batch in dataloader:
        print("\nBatch information:")
        print(f"Mel shape: {batch['mel'].shape}")
        print(f"BN features shape: {batch['bn_features'].shape}")
        print(f"Speaker IDs: {batch['speaker_ids']}")
        print(f"Accent IDs: {batch['accent_ids']}")
        print(f"Number of audio files: {len(batch['audio_infos'])}")
        break

if __name__ == "__main__":
    explore_dataset()