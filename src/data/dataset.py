import os
import torch
import torchaudio
from torch.utils.data import Dataset
from datasets import load_dataset
from src.utils.config import Config
from src.utils.audio import load_wav, mel_spectrogram
from src.data.preprocessing import BNFeatureExtractor
from src.utils.text import TextProcessor

class GLOBEDataset(Dataset):
    def __init__(self, split="train", config=None, extract_features=True):
        self.config = config or Config()
        self.split = split

        # Initialize feature extractor
        if extract_features:
            self.bn_extractor = BNFeatureExtractor(self.config)
        else:
            self.bn_extractor = None

        # Initialize text processor for G2P
        self.text_processor = TextProcessor()

        # Load dataset
        print(f"Loading GLOBE dataset ({self.config.dataset_name})...")
        self.dataset = load_dataset(
            self.config.dataset_name,
            split=split,
            cache_dir=self.config.cache_dir,
            verification_mode="no_checks"
        )

        # Filter for target accents
        if self.config.source_accents:
            print(f"Filtering for accents: {self.config.source_accents}")
            self.dataset = self.dataset.filter(
                lambda ex: ex.get('accent', '').lower() in [a.lower() for a in self.config.source_accents]
            )

        # Cache directory for processed features
        self.cache_dir = os.path.join(self.config.output_dir, "processed_features")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Create speaker and accent maps
        self._create_speaker_accent_maps()
        print(f"Dataset loaded with {len(self.dataset)} samples")

    def _create_speaker_accent_maps(self):
        speakers = set()
        for ex in self.dataset:
            speakers.add(ex.get('speaker_id', ex.get('speaker', '')))
        self.speaker_map = {s: i for i, s in enumerate(sorted(speakers))}

        accents = set(self.config.source_accents)
        self.accent_map = {a.lower(): i for i, a in enumerate(sorted(accents))}
        print(f"Found {len(self.speaker_map)} speakers and {len(self.accent_map)} accents")

    def _get_cache_path(self, example_id, feature_type):
        return os.path.join(self.cache_dir, f"{example_id}_{feature_type}.pt")

    def _extract_or_load_features(self, example):
        # Generate unique example ID
        if 'id' in example:
            example_id = str(example['id'])
        elif 'speaker_id' in example:
            example_id = f"{example['speaker_id']}_{hash(str(example))}"
        else:
            example_id = str(hash(str(example)))

        mel_path = self._get_cache_path(example_id, "mel")
        bn_path  = self._get_cache_path(example_id, "bn")

        if os.path.exists(mel_path) and os.path.exists(bn_path):
            mel = torch.load(mel_path)
            bn  = torch.load(bn_path)
        else:
            # Try HuggingFace audio format
            audio_data = None
            audio_sr   = None
            if 'audio' in example and isinstance(example['audio'], dict):
                audio_data = example['audio'].get('array', None)
                audio_sr   = example['audio'].get('sampling_rate', None)
            if audio_data is not None:
                waveform = torch.tensor(audio_data).float()
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                waveform = load_wav(waveform, audio_sr, self.config.target_sampling_rate)
                mel = mel_spectrogram(waveform, self.config)
                bn  = self.bn_extractor.extract_features(waveform) if self.bn_extractor else None
                torch.save(mel, mel_path)
                if bn is not None:
                    torch.save(bn, bn_path)
            else:
                # Fallback to file path
                audio_path = None
                for key in ['path','file','filename','audio_path']:
                    if key in example and isinstance(example[key], str):
                        audio_path = example[key]
                        break
                if audio_path:
                    waveform, sr = torchaudio.load(audio_path)
                    waveform = load_wav(waveform, sr, self.config.target_sampling_rate)
                    mel = mel_spectrogram(waveform, self.config)
                    bn  = self.bn_extractor.extract_features(waveform) if self.bn_extractor else None
                    torch.save(mel, mel_path)
                    if bn is not None:
                        torch.save(bn, bn_path)
                else:
                    print(f"Warning: No audio for example {example_id}")
                    mel = torch.zeros((self.config.n_mels, 50))
                    bn  = torch.zeros((50, self.config.bn_feature_dim))
        return mel, bn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Speaker & accent indices
        speaker = example.get('speaker_id', example.get('speaker', ''))
        speaker_id = self.speaker_map.get(speaker, 0)
        accent = example.get('accent', '')
        accent_id = self.accent_map.get(accent.lower(), 0)

        # Load features
        mel, bn_features = self._extract_or_load_features(example)

        # Transcript
        transcript = example.get('text', example.get('transcript', ''))

        # Compute phoneme IDs
        phoneme_ids = self.text_processor.text_to_sequence(transcript)
        num_ph = len(phoneme_ids) - 1 # Exclude EOS token for duration calculation

        # Get total frames either from dataset duration or mel length
        duration_sec = example.get('duration', None)
        if duration_sec is not None:
            # Convert seconds to mel frames: frame_shift = hop_length / sampling_rate
            frame_shift = self.config.hop_length / self.config.target_sampling_rate
            total_frames = int(round(duration_sec / frame_shift))
        else:
            total_frames = mel.size(1)
        # Evenly split total frames across phonemes
        # Handle potential division by zero if num_ph is 0 (empty transcript?)
        if num_ph > 0:
            base = total_frames // num_ph
            extra = total_frames % num_ph
            dur_list = [base + (1 if i < extra else 0) for i in range(num_ph)]
        else:
            dur_list = []
        durations = torch.LongTensor(dur_list)

        # Audio info
        audio_info = example.get('audio', {}) if isinstance(example.get('audio', {}), dict) else {}
        if not audio_info:
            for key in ['path','file','filename','audio_path']:
                if key in example and isinstance(example[key], str):
                    audio_info = {'path': example[key]}
                    break

        return {
            'mel': mel,
            'bn_features': bn_features,
            'speaker_id': speaker_id,
            'accent_id': accent_id,
            'phoneme_ids': torch.LongTensor(phoneme_ids), # Return IDs as LongTensor
            'durations': durations,
            'audio_info': audio_info
            # Optionally keep transcript for debugging/info
            # 'transcript_text': transcript 
        }
