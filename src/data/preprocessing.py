import torch
from transformers import HubertModel, Wav2Vec2Processor
from src.utils.config import Config
from src.utils.audio import load_wav

class BNFeatureExtractor:
    def __init__(self, config=None):
        self.config = config or Config()
        
        # Load HuBERT model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(self.config.asr_model_name)
        self.model = HubertModel.from_pretrained(self.config.asr_model_name)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def extract_features(self, waveform, sr=None):
        """
        Extract bottleneck features from waveform.
        
        Args:
            waveform: Audio waveform tensor
            sr: Sample rate (if None, uses config.target_sampling_rate)
            
        Returns:
            bn_features: Bottleneck features tensor of shape [time, feature_dim]
        """
        # Handle sample rate
        sr = sr or self.config.target_sampling_rate
        
        # Process waveform
        if not isinstance(waveform, torch.Tensor):
            waveform = torch.tensor(waveform)
        
        # Ensure correct sample rate
        waveform = load_wav(waveform, sr, self.config.target_sampling_rate)
        
        # Prepare input for HuBERT
        inputs = self.processor(waveform, sampling_rate=self.config.target_sampling_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get features from specified layer
        bn_features = outputs.hidden_states[self.config.bn_layer]
        
        # Move back to CPU and remove batch dimension
        bn_features = bn_features.squeeze(0).cpu()
        
        return bn_features