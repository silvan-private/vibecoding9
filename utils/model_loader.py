"""Windows-safe model loading utilities."""
import os
import logging
import shutil
from pathlib import Path
import torch
import yaml
from huggingface_hub import hf_hub_download
import speechbrain as sb
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.processing.features import InputNormalization

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dir(path_str):
    """Create directory if it doesn't exist."""
    path = Path(path_str).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

def copy_file(src, dst):
    """Copy file with proper directory creation."""
    dst_dir = os.path.dirname(dst)
    ensure_dir(dst_dir)
    shutil.copy2(src, dst)
    return dst

class CustomNormalization(torch.nn.Module):
    """Custom normalization module that stores running stats."""
    def __init__(self, input_size=80):
        super().__init__()
        self.input_size = input_size
        self.register_buffer("count", torch.tensor(0.0))
        self.register_buffer("glob_mean", torch.zeros(input_size))
        self.register_buffer("glob_std", torch.ones(input_size))
        
    def forward(self, x, lengths=None):
        """Normalize input tensor."""
        return (x - self.glob_mean.unsqueeze(0)) / self.glob_std.unsqueeze(0)
        
    def load_state(self, state_dict):
        """Load normalization statistics."""
        # Convert statistics to tensors if they aren't already
        self.count.copy_(torch.tensor(state_dict["count"], dtype=torch.float))
        
        # Handle mean statistics
        mean = torch.as_tensor(state_dict["glob_mean"], dtype=torch.float)
        if mean.dim() == 0:
            mean = mean.expand(self.input_size)
        elif mean.dim() == 1 and mean.shape[0] != self.input_size:
            mean = mean.mean().expand(self.input_size)
        self.glob_mean.copy_(mean)
        
        # Handle standard deviation statistics
        std = torch.as_tensor(state_dict["glob_std"], dtype=torch.float)
        if std.dim() == 0:
            std = std.expand(self.input_size)
        elif std.dim() == 1 and std.shape[0] != self.input_size:
            std = std.mean().expand(self.input_size)
        self.glob_std.copy_(std)

class CustomECAPAModel(torch.nn.Module):
    """Custom ECAPA-TDNN model loader that avoids SpeechBrain's fetching mechanism."""
    
    def __init__(self, model_path, mean_var_norm_path):
        super().__init__()
        # Load the model state dict
        model_state = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Initialize the ECAPA-TDNN model
        self.embedding_model = ECAPA_TDNN(
            input_size=80,
            channels=[1024, 1024, 1024, 1024, 3072],
            kernel_sizes=[5, 3, 3, 3, 1],
            dilations=[1, 2, 3, 4, 1],
            attention_channels=128,
            lin_neurons=192,
        )
        
        # Remove 'embedding_model.' prefix from state dict keys if present
        cleaned_state_dict = {}
        for key, value in model_state.items():
            if key.startswith('embedding_model.'):
                cleaned_key = key.replace('embedding_model.', '')
                cleaned_state_dict[cleaned_key] = value
            else:
                cleaned_state_dict[key] = value
        
        # Load the model weights
        self.embedding_model.load_state_dict(cleaned_state_dict)
        
        # Set model to evaluation mode
        self.embedding_model.eval()
        
        # Load mean/var normalization stats
        mean_var_norm_state = torch.load(mean_var_norm_path, map_location=torch.device('cpu'))
        
        # Initialize feature extraction
        from speechbrain.lobes.features import Fbank
        self.compute_features = Fbank(
            n_mels=80,
            left_frames=0,
            right_frames=0,
            deltas=False,
        )
        
        # Initialize mean/var normalization
        self.mean_var_norm = InputNormalization(norm_type="global")
        if isinstance(mean_var_norm_state, dict):
            if "mean" in mean_var_norm_state:
                self.mean_var_norm.running_mean = mean_var_norm_state["mean"]
                self.mean_var_norm.running_std = mean_var_norm_state["std"]
            elif "glob_mean" in mean_var_norm_state:
                self.mean_var_norm.running_mean = mean_var_norm_state["glob_mean"]
                self.mean_var_norm.running_std = mean_var_norm_state["glob_std"]
            else:
                # Try to find in nested dicts
                for key, value in mean_var_norm_state.items():
                    if isinstance(value, dict) and "glob_mean" in value:
                        self.mean_var_norm.running_mean = value["glob_mean"]
                        self.mean_var_norm.running_std = value["glob_std"]
                        break
        else:
            raise ValueError("Could not find mean/std values in the state dictionary")
    
    def encode_batch(self, wavs, lengths=None):
        """Encode a batch of wavs into embeddings."""
        with torch.no_grad():
            # Extract features
            feats = self.compute_features(wavs)
            if lengths is None:
                lengths = torch.ones(wavs.shape[0])
            
            # Apply mean/var normalization
            feats = self.mean_var_norm(feats, lengths)
            
            # Get embeddings from the model
            embeddings = self.embedding_model(feats)
            return embeddings
    
    def compute_embeddings(self, wavs, lengths=None):
        """Compute embeddings from raw waveforms."""
        return self.encode_batch(wavs, lengths)

def load_model_safely(model_name="speechbrain/spkrec-ecapa-voxceleb", cache_dir="speaker_data/models/ecapa-tdnn"):
    """
    Load the ECAPA-TDNN model safely on Windows by downloading files directly and using a custom loader.
    """
    logger.info(f"Loading model {model_name} with Windows-safe operations")
    
    # Convert to absolute path and ensure directory exists
    cache_dir = ensure_dir(cache_dir)
    logger.info(f"Using cache directory: {cache_dir}")
    
    try:
        # Download required files
        files_to_get = ["embedding_model.ckpt", "mean_var_norm_emb.ckpt"]
        local_files = {}
        
        for file in files_to_get:
            # Download from HuggingFace
            downloaded_path = hf_hub_download(
                repo_id=model_name,
                filename=file,
                cache_dir="./huggingface_cache"
            )
            
            # Copy to our cache directory
            dest_path = os.path.join(cache_dir, file)
            local_files[file] = copy_file(downloaded_path, dest_path)
            logger.info(f"Successfully copied {file} to {dest_path}")
        
        # Initialize our custom model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CustomECAPAModel(
            model_path=local_files["embedding_model.ckpt"],
            mean_var_norm_path=local_files["mean_var_norm_emb.ckpt"]
        )
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise 