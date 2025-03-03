"""Audio chunk processing module for real-time audio analysis."""
import numpy as np
import librosa
import logging
from typing import Dict, Union

logger = logging.getLogger(__name__)

def process_chunk(raw_audio: Union[bytes, np.ndarray], sample_rate: int) -> Dict[str, Union[np.ndarray, int]]:
    """Process a chunk of audio data.
    
    Args:
        raw_audio: Audio data as bytes or numpy array
        sample_rate: Original sample rate of the audio
        
    Returns:
        Dict containing:
            - audio: Processed audio as numpy array
            - sample_rate: Output sample rate (always 16000)
            
    Raises:
        ValueError: If input audio is invalid
    """
    TARGET_SAMPLE_RATE = 16000
    
    try:
        # Convert bytes to numpy array if needed
        if isinstance(raw_audio, bytes):
            audio = np.frombuffer(raw_audio, dtype=np.float32)
        else:
            audio = raw_audio.astype(np.float32)
        
        # Validate input
        if len(audio) == 0:
            raise ValueError("Empty audio chunk")
        
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains invalid values")
        
        # Normalize audio to [-1, 1] range
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
        
        # Resample if needed
        if sample_rate != TARGET_SAMPLE_RATE:
            logger.debug(f"Resampling from {sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz")
            audio = librosa.resample(
                y=audio,
                orig_sr=sample_rate,
                target_sr=TARGET_SAMPLE_RATE
            )
        
        # Apply pre-emphasis filter
        audio = librosa.effects.preemphasis(audio)
        
        # Ensure we have exactly 1 second of audio
        target_length = TARGET_SAMPLE_RATE
        if len(audio) > target_length:
            audio = audio[:target_length]
        elif len(audio) < target_length:
            # Pad with zeros if chunk is too short
            audio = np.pad(audio, (0, target_length - len(audio)))
        
        return {
            "audio": audio,
            "sample_rate": TARGET_SAMPLE_RATE
        }
        
    except Exception as e:
        logger.error(f"Error processing audio chunk: {str(e)}")
        raise 