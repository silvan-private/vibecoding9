"""Tests for audio chunk processing."""
import pytest
import numpy as np
from audio_chunk_processor import process_chunk

def test_process_chunk_basic():
    """Test basic audio chunk processing."""
    # Create 1 second of silence at 16kHz
    test_audio = np.zeros(16000, dtype=np.float32)
    result = process_chunk(test_audio, 16000)
    
    assert result["sample_rate"] == 16000
    assert len(result["audio"]) == 16000
    assert isinstance(result["audio"], np.ndarray)
    assert result["audio"].dtype == np.float32

def test_process_chunk_resampling():
    """Test resampling from different sample rates."""
    # Test 44.1kHz → 16kHz
    samples_44k = int(44100 * 1.0)  # 1 second at 44.1kHz
    test_audio_44k = np.random.rand(samples_44k).astype(np.float32)
    result_44k = process_chunk(test_audio_44k, 44100)
    
    assert result_44k["sample_rate"] == 16000
    assert len(result_44k["audio"]) == 16000
    
    # Test 8kHz → 16kHz
    samples_8k = int(8000 * 1.0)  # 1 second at 8kHz
    test_audio_8k = np.random.rand(samples_8k).astype(np.float32)
    result_8k = process_chunk(test_audio_8k, 8000)
    
    assert result_8k["sample_rate"] == 16000
    assert len(result_8k["audio"]) == 16000

def test_process_chunk_normalization():
    """Test audio normalization."""
    # Create audio with values > 1.0
    test_audio = np.random.rand(16000) * 10  # Values between 0 and 10
    result = process_chunk(test_audio, 16000)
    
    assert np.abs(result["audio"]).max() <= 1.0

def test_process_chunk_bytes():
    """Test processing audio from bytes."""
    # Create audio bytes
    test_audio = np.zeros(16000, dtype=np.float32)
    audio_bytes = test_audio.tobytes()
    result = process_chunk(audio_bytes, 16000)
    
    assert result["sample_rate"] == 16000
    assert len(result["audio"]) == 16000

def test_process_chunk_errors():
    """Test error handling."""
    # Test empty audio
    with pytest.raises(ValueError):
        process_chunk(np.array([], dtype=np.float32), 16000)
    
    # Test invalid values
    with pytest.raises(ValueError):
        test_audio = np.array([np.nan, np.inf], dtype=np.float32)
        process_chunk(test_audio, 16000)

def test_process_chunk_padding():
    """Test padding of short chunks."""
    # Test short chunk (0.5 seconds)
    test_audio = np.random.rand(8000).astype(np.float32)  # 0.5s at 16kHz
    result = process_chunk(test_audio, 16000)
    
    assert len(result["audio"]) == 16000
    # Check that the first 8000 samples are not zero and the rest are
    assert not np.allclose(result["audio"][:8000], 0)
    assert np.allclose(result["audio"][8000:], 0)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 