"""Integration tests for the voice analysis system."""
import os
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

from db.speaker_db import SpeakerDatabase
from voice_match import VoiceMatcher
from audio_chunk_processor import process_chunk
from utils.time_helpers import TimeFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestProgress:
    """Track test progress and checkpoints."""
    def __init__(self):
        self.checkpoints = {}
        
    def mark(self, checkpoint: str, success: bool, details: str = ""):
        """Mark a checkpoint as passed/failed."""
        self.checkpoints[checkpoint] = {
            'success': success,
            'timestamp': datetime.now(),
            'details': details
        }
        
        # Log the result
        status = "✓" if success else "✗"
        logger.info(f"{status} {checkpoint}: {details}")
        
    def summary(self) -> str:
        """Generate test progress summary."""
        lines = ["\nTEST PROGRESS SUMMARY", "=" * 50]
        
        for checkpoint, data in self.checkpoints.items():
            status = "PASS" if data['success'] else "FAIL"
            lines.append(f"{status}: {checkpoint}")
            if data['details']:
                lines.append(f"     {data['details']}")
        
        return "\n".join(lines)

def generate_test_audio(duration_seconds: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate test audio data (simple sine wave)."""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)

def test_full_flow():
    """Test complete system flow from audio to database matching."""
    progress = TestProgress()
    test_db_path = "test_integration.db"
    
    try:
        # Clean up any existing test database
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        # 1. Process sample audio
        logger.info("\nStep 1: Processing audio...")
        test_audio = generate_test_audio()
        processed = process_chunk(test_audio, sample_rate=16000)
        
        progress.mark(
            "Audio Processing",
            success=processed is not None and len(processed['audio']) == 16000,
            details=f"Processed audio shape: {processed['audio'].shape}"
        )
        
        # 2. Generate voice print (simulated embedding)
        logger.info("\nStep 2: Generating voice print...")
        embedding_dim = 192
        voiceprint = np.random.randn(embedding_dim)
        voiceprint = voiceprint / np.linalg.norm(voiceprint)
        
        progress.mark(
            "Voice Print Generation",
            success=len(voiceprint) == embedding_dim,
            details=f"Embedding dimension: {embedding_dim}"
        )
        
        # 3. Store in database
        logger.info("\nStep 3: Storing in database...")
        db = SpeakerDatabase(test_db_path)
        speaker_name = "Test Speaker"
        
        # Ensure voiceprint is float32 before storing
        voiceprint = voiceprint.astype(np.float32)
        speaker_id = db.add_speaker(speaker_name, voiceprint.tobytes())
        
        # Add a test segment
        start_time = 0.0
        end_time = 1.0
        text = "Test speech segment"
        segment_id = db.add_segment(
            speaker_id=speaker_id,
            start=start_time,
            end=end_time,
            text=text,
            confidence=0.95
        )
        
        progress.mark(
            "Database Storage",
            success=speaker_id is not None and segment_id is not None,
            details=f"Speaker ID: {speaker_id}, Segment ID: {segment_id}"
        )
        
        # 4. Query and match
        logger.info("\nStep 4: Testing voice matching...")
        stored_speaker = db.get_speaker(speaker_id)
        stored_voiceprint = np.frombuffer(stored_speaker[2], dtype=np.float32)
        
        # Ensure stored voiceprint is properly normalized
        stored_voiceprint = stored_voiceprint / np.linalg.norm(stored_voiceprint)
        matcher = VoiceMatcher(stored_voiceprint)
        match_result = matcher.is_match(voiceprint)
        
        progress.mark(
            "Voice Matching",
            success=match_result.is_match,
            details=f"Similarity: {match_result.similarity:.3f}"
        )
        
        # 5. Verify timestamps
        logger.info("\nStep 5: Verifying timestamp formatting...")
        segments = db.get_segments(speaker_id)
        formatted_time = TimeFormatter.format_timestamp(segments[0][2])  # Format start time
        parsed_time = TimeFormatter.parse_timestamp(formatted_time)
        
        progress.mark(
            "Timestamp Formatting",
            success=abs(parsed_time - segments[0][2]) < 0.001,
            details=f"Formatted: {formatted_time}, Parsed: {parsed_time:.3f}"
        )
        
        # Print final summary
        logger.info(progress.summary())
        
        # Verify all checkpoints passed
        assert all(cp['success'] for cp in progress.checkpoints.values()), \
            "Not all checkpoints passed"
            
    except Exception as e:
        logger.error(f"Integration test failed: {str(e)}")
        raise
    finally:
        # Clean up
        if 'db' in locals():
            db.close()  # Close database connection
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

if __name__ == "__main__":
    test_full_flow() 