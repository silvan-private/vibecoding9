"""Script to analyze videos for speaker identification."""
import logging
from speaker_db import SpeakerKnowledgeDB
from tqdm import tqdm
import time
from pathlib import Path
import datetime
import traceback
import sqlite3
import gc
import psutil
import os
import json
import hashlib
from urllib.parse import urlparse, parse_qs
import re
from audio_processor import AudioProcessor
from utils.time_helpers import TimeFormatter
import whisper
from typing import Optional, Dict, List, NamedTuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('speaker_data/analysis_progress.log')
    ]
)
logger = logging.getLogger(__name__)

class ProcessedChunk(NamedTuple):
    """Container for processed audio chunk data."""
    embedding: np.ndarray
    segments: List[Dict]
    start_time: float
    duration: float

# Constants
SAMPLE_RATE = 16000
DEVICE = "cpu"

class ProgressTracker:
    def __init__(self):
        self.pbar = None
        self.current_stage = None
        self.total = None
        self.progress_file = Path("speaker_data/progress.txt")
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize progress file
        self._write_progress("Starting analysis...", 0, 0)
        
    def _write_progress(self, stage, progress=None, total=None):
        """Write progress to file for external monitoring"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                f.write(f"Current Stage: {stage}\n")
                if progress is not None and total is not None and total > 0:  # Prevent division by zero
                    percentage = (progress / total) * 100
                    f.write(f"Progress: {percentage:.1f}% ({progress}/{total})\n")
                f.write(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                # Add recent log messages
                f.write("\nRecent Updates:\n")
                if hasattr(self, 'recent_updates'):
                    for update in self.recent_updates[-5:]:
                        f.write(f"- {update}\n")
        except Exception as e:
            logger.error(f"Error writing progress: {str(e)}")
        
    def update_progress(self, stage, progress=None, total=None, status=None):
        try:
            # Update terminal progress bar
            if stage != self.current_stage or (total is not None and total != self.total):
                if self.pbar is not None:
                    self.pbar.close()
                if total is not None:
                    self.pbar = tqdm(total=total, desc=stage, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
                    self.total = total
                self.current_stage = stage
            
            if progress is not None and self.pbar is not None:
                self.pbar.n = progress
                self.pbar.refresh()
            
            # Keep track of recent updates - only add if status is different from last update
            if status:
                if not hasattr(self, 'recent_updates'):
                    self.recent_updates = []
                if not self.recent_updates or status != self.recent_updates[-1].split(' - ')[1]:
                    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                    self.recent_updates.append(f"{timestamp} - {status}")
                    if len(self.recent_updates) > 10:
                        self.recent_updates = self.recent_updates[-10:]
            
            # Update progress file
            self._write_progress(stage, progress, total)
        except Exception as e:
            logger.error(f"Error updating progress: {str(e)}\n{traceback.format_exc()}")
            
    def close(self):
        try:
            if self.pbar is not None:
                self.pbar.close()
                self.pbar = None
            if self.progress_file.exists():
                self.progress_file.write_text("Analysis completed at " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        except Exception as e:
            logger.error(f"Error closing progress tracker: {str(e)}")

class TranscriptManager:
    def __init__(self, base_dir: Path = Path("speaker_data")):
        self.base_dir = base_dir
        self.transcripts_dir = base_dir / "transcripts"
        self.db_path = base_dir / "sam_knowledge.db"
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    def _get_video_id(self, url: str) -> str:
        """Extract video ID from URL and create a safe directory name."""
        # Try to extract video ID from YouTube URL
        parsed_url = urlparse(url)
        if 'youtube.com' in parsed_url.netloc or 'youtu.be' in parsed_url.netloc:
            if 'youtube.com' in parsed_url.netloc:
                query = parse_qs(parsed_url.query)
                video_id = query.get('v', [''])[0]
            else:
                video_id = parsed_url.path.lstrip('/')
        else:
            # For non-YouTube URLs, create a hash
            video_id = hashlib.md5(url.encode()).hexdigest()[:10]
        
        return video_id

    def get_video_dir(self, url: str) -> Path:
        """Get the directory for storing transcripts for a specific video."""
        video_id = self._get_video_id(url)
        video_dir = self.transcripts_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        return video_dir

    def save_metadata(self, url: str, metadata: Dict):
        """Save video metadata to JSON file."""
        video_dir = self.get_video_dir(url)
        with open(video_dir / "metadata.json", "w", encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def save_raw_transcript(self, url: str, segments: List[Dict]):
        """Save raw transcript with detailed timing information."""
        video_dir = self.get_video_dir(url)
        with open(video_dir / "raw.txt", "w", encoding='utf-8') as f:
            f.write("=== Raw Transcript ===\n\n")
            for segment in segments:
                start_time = TimeFormatter.format_timestamp(segment['start'])
                end_time = TimeFormatter.format_timestamp(segment['end'])
                confidence = segment.get('similarity', 0) * 100
                f.write(f"[{start_time} -> {end_time}]\n")
                f.write(f"Speaker: {segment.get('speaker', 'Unknown')} ({confidence:.1f}% confidence)\n")
                f.write(f"{segment['text']}\n\n")

    def save_grouped_transcript(self, url: str):
        """Save grouped transcript from database."""
        video_dir = self.get_video_dir(url)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Query to group consecutive segments by the same speaker
        query = """
        WITH grouped AS (
            SELECT 
                speaker,
                group_concat(text, ' ') as full_text,
                min(timestamp) as start_time,
                max(end_timestamp) as end_time,
                round(avg(confidence)*100,1) as avg_confidence
            FROM transcripts
            WHERE video_url = ?
            GROUP BY speaker, (
                SELECT count(*)
                FROM transcripts t2
                WHERE t2.timestamp <= transcripts.timestamp
                AND t2.speaker != transcripts.speaker
                AND t2.video_url = ?
            )
        )
        SELECT 
            start_time,
            end_time,
            speaker,
            avg_confidence,
            full_text
        FROM grouped
        ORDER BY start_time;
        """
        
        cursor.execute(query, (url, url))
        segments = cursor.fetchall()
        
        with open(video_dir / "grouped.txt", "w", encoding='utf-8') as f:
            f.write("=== Grouped Transcript ===\n\n")
            for start, end, speaker, confidence, text in segments:
                start_time = TimeFormatter.format_timestamp(start)
                end_time = TimeFormatter.format_timestamp(end)
                f.write(f"[{start_time} -> {end_time}]\n")
                f.write(f"Speaker: {speaker} ({confidence:.1f}% confidence)\n")
                f.write(f"{text}\n\n")
        
        conn.close()

    def save_html_viewer(self, url: str):
        """Generate an HTML viewer for the transcript."""
        video_dir = self.get_video_dir(url)
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Get video metadata
        metadata = {}
        if (video_dir / "metadata.json").exists():
            with open(video_dir / "metadata.json", "r", encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Query for transcript data
        cursor.execute("""
            SELECT timestamp, end_timestamp, speaker, text, confidence
            FROM transcripts
            WHERE video_url = ?
            ORDER BY timestamp
        """, (url,))
        segments = cursor.fetchall()
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Transcript Viewer</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                }}
                .metadata {{
                    background: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .segment {{
                    margin-bottom: 20px;
                    padding: 10px;
                    border-left: 3px solid #ddd;
                }}
                .timestamp {{
                    color: #666;
                    font-size: 0.9em;
                }}
                .speaker {{
                    font-weight: bold;
                    color: #2c5282;
                }}
                .confidence {{
                    color: #718096;
                    font-size: 0.9em;
                }}
                .text {{
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="metadata">
                <h1>Transcript Viewer</h1>
                <p>URL: <a href="{url}" target="_blank">{url}</a></p>
                <p>Analysis Date: {metadata.get('analysis_date', 'Unknown')}</p>
                <p>Duration: {metadata.get('duration', 'Unknown')}</p>
            </div>
            
            <div class="transcript">
        """
        
        for timestamp, end_timestamp, speaker, text, confidence in segments:
            start_time = TimeFormatter.format_timestamp(timestamp)
            end_time = TimeFormatter.format_timestamp(end_timestamp)
            html_content += f"""
                <div class="segment">
                    <div class="timestamp">[{start_time} -> {end_time}]</div>
                    <div class="speaker">{speaker}</div>
                    <div class="confidence">Confidence: {confidence*100:.1f}%</div>
                    <div class="text">{text}</div>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(video_dir / "viewer.html", "w", encoding='utf-8') as f:
            f.write(html_content)
        
        conn.close()

    def create_index_page(self):
        """Create an index page listing all analyzed videos."""
        videos = []
        for video_dir in self.transcripts_dir.iterdir():
            if video_dir.is_dir():
                metadata_file = video_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r", encoding='utf-8') as f:
                        metadata = json.load(f)
                        videos.append({
                            'id': video_dir.name,
                            'url': metadata.get('url', 'Unknown'),
                            'title': metadata.get('title', 'Unknown'),
                            'analysis_date': metadata.get('analysis_date', 'Unknown'),
                            'duration': metadata.get('duration', 'Unknown')
                        })
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Analyzed Videos Index</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                }
                .video-list {
                    display: grid;
                    gap: 20px;
                }
                .video-card {
                    border: 1px solid #ddd;
                    padding: 15px;
                    border-radius: 5px;
                }
                .video-title {
                    font-size: 1.2em;
                    font-weight: bold;
                    margin-bottom: 10px;
                }
                .video-meta {
                    color: #666;
                    font-size: 0.9em;
                }
                .video-links {
                    margin-top: 10px;
                }
                .video-links a {
                    margin-right: 15px;
                    color: #2c5282;
                    text-decoration: none;
                }
                .video-links a:hover {
                    text-decoration: underline;
                }
            </style>
        </head>
        <body>
            <h1>Analyzed Videos Index</h1>
            <div class="video-list">
        """
        
        for video in sorted(videos, key=lambda x: x['analysis_date'], reverse=True):
            html_content += f"""
                <div class="video-card">
                    <div class="video-title">{video['title']}</div>
                    <div class="video-meta">
                        <div>Analysis Date: {video['analysis_date']}</div>
                        <div>Duration: {video['duration']}</div>
                        <div>URL: <a href="{video['url']}" target="_blank">{video['url']}</a></div>
                    </div>
                    <div class="video-links">
                        <a href="transcripts/{video['id']}/viewer.html">View Transcript</a>
                        <a href="transcripts/{video['id']}/raw.txt">Raw Transcript</a>
                        <a href="transcripts/{video['id']}/grouped.txt">Grouped Transcript</a>
                    </div>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(self.base_dir / "index.html", "w", encoding='utf-8') as f:
            f.write(html_content)

class AudioProcessor:
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the audio processor."""
        try:
            # Set up logging
            self.logger = logging.getLogger(__name__)
            
            # Set up output directory
            self.output_dir = Path(output_dir or "speaker_data")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create required subdirectories
            self.models_dir = self.output_dir / "models"
            self.cache_dir = self.output_dir / "cache"
            self.embeddings_dir = self.output_dir / "embeddings"
            
            for directory in [self.models_dir, self.cache_dir, self.embeddings_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize models
            self.logger.info("Loading Whisper model on cpu...")
            self.whisper = whisper.load_model("base", device=DEVICE)
            self.logger.info("Whisper model loaded successfully")
            
            # Initialize ECAPA-TDNN model
            self.logger.info("Loading ECAPA-TDNN model on cpu...")
            self.classifier = self._load_ecapa_tdnn()
            self.logger.info("ECAPA-TDNN model loaded successfully")
            
            # Initialize reference speaker
            self.reference_embedding = None
            self.similarity_threshold = 0.75  # Increased threshold for more strict matching
            
        except Exception as e:
            self.logger.error(f"Error initializing AudioProcessor: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _load_ecapa_tdnn(self):
        """Load ECAPA-TDNN speaker recognition model."""
        try:
            from speechbrain.pretrained import EncoderClassifier
            
            # Download and load the model
            classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(self.models_dir / "ecapa-tdnn"),
                run_opts={"device": DEVICE}
            )
            
            return classifier
            
        except Exception as e:
            self.logger.error(f"Error loading ECAPA-TDNN model: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _extract_voice_embedding(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """Extract voice embedding from audio chunk."""
        try:
            if self.classifier is None:
                self.logger.error("ECAPA-TDNN classifier not initialized")
                return None
                
            # Ensure chunk is in the right format
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.classifier.encode_batch(torch.from_numpy(chunk).unsqueeze(0))
                return embedding.squeeze().cpu().numpy()
                
        except Exception as e:
            self.logger.error(f"Error extracting voice embedding: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def _transcribe_audio(self, chunk: np.ndarray) -> Optional[Dict]:
        """Transcribe audio chunk using Whisper."""
        try:
            if self.whisper is None:
                self.logger.error("Whisper model not initialized")
                return None
                
            # Ensure chunk is in the right format
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)
            
            # Transcribe
            result = self.whisper.transcribe(chunk, language="en")
            return result
            
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def process_chunk(self, chunk: np.ndarray, chunk_start: float = 0.0) -> Optional[ProcessedChunk]:
        """Process a single chunk of audio data."""
        try:
            self.logger.info(f"\nProcessing chunk starting at {chunk_start:.2f}s")
            self.logger.info(f"Chunk shape: {chunk.shape}, dtype: {chunk.dtype}")
            self.logger.info(f"Chunk stats - min: {chunk.min():.3f}, max: {chunk.max():.3f}, mean: {chunk.mean():.3f}")
            
            # First extract voice embedding
            self.logger.info("Extracting voice embedding...")
            try:
                if self.classifier is None:
                    self.logger.error("ECAPA-TDNN classifier is not initialized")
                    return None
                    
                embedding = self._extract_voice_embedding(chunk)
                if embedding is None:
                    self.logger.error("Voice embedding extraction failed - no embedding returned")
                    return None
                self.logger.info(f"Embedding extracted with shape {embedding.shape}")
            except Exception as e:
                self.logger.error(f"Voice embedding extraction failed with error: {str(e)}")
                self.logger.error("Full traceback:")
                import traceback
                self.logger.error(traceback.format_exc())
                return None

            # Then transcribe
            self.logger.info("Transcribing audio...")
            try:
                result = self._transcribe_audio(chunk)
                if not result:
                    self.logger.error("Transcription failed")
                    return None
                self.logger.info(f"Transcription completed with {len(result['segments'])} segments")
            except Exception as e:
                self.logger.error(f"Transcription failed with error: {str(e)}")
                self.logger.error("Full traceback:")
                import traceback
                self.logger.error(traceback.format_exc())
                return None

            # Process segments
            try:
                matching_segments = []
                for segment in result['segments']:
                    # Add chunk_start to segment times
                    segment['start'] += chunk_start
                    segment['end'] += chunk_start
                    
                    # Calculate similarity if we have a reference embedding
                    if self.reference_embedding is not None:
                        similarity = cosine_similarity(
                            embedding.reshape(1, -1),
                            self.reference_embedding.reshape(1, -1)
                        )[0][0]
                        
                        # Enhanced logging for speaker detection
                        self.logger.info(f"\nSegment {TimeFormatter.format(segment['start'])} -> {TimeFormatter.format(segment['end'])}")
                        self.logger.info(f"Text: {segment['text']}")
                        self.logger.info(f"Similarity score: {similarity:.4f} (threshold: {self.similarity_threshold:.4f})")
                        
                        if similarity >= self.similarity_threshold:
                            segment['similarity'] = float(similarity)
                            matching_segments.append(segment)
                            self.logger.info("✓ Segment matched!")
                        else:
                            self.logger.info("✗ Segment below threshold")
                            # Add segment with low similarity for debugging
                            segment['similarity'] = float(similarity)
                            matching_segments.append(segment)  # Include non-matching segments
                    else:
                        # If no reference embedding, keep all segments
                        matching_segments.append(segment)

                self.logger.info(f"\nFound {len(matching_segments)} segments in chunk")
                
                return ProcessedChunk(
                    embedding=embedding,
                    segments=matching_segments,
                    start_time=chunk_start,
                    duration=len(chunk) / SAMPLE_RATE
                )
            except Exception as e:
                self.logger.error(f"Error processing segments: {str(e)}")
                self.logger.error("Full traceback:")
                import traceback
                self.logger.error(traceback.format_exc())
                return None
        except Exception as e:
            self.logger.error(f"Error in process_chunk: {str(e)}")
            self.logger.error("Full traceback:")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def create_speaker_profile(self, speaker_name: str, video_url: str, start_time: float = 0, duration: float = 60) -> Optional[str]:
        """Create a speaker profile from a reference video segment."""
        try:
            self.logger.info(f"Creating speaker profile for {speaker_name} from {video_url}")
            self.logger.info(f"Using segment: {start_time}s to {start_time + duration}s")
            
            # Download and process reference video
            audio_file = self.download_video(video_url, start_time, duration)
            if not audio_file:
                self.logger.error("Failed to download reference video")
                return None
            
            # Process reference audio to get embedding
            results = self.process_audio_file(audio_file, start_time, duration)
            if not results or not results[0]:
                self.logger.error("Failed to process reference audio")
                return None
            
            # Store reference embedding
            self.reference_embedding = results[0].embedding
            
            # Save embedding for future use
            embedding_file = self.embeddings_dir / f"{speaker_name}_reference.npy"
            np.save(str(embedding_file), self.reference_embedding)
            
            self.logger.info(f"Successfully created speaker profile for {speaker_name}")
            return speaker_name
            
        except Exception as e:
            self.logger.error(f"Error creating speaker profile: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def download_video(self, url: str, start_time: float = 0, duration: Optional[float] = None) -> Optional[Path]:
        """Download video and extract audio segment."""
        try:
            from yt_dlp import YoutubeDL
            import ffmpeg
            
            # Download video
            output_file = self.cache_dir / f"{hashlib.md5(url.encode()).hexdigest()}.wav"
            if not output_file.exists():
                ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                    }],
                    'outtmpl': str(output_file.with_suffix('')),
                }
                
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            
            # Extract segment if needed
            if start_time > 0 or duration is not None:
                segment_file = self.cache_dir / f"{output_file.stem}_{start_time}_{duration or 'end'}.wav"
                if not segment_file.exists():
                    stream = ffmpeg.input(str(output_file))
                    if duration:
                        stream = stream.filter('atrim', start=start_time, duration=duration)
                    else:
                        stream = stream.filter('atrim', start=start_time)
                    stream = stream.filter('asetpts', 'PTS-STARTPTS')
                    stream.output(str(segment_file)).overwrite_output().run(capture_stdout=True, capture_stderr=True)
                return segment_file
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error downloading video: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None

    def process_audio_file(self, audio_file: Path, start_time: float = 0, duration: Optional[float] = None) -> List[ProcessedChunk]:
        """Process audio file and return chunks with embeddings and transcripts."""
        try:
            import soundfile as sf
            
            # Load audio file
            audio, sample_rate = sf.read(str(audio_file))
            if sample_rate != SAMPLE_RATE:
                # Resample if needed
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * SAMPLE_RATE / sample_rate))
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Process in chunks
            chunk_duration = 30  # seconds
            chunk_size = chunk_duration * SAMPLE_RATE
            chunks = []
            
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                if len(chunk) < SAMPLE_RATE:  # Skip chunks shorter than 1 second
                    continue
                    
                chunk_start = i / SAMPLE_RATE
                result = self.process_chunk(chunk, chunk_start)
                if result:
                    chunks.append(result)
                    
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing audio file: {str(e)}")
            self.logger.error(traceback.format_exc())
            return []

    def set_reference_speaker(self, speaker_id: str):
        """Set the reference speaker for comparison."""
        try:
            # Load the saved embedding
            embedding_file = self.embeddings_dir / f"{speaker_id}_reference.npy"
            if not embedding_file.exists():
                self.logger.error(f"No embedding found for speaker {speaker_id}")
                return False
                
            self.reference_embedding = np.load(str(embedding_file))
            self.logger.info(f"Loaded reference embedding for {speaker_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting reference speaker: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

def format_transcript(knowledge, output_file=None):
    """Format transcript with clear speaker labels and optionally save to file."""
    try:
        header = "\nTRANSCRIPT WITH SPEAKER IDENTIFICATION"
        separator = "=" * 80
        
        if not knowledge:
            message = "No transcript segments found."
            print(header)
            print(separator)
            print(message)
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"{header}\n{separator}\n{message}\n")
            return
        
        # Sort entries by timestamp
        sorted_entries = sorted(knowledge, key=lambda x: x['timestamp'])
        
        # Prepare transcript content
        transcript_lines = [header, separator]
        
        current_speaker = None
        for entry in sorted_entries:
            timestamp = entry['timestamp']
            text = entry['text'].strip()
            confidence = entry['confidence']
            is_sam = entry.get('is_sam', True)  # Default to SAM if not specified
            
            # Add newline between different speakers
            if current_speaker is not None and current_speaker != is_sam:
                transcript_lines.append("")
                
            # Format speaker label
            speaker_label = "SAM" if is_sam else "Other Speaker"
            current_speaker = is_sam
            
            # Format timestamp as MM:SS
            minutes = int(timestamp) // 60
            seconds = int(timestamp) % 60
            timestamp_str = f"{minutes:02d}:{seconds:02d}"
            
            # Add formatted entry
            transcript_lines.extend([
                f"[{speaker_label}] ({timestamp_str})",
                f"  {text}",
                f"  Confidence: {confidence:.2f}",
                "-" * 40
            ])
        
        # Join all lines
        transcript_content = "\n".join(transcript_lines)
        
        # Print to console
        print(transcript_content)
        
        # Save to file if specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(transcript_content)
            logger.info(f"Transcript saved to: {output_file}")
            
    except Exception as e:
        logger.error(f"Error formatting transcript: {str(e)}\n{traceback.format_exc()}")
        raise

def analyze_video(processor, url, start_time=0, duration=None, speakers=None):
    """Analyze a video for speaker segments."""
    logger.info(f"\nAnalyzing video: {url}")
    logger.info(f"Time range: {start_time}s to {start_time + (duration or 0)}s")
    logger.info(f"Looking for {len(speakers or [])} speakers...")
    
    # Set reference speakers
    for speaker in (speakers or []):
        processor.set_reference_speaker(speaker)
    
    # Download and process video
    audio_file = processor.download_video(url, start_time, duration)
    if not audio_file:
        logger.error("Failed to download video")
        return None
        
    # Process audio file
    results = processor.process_audio_file(audio_file, start_time, duration)
    if not results:
        logger.error("Failed to process audio")
        return None
        
    # Combine results
    all_segments = []
    for result in results:
        if result and result.segments:
            all_segments.extend(result.segments)
            
    logger.info(f"Found {len(all_segments)} segments")
    return all_segments

def main():
    """Main entry point."""
    try:
        # Initialize transcript manager
        transcript_manager = TranscriptManager()
        
        # Initialize SQLite database
        db_path = Path("speaker_data/sam_knowledge.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create transcripts table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_url TEXT,
                timestamp REAL,
                end_timestamp REAL,
                speaker TEXT,
                text TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        
        logger.info("Initializing speaker analysis system...")
        processor = AudioProcessor()
        
        # Create speaker profile
        logger.info("\nCreating speaker profile for SAM...")
        speaker_id = processor.create_speaker_profile(
            "SAM",
            "https://www.youtube.com/watch?v=cfAUbJgR0pE",
            start_time=0,
            duration=60  # Analyze first 60 seconds for reference
        )
        if not speaker_id:
            logger.error("Failed to create speaker profile")
            return
            
        # Analyze target video
        logger.info("\nAnalyzing target video for SAM's segments...")
        logger.info(f"Memory usage before chunk 1: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB\n")
        
        target_url = "https://www.youtube.com/watch?v=gPKkIkEnZw8"
        segments = analyze_video(
            processor,
            target_url,
            start_time=0,     # Start from beginning
            duration=None,    # Analyze entire video (changed from 60)
            speakers=[speaker_id]
        )
        
        logger.info("Video analysis completed")
        
        # Save results to database and generate transcripts
        if segments:
            # Save metadata
            metadata = {
                'url': target_url,
                'title': 'SAM Analysis',  # You might want to get this from YouTube
                'analysis_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration': 'Full video',  # Updated to reflect full video analysis
                'total_segments': len(segments)
            }
            transcript_manager.save_metadata(target_url, metadata)
            
            # Save raw transcript
            transcript_manager.save_raw_transcript(target_url, segments)
            
            logger.info("\nSaving transcribed segments to database...")
            for segment in segments:
                # Save to database
                cursor.execute('''
                    INSERT INTO transcripts (video_url, timestamp, end_timestamp, speaker, text, confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    target_url,
                    segment['start'],
                    segment['end'],
                    'SAM' if segment.get('similarity', 0) >= processor.similarity_threshold else 'Other Speaker',
                    segment['text'],
                    segment.get('similarity', 0)
                ))
            
            conn.commit()
            
            # Generate grouped transcript and HTML viewer
            transcript_manager.save_grouped_transcript(target_url)
            transcript_manager.save_html_viewer(target_url)
            
            # Update index page
            transcript_manager.create_index_page()
            
            logger.info(f"\nTranscripts and viewer saved in: {transcript_manager.get_video_dir(target_url)}")
            logger.info(f"View the transcript at: {transcript_manager.base_dir}/index.html")
            
        else:
            logger.warning("No matching segments found")
            
        conn.close()
        
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main() 