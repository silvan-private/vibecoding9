import os
# Only set essential environment variables
os.environ['HF_HOME'] = './huggingface_cache'
os.environ['SPEECHBRAIN_CACHE_DIR'] = './model_cache'

import sys
import logging
import torch
import torchaudio
import numpy as np
from pathlib import Path
import yt_dlp
from pydub import AudioSegment
import whisper
import time
import requests
from yt_dlp import YoutubeDL
import json
from datetime import datetime
from speechbrain.inference import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import argparse
import shutil
from huggingface_hub import hf_hub_download
import traceback
from utils.time_helpers import TimeFormatter
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import nullcontext
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class ProcessedChunk:
    """Class to hold processed chunk data."""
    embedding: np.ndarray
    segments: List[Dict[str, Any]]
    start_time: float
    duration: float

# Configure device and optimization settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_THREADS = min(multiprocessing.cpu_count(), 8)  # Use up to 8 CPU threads
ENABLE_FP16 = DEVICE.type == 'cuda'  # Enable FP16 on GPU only
CHUNK_DURATION = 30.0  # Process in 30-second chunks
SAMPLE_RATE = 16000  # 16kHz sample rate for models

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Add stdout handler
        logging.FileHandler('audio_processing.log')  # Also save to file
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Ensure logger level is set to INFO

# Create a minimal custom.py module
def create_custom_module(model_dir):
    """Create a minimal custom.py module for SpeechBrain"""
    custom_py = model_dir / "custom.py"
    if not custom_py.exists():
        with open(custom_py, "w") as f:
            f.write("""
# Minimal custom module for SpeechBrain
def custom_func(*args, **kwargs):
    pass
""")
    return custom_py

# Patch SpeechBrain's file handling to use copying instead of symlinks
import speechbrain.utils.fetching as sb_fetching
def safe_link_strategy(src, dst):
    """Safe linking strategy that uses copying instead of symlinks"""
    try:
        # Convert to Path objects and resolve
        src_path = Path(src).resolve()
        dst_path = Path(dst).resolve()
        
        # Create parent directory if it doesn't exist
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # If source is custom.py and doesn't exist, create it
        if src_path.name == "custom.py" and not src_path.exists():
            src_path = create_custom_module(src_path.parent)
        
        # Check if source exists
        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {src_path}")
        
        # Copy file
        logger.info(f"Copying {src_path} -> {dst_path}")
        shutil.copy2(str(src_path), str(dst_path))
        return dst_path
        
    except Exception as e:
        logger.error(f"Error copying file: {str(e)}")
        raise

# Override SpeechBrain's file handling
sb_fetching.SYMLINK_STRATEGY = "copy"
sb_fetching.link_with_strategy = lambda src, dst, _: safe_link_strategy(src, dst)

def download_model_files(model_dir):
    """Download ECAPA-TDNN model files with Windows-safe copy operations"""
    try:
        # Setup directories
        model_dir = Path(model_dir).resolve()
        cache_dir = model_dir / "cache"
        model_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create custom module
        create_custom_module(model_dir)
        
        # Define model files to download
        model_files = {
            "hyperparams.yaml": "hyperparams.yaml",
            "embedding_model.ckpt": "embedding_model.ckpt",
            "mean_var_norm_emb.ckpt": "mean_var_norm_emb.ckpt",
            "classifier.ckpt": "classifier.ckpt",
            "label_encoder.txt": "label_encoder.txt"
        }
        
        model_repo = "speechbrain/spkrec-ecapa-voxceleb"
        
        # Download and copy each file
        for local_file, remote_file in model_files.items():
            target_file = model_dir / local_file
            if not target_file.exists():
                try:
                    logger.info(f"Downloading {local_file}...")
                    # Download to cache directory first
                    tmp_file = hf_hub_download(
                        repo_id=model_repo,
                        filename=remote_file,
                        cache_dir=cache_dir,
                        local_dir_use_symlinks=False
                    )
                    
                    # Copy file to final location
                    safe_link_strategy(tmp_file, target_file)
                    logger.info(f"✓ Successfully downloaded and copied {local_file}")
                    
                except Exception as e:
                    logger.error(f"Error downloading {local_file}: {str(e)}")
                    return False
        
        # Verify all files exist
        missing_files = [f for f in model_files.keys() if not (model_dir / f).exists()]
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model files: {str(e)}")
        return False

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
            self.similarity_threshold = 0.65  # Lower threshold for better matching
            
        except Exception as e:
            self.logger.error(f"Error initializing AudioProcessor: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def set_reference_speaker(self, speaker_id: str) -> bool:
        """Set the reference speaker for comparison."""
        try:
            # Load embedding from file
            embedding_file = self.embeddings_dir / f"speaker_{speaker_id}.npz"
            if not embedding_file.exists():
                self.logger.error(f"No embedding file found for speaker {speaker_id}")
                return False
            
            # Load embedding
            data = np.load(str(embedding_file))
            self.reference_embedding = data['embedding']
            self.logger.info(f"Set reference speaker {speaker_id} with embedding shape {self.reference_embedding.shape}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error setting reference speaker: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

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
                if self.whisper is None:
                    self.logger.error("Whisper model is not initialized")
                    return None
                    
                result = self._transcribe_audio(chunk)
                if not result or not result.get('segments'):
                    self.logger.error("Transcription failed - no segments returned")
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
                        
                        # Log similarity score for each segment
                        self.logger.info(f"\nSegment {TimeFormatter.format(segment['start'])} -> {TimeFormatter.format(segment['end'])}")
                        self.logger.info(f"Text: {segment['text']}")
                        self.logger.info(f"Similarity score: {similarity:.4f} (threshold: {self.similarity_threshold:.4f})")
                        
                        if similarity >= self.similarity_threshold:
                            segment['similarity'] = float(similarity)
                            matching_segments.append(segment)
                            self.logger.info("✓ Segment matched!")
                        else:
                            self.logger.info("✗ Segment below threshold")
                    else:
                        # If no reference embedding, keep all segments
                        matching_segments.append(segment)

                self.logger.info(f"\nFound {len(matching_segments)} matching segments in chunk")
                
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
            self.logger.error(f"Unexpected error in process_chunk: {str(e)}")
            self.logger.error("Full traceback:")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _transcribe_audio(self, audio_data: np.ndarray) -> Optional[dict]:
        """Transcribe audio data using Whisper."""
        try:
            self.logger.info("Starting Whisper transcription...")
            
            # Convert to float32 if not already
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio to [-1, 1] range if not already
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
                
            # Run transcription
            result = self.whisper.transcribe(
                audio_data,
                language='en',
                task='transcribe',
                fp16=ENABLE_FP16 and DEVICE.type == 'cuda'
            )
            
            if not result or not isinstance(result, dict):
                self.logger.error("Transcription failed - invalid result format")
                return None
                
            if 'segments' not in result:
                self.logger.error("Transcription failed - no segments in result")
                return None
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in transcription: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def _extract_voice_embedding(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Extract voice embedding from audio data."""
        try:
            self.logger.info("Starting voice embedding extraction...")
            
            # Convert to float32 if not already
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Apply preprocessing steps
            # 1. Normalize audio to [-1, 1] range first
            audio_data = librosa.util.normalize(audio_data)
            
            # 2. Apply voice activity detection to remove silence
            intervals = librosa.effects.split(
                audio_data,
                top_db=20,  # More lenient threshold
                frame_length=2048,
                hop_length=512
            )
            
            if len(intervals) > 0:
                # Concatenate non-silent intervals
                audio_data = np.concatenate([audio_data[start:end] for start, end in intervals])
                self.logger.info(f"After VAD: kept {len(audio_data)/SAMPLE_RATE:.2f}s of audio")
            
            # 3. Pre-emphasis filter to boost high frequencies
            audio_data = librosa.effects.preemphasis(audio_data, coef=0.95)
            
            # Convert to tensor and add batch dimension
            waveform = torch.tensor(audio_data).unsqueeze(0)
            
            # Move to device and convert to FP16 if enabled
            with torch.no_grad():
                waveform = waveform.to(DEVICE)
                if ENABLE_FP16 and DEVICE.type == 'cuda':
                    waveform = waveform.half()
                
                # Extract embedding
                embedding = self.classifier.encode_batch(waveform)
                embedding = embedding.cpu().numpy()
                
                if embedding is None or embedding.size == 0:
                    self.logger.error("Embedding extraction failed - empty result")
                    return None
                    
                return embedding.squeeze()
                
        except Exception as e:
            self.logger.error(f"Error in voice embedding extraction: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def set_progress_callback(self, callback):
        """Set a callback function to report progress"""
        self.progress_callback = callback

    def update_progress(self, stage, progress=None):
        """Update progress through callback if set"""
        if self.progress_callback:
            self.progress_callback(stage, progress)

    def _get_cached_audio(self, url, start_time, end_time):
        """Try to get cached audio file for the given parameters"""
        import hashlib
        
        # Ensure consistent numeric values for cache key
        start_seconds = float(start_time) if start_time is not None else 0
        end_seconds = float(end_time) if end_time is not None else float('inf')
        
        # Create a unique cache key based on URL and time range
        cache_key = f"{url}_{start_seconds:.3f}_{end_seconds:.3f}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = self.cache_dir / f"audio_cache_{cache_hash}.wav"
        
        logger.info(f"Checking cache with key: {cache_key}")
        logger.info(f"Looking for cache file: {cache_file}")
        
        if cache_file.exists():
            # Verify the cached file
            try:
                audio_data, sr = sf.read(str(cache_file))
                duration = len(audio_data) / sr
                if end_seconds != float('inf') and duration < end_seconds:
                    logger.info(f"Cache file too short ({duration:.2f}s < {end_seconds:.2f}s) - will re-download")
                    return None
                logger.info(f"✓ Cache hit! Found cached audio file: {cache_file}")
                return str(cache_file)
            except Exception as e:
                logger.error(f"Error verifying cache file: {str(e)}")
                return None
        
        logger.info("✗ Cache miss - will need to download")
        return None

    def _save_to_cache(self, audio_file, url, start_time, end_time):
        """Save processed audio file to cache"""
        import hashlib
        import shutil
        
        # Ensure consistent numeric values for cache key
        start_seconds = float(start_time) if start_time is not None else 0
        end_seconds = float(end_time) if end_time is not None else float('inf')
        
        # Create a unique cache key based on URL and time range
        cache_key = f"{url}_{start_seconds:.3f}_{end_seconds:.3f}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = self.cache_dir / f"audio_cache_{cache_hash}.wav"
        
        logger.info(f"Saving to cache with key: {cache_key}")
        logger.info(f"Cache file path: {cache_file}")
        
        # Copy the file to cache
        shutil.copy2(audio_file, cache_file)
        logger.info(f"✓ Successfully saved audio file to cache: {cache_file}")
        return str(cache_file)

    def download_audio(self, url, start_time=None, end_time=None):
        """Download audio from YouTube URL using yt-dlp and process with FFmpeg"""
        try:
            logger.info(f"Downloading audio from {url}")
            logger.info(f"Time range: start={start_time}, end={end_time}")
            
            # Convert time strings to seconds if needed
            if isinstance(start_time, str):
                h, m, s = map(int, start_time.split(':'))
                start_seconds = h * 3600 + m * 60 + s
            else:
                start_seconds = start_time

            if isinstance(end_time, str):
                h, m, s = map(int, end_time.split(':'))
                end_seconds = h * 3600 + m * 60 + s
            else:
                end_seconds = end_time
            
            logger.info(f"Converted time range: {start_seconds}s to {end_seconds}s")
            
            # Check cache first
            cached_file = self._get_cached_audio(url, start_seconds, end_seconds)
            if cached_file:
                logger.info("Using cached audio file")
                return cached_file
            
            logger.info("No cache found - proceeding with download...")
            
            # If not in cache, proceed with download
            self.output_dir.mkdir(exist_ok=True)
            temp_file = str(self.output_dir / 'full_audio.%(ext)s')
            intermediate_file = str(self.output_dir / 'temp_audio_raw.wav')
            final_file = str(self.output_dir / 'temp_audio.wav')
            
            # Base YDL options
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': temp_file,
                'quiet': True,
                'no_warnings': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'prefer_ffmpeg': True,
            }

            downloaded_file = None
            last_error = None
            
            # Try different cookie methods in order of preference
            methods = [
                ('cookies_file', lambda: self._try_cookies_file(ydl_opts, url)),
                ('browser_cookies', lambda: self._try_browser_cookies(ydl_opts, url)),
                ('no_cookies', lambda: self._try_no_cookies(ydl_opts, url))
            ]
            
            for method_name, method in methods:
                try:
                    logger.info(f"Attempting download using {method_name}...")
                    downloaded_file = method()
                    if downloaded_file and os.path.exists(downloaded_file):
                        logger.info(f"Successfully downloaded using {method_name}")
                        break
                except Exception as e:
                    last_error = str(e)
                    logger.info(f"Failed with {method_name}: {str(e)}")
                    continue

            if not downloaded_file or not os.path.exists(downloaded_file):
                raise Exception(f"Failed to download audio: {last_error}\nPlease try one of the following:\n"
                              "1. Log into YouTube in your browser\n"
                              "2. Export cookies to cookies.txt (see https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp)\n"
                              "3. Place cookies.txt in the same directory as the script")

            # If time range is specified, extract the segment using FFmpeg
            if start_seconds is not None and end_seconds is not None:
                duration = end_seconds - start_seconds
                import subprocess

                # Add a small buffer before the start time for better extraction
                buffer_start = max(0, start_seconds - 0.5)
                buffer_duration = duration + 1  # Add 1 second buffer

                # First pass: Extract time range
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-i', downloaded_file,
                    '-ss', str(buffer_start),
                    '-t', str(buffer_duration),
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    intermediate_file
                ]

                logger.info(f"Extracting time range {start_seconds} to {end_seconds}...")
                try:
                    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"FFmpeg error: {e.stderr.decode()}")
                    raise Exception("Failed to extract audio segment")

                # Second pass: Apply audio normalization and preprocessing
                try:
                    # Load audio with soundfile
                    audio, sr = sf.read(intermediate_file)
                    
                    # Ensure mono
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    
                    # Normalize audio
                    audio = librosa.util.normalize(audio)
                    
                    # Apply pre-emphasis filter
                    audio = librosa.effects.preemphasis(audio)
                    
                    # Save processed audio
                    sf.write(final_file, audio, sr, subtype='PCM_16')
                    
                    # Clean up intermediate files
                    os.remove(downloaded_file)
                    os.remove(intermediate_file)
                    
                except Exception as e:
                    logger.error(f"Error processing audio: {str(e)}")
                    raise
            else:
                # If no time range specified, just use the full audio
                shutil.move(downloaded_file, final_file)

            if not os.path.exists(final_file):
                raise Exception("Failed to create final audio file")

            # Verify the audio file
            try:
                audio, sr = sf.read(final_file)
                if len(audio) == 0:
                    raise Exception("Downloaded audio file is empty")
                logger.info(f"Audio file verified: {len(audio)/sr:.2f} seconds, {sr}Hz")
                
                # Plot audio waveform for debugging
                plt.figure(figsize=(10, 4))
                plt.plot(audio)
                plt.title("Audio Waveform")
                plt.savefig(str(self.output_dir / "debug_waveform.png"))
                plt.close()
                
            except Exception as e:
                logger.error(f"Error verifying audio file: {str(e)}")
                raise

            logger.info(f"Download completed. Audio file: {final_file}")

            # Save to cache before returning
            final_file = self._save_to_cache(final_file, url, start_seconds, end_seconds)
            return final_file
            
        except Exception as e:
            logger.error(f"Error downloading audio: {str(e)}")
            raise Exception(f"Error downloading audio: {str(e)}")

    def _try_cookies_file(self, ydl_opts, url):
        """Try downloading using cookies.txt file"""
        cookies_file = Path("cookies.txt")
        if not cookies_file.exists():
            cookies_file = self.output_dir / "cookies.txt"
        
        if cookies_file.exists():
            ydl_opts['cookiefile'] = str(cookies_file)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
        else:
            raise Exception("No cookies.txt file found")
            
    def _try_browser_cookies(self, ydl_opts, url):
        """Try downloading using browser cookies"""
        browsers = ['chrome', 'firefox', 'edge', 'safari', 'opera', 'chromium', 'brave']
        last_error = None
        
        for browser in browsers:
            try:
                ydl_opts['cookiesfrombrowser'] = (browser,)
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    return ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')
            except Exception as e:
                last_error = str(e)
                continue
                
        raise Exception(f"All browser cookie attempts failed: {last_error}")
        
    def _try_no_cookies(self, ydl_opts, url):
        """Try downloading without cookies"""
        ydl_opts.pop('cookiesfrombrowser', None)
        ydl_opts.pop('cookiefile', None)
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info).replace('.webm', '.wav').replace('.m4a', '.wav')

    def _segment_audio(self, audio_path, window_size=1.5, overlap=0.25):
        """Segment audio into overlapping windows using extremely lenient VAD parameters"""
        logger.info("Segmenting audio file...")
        
        # Load and preprocess audio
        audio, sample_rate = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Test mode adjustments
        if self.test_mode:
            logger.info("\nTEST MODE SEGMENTATION:")
            logger.info("-" * 50)
            # Limit audio duration to 3 seconds
            max_duration = 3  # seconds
            max_samples = int(max_duration * sample_rate)
            audio = audio[:max_samples]
            logger.info(f"Limited audio to {max_duration} seconds")
            
            # Use larger windows with less overlap
            window_size = 1.0  # 1 second windows
            overlap = 0.1      # 10% overlap
            logger.info(f"Using test parameters: window_size={window_size}s, overlap={overlap*100}%")
        
        # Convert parameters to samples
        window_samples = int(window_size * sample_rate)
        overlap_samples = int(overlap * window_samples)
        stride = window_samples - overlap_samples
        
        segments = []
        timestamps = []
        current_sample = 0
        total_samples = len(audio)
        
        # Pre-emphasis filter
        audio = librosa.effects.preemphasis(audio)
        
        while current_sample + window_samples <= total_samples:
            segment = audio[current_sample:current_sample + window_samples]
            
            # In test mode, strictly limit to 5 segments
            if self.test_mode and len(segments) >= 5:
                logger.info("Test mode: Reached maximum 5 segments - stopping")
                break
            
            segments.append(torch.FloatTensor(segment))
            timestamps.append({
                "start": current_sample / sample_rate,
                "end": (current_sample + window_samples) / sample_rate
            })
            
            current_sample += stride
        
        num_segments = len(segments)
        logger.info(f"\nSegmentation complete:")
        logger.info(f"Created {num_segments} segments" + (" (limited by test mode)" if self.test_mode else ""))
        logger.info(f"Audio duration: {total_samples/sample_rate:.2f} seconds")
        logger.info(f"Window size: {window_size} seconds")
        logger.info(f"Overlap: {overlap*100}%")
        
        return segments, timestamps

    def _create_embedding(self, audio_segment):
        """Create embedding for audio segment using Wav2Vec2"""
        try:
            if audio_segment.shape[1] > 0:  # Check if segment is not empty
                # Process through Wav2Vec2
                inputs = self.processor(audio_segment.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.audio_model(**inputs)
                    
                # Use the mean of hidden states as the embedding
                embedding = outputs.last_hidden_state.mean(dim=1)
                return embedding
            return None
        except Exception as e:
            logger.error(f"Error creating audio embedding: {str(e)}")
            return None

    def _compute_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between two embeddings"""
        return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

    def _save_embedding(self, embedding, speaker_id):
        """Save speaker embedding to file"""
        try:
            # Ensure both paths are absolute and normalized
            embedding_path = (self.embeddings_dir / f"{speaker_id}_ecapa.npz").resolve()
            output_dir_path = self.output_dir.resolve()
            
            # Save the embedding
            np.savez(embedding_path, embedding=embedding)
            
            # Get relative path using parts to handle Windows paths correctly
            rel_path = embedding_path.relative_to(output_dir_path)
            return str(rel_path)
        except Exception as e:
            logger.error(f"Error saving embedding: {str(e)}")
            # Fallback: return the path relative to embeddings dir
            return f"embeddings/{speaker_id}_ecapa.npz"

    def _cluster_speakers(self, embeddings, reference_embedding=None, similarity_threshold=0.75):
        """Compare embeddings against reference speaker"""
        if reference_embedding is not None:
            # Compare each segment against reference embedding
            similarities = cosine_similarity(embeddings, reference_embedding.reshape(1, -1))
            # Assign segments to speaker if similarity exceeds threshold
            labels = (similarities.squeeze() >= similarity_threshold).astype(int)
            logger.info(f"Found {np.sum(labels)} segments matching reference speaker")
            return labels
        else:
            # If no reference, treat all as same speaker
            return np.zeros(len(embeddings))

    def _find_text_in_timerange(self, transcript, start, end):
        """Find transcribed text within time range with improved overlap detection"""
        logger.info(f"\nSearching for text between {TimeFormatter.format(start)} and {TimeFormatter.format(end)}")
        logger.info(f"Number of transcript segments to search: {len(transcript.get('segments', []))}")
        
        matching_text = []
        
        if not transcript or 'segments' not in transcript:
            return ""
        
        logger.info("\nAll transcript segments:")
        for i, seg in enumerate(transcript['segments']):
            logger.info(f"Transcript {i}: {TimeFormatter.format(seg['start'])} -> {TimeFormatter.format(seg['end'])}: {seg['text']}")
        
        for segment in transcript['segments']:
            # Check if segment overlaps with our time range
            if segment['start'] <= end and segment['end'] >= start:
                logger.info(f"\nChecking transcript segment: {TimeFormatter.format(segment['start'])} -> {TimeFormatter.format(segment['end'])}")
                logger.info(f"Segment text: {segment['text']}")
                
                # If we have word-level timestamps, use them for more precise text selection
                if 'words' in segment:
                    matching_words = []
                    for word_data in segment['words']:
                        if word_data['start'] <= end and word_data['end'] >= start:
                            matching_words.append(word_data['word'])
                    if matching_words:
                        matching_text.append(' '.join(matching_words))
                else:
                    # If no word timestamps, use the entire segment text
                    matching_text.append(segment['text'])
                
                logger.info("Checking overlap conditions:")
                logger.info(f"1. segment_start ({TimeFormatter.format(segment['start'])}) <= end ({TimeFormatter.format(end)}): {segment['start'] <= end}")
                logger.info(f"2. segment_end ({TimeFormatter.format(segment['end'])}) >= start ({TimeFormatter.format(start)}): {segment['end'] >= start}")
                logger.info("✓ Found overlapping segment!")
        
        final_text = ' '.join(matching_text).strip()
        logger.info(f"\nFinal text found: {final_text}")
        return final_text

    def process_video(self, url, start_time=None, end_time=None, speaker_id=None, progress_callback=None):
        """Process a video segment to extract voice print and transcription."""
        try:
            # Download video
            if progress_callback:
                progress_callback(0)
            logger.info(f"Downloading video: {url}")
            
            # Check cache first
            cached_audio = self._get_cached_audio(url, start_time, end_time)
            if cached_audio:
                audio_file = cached_audio
                if progress_callback:
                    progress_callback(20)
            else:
                # Download if not cached
                audio_file = self.download_audio(url, start_time, end_time)
                if progress_callback:
                    progress_callback(20)
            
            # Load and process audio
            logger.info("Processing audio...")
            logger.info(f"Loading audio file: {audio_file}")
            
            if not Path(audio_file).exists():
                raise Exception(f"Audio file does not exist: {audio_file}")
                
            audio_data, sample_rate = sf.read(audio_file)
            logger.info(f"Audio loaded - Shape: {audio_data.shape}, Sample rate: {sample_rate}Hz")
            logger.info(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                logger.info(f"Converting {audio_data.shape[1]} channels to mono")
                audio_data = np.mean(audio_data, axis=1)
            
            # Extract segment if time range specified
            if start_time is not None and end_time is not None:
                logger.info(f"Extracting segment from {start_time}s to {end_time}s")
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                if start_sample >= len(audio_data):
                    raise Exception(f"Start time {start_time}s is beyond audio length {len(audio_data)/sample_rate}s")
                if end_sample > len(audio_data):
                    end_sample = len(audio_data)
                audio_data = audio_data[start_sample:end_sample]
                logger.info(f"Extracted segment - New duration: {len(audio_data)/sample_rate:.2f} seconds")
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                logger.info(f"Resampling from {sample_rate}Hz to 16000Hz")
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            if progress_callback:
                progress_callback(40)
            
            # Process audio in parallel chunks
            chunk_size = 30 * sample_rate  # 30 seconds
            chunks = []
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:min(i + chunk_size, len(audio_data))]
                if len(chunk) >= sample_rate:  # Only include chunks >= 1 second
                    chunks.append(chunk)
            
            logger.info(f"Created {len(chunks)} chunks of {chunk_size/sample_rate:.1f} seconds each")
            
            if not chunks:
                raise Exception(f"No valid chunks to process")
            
            # Process chunks in parallel batches
            batch_size = min(NUM_THREADS, len(chunks))
            results = []
            
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                # Submit all chunks for processing
                future_to_chunk = {
                    executor.submit(self.process_chunk, chunk, i / sample_rate): i 
                    for i, chunk in enumerate(chunks)
                }
                
                # Process results as they complete
                for future in future_to_chunk:
                    chunk_idx = future_to_chunk[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                        if progress_callback:
                            progress = 40 + int((chunk_idx + 1) * 50 / len(chunks))
                            progress_callback(progress)
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                        continue
            
            if not results:
                raise Exception("No valid results from audio processing")
            
            # Combine results
            embedding_file = self.embeddings_dir / f"speaker_{speaker_id}.npz"
            self.embeddings_dir.mkdir(parents=True, exist_ok=True)
            
            # Save combined embedding (average of all chunk embeddings)
            embeddings = np.vstack([r['embedding'] for r in results])
            mean_embedding = np.mean(embeddings, axis=0)
            np.savez(str(embedding_file), embedding=mean_embedding)
            logger.info(f"Saved voice print to {embedding_file}")
            
            # Combine segments
            segments = []
            for r in results:
                segments.extend(r['segments'])
            
            # Sort segments by start time
            segments.sort(key=lambda x: x['start'])
            
            combined_results = {
                'embedding_file': str(embedding_file),
                'segments': segments
            }
            
            if progress_callback:
                progress_callback(100)
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            if 'embedding_file' in locals() and embedding_file.exists():
                try:
                    embedding_file.unlink()
                except:
                    pass
            return None

    def _verify_time_formatter(self):
        """Verify that the TimeFormatter is available and working."""
        try:
            from utils import TimeFormatter
            logging.info("\n[SUCCESS] OpenAI whisper format_timestamp found")
            logging.info("[TEST] format_timestamp(123.456) = " + TimeFormatter.format_timestamp(123.456))
        except ImportError as e:
            logging.error("Failed to import TimeFormatter:", str(e))
            raise 

    def process_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> List[ProcessedChunk]:
        """Process audio data in chunks."""
        try:
            # Create chunks
            chunk_size = int(CHUNK_DURATION * sample_rate)
            chunks = []
            
            # Split audio into chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    # Pad last chunk if needed
                    padding = chunk_size - len(chunk)
                    chunk = np.pad(chunk, (0, padding))
                chunks.append(chunk)
            
            self.logger.info(f"Created {len(chunks)} chunks of {CHUNK_DURATION:.1f} seconds each")
            
            # Process chunks sequentially
            results = []
            for i, chunk in enumerate(chunks):
                self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                result = self.process_chunk(chunk, i * CHUNK_DURATION)
                if result is not None:
                    results.append(result)
                else:
                    self.logger.warning(f"Failed to process chunk {i+1}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return [] 

    def process_audio_file(self, audio_file: Path, start_time: float = 0.0, duration: Optional[float] = None) -> List[ProcessedChunk]:
        """Process an audio file."""
        try:
            self.logger.info("Processing audio...")
            
            # Load audio file
            self.logger.info(f"Loading audio file: {audio_file}")
            audio_data, sample_rate = librosa.load(audio_file, sr=None)
            self.logger.info(f"Audio loaded - Shape: {audio_data.shape}, Sample rate: {sample_rate}Hz")
            self.logger.info(f"Duration: {len(audio_data)/sample_rate:.2f} seconds")
            
            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                self.logger.info("Converting 2 channels to mono")
                audio_data = np.mean(audio_data, axis=1)
            
            # Extract segment if start_time or duration is specified
            if start_time > 0 or duration is not None:
                self.logger.info(f"Extracting segment from {start_time}s to {start_time + (duration or 0)}s")
                start_sample = int(start_time * sample_rate)
                if duration:
                    end_sample = start_sample + int(duration * sample_rate)
                    audio_data = audio_data[start_sample:end_sample]
                else:
                    audio_data = audio_data[start_sample:]
                self.logger.info(f"Extracted segment - New duration: {len(audio_data)/sample_rate:.2f} seconds")
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                self.logger.info(f"Resampling from {sample_rate}Hz to 16000Hz")
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Process audio
            return self.process_audio(audio_data, sample_rate)
            
        except Exception as e:
            self.logger.error(f"Error processing audio file: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return [] 

    def create_speaker_profile(self, name: str, url: str, start_time: float = 0.0, duration: float = 30.0) -> Optional[str]:
        """Create a speaker profile from a video segment."""
        try:
            self.logger.info(f"\nExtracting voice print for speaker: {name}")
            self.logger.info(f"Source: {url}")
            self.logger.info(f"Time range: {start_time} to {start_time + duration}")
            
            # Download video
            audio_file = self.download_video(url, start_time, duration)
            if not audio_file:
                self.logger.error("Failed to download video")
                return None
            
            # Process audio file
            results = self.process_audio_file(audio_file, start_time, duration)
            if not results or not results[0]:
                self.logger.error("Failed to process audio")
                return None
            
            # Get embedding from first chunk
            embedding = results[0].embedding
            
            # Generate speaker ID
            speaker_id = str(abs(hash(name + url + str(start_time))))
            
            # Save embedding
            embedding_file = self.embeddings_dir / f"speaker_{speaker_id}.npz"
            embedding_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(embedding_file), embedding=embedding)
            self.logger.info(f"Saved voice print to {embedding_file}")
            
            # Set as reference speaker
            self.set_reference_speaker(speaker_id)
            
            print(f"✓ Successfully added speaker: {name} (ID: {speaker_id})")
            return speaker_id
            
        except Exception as e:
            self.logger.error(f"Error creating speaker profile: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None 

    def download_video(self, url: str, start_time: float = 0.0, duration: Optional[float] = None) -> Optional[Path]:
        """Download a video segment and return the path to the audio file."""
        try:
            self.logger.info(f"Downloading video: {url}")
            
            # Generate cache key
            cache_key = f"{url}_{start_time:.3f}_{duration or 0:.3f}"
            self.logger.info(f"Checking cache with key: {cache_key}")
            
            # Create cache directory
            cache_dir = self.output_dir / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate cache file path
            import hashlib
            cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_file = cache_dir / f"audio_cache_{cache_key_hash}.wav"
            
            self.logger.info(f"Looking for cache file: {cache_file}")
            
            # Check if cached
            if cache_file.exists():
                self.logger.info(f"✓ Cache hit! Found cached audio file: {cache_file}")
                return cache_file
            
            # Download video
            import yt_dlp
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }],
                'outtmpl': str(cache_file.with_suffix('')),
                'quiet': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            self.logger.info(f"✓ Successfully downloaded video to {cache_file}")
            return cache_file
            
        except Exception as e:
            self.logger.error(f"Error downloading video: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None 

    def _load_ecapa_tdnn(self) -> Any:
        """Load the ECAPA-TDNN model."""
        try:
            from speechbrain.pretrained import EncoderClassifier
            
            # Set up model directory
            model_dir = self.models_dir / "ecapa-tdnn"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Download model files
            model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(model_dir),
                run_opts={"device": DEVICE}
            )
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading ECAPA-TDNN model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise 