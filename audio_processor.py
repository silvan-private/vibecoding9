import os
# Set environment variables before importing speechbrain
os.environ['HF_HUB_ENABLE_SYMLINKS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HOME'] = './pretrained_models/huggingface'
os.environ['SPEECHBRAIN_CACHE_DIR'] = './pretrained_models/speechbrain'

import sys
import logging

# Debug Python path and module locations
logger = logging.getLogger(__name__)
logger.info("="*80)
logger.info("MODULE IMPORT DIAGNOSTICS")
logger.info("="*80)
logger.info(f"Python Path: {sys.path}")
try:
    import whisper
    logger.info(f"Whisper Module: {whisper.__file__}")
    logger.info(f"Whisper Utils: {whisper.utils.__file__}")
except ImportError:
    logger.info("Whisper module not found")
try:
    import whisper_timestamped
    logger.info(f"Whisper Timestamped Module: {whisper_timestamped.__file__}")
except ImportError:
    logger.info("Whisper Timestamped module not found")
logger.info("="*80)

import torch
import torchaudio
import numpy as np
from pathlib import Path
import yt_dlp
from pydub import AudioSegment
from faster_whisper import WhisperModel
import time
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import requests
from yt_dlp import YoutubeDL
import json
from datetime import datetime
from speechbrain.pretrained import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import argparse
import shutil
from huggingface_hub import hf_hub_download
from speechbrain.utils.fetching import fetch
import traceback
from utils.time_helpers import TimeFormatter

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

def download_model_files():
    """Download ECAPA-TDNN model files with Windows-safe copy operations"""
    try:
        # Setup directories
        model_dir = Path("./pretrained_models/ecapa-tdnn").resolve()
        cache_dir = Path("./pretrained_models/huggingface").resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
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
                    shutil.copy2(tmp_file, target_file)
                    logger.info(f"✓ Successfully downloaded and copied {local_file}")
                    
                except Exception as e:
                    logger.error(f"Error downloading {local_file}: {str(e)}")
                    return False
        
        # Verify all files exist
        missing_files = [f for f in model_files.keys() if not (model_dir / f).exists()]
        if missing_files:
            logger.error(f"Missing required files: {missing_files}")
            return False
        
        # Verify file permissions
        for file_name in model_files.keys():
            file_path = model_dir / file_name
            if not os.access(file_path, os.R_OK):
                logger.error(f"Cannot read file: {file_name}")
                return False
        
        logger.info("✓ All model files downloaded and verified successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading model files: {str(e)}")
        return False

class AudioProcessor:
    def __init__(self, output_dir="speaker_data", embedding_dir="embeddings", mode="diarize", target_embedding=None, test_mode=False):
        logger.info("Initializing AudioProcessor...")
        # Resolve absolute paths
        self.output_dir = Path(output_dir).resolve()
        self.embedding_dir = (self.output_dir / embedding_dir).resolve()
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.embedding_dir.mkdir(exist_ok=True)
        
        # Create cache directories
        self.cache_dir = Path("./cache").resolve()
        self.cache_dir.mkdir(exist_ok=True)
        for cache_dir in ['./pretrained_models/huggingface', './pretrained_models/speechbrain', './cache']:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        
        self.mode = mode
        self.target_embedding = target_embedding
        self.progress_callback = None
        self.test_mode = test_mode  # Add test mode flag
        
        # Initialize models
        logger.info("Loading Whisper model...")
        self.whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        logger.info("Whisper model loaded successfully")
        
        # Load ECAPA-TDNN model with direct file copying
        logger.info("Loading ECAPA-TDNN model...")
        try:
            # Create directories if they don't exist
            model_dir = Path("./pretrained_models/ecapa-tdnn").resolve()
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # First, download the model files
            if not download_model_files():
                raise Exception("Failed to download model files")
            
            # Initialize the classifier with local files
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(model_dir),  # Convert to string to avoid Path issues
                run_opts={"device": "cpu"}
            )
            
            logger.info("ECAPA-TDNN model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ECAPA-TDNN model: {str(e)}")
            raise
        
        # Initialize speaker recognition with Wav2Vec2
        try:
            logger.info("Initializing audio processing model...")
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.audio_model.eval()  # Set to evaluation mode
            logger.info("Audio processing model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing audio processing model: {str(e)}")
            print("Error initializing audio processing model. Please ensure you have internet access.")
            raise e

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
            logger.info(f"✓ Cache hit! Found cached audio file: {cache_file}")
            return str(cache_file)
        
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

    def _extract_voice_embedding(self, audio_segment):
        """Extract ECAPA-TDNN embedding from audio segment"""
        try:
            # Convert tensor to numpy and ensure correct shape
            if isinstance(audio_segment, torch.Tensor):
                audio_np = audio_segment.squeeze().numpy()
            else:
                audio_np = audio_segment
            
            # Ensure audio is mono
            if len(audio_np.shape) > 1 and audio_np.shape[0] > 1:
                audio_np = audio_np[0]
            
            # Apply preprocessing
            audio_np = librosa.util.normalize(audio_np)
            audio_np = librosa.effects.preemphasis(audio_np)
            
            # Convert to torch tensor and add batch dimension
            audio_tensor = torch.FloatTensor(audio_np).unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.classifier.encode_batch(audio_tensor)
                embedding_np = embedding.squeeze().cpu().numpy()
            
            # Plot embedding distribution for debugging
            if not hasattr(self, '_embedding_count'):
                self._embedding_count = 0
            self._embedding_count += 1
            
            if self._embedding_count % 10 == 0:  # Plot every 10th embedding
                plt.figure(figsize=(6, 6))
                plt.scatter(embedding_np[:50], embedding_np[50:100])
                plt.title(f"Embedding Distribution (Sample {self._embedding_count})")
                plt.savefig(str(self.output_dir / f"debug_embedding_{self._embedding_count}.png"))
                plt.close()
            
            return embedding_np
        except Exception as e:
            logger.error(f"Error extracting voice embedding: {str(e)}")
            return None

    def _compute_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between two embeddings"""
        return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

    def _save_embedding(self, embedding, speaker_id):
        """Save speaker embedding to file"""
        try:
            # Ensure both paths are absolute and normalized
            embedding_path = (self.embedding_dir / f"{speaker_id}_ecapa.npz").resolve()
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

    def _cluster_speakers(self, embeddings, max_speakers=5):
        """Cluster speaker embeddings using extremely conservative thresholds"""
        if len(embeddings) < 2:
            return [0] * len(embeddings)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Force very conservative speaker count
        similarity_threshold = 0.3  # Extremely low threshold to merge speakers
        
        # Always start with minimum speakers
        estimated_speakers = 1  # Force single speaker
        
        # Initialize clustering with minimum speakers
        clustering = SpectralClustering(
            n_clusters=estimated_speakers,
            affinity='precomputed',
            random_state=42,
            n_init=100,
            assign_labels='discretize'
        )
        
        # Normalize similarity matrix
        similarity_matrix = np.maximum(similarity_matrix, similarity_matrix.T)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Debug logging for similarity matrix
        logger.info(f"Similarity matrix stats - Mean: {np.mean(similarity_matrix):.3f}, Min: {np.min(similarity_matrix):.3f}, Max: {np.max(similarity_matrix):.3f}")
        
        # Fit and predict
        labels = clustering.fit_predict(similarity_matrix)
        
        # Force all segments to same speaker
        return np.zeros_like(labels)  # Return all zeros to assign everything to first speaker

    def _find_text_in_timerange(self, transcript, start, end):
        """Find transcribed text within time range with improved overlap detection"""
        text_segments = []
        
        # Use TimeFormatter consistently
        start_time = TimeFormatter.format(start, ms_precision=True)
        end_time = TimeFormatter.format(end, ms_precision=True)
        
        logger.info(f"\nSearching for text between {start_time} and {end_time}")
        logger.info(f"Number of transcript segments to search: {len(transcript['segments'])}")
        
        # Debug: Print all transcript segments
        logger.info("\nAll transcript segments:")
        for i, seg in enumerate(transcript["segments"]):
            seg_start = TimeFormatter.format(seg['start'], ms_precision=True)
            seg_end = TimeFormatter.format(seg['end'], ms_precision=True)
            logger.info(f"Transcript {i}: {seg_start} -> {seg_end}: {seg['text'][:50]}...")
        
        for segment in transcript["segments"]:
            segment_start = segment["start"]
            segment_end = segment["end"]
            seg_start_fmt = TimeFormatter.format(segment_start, ms_precision=True)
            seg_end_fmt = TimeFormatter.format(segment_end, ms_precision=True)
            
            logger.info(f"\nChecking transcript segment: {seg_start_fmt} -> {seg_end_fmt}")
            logger.info(f"Segment text: {segment['text']}")
            logger.info(f"Checking overlap conditions:")
            logger.info(f"1. segment_start ({seg_start_fmt}) <= end ({end_time}): {segment_start <= end}")
            logger.info(f"2. segment_end ({seg_end_fmt}) >= start ({start_time}): {segment_end >= start}")
            
            # If there's any overlap between the segments
            if (segment_start <= end and segment_end >= start):
                logger.info(f"✓ Found overlapping segment!")
                text_segments.append(segment["text"])
            else:
                logger.info("✗ No overlap found")
        
        result = " ".join(text_segments)
        logger.info(f"\nFinal text found: {result[:100]}..." if result else "No text found in this range")
        return result

    def process_video(self, url, start_time=None, end_time=None):
        """Process video URL and extract speaker information with improved diarization"""
        start_time = start_time or 0
        end_time = end_time or float('inf')
        
        start_processing = time.time()
        logger.info("="*80)
        logger.info("STARTING VIDEO PROCESSING")
        logger.info(f"URL: {url}")
        logger.info(f"Time range: {start_time} to {end_time}")
        logger.info(f"Test mode: {'ENABLED' if self.test_mode else 'DISABLED'}")
        logger.info("="*80)

        # Initialize format_timestamp at the start
        try:
            from whisper.utils import format_timestamp
            logger.info("[SUCCESS] OpenAI whisper format_timestamp found")
        except ImportError:
            try:
                from whisper_timestamped import format_timestamp
                logger.info("[SUCCESS] whisper-timestamped format_timestamp found")
            except ImportError:
                logger.info("[FAIL] No format_timestamp in standard locations")
                def format_timestamp(seconds: float):
                    return f"{seconds:.2f}s"
                logger.info("[INFO] Using basic format_timestamp implementation")
        
        # Make format_timestamp available globally
        globals()['format_timestamp'] = format_timestamp

        # Test the format_timestamp function
        test_time = 123.456
        try:
            formatted = format_timestamp(test_time)
            logger.info(f"[TEST] format_timestamp({test_time}) = {formatted}")
        except Exception as e:
            logger.error(f"[TEST] format_timestamp test failed: {str(e)}")

        try:
            # Download and prepare audio
            logger.info("\n[1] Downloading audio...")
            audio_path = self.download_audio(url, start_time, end_time)
            
            # Generate transcript using Whisper
            logger.info("\n[1.5] Generating transcript...")
            logger.info("Audio file being transcribed: %s", audio_path)
            
            # Verify audio file before transcription
            try:
                audio_info = sf.info(audio_path)
                logger.info("Audio file verification:")
                logger.info(f"- Duration: {audio_info.duration:.2f} seconds")
                logger.info(f"- Sample rate: {audio_info.samplerate} Hz")
                logger.info(f"- Channels: {audio_info.channels}")
            except Exception as e:
                logger.error(f"Error verifying audio file: {str(e)}")
            
            # Set Whisper parameters for better results on short segments
            whisper_params = {
                "beam_size": 5,
                "best_of": 5,
                "temperature": 0.0,  # Use single temperature for faster-whisper
                "condition_on_previous_text": True,
                "vad_filter": True,
                "vad_parameters": {
                    "threshold": 0.1,        # More lenient
                    "min_speech_duration_ms": 100,
                    "max_speech_duration_s": 10,
                    "min_silence_duration_ms": 50
                }
            }
            
            logger.info("\nStarting Whisper transcription with parameters:")
            for k, v in whisper_params.items():
                logger.info(f"- {k}: {v}")
            
            try:
                segments, info = self.whisper_model.transcribe(
                    audio_path,
                    beam_size=whisper_params["beam_size"],
                    best_of=whisper_params["best_of"],
                    temperature=whisper_params["temperature"],
                    condition_on_previous_text=whisper_params["condition_on_previous_text"],
                    vad_filter=whisper_params["vad_filter"],
                    vad_parameters=whisper_params["vad_parameters"]
                )
                # Convert generator to list immediately
                segments = list(segments)
                logger.info("\nWhisper transcription completed")
                logger.info(f"Info returned: {info}")
                logger.info(f"Number of segments: {len(segments)}")
                
                # Debug: Print raw segment attributes
                if segments:
                    logger.info("\nFirst segment attributes:")
                    first_seg = segments[0]
                    for attr in dir(first_seg):
                        if not attr.startswith('_'):
                            try:
                                value = getattr(first_seg, attr)
                                logger.info(f"- {attr}: {value}")
                            except Exception as e:
                                logger.info(f"- {attr}: <error getting value: {e}>")
                
                for i, seg in enumerate(segments):
                    logger.info(f"Segment {i}:")
                    logger.info(f"- Time: {seg.start:.2f}s -> {seg.end:.2f}s")
                    logger.info(f"- Text: '{seg.text}'")
                    logger.info(f"- Word count: {len(seg.text.split())}")
                    logger.info(f"- Avg logprob: {seg.avg_logprob}")
                    logger.info(f"- No speech prob: {seg.no_speech_prob}")
            except Exception as e:
                logger.error(f"Error during transcription: {str(e)}")
                logger.error(traceback.format_exc())
                raise
            
            # Convert faster_whisper segments to our format and add debug logging
            logger.info("\nTranscript Generation Details:")
            logger.info("-" * 50)
            logger.info("Raw segments from Whisper:")
            
            # Debug: Print raw segment attributes
            if segments:  # Changed from len(segments) > 0
                logger.info("\nFirst segment attributes:")
                first_seg = segments[0]
                for attr in dir(first_seg):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(first_seg, attr)
                            logger.info(f"- {attr}: {value}")
                        except Exception as e:
                            logger.info(f"- {attr}: <error getting value: {e}>")
            
            for i, seg in enumerate(segments):
                logger.info(f"Segment {i}:")
                logger.info(f"- Time: {seg.start:.2f}s -> {seg.end:.2f}s")
                logger.info(f"- Text: '{seg.text}'")
                logger.info(f"- Word count: {len(seg.text.split())}")
                logger.info(f"- Avg logprob: {seg.avg_logprob}")
                logger.info(f"- No speech prob: {seg.no_speech_prob}")
                
            transcript = {
                'segments': []
            }
            
            # Process each segment and ensure valid timestamps
            for seg in segments:
                if seg.text and seg.text.strip():  # Only include non-empty segments
                    transcript['segments'].append({
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": seg.text.strip(),
                        "avg_logprob": seg.avg_logprob,
                        "no_speech_prob": seg.no_speech_prob
                    })
            
            logger.info(f"\nProcessed {len(transcript['segments'])} valid text segments")
            if transcript['segments']:
                logger.info("First few processed segments:")
                for i, seg in enumerate(transcript['segments'][:3]):
                    logger.info(f"Segment {i}:")
                    logger.info(f"- Time: {seg['start']:.2f}s -> {seg['end']:.2f}s")
                    logger.info(f"- Text: '{seg['text']}'")
                    logger.info(f"- Avg logprob: {seg['avg_logprob']:.2f}")
                    logger.info(f"- No speech prob: {seg['no_speech_prob']:.2f}")
            else:
                logger.warning("No valid text segments found in transcription!")
            logger.info("-" * 50)
            
            # Segment the audio
            logger.info("\n[2] Segmenting audio...")
            segments, timestamps = self._segment_audio(audio_path)
            
            if len(segments) == 0:
                raise Exception("No segments were created from the audio")
            
            logger.info(f"Created {len(segments)} segments" + (" (limited by test mode)" if self.test_mode else ""))
            
            # Initialize result structure
            result = {
                'speakers': [],
                'segments': []
            }
            
            # Process segments with detailed status tracking
            speakers = {}
            processed_segments = []
            all_embeddings = []
            embedding_map = []
            
            # Pre-processing diagnostics
            logger.info("\nPRE-PROCESSING DIAGNOSTICS")
            logger.info("-" * 50)
            logger.info(f"Total segments to process: {len(segments)}")
            logger.info(f"Test mode enabled: {self.test_mode}")
            logger.info(f"Transcript segments available: {len(transcript['segments'])}")
            logger.info("-" * 50 + "\n")
            
            # Process each segment
            for i, (segment, time_info) in enumerate(zip(segments, timestamps)):
                try:
                    logger.debug(f"\nProcessing segment {i+1}...")
                    logger.debug(f"Time info: start={time_info['start']:.3f}s, end={time_info['end']:.3f}s")
                    
                    # Format timestamps using the determined format_timestamp function
                    start_formatted = format_timestamp(time_info['start'])
                    end_formatted = format_timestamp(time_info['end'])
                    
                    segment_status = {
                        "Time Range": {
                            "status": "success",
                            "details": f"{start_formatted} -> {end_formatted}"
                        },
                        "Audio Properties": {
                            "status": "success",
                            "details": f"Length: {len(segment)}, Shape: {segment.shape}"
                        }
                    }
                    
                    # Extract voice embedding
                    voice_embedding = self._extract_voice_embedding(segment)
                    if voice_embedding is not None:
                        segment_status["Voice Embedding"] = {
                            "status": "success",
                            "details": f"Shape: {voice_embedding.shape}, Mean: {np.mean(voice_embedding):.3f}"
                        }
                        all_embeddings.append(voice_embedding)
                        embedding_map.append((voice_embedding, time_info))
                    else:
                        segment_status["Voice Embedding"] = {
                            "status": "failure",
                            "details": "Failed to extract embedding"
                        }
                        continue
                    
                    # Find text in time range using the transcript
                    text = self._find_text_in_timerange(transcript, time_info["start"], time_info["end"])
                    if text.strip():
                        segment_status["Text Found"] = {
                            "status": "success",
                            "details": f"Length: {len(text)} chars | Preview: {text[:50]}..."
                        }
                    else:
                        segment_status["Text Found"] = {
                            "status": "failure",
                            "details": "No text found in this time range"
                        }
                    
                    # Print status visualization
                    logger.info("\n" + "="*80)
                    logger.info(f"SEGMENT {i+1} STATUS:")
                    logger.info("="*80)
                    for step, step_status in segment_status.items():
                        symbol = "✓" if step_status["status"] == "success" else "✗"
                        logger.info(f"{symbol} {step:<30} | {step_status['details']}")
                    logger.info("-"*80)
                    
                    # Only continue if we have both embedding and text
                    if voice_embedding is not None and text.strip():
                        processed_segments.append({
                            "start": time_info["start"],
                            "end": time_info["end"],
                            "text": text,
                            "embedding": voice_embedding
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing segment {i+1}: {str(e)}\n{traceback.format_exc()}")
                    continue

            # Update result with speaker information
            if len(all_embeddings) > 0:
                # Save the first speaker's embedding
                speaker_id = f"speaker_{int(time.time())}"
                embedding_file = self._save_embedding(all_embeddings[0], speaker_id)
                result['speakers'].append({
                    "id": speaker_id,
                    "embedding_file": embedding_file,
                    "segments": processed_segments
                })
            
            # Final summary
            logger.info("\n" + "="*80)
            logger.info("PROCESSING SUMMARY")
            logger.info("="*80)
            logger.info(f"Total segments analyzed: {len(segments)}")
            logger.info(f"Successful embeddings: {len(all_embeddings)}")
            logger.info(f"Segments with text: {len(processed_segments)}")
            logger.info(f"Processing time: {time.time() - start_processing:.2f} seconds")
            logger.info("="*80)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in process_video: {str(e)}\n{traceback.format_exc()}")
            raise 