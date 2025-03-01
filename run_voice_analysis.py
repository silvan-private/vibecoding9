import os
from pathlib import Path
import json
from audio_processor import AudioProcessor
from voice_analyzer import extract_speaker, analyze_target
from utils.time_helpers import TimeFormatter
import numpy as np
from datetime import datetime
import sys
import time
import threading
from itertools import cycle
import logging
import traceback

class Spinner:
    """A simple spinner class for showing progress"""
    def __init__(self, message="Processing", delay=0.1):
        self.spinner = cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.delay = delay
        self.busy = False
        self.spinner_visible = False
        self.message = message
        sys.stdout.write('\n')  # Print a newline first
        
    def write_next(self):
        with self._screen_lock:
            if not self.spinner_visible:
                sys.stdout.write(f"\r{next(self.spinner)} {self.message}")
                self.spinner_visible = True
                sys.stdout.flush()
    
    def remove_spinner(self, cleanup=False):
        with self._screen_lock:
            if self.spinner_visible:
                sys.stdout.write('\r')
                if cleanup:
                    sys.stdout.write(' ' * (len(self.message) + 2))
                    sys.stdout.write('\r')
                sys.stdout.flush()
                self.spinner_visible = False
    
    def update_message(self, message):
        """Update the message displayed next to the spinner"""
        self.message = message
        self.remove_spinner(cleanup=True)
        
    def spinner_task(self):
        while self.busy:
            self.write_next()
            time.sleep(self.delay)
            self.remove_spinner()
    
    def __enter__(self):
        if sys.stdout.isatty():
            self._screen_lock = threading.Lock()
            self.busy = True
            self.thread = threading.Thread(target=self.spinner_task)
            self.thread.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if sys.stdout.isatty():
            self.busy = False
            time.sleep(self.delay)
            self.remove_spinner(cleanup=True)
            if exc_type is not None:
                return False
            
def show_stage_progress(message):
    """Show a progress message with timestamp"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"\r[{timestamp}] {message}")

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the application header"""
    print("\n" + "="*50)
    print("Voice Analysis Tool - Interactive Menu")
    print("="*50 + "\n")

def get_time_input(prompt):
    """Get and validate time input in HH:MM:SS format"""
    while True:
        time_str = input(prompt + " (HH:MM:SS format): ")
        try:
            time_obj = datetime.strptime(time_str, '%H:%M:%S')
            seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
            return time_str, seconds
        except ValueError:
            print("Invalid time format. Please use HH:MM:SS (e.g., 00:01:30)")

def get_url_input(prompt):
    """Get and validate URL input"""
    while True:
        url = input(prompt + ": ").strip()
        if url.startswith(("http://", "https://")) and ("youtube.com" in url or "youtu.be" in url):
            return url
        print("Invalid URL. Please enter a valid YouTube URL")

def show_main_menu():
    """Display the main menu and get user choice"""
    print("\nAvailable Operations:")
    print("1. Extract Speaker Voice Print")
    print("2. Analyze Video with Existing Voice Print")
    print("3. Full Process (Extract and Analyze)")
    print("4. View Saved Voice Prints")
    print("5. Exit")
    print("6. Test Extract (Pre-filled values)")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-6): "))
            if 1 <= choice <= 6:
                return choice
            print("Please enter a number between 1 and 6")
        except ValueError:
            print("Please enter a valid number")

def list_voice_prints(output_dir="speaker_data"):
    """List all saved voice prints"""
    print("\nSaved Voice Prints:")
    print("-" * 50)
    
    voice_prints_dir = Path(output_dir)
    if not voice_prints_dir.exists():
        print("No voice prints found")
        return []
    
    voice_prints = list(voice_prints_dir.glob("*_ecapa.npz"))
    if not voice_prints:
        print("No voice prints found")
        return []
    
    for i, vp in enumerate(voice_prints, 1):
        print(f"{i}. {vp.name}")
    return voice_prints

def extract_speaker_workflow():
    """Handle the speaker extraction workflow"""
    print("\nSpeaker Extraction")
    print("-" * 50)
    
    url = get_url_input("Enter YouTube URL")
    start_time, start_seconds = get_time_input("Enter start time")
    end_time, end_seconds = get_time_input("Enter end time")
    
    with Spinner("Initializing models...") as spinner:
        embedding_file = None
        try:
            # Start the extraction process
            spinner.update_message("Downloading audio from YouTube...")
            processor = AudioProcessor(output_dir="speaker_data", mode="diarize")
            results = processor.process_video(url, start_time=start_seconds, end_time=end_seconds)
            
            if results and results.get("speakers") and len(results["speakers"]) > 0:
                speaker_data = results["speakers"][0]
                embedding_file = Path("speaker_data") / speaker_data["embedding_file"]
                spinner.update_message("Speaker voiceprint extracted successfully!")
                time.sleep(1)  # Show success message briefly
            else:
                spinner.update_message("No speaker detected in the specified time range")
                time.sleep(2)
                
        except Exception as e:
            spinner.update_message(f"Error: {str(e)}")
            time.sleep(2)
    
    input("\nPress Enter to continue...")
    return embedding_file

def analyze_video_workflow(embedding_file=None):
    """Handle the video analysis workflow"""
    print("\nVideo Analysis")
    print("-" * 50)
    
    if not embedding_file:
        voice_prints = list_voice_prints()
        if not voice_prints:
            input("\nPress Enter to continue...")
            return
        
        while True:
            try:
                choice = int(input("\nSelect a voice print number (or 0 to cancel): "))
                if choice == 0:
                    return
                if 1 <= choice <= len(voice_prints):
                    embedding_file = voice_prints[choice - 1]
                    break
                print("Invalid choice")
            except ValueError:
                print("Please enter a valid number")
    
    url = get_url_input("Enter YouTube URL to analyze")
    
    with Spinner("Analyzing video...") as spinner:
        try:
            # Load the embedding
            embedding_data = np.load(embedding_file)
            embedding = embedding_data["embedding"]
            
            # Analyze target video
            processor = AudioProcessor(
                output_dir="speaker_data",
                mode="match",
                target_embedding=embedding
            )
            results = processor.process_video(url)
            
            if results and results.get("segments"):
                spinner.update_message("Analysis complete! Processing results...")
                time.sleep(1)
                
                print("\nMatching segments found:")
                print("-" * 50)
                matches_found = False
                
                for segment in results["segments"]:
                    score = segment.get("voiceprint", {}).get("match_score", 0)
                    if score > 0.75:
                        matches_found = True
                        start = TimeFormatter.format(segment["start"], ms_precision=True)
                        end = TimeFormatter.format(segment["end"], ms_precision=True)
                        print(f"[{start} -> {end}] (match: {score:.2f})")
                        print(f"Text: {segment['text']}\n")
                
                if not matches_found:
                    print("No high-confidence matches found")
            else:
                spinner.update_message("No results found")
                time.sleep(1)
                
        except Exception as e:
            spinner.update_message(f"Error: {str(e)}")
            time.sleep(2)
    
    input("\nPress Enter to continue...")

def test_extract_workflow():
    """Handle the test speaker extraction workflow with pre-filled values"""
    print("\nTest Speaker Extraction")
    print("-" * 50)
    
    # Diagnostic code to check format_timestamp availability
    print("\nChecking format_timestamp availability:")
    print("-" * 50)
    try:
        from whisper.utils import format_timestamp
        print("[SUCCESS] OpenAI whisper format_timestamp found")
    except ImportError:
        try:
            from whisper_timestamped import format_timestamp
            print("[SUCCESS] whisper-timestamped format_timestamp found")
        except ImportError:
            print("[FAIL] No format_timestamp in standard locations")
            # Implement basic version
            def format_timestamp(seconds: float):
                return f"{seconds:.2f}s"
            print("[INFO] Using basic format_timestamp implementation")
            # Make format_timestamp available globally
            globals()['format_timestamp'] = format_timestamp
    
    # Pre-filled test values
    url = "https://www.youtube.com/watch?v=gPKkIkEnZw8"
    start_time = "00:04:00"  # 4 minute mark
    end_time = "00:04:03"    # 4 minute 3 second mark (just 3 seconds)
    
    # Configure debug logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.debug("Starting test extraction with debug logging enabled")
    
    # Convert times to seconds
    try:
        start_seconds = TimeFormatter.parse(start_time)
        end_seconds = TimeFormatter.parse(end_time)
        logger.debug(f"Converted time range: {start_seconds}s to {end_seconds}s")
    except Exception as e:
        logger.error(f"Time conversion error: {str(e)}")
        return None
    
    print(f"Using test values:")
    print(f"URL: {url}")
    print(f"Time range: {start_time} to {end_time}")
    print("Test mode: ENABLED (max 5 segments)")
    
    with Spinner("Initializing models...") as spinner:
        embedding_file = None
        try:
            # Start the extraction process
            spinner.update_message("Downloading audio from YouTube...")
            processor = AudioProcessor(output_dir="speaker_data", mode="diarize", test_mode=True)  # Enable test mode
            
            # Add debug callback
            def progress_callback(stage, progress=None):
                try:
                    if progress is not None:
                        spinner.update_message(f"{stage}: {progress:.1f}%")
                    else:
                        spinner.update_message(stage)
                    logger.debug(f"Progress update - Stage: {stage}, Progress: {progress}")
                except Exception as e:
                    logger.error(f"Progress callback error: {str(e)}")
            processor.set_progress_callback(progress_callback)
            
            logger.debug("Starting video processing...")
            results = processor.process_video(url, start_time=start_seconds, end_time=end_seconds)
            
            if results and results.get("speakers") and len(results["speakers"]) > 0:
                speaker_data = results["speakers"][0]
                embedding_file = Path("speaker_data") / speaker_data["embedding_file"]
                spinner.update_message("Speaker voiceprint extracted successfully!")
                time.sleep(1)  # Show success message briefly
                
                # Print debug information about segments
                if results.get("segments"):
                    print("\nSegment Information:")
                    print("-" * 50)
                    for i, seg in enumerate(results["segments"][:5], 1):
                        try:
                            print(f"Segment {i}:")
                            logger.debug(f"Processing segment {i} timestamps...")
                            start_time = TimeFormatter.format(seg['start'], ms_precision=True)
                            end_time = TimeFormatter.format(seg['end'], ms_precision=True)
                            print(f"Time: {start_time} -> {end_time}")
                            print(f"Speaker: {seg.get('speaker', 'Unknown')}")
                            if 'text' in seg:
                                print(f"Text: {seg['text'][:100]}...")
                            print()
                        except Exception as e:
                            logger.error(f"Error processing segment {i}: {str(e)}\nSegment data: {seg}")
                            continue
                    
                    if len(results["segments"]) > 5:
                        print(f"... and {len(results['segments']) - 5} more segments")
            else:
                spinner.update_message("No speaker detected in the specified time range")
                time.sleep(2)
                
        except Exception as e:
            logger.error(f"Processing error: {str(e)}\n{traceback.format_exc()}")
            spinner.update_message(f"Error: {str(e)}")
            time.sleep(2)
    
    input("\nPress Enter to continue...")
    return embedding_file

def main():
    # Create output directory
    Path("speaker_data").mkdir(exist_ok=True)
    
    while True:
        clear_screen()
        print_header()
        
        choice = show_main_menu()
        
        if choice == 1:  # Extract
            embedding_file = extract_speaker_workflow()
        
        elif choice == 2:  # Analyze
            analyze_video_workflow()
        
        elif choice == 3:  # Full Process
            embedding_file = extract_speaker_workflow()
            if embedding_file:
                analyze_video_workflow(embedding_file)
        
        elif choice == 4:  # View Saved
            list_voice_prints()
            input("\nPress Enter to continue...")
        
        elif choice == 6:  # Test Extract
            embedding_file = test_extract_workflow()
            
        else:  # Exit
            print("\nThank you for using Voice Analysis Tool!")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1) 