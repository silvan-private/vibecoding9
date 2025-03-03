import argparse
from pathlib import Path
import json
from audio_processor import AudioProcessor
import sys
from datetime import datetime
import numpy as np
from utils.time_helpers import TimeFormatter

def parse_time(time_str):
    """Parse time string to seconds"""
    return TimeFormatter.parse(time_str)

def extract_speaker(url, start_time, end_time, output_dir="speaker_data"):
    """Extract speaker voice print from a specific time range"""
    print(f"\nExtracting speaker voiceprint from {url}")
    print(f"Time range: {start_time} to {end_time}")
    
    # Convert times to seconds
    start_seconds = parse_time(start_time)
    end_seconds = parse_time(end_time)
    
    # Initialize processor in diarization mode
    processor = AudioProcessor(output_dir=output_dir, mode="diarize")
    
    # Process video with time range
    results = processor.process_video(url, start_time=start_seconds, end_time=end_seconds)
    
    # Get the first detected speaker's embedding file
    if results and results.get("speakers") and len(results["speakers"]) > 0:
        speaker_data = results["speakers"][0]
        embedding_file = Path(output_dir) / speaker_data["embedding_file"]
        print(f"\nSpeaker voiceprint extracted and saved to: {embedding_file}")
        return embedding_file
    else:
        print("\nError: No speaker detected in the specified time range")
        return None

def analyze_target(url, speaker_embedding, output_dir="speaker_data"):
    """Analyze target video for the extracted speaker"""
    print(f"\nAnalyzing target video: {url}")
    
    # Initialize processor in matching mode with the speaker embedding
    processor = AudioProcessor(
        output_dir=output_dir,
        mode="match",
        target_embedding=speaker_embedding
    )
    
    # Process video
    results = processor.process_video(url)
    
    # Print matching segments
    if results and results.get("segments"):
        print("\nMatching segments found:")
        print("-" * 50)
        for segment in results["segments"]:
            score = segment.get("voiceprint", {}).get("match_score", 0)
            if score > 0.75:  # Only show high confidence matches
                start = TimeFormatter.format(segment["start"], ms_precision=True)
                end = TimeFormatter.format(segment["end"], ms_precision=True)
                print(f"[{start} -> {end}] (match: {score:.2f})")
                print(f"Text: {segment['text']}\n")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Voice Analysis Tool")
    parser.add_argument("--mode", choices=["extract", "analyze", "full"], 
                       default="full", help="Operation mode")
    parser.add_argument("--url1", help="URL for speaker extraction")
    parser.add_argument("--start-time", help="Start time for extraction (HH:MM:SS)")
    parser.add_argument("--end-time", help="End time for extraction (HH:MM:SS)")
    parser.add_argument("--url2", help="URL for analysis")
    parser.add_argument("--embedding", help="Path to existing speaker embedding file")
    parser.add_argument("--output-dir", default="speaker_data", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    if args.mode in ["extract", "full"]:
        if not all([args.url1, args.start_time, args.end_time]):
            print("Error: For extraction mode, --url1, --start-time, and --end-time are required")
            parser.print_help()
            sys.exit(1)
        
        embedding_file = extract_speaker(
            args.url1, 
            args.start_time, 
            args.end_time, 
            args.output_dir
        )
        
        if args.mode == "extract":
            return
    
    if args.mode in ["analyze", "full"]:
        if args.mode == "analyze" and not args.embedding:
            print("Error: For analyze mode, --embedding is required")
            parser.print_help()
            sys.exit(1)
        
        if not args.url2:
            print("Error: --url2 is required for analysis")
            parser.print_help()
            sys.exit(1)
        
        # Load embedding from file
        embedding_path = args.embedding if args.mode == "analyze" else embedding_file
        if not embedding_path or not embedding_path.exists():
            print("Error: No valid speaker embedding available")
            sys.exit(1)
            
        # Load the embedding
        embedding_data = np.load(embedding_path)
        embedding = embedding_data["embedding"]
        
        # Analyze target video
        results = analyze_target(args.url2, embedding, args.output_dir)
        
        # Print matching segments
        if results and results.get("segments"):
            print("\nMatching segments found:")
            print("-" * 50)
            for segment in results["segments"]:
                score = segment.get("voiceprint", {}).get("match_score", 0)
                if score > 0.75:  # Only show high confidence matches
                    start = TimeFormatter.format(segment["start"], ms_precision=True)
                    end = TimeFormatter.format(segment["end"], ms_precision=True)
                    print(f"[{start} -> {end}] (match: {score:.2f})")
                    print(f"Text: {segment['text']}\n")

if __name__ == "__main__":
    main() 