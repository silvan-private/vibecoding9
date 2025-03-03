from audio_processor import AudioProcessor
import json
from pathlib import Path
import logging
import traceback
import argparse
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_target_embedding(embedding_path):
    """Load target speaker embedding from file"""
    try:
        data = np.load(embedding_path)
        return data['embedding']
    except Exception as e:
        logger.error(f"Error loading target embedding: {str(e)}")
        return None

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process video audio for speaker diarization or matching')
    parser.add_argument('--url', type=str, default="https://www.youtube.com/watch?v=cfAUbJgR0pE",
                      help='YouTube video URL to process')
    parser.add_argument('--mode', type=str, choices=['diarize', 'match'], default='diarize',
                      help='Processing mode: diarization or speaker matching')
    parser.add_argument('--target', type=str, help='Path to target speaker embedding file for matching mode')
    parser.add_argument('--output-dir', type=str, default='speaker_data',
                      help='Directory for output files')
    parser.add_argument('--embedding-dir', type=str, default='embeddings',
                      help='Directory for speaker embeddings')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'match' and not args.target:
        logger.error("Target embedding file must be specified in matching mode")
        return
    
    # Load target embedding if in matching mode
    target_embedding = None
    if args.mode == 'match':
        target_embedding = load_target_embedding(args.target)
        if target_embedding is None:
            return
    
    logger.info("Starting video analysis...")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Embedding directory: {args.embedding_dir}")
    
    # Initialize the AudioProcessor
    try:
        processor = AudioProcessor(
            output_dir=args.output_dir,
            embedding_dir=args.embedding_dir,
            mode=args.mode,
            target_embedding=target_embedding
        )
    except Exception as e:
        logger.error(f"Failed to initialize AudioProcessor: {str(e)}")
        logger.error(traceback.format_exc())
        return
    
    try:
        logger.info(f"Processing video: {args.url}")
        # Process the video
        results = processor.process_video(args.url)
        
        # Save results to a JSON file for reference
        output_path = Path(f"{args.output_dir}/analysis_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
            
        print("\nAnalysis Results:")
        print("-----------------")
        print(f"Number of speakers detected: {len(results['speakers'])}")
        print(f"Total transcript length: {len(results['transcript'])} characters")
        print(f"Number of timestamped segments: {len(results['timestamps'])}")
        print("\nResults have been saved to:", output_path)
        print("Speaker embeddings have been saved as .npy files in the speaker_data directory")
        logger.info("Analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 