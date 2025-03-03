"""Script to test speaker detection around a specific timestamp."""
import logging
from pathlib import Path
from analyze_speaker import AudioProcessor, TranscriptManager, analyze_video
import datetime
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('speaker_data/test_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

def test_speaker_detection():
    """Test speaker detection around the 4:23 mark."""
    try:
        # Initialize transcript manager
        transcript_manager = TranscriptManager()
        
        # Initialize SQLite database for test results
        db_path = Path("speaker_data/test_results.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create transcripts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_transcripts (
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
        
        # Create speaker profile using the first minute of the reference video
        logger.info("\nCreating speaker profile for SAM...")
        speaker_id = processor.create_speaker_profile(
            "SAM",
            "https://www.youtube.com/watch?v=cfAUbJgR0pE",
            start_time=0,
            duration=60
        )
        if not speaker_id:
            logger.error("Failed to create speaker profile")
            return
            
        # Analyze target video around 4:23 (263 seconds)
        # We'll take a 20-second window (253s to 273s)
        logger.info("\nAnalyzing target video segment around speaker transition...")
        
        target_url = "https://www.youtube.com/watch?v=gPKkIkEnZw8"
        segments = analyze_video(
            processor,
            target_url,
            start_time=253,    # 10 seconds before transition
            duration=20,       # 20 second window
            speakers=[speaker_id]
        )
        
        logger.info("Video segment analysis completed")
        
        # Save results
        if segments:
            # Save metadata for test
            metadata = {
                'url': target_url,
                'title': 'Speaker Transition Test',
                'analysis_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'test_window': '4:13 - 4:33',
                'total_segments': len(segments)
            }
            transcript_manager.save_metadata(target_url + "_test", metadata)
            
            # Save raw transcript
            transcript_manager.save_raw_transcript(target_url + "_test", segments)
            
            logger.info("\nSaving test segments to database...")
            for segment in segments:
                # Save to database with clear speaker labels
                cursor.execute('''
                    INSERT INTO test_transcripts (video_url, timestamp, end_timestamp, speaker, text, confidence)
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
            
            # Generate HTML viewer for test results
            transcript_manager.save_html_viewer(target_url + "_test")
            
            logger.info("\nAnalysis Results:")
            logger.info("-" * 40)
            for segment in segments:
                start_time = int(segment['start'])
                minutes = start_time // 60
                seconds = start_time % 60
                timestamp = f"{minutes:02d}:{seconds:02d}"
                
                speaker = 'SAM' if segment.get('similarity', 0) >= processor.similarity_threshold else 'Other Speaker'
                confidence = segment.get('similarity', 0) * 100
                
                logger.info(f"Time: {timestamp}")
                logger.info(f"Speaker: {speaker} (confidence: {confidence:.1f}%)")
                logger.info(f"Text: {segment['text']}")
                logger.info("-" * 40)
            
            logger.info(f"\nTest results saved in: {transcript_manager.get_video_dir(target_url + '_test')}")
            
        else:
            logger.warning("No segments found in test window")
            
        conn.close()
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    test_speaker_detection() 