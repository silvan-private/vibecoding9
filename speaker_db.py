import os
import sqlite3
import logging
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from audio_processor import AudioProcessor, ProcessedChunk
import time
from typing import Optional, List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeakerKnowledgeDB:
    def __init__(self, db_path="speaker_knowledge.db"):
        """Initialize the speaker knowledge database"""
        self.db_path = db_path
        self.setup_database()
        self.audio_processor = AudioProcessor(output_dir="speaker_data")

    def setup_database(self):
        """Create the database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            
            # Create speakers table
            c.execute('''
                CREATE TABLE IF NOT EXISTS speakers (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    reference_url TEXT,
                    reference_start_time REAL,
                    reference_end_time REAL,
                    voice_print_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create segments table for storing transcribed segments
            c.execute('''
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY,
                    speaker_id INTEGER,
                    source_url TEXT,
                    start_time REAL,
                    end_time REAL,
                    text TEXT,
                    match_confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (speaker_id) REFERENCES speakers(id)
                )
            ''')
            
            conn.commit()

    def add_speaker(self, name, url, start_time, end_time):
        """Add a new speaker from a reference video segment"""
        print(f"\nExtracting voice print for speaker: {name}")
        print(f"Source: {url}")
        print(f"Time range: {start_time} to {end_time}")
        
        try:
            # Generate a unique speaker ID
            speaker_id = int(time.time())
            
            # Process the video segment to get speaker voice print
            results = self.audio_processor.process_video(url, start_time, end_time, speaker_id=speaker_id)
            
            if not results or not results.get('embedding_file'):
                raise Exception("No speaker voice print extracted")
            
            # Get the embedding file path
            voice_print_path = results['embedding_file']
            
            # Verify the embedding file exists
            if not Path(voice_print_path).exists():
                raise Exception(f"Voice print file not found at {voice_print_path}")
            
            # Load the embedding to set as reference
            embedding_data = np.load(voice_print_path)
            self.audio_processor.set_reference_speaker(speaker_id, embedding_data['embedding'])
            
            # Store speaker in database
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    INSERT INTO speakers (id, name, reference_url, reference_start_time, 
                                       reference_end_time, voice_print_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (speaker_id, name, url, start_time, end_time, voice_print_path))
                conn.commit()
            
            print(f"✓ Successfully added speaker: {name} (ID: {speaker_id})")
            return speaker_id
            
        except Exception as e:
            logger.error(f"Error adding speaker: {str(e)}")
            return None

    def analyze_video(self, url, speaker_ids=None, start_time=None, end_time=None, progress_callback=None):
        """Analyze a video for known speakers and store their segments"""
        print(f"\nAnalyzing video: {url}")
        if start_time is not None and end_time is not None:
            print(f"Time range: {start_time}s to {end_time}s")
        
        try:
            # Get speaker voice prints
            speakers = []
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                query = '''
                    SELECT id, name, voice_print_path 
                    FROM speakers
                '''
                if speaker_ids:
                    query += ' WHERE id IN (' + ','.join('?' * len(speaker_ids)) + ')'
                    c.execute(query, speaker_ids)
                else:
                    c.execute(query)
                    
                speakers = c.fetchall()
            
            if not speakers:
                print("No speakers found in database")
                return []
            
            print(f"Looking for {len(speakers)} speakers...")
            
            # Load voice prints and set reference speaker
            for speaker_id, name, vp_path in speakers:
                try:
                    if not vp_path:
                        logger.error(f"No voice print path found for speaker {name}")
                        continue
                        
                    if not Path(vp_path).exists():
                        logger.error(f"Voice print file not found at {vp_path}")
                        continue
                        
                    # Load and set reference speaker
                    vp_data = np.load(vp_path)
                    self.audio_processor.set_reference_speaker(speaker_id, vp_data['embedding'])
                    break  # Only use the first speaker for now
                    
                except Exception as e:
                    logger.error(f"Error loading voice print for {name}: {str(e)}")
                    continue
            
            # Process video with reference speaker
            results = self.audio_processor.process_video(
                url=url,
                start_time=start_time,
                end_time=end_time,
                speaker_id=speaker_id,
                progress_callback=progress_callback
            )
            
            if not results or not results.get('segments'):
                print(f"No segments found")
                return []
            
            # Store matching segments
            all_knowledge = []
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                for segment in results['segments']:
                    if segment.get('speaker_id'):
                        c.execute('''
                            INSERT INTO segments (
                                speaker_id, source_url, start_time, end_time, 
                                text, match_confidence
                            )
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            segment['speaker_id'], url, segment['start'], segment['end'],
                            segment['text'], segment.get('confidence', 0.75)
                        ))
                        
                        # Add to knowledge base
                        all_knowledge.append({
                            'speaker': name,
                            'source': url,
                            'timestamp': segment['start'],
                            'text': segment['text'],
                            'confidence': segment.get('confidence', 0.75)
                        })
                    
                conn.commit()
            
            print(f"✓ Saved matching segments for {name}")
            return all_knowledge
            
        except Exception as e:
            logger.error(f"Error analyzing video: {str(e)}")
            return []

    def get_speaker_knowledge(self, speaker_id=None, min_confidence=0.75):
        """Retrieve all transcribed segments for a speaker"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                
                query = '''
                    SELECT 
                        speakers.name,
                        segments.source_url,
                        segments.start_time,
                        segments.end_time,
                        segments.text,
                        segments.match_confidence,
                        segments.created_at
                    FROM segments
                    JOIN speakers ON segments.speaker_id = speakers.id
                    WHERE segments.match_confidence >= ?
                '''
                params = [min_confidence]
                
                if speaker_id:
                    query += ' AND speakers.id = ?'
                    params.append(speaker_id)
                
                query += ' ORDER BY segments.created_at DESC'
                
                c.execute(query, params)
                results = c.fetchall()
                
                knowledge_base = []
                for row in results:
                    knowledge_base.append({
                        'speaker': row[0],
                        'source': row[1],
                        'timestamp': f"{row[2]:.2f} -> {row[3]:.2f}",
                        'text': row[4],
                        'confidence': row[5],
                        'added': row[6]
                    })
                
                return knowledge_base
                
        except Exception as e:
            print(f"Error retrieving knowledge base: {str(e)}")
            return []

    def export_knowledge_base(self, output_file="knowledge_base.json"):
        """Export the entire knowledge base to a JSON file"""
        try:
            knowledge_base = {
                'speakers': [],
                'segments': []
            }
            
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                
                # Get all speakers
                c.execute('SELECT * FROM speakers')
                speakers = c.fetchall()
                for speaker in speakers:
                    knowledge_base['speakers'].append({
                        'id': speaker[0],
                        'name': speaker[1],
                        'reference': {
                            'url': speaker[2],
                            'start_time': speaker[3],
                            'end_time': speaker[4]
                        },
                        'voice_print': speaker[5],
                        'added': speaker[6]
                    })
                
                # Get all segments
                c.execute('''
                    SELECT 
                        segments.*,
                        speakers.name as speaker_name
                    FROM segments
                    JOIN speakers ON segments.speaker_id = speakers.id
                ''')
                segments = c.fetchall()
                for segment in segments:
                    knowledge_base['segments'].append({
                        'id': segment[0],
                        'speaker': {
                            'id': segment[1],
                            'name': segment[-1]
                        },
                        'source': segment[2],
                        'timestamp': {
                            'start': segment[3],
                            'end': segment[4]
                        },
                        'text': segment[5],
                        'confidence': segment[6],
                        'added': segment[7]
                    })
            
            # Save to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
            
            print(f"✓ Knowledge base exported to: {output_file}")
            
        except Exception as e:
            print(f"Error exporting knowledge base: {str(e)}") 