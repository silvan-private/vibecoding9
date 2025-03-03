# Speaker Detection and Transcription System

A Python-based system for analyzing videos, detecting speakers, and generating transcripts with speaker identification.

## Features

- Speaker identification using ECAPA-TDNN model
- Audio transcription using OpenAI's Whisper model
- Multiple output formats:
  - Raw transcripts with timestamps
  - Grouped transcripts by speaker
  - Interactive HTML viewer
- Speaker similarity scoring
- Progress tracking and logging
- Video segment analysis
- Caching of downloaded audio and embeddings

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd speaker-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
# Analyze a video with speaker detection
python analyze_speaker.py
```

### Testing Speaker Detection

```python
# Test speaker detection on a specific segment
python test_speaker_detection.py
```

## Project Structure

- `analyze_speaker.py`: Main script for video analysis
- `test_speaker_detection.py`: Script for testing speaker detection
- `speaker_data/`: Directory for storing outputs
  - `models/`: Cached ML models
  - `cache/`: Downloaded audio files
  - `embeddings/`: Speaker embeddings
  - `transcripts/`: Generated transcripts

## Output Formats

1. **Raw Transcript** (`raw.txt`):
   - Detailed transcript with timestamps
   - Speaker identification and confidence scores
   - Individual segments

2. **Grouped Transcript** (`grouped.txt`):
   - Segments grouped by speaker
   - Merged consecutive segments from same speaker
   - Average confidence scores

3. **HTML Viewer** (`viewer.html`):
   - Interactive web interface
   - Styled transcript display
   - Speaker highlighting
   - Timestamp navigation

## Technical Details

- Speaker Recognition: ECAPA-TDNN model from SpeechBrain
- Transcription: OpenAI's Whisper model
- Audio Processing: librosa and soundfile
- Similarity Threshold: 0.75 (configurable)
- Chunk Duration: 30 seconds

## Dependencies

- PyTorch
- SpeechBrain
- OpenAI Whisper
- yt-dlp
- scikit-learn
- And more (see requirements.txt)

## License

MIT License 