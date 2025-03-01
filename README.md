# Voice Analysis Tool

A Python tool for analyzing speaker voice prints from YouTube videos. This tool can extract speaker voice prints and analyze videos to find matching segments.

## Features

- Extract speaker voice prints from YouTube video segments
- Analyze videos to find segments matching a voice print
- Support for caching audio files
- Test mode for quick verification
- Detailed logging and diagnostics

## Requirements

- Python 3.8+
- FFmpeg installed and in PATH
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/silvan-private/vibecoding9.git
cd vibecoding9
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the interactive menu:
```bash
python run_voice_analysis.py
```

Available operations:
1. Extract Speaker Voice Print
2. Analyze Video with Existing Voice Print
3. Full Process (Extract and Analyze)
4. View Saved Voice Prints
5. Exit
6. Test Extract (Pre-filled values)

## Project Structure

- `run_voice_analysis.py`: Main script with interactive menu
- `audio_processor.py`: Core audio processing functionality
- `utils/`: Utility functions and helpers
  - `time_helpers.py`: Time formatting utilities

## Models Used

- Whisper (tiny.en) for transcription
- ECAPA-TDNN for voice embeddings
- Wav2Vec2 for audio processing

## License

MIT License 