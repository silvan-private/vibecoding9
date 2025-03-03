from audio_processor import download_model_files
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print("Testing model download...")
    result = download_model_files()
    print(f"Download result: {result}")

if __name__ == "__main__":
    main() 