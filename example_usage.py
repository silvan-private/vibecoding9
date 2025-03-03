from audio_processor import AudioProcessor

processor = AudioProcessor()
result = processor.process_video("https://www.youtube.com/watch?v=cfAUbJgR0pE")

print("Transcript:", result["transcript"])
print("\nSpeakers:", result["speakers"])
print("\nTimestamps:")
for segment in result["timestamps"]:
    print(f"{segment['speaker']} ({segment['start']:.2f} - {segment['end']:.2f}): {segment['text']}") 