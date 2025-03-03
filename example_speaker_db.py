from speaker_db import SpeakerKnowledgeDB

def main():
    # Initialize the database
    db = SpeakerKnowledgeDB()
    
    # Example: Add a new speaker from a reference video
    # This should be a clear segment where only the target speaker is talking
    speaker_id = db.add_speaker(
        name="Example Speaker",
        url="https://www.youtube.com/watch?v=EXAMPLE_ID",
        start_time=65,  # Start at 1:05
        end_time=95     # End at 1:35
    )
    
    if speaker_id:
        print(f"\nSuccessfully added speaker with ID: {speaker_id}")
        
        # Now analyze another video for this speaker
        db.analyze_video(
            url="https://www.youtube.com/watch?v=ANOTHER_VIDEO_ID",
            speaker_ids=[speaker_id]  # Optionally focus on specific speakers
        )
        
        # Get all transcribed segments for this speaker
        knowledge = db.get_speaker_knowledge(speaker_id)
        
        print("\nKnowledge base entries:")
        for entry in knowledge:
            print("\n---")
            print(f"Source: {entry['source']}")
            print(f"Time: {entry['timestamp']}")
            print(f"Text: {entry['text']}")
            print(f"Confidence: {entry['confidence']:.2f}")
        
        # Export the entire knowledge base to JSON
        db.export_knowledge_base()

if __name__ == "__main__":
    main() 