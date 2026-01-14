from dataset import retrieveData
from storyStandardization import StoryAnalyzer

def main():
    # Initialize the story analyzer
    print("Initializing Story Analyzer with NER model...")
    analyzer = StoryAnalyzer()
    
    # Retrieve book data
    print("\nRetrieving book 1...")
    data = retrieveData.retrieveBook(1)
    
    # Get the story text from the data
    story_text = data['story'].get('story', '') if 'story' in data['story'] else str(data['story'])
    
    print(f"\nBook Info: {data['book'].get('title', 'Unknown')}")
    print(f"Story preview: {story_text[:200]}...")
    
    # Analyze the story
    print("\nAnalyzing story for named entities...")
    analysis = analyzer.analyze_story(story_text)
    
    # Print the analysis summary
    analyzer.print_analysis_summary(analysis)
    
    # Extract specific entity types
    print("\n" + "="*60)
    print("CHARACTERS:")
    characters = analyzer.get_characters(story_text)
    for char in characters:
        print(f"  - {char}")
    
    print("\nLOCATIONS:")
    locations = analyzer.get_locations(story_text)
    for loc in locations:
        print(f"  - {loc}")
    print("="*60)

    for unit in timeline:
        audio_plan = audio_decoder.decode_audio_unit(unit["state_snapshot"])
        print(f"Unit {unit['uid']} ({unit['speaker']}): pitch {audio_plan['pitch_curve'][:3]}, "
            f"rate {audio_plan['rate_curve'][:3]}, volume {audio_plan['volume_curve'][:3]}")

if __name__ == "__main__":
    main()
