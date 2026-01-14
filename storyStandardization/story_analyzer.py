"""
Story Analyzer
Analyzes stories using NER to extract and categorize entities
"""
import re
from typing import Dict, List, Optional
from models.ner_model import NERModel


class StoryAnalyzer:
    """
    Analyzes story text to extract named entities and metadata
    """
    
    def __init__(self):
        """Initialize the story analyzer with NER model"""
        self.ner_model = NERModel()
        self.ner_model.load_model()
    
    def extract_book_metadata(self, story_text: str) -> Dict[str, str]:
        """
        Extract book metadata from story text (between *** markers)
        
        Args:
            story_text: Raw story text with metadata
            
        Returns:
            Dictionary with 'metadata' and 'story' keys
        """
        # Extract text between *** markers
        metadata_match = re.search(r'\*\*\*(.*?)\*\*\*', story_text, re.DOTALL)
        
        if metadata_match:
            metadata = metadata_match.group(1).strip()
            # Remove metadata from story
            clean_story = re.sub(r'\*\*\*.*?\*\*\*', '', story_text, flags=re.DOTALL).strip()
            return {
                'metadata': metadata,
                'story': clean_story
            }
        
        return {
            'metadata': '',
            'story': story_text
        }
    
    def analyze_story(self, story_text: str) -> Dict:
        """
        Analyze story and extract all named entities
        
        Args:
            story_text: Story text to analyze
            
        Returns:
            Dictionary containing metadata, entities, and entity groupings
        """
        # Extract metadata and clean story
        parsed = self.extract_book_metadata(story_text)
        clean_story = parsed['story']
        
        # Extract entities
        entities = self.ner_model.extract_entities(clean_story)
        entities_by_type = self.ner_model.get_entities_by_type(clean_story)
        
        return {
            'metadata': parsed['metadata'],
            'total_entities': len(entities),
            'entities': entities,
            'entities_by_type': entities_by_type,
            'story_length': len(clean_story)
        }
    
    def get_characters(self, story_text: str) -> List[str]:
        """
        Extract character names (PERSON entities) from story
        
        Args:
            story_text: Story text to analyze
            
        Returns:
            List of character names
        """
        parsed = self.extract_book_metadata(story_text)
        entities_by_type = self.ner_model.get_entities_by_type(parsed['story'])
        return entities_by_type.get('PERSON', [])
    
    def get_locations(self, story_text: str) -> List[str]:
        """
        Extract locations (GPE, LOC entities) from story
        
        Args:
            story_text: Story text to analyze
            
        Returns:
            List of location names
        """
        parsed = self.extract_book_metadata(story_text)
        entities_by_type = self.ner_model.get_entities_by_type(parsed['story'])
        
        locations = []
        locations.extend(entities_by_type.get('GPE', []))  # Geopolitical entities
        locations.extend(entities_by_type.get('LOC', []))  # Locations
        return locations
    
    def print_analysis_summary(self, analysis: Dict):
        """
        Print a formatted summary of story analysis
        
        Args:
            analysis: Analysis dictionary from analyze_story()
        """
        print("\n" + "="*60)
        print("STORY ANALYSIS SUMMARY")
        print("="*60)
        
        if analysis['metadata']:
            print(f"\nMetadata: {analysis['metadata']}")
        
        print(f"\nStory Length: {analysis['story_length']} characters")
        print(f"Total Entities Found: {analysis['total_entities']}")
        
        print("\nEntities by Type:")
        print("-" * 60)
        for entity_type, entities in analysis['entities_by_type'].items():
            print(f"\n{entity_type} ({len(entities)}):")
            for entity in entities[:10]:  # Show first 10
                print(f"  - {entity}")
            if len(entities) > 10:
                print(f"  ... and {len(entities) - 10} more")
        
        print("\n" + "="*60)
