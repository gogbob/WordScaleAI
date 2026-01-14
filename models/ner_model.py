"""
Named Entity Recognition Model
Provides NER capabilities for story text analysis
"""
import spacy
from typing import List, Dict, Optional


class NERModel:
    """
    Named Entity Recognition model using spaCy
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the NER model
        
        Args:
            model_name: spaCy model to use (default: en_core_web_sm)
        """
        self.model_name = model_name
        self.nlp = None
        
    def load_model(self):
        """Load the spaCy NER model"""
        try:
            self.nlp = spacy.load(self.model_name)
            print(f"NER model '{self.model_name}' loaded successfully")
        except OSError:
            print(f"Model '{self.model_name}' not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", self.model_name])
            self.nlp = spacy.load(self.model_name)
            print(f"NER model '{self.model_name}' downloaded and loaded")
    
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract named entities from text
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of dictionaries containing entity information
        """
        if self.nlp is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities
    
    def get_entities_by_type(self, text: str) -> Dict[str, List[str]]:
        """
        Group entities by their type
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with entity types as keys and lists of entity texts as values
        """
        entities = self.extract_entities(text)
        grouped = {}
        
        for entity in entities:
            label = entity['label']
            if label not in grouped:
                grouped[label] = []
            if entity['text'] not in grouped[label]:
                grouped[label].append(entity['text'])
        
        return grouped
