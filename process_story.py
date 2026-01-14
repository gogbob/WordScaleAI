import spacy
from allennlp.predictors.predictor import Predictor

# 1. Load NER model
nlp = spacy.load("en_core_web_sm")  # or a larger NER model

# 2. Load coreference model
predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz"
)

def preprocess_story(text):
    # Step 1: sentence split (simple)
    sentences = [s.text for s in nlp(text).sents]

    # Step 2: NER + coref (example for tracking entities)
    coref_results = predictor.predict(document=text)

    # Step 3: Build speaker attribution (rule-based for demo)
    timeline = []
    current_speaker = None
    for sentence in sentences:
        if "Alice" in sentence:
            current_speaker = "Alice"
        elif "Bob" in sentence:
            current_speaker = "Bob"
        timeline.append({"speaker": current_speaker, "text": sentence})
    return timeline

