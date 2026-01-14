from models.emotion_embedding import get_emotion_embedding
from models.prosody_embedding import get_prosody_embedding
from models.sematic_embedding import get_semantic_embedding


def semantic_embedding(text: str):
    """Wrapper around the OpenAI semantic embedding model."""
    return get_semantic_embedding(text)


def emotion_embedding(text: str):
    """Wrapper around the DistilRoBERTa-based emotion embedding model."""
    return get_emotion_embedding(text)


def prosody_embedding(text: str):
    """Wrapper around handcrafted prosody features."""
    return get_prosody_embedding(text)
