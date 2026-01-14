import numpy as np


def get_prosody_embedding(text: str) -> np.ndarray:
    """Derive prosody features (length, punctuation cues) and project to 128-d."""
    words = text.split()
    length = max(len(words), 1)
    exclamations = text.count("!") / length
    question = 1.0 if "?" in text else 0.0
    ellipses = text.count("...") / length
    vector = np.array([float(length), exclamations, question, ellipses], dtype=np.float32)
    return np.pad(vector, (0, 124))
