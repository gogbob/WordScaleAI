from functools import lru_cache
from typing import Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


@lru_cache(maxsize=1)
def _load_model() -> tuple[AutoTokenizer, AutoModel]:
    """Lazily load the emotion classification model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    return tokenizer, model


def get_emotion_embedding(text: Union[str, list[str]]) -> np.ndarray:
    """Generate a 256-d emotion embedding from text."""
    tokenizer, model = _load_model()
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**encoded)
        cls_state = outputs.last_hidden_state[:, 0, :]
        normalized = torch.nn.functional.normalize(cls_state, p=2, dim=-1)
    vector = normalized[..., :256].cpu().numpy()
    # Ensure a 1-D vector for single text inputs
    return vector[0] if isinstance(text, str) else vector
