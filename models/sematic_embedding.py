import os
from typing import Any, Optional

import numpy as np

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - fallback for legacy client
    OpenAI = None
    import openai  # type: ignore


MODEL_NAME = "text-embedding-3-large"


def _resolve_client(provided: Optional[Any] = None) -> Any:
    if provided is not None:
        return provided
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    if OpenAI is not None:
        return OpenAI(api_key=api_key)
    openai.api_key = api_key
    return openai


def get_semantic_embedding(text: str, *, client: Optional[Any] = None, model: str = MODEL_NAME) -> np.ndarray:
    """Query OpenAI embeddings API for a semantic representation."""
    resolved_client = _resolve_client(client)
    if OpenAI is not None and isinstance(resolved_client, OpenAI):
        response = resolved_client.embeddings.create(input=text, model=model)
        embedding = response.data[0].embedding
    else:
        response = resolved_client.Embedding.create(input=text, model=model)
        embedding = response["data"][0]["embedding"]
    return np.asarray(embedding, dtype=np.float32)
