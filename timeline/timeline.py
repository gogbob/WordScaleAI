from embeddings import semantic_embedding, emotion_embedding, prosody_embedding
from lns import CharacterState
import numpy as np

# Example short story
story = [
    {"speaker": "Alice", "text": "Hello there!"},
    {"speaker": "Alice", "text": "It's a dark rainy night..."},
    {"speaker": "Bob", "text": "I hope we reach the castle soon."},
]

# Initialize character states
characters = {name: CharacterState(name) for name in ["Alice", "Bob"]}

# Build timeline with integrated states
timeline = []

for i, unit in enumerate(story):
    speaker = characters[unit["speaker"]]

    delta_voice = semantic_embedding(unit["text"])[:512] # first 512 dims
    delta_emotion = emotion_embedding(unit["text"])
    delta_prosody = prosody_embedding(unit["text"])

    # Integrate into character state
    speaker.integrate_deltas(delta_voice, delta_emotion, delta_prosody)

    # Store snapshot
    timeline.append({
        "uid": f"U{i+1:03d}",
        "text": unit["text"],
        "speaker": unit["speaker"],
        "delta_voice": delta_voice,
        "delta_emotion": delta_emotion,
        "delta_prosody": delta_prosody,
        "state_snapshot": {
            "voice": speaker.voice.copy(),
            "emotion_pos": speaker.emotion_pos.copy(),
            "emotion_vel": speaker.emotion_vel.copy(),
            "prosody": speaker.prosody.copy()
        }
    })

# Timeline now has delta vectors and integrated states
print("Timeline prepared with latent states.")
