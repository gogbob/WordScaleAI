import numpy as np

def decode_audio_unit(state_snapshot):
    """
    Map latent state to audio parameters.
    For demonstration, we project vectors to pitch, rate, volume curves.
    """
    voice_embedding = state_snapshot["voice"]
    pitch_curve = 100 + state_snapshot["emotion_pos"][:20]       # Hz
    rate_curve = 1.0 + state_snapshot["prosody"][:20] * 0.01
    volume_curve = 0.7 + state_snapshot["prosody"][:20] * 0.01

    return {
        "tts_embedding": voice_embedding,
        "pitch_curve": pitch_curve,
        "rate_curve": rate_curve,
        "volume_curve": volume_curve
    }
