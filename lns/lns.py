import numpy as np

class CharacterState:
    def __init__(self, name):
        self.name = name
        self.voice = np.zeros(512, dtype=np.float32)       # slow identity
        self.emotion_pos = np.zeros(256, dtype=np.float32) # emotion position
        self.emotion_vel = np.zeros(256, dtype=np.float32) # emotion velocity
        self.prosody = np.zeros(128, dtype=np.float32)     # prosody

        # Parameters
        self.alpha_voice = 0.99
        self.beta_emotion = 0.85
        self.alpha_prosody = 0.5

    def integrate_deltas(self, delta_voice, delta_emotion, delta_prosody):
        # Voice identity: slow convergence
        self.voice = self.alpha_voice * self.voice + (1 - self.alpha_voice) * delta_voice

        # Emotion: second-order dynamics
        self.emotion_vel = self.beta_emotion * self.emotion_vel + delta_emotion
        self.emotion_pos += self.emotion_vel

        # Prosody: local fast-changing
        self.prosody = self.alpha_prosody * self.prosody + (1 - self.alpha_prosody) * delta_prosody
