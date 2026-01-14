# Specifications

## Dimensional Spaces

| Space                   | Symbol | Dim  | Purpose                        |
| ----------------------- | ------ | ---- | ------------------------------ |
| Semantic                | `S`    | 3072 | Meaning, reference, retrieval  |
| Voice Identity          | `V`    | 512  | Accent, age, sociolect, timbre |
| Emotion Position        | `E`    | 256  | Current affect                 |
| Emotion Velocity        | `Ė`    | 256  | Emotional momentum             |
| Prosody                 | `P`    | 128  | Rhythm, stress, pacing         |
| Environment             | `Env`  | 128  | Ambience, SFX field            |
| Interaction             | `I`    | 128  | Power, tension, proximity      |
| (Future) Visual Posture | `K`    | 128  | Body language                  |
| (Future) Gaze           | `G`    | 64   | Attention, focus               |


## Voice Identity

### State

V_t ∈ ℝ⁵¹²

### Update

V_t = α_v · V_{t-1} + (1 − α_v) · ΔV_t

where:
ΔV_t = inferred voice signal from text
α_v ∈ [0.97, 0.995]

## Emotion Position

### State

E_t  = emotion position
Ė_t = emotion velocity

### Update

Ė_t = β · Ė_{t-1} + ΔE_t
E_t = E_{t-1} + Ė_t

where:
β ∈ [0.6, 0.9] (emotional inertia)
ΔE_t = emotional impulse from text

## Prosody

### State

P_t ∈ ℝ¹²⁸

### Update

P_t = α_p · P_{t-1} + (1 − α_p) · ΔP_t

where:
α_p ∈ [0.3, 0.6]

## Environment

### State

Env_t ∈ ℝ¹²⁸

### Update

Env_t = α_env · Env_{t-1} + (1 − α_env) · ΔEnv_t

where:
α_env ∈ [0.8, 0.95]

## Interaction

### State

I_t(A,B) ∈ ℝ¹²⁸

### Update

I_t = α_i · I_{t-1} + (1 − α_i) · ΔI_t

where:
α_i ∈ [0.7, 0.9]

## How Δ (Delta) Vectors Are Produced
ΔV_t  = f_voice(semantic_embedding, dialogue_style)
ΔE_t  = f_emotion(semantic_embedding, context)
ΔP_t  = f_prosody(punctuation, pacing)
ΔEnv_t = f_environment(action verbs, setting)

## Offline smoothing

After forward integration, run a backward pass:

E'_t = γ · E_t + (1 − γ) · E_{t+1}

## Decoder Projection

[V_t, E_t, P_t] → audio parameters
[E_t, I_t, Env_t] → visuals (future)


