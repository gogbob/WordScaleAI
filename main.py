from timeline import timeline
from audio_decoder import decode_audio_unit

for unit in timeline:
    audio_plan = decode_audio_unit(unit["state_snapshot"])
    print(f"Unit {unit['uid']} ({unit['speaker']}): pitch {audio_plan['pitch_curve'][:3]}, "
          f"rate {audio_plan['rate_curve'][:3]}, volume {audio_plan['volume_curve'][:3]}")
