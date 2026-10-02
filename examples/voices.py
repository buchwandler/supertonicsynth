"""List voices from a bundle, installing it first if the model is not available locally."""

from supertonicsynth import SupertonicRuntime

with SupertonicRuntime.from_pretrained("supertonic:supertonic-3") as tts:
    for voice in tts.voice_names:
        print(voice)
