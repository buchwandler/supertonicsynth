from supertonicsynth import SupertonicRuntime

with SupertonicRuntime.from_pretrained("supertonic-3") as tts:
    for voice in tts.voice_names:
        print(voice)
