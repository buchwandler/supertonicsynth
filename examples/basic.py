from supertonicsynth import SupertonicRuntime

with SupertonicRuntime.from_pretrained("supertonic-3") as tts:
    result = tts.synthesize_text(
        "Welcome to SupertonicSynth.", voice="M1", language="en", seed=1234
    )
    result.write_wav("basic.wav")
    print(result.duration)
