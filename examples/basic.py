from supertonicsynth import SupertonicRuntime, SynthesisConfig, VoiceLevelConfig

config = SynthesisConfig(
    voice_level=VoiceLevelConfig(mode="calibrated"),
)

with SupertonicRuntime.from_pretrained("supertonic-3") as tts:
    result = tts.synthesize_text(
        "A prepared German sentence.",
        voice="F1",
        language="de",
        config=config,
        seed=1234,
    )
    result.write_wav("basic.wav")
    print(result.metadata["voice_ref"])
    print(result.metadata["voice_level"]["calibration_key"])
