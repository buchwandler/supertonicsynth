from supertonicsynth import SupertonicRuntime, SynthesisConfig, VoiceLevelConfig

examples = [
    ("en", "A prepared English sentence."),
    ("de", "Ein vorbereiteter deutscher Satz."),
    ("es", "Una frase española preparada."),
]
config = SynthesisConfig(voice_level=VoiceLevelConfig(mode="calibrated"))

with SupertonicRuntime.from_pretrained("supertonic-3") as tts:
    for language, text in examples:
        result = tts.synthesize_text(
            text,
            voice="F1",
            language=language,
            config=config,
        )
        result.write_wav(f"example_{language}.wav")
        print(result.metadata["voice_level"]["calibration_key"])
