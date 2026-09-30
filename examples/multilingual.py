from supertonicsynth import SupertonicRuntime

examples = [
    ("en", "Hello from Supertonic."),
    ("de", "Hallo von Supertonic."),
    ("fr", "Bonjour de Supertonic."),
]
with SupertonicRuntime.from_pretrained("supertonic-3") as tts:
    for lang, text in examples:
        tts.synthesize_text(text, voice="F1", language=lang).write_wav(f"example_{lang}.wav")
