from __future__ import annotations

SUPPORTED_LANGUAGES = (
    "en",
    "ko",
    "ja",
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "lt",
    "lv",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "vi",
)
UNKNOWN_LANGUAGE = "na"
AVAILABLE_LANGUAGES = (*SUPPORTED_LANGUAGES, UNKNOWN_LANGUAGE)
DEFAULT_MODEL = "supertonic-3"
DEFAULT_VOICE = "M1"
DEFAULT_LANGUAGE = UNKNOWN_LANGUAGE
DEFAULT_STEPS = 5
DEFAULT_SPEED = 1.05
DEFAULT_MAX_CHUNK_LENGTH = 300
DEFAULT_MAX_CHUNK_LENGTH_KO = 120
DEFAULT_SILENCE_DURATION = 0.3
MIN_SPEED = 0.7
MAX_SPEED = 2.0
MIN_STEPS = 1
MAX_STEPS = 100
MAX_TEXT_LENGTH = 100_000
SAMPLE_RATE = 44_100
