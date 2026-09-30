from supertonicsynth.config import AVAILABLE_LANGUAGES


def test_languages():
    assert len(AVAILABLE_LANGUAGES) == 32
    assert "en" in AVAILABLE_LANGUAGES
    assert "na" in AVAILABLE_LANGUAGES
