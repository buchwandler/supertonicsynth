from types import SimpleNamespace

import pytest

import supertonicsynth
import supertonicsynth.__main__ as cli
from supertonicsynth.voice_level import VoiceLevelConfig


def test_public_voice_level_exports():
    names = (
        "CalibrationDataError",
        "VoiceCalibrationCatalog",
        "VoiceCalibrationKey",
        "VoiceLevelApplication",
        "VoiceLevelCalibration",
        "VoiceLevelConfig",
        "VoiceLevelMode",
        "apply_voice_level_calibration",
        "default_voice_calibration",
        "load_voice_calibration",
    )

    assert set(names) <= set(supertonicsynth.__all__)
    assert all(hasattr(supertonicsynth, name) for name in names)


def test_cli_parses_calibration_and_audio_controls():
    args = cli.build_parser().parse_args(
        [
            "synthesize",
            "prepared text",
            "-o",
            "output.wav",
            "--voice-level",
            "calibrated",
            "--voice-gain-db",
            "-2.5",
            "--output-gain",
            "0.75",
            "--no-normalize-audio",
        ]
    )

    assert args.voice_level == "calibrated"
    assert args.voice_gain_db == -2.5
    assert args.output_gain == 0.75
    assert args.normalize_audio is False


def test_cli_passes_postprocessing_controls_to_synthesis(monkeypatch, tmp_path):
    class FakeRuntime:
        def __init__(self):
            self.request = None

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def synthesize_text(self, text, **kwargs):
            self.request = (text, kwargs)
            return SimpleNamespace(
                duration=1.0,
                sample_rate=22050,
                write_wav=lambda path: None,
            )

    runtime = FakeRuntime()
    monkeypatch.setattr(cli.SupertonicRuntime, "from_pretrained", lambda ref: runtime)
    output = tmp_path / "out.wav"

    assert (
        cli.main(
            [
                "synthesize",
                "prepared text",
                "-o",
                str(output),
                "--voice-level",
                "calibrated",
                "--voice-gain-db",
                "-2.5",
                "--output-gain",
                "0.75",
                "--no-normalize-audio",
            ]
        )
        == 0
    )

    text, request = runtime.request
    assert text == "prepared text"
    config = request["config"]
    assert config.normalize_audio is False
    assert config.output_gain == 0.75
    assert config.voice_level == VoiceLevelConfig(mode="calibrated", gain_db=-2.5)


def test_voices_cli_uses_metadata_only_discovery(monkeypatch, tmp_path, capsys):
    model = SimpleNamespace(
        id="supertonic-3",
        ref="supertonic:supertonic-3",
        aliases=("st3",),
        voice_ids=("F1", "M1"),
    )
    discovery_calls = []

    def discover(**kwargs):
        discovery_calls.append(kwargs)
        return (model,)

    def fail_runtime(*_args, **_kwargs):
        raise AssertionError("voices must not open a runtime")

    monkeypatch.setattr(cli, "discover_models", discover)
    monkeypatch.setattr(cli.SupertonicRuntime, "from_pretrained", fail_runtime)
    cache_dir = tmp_path / "cache"
    catalog_path = tmp_path / "catalog.json"

    assert (
        cli.main(
            [
                "voices",
                "--model",
                "supertonic:st3",
                "--offline",
                "--refresh-catalog",
                "--cache-dir",
                str(cache_dir),
                "--catalog-path",
                str(catalog_path),
            ]
        )
        == 0
    )

    assert capsys.readouterr().out == "F1\nM1\n"
    assert discovery_calls == [
        {
            "offline": True,
            "refresh": True,
            "cache_dir": cache_dir,
            "catalog_path": catalog_path,
        }
    ]


def test_voices_cli_help_describes_metadata_only_behavior(capsys):

    with pytest.raises(SystemExit):
        cli.main(["voices", "--help"])

    assert "without installing model assets" in capsys.readouterr().out
