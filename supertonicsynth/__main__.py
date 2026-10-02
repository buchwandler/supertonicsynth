from __future__ import annotations

import argparse
from pathlib import Path

from .runtime import SupertonicRuntime
from .types import SynthesisConfig
from .voice_level import VoiceLevelConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="supertonicsynth")
    sub = parser.add_subparsers(dest="command", required=True)
    synth = sub.add_parser("synthesize", help="synthesize text")
    synth.add_argument("text")
    synth.add_argument("-o", "--output", type=Path, required=True)
    synth.add_argument("--model", default="supertonic-3")
    synth.add_argument("--voice", default="M1")
    synth.add_argument("--language", default="na")
    synth.add_argument("--steps", type=int, default=5)
    synth.add_argument("--speed", type=float, default=1.05)
    synth.add_argument("--seed", type=int)
    synth.add_argument("--voice-level", choices=("off", "calibrated"), default="off")
    synth.add_argument("--voice-gain-db", type=float)
    synth.add_argument("--output-gain", type=float, default=1.0)
    synth.add_argument("--normalize-audio", action=argparse.BooleanOptionalAction, default=None)
    voices = sub.add_parser(
        "voices",
        help="install/open a model bundle and list its voices",
        description="Installs and downloads the selected bundle if needed, then lists its voices.",
    )
    voices.add_argument("--model", default="supertonic-3")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    with SupertonicRuntime.from_pretrained(args.model) as runtime:
        if args.command == "voices":
            for name in runtime.voice_names:
                print(name)
            return 0
        config = SynthesisConfig(
            steps=args.steps,
            speed=args.speed,
            seed=args.seed,
            normalize_audio=(True if args.normalize_audio is None else args.normalize_audio),
            output_gain=args.output_gain,
            voice_level=VoiceLevelConfig(
                mode=args.voice_level,
                gain_db=args.voice_gain_db,
            ),
        )
        result = runtime.synthesize_text(
            args.text, voice=args.voice, language=args.language, config=config
        )
        result.write_wav(args.output)
        print(f"{args.output} ({result.duration:.2f}s, {result.sample_rate} Hz)")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
