from __future__ import annotations

import argparse
from pathlib import Path

from .discovery import discover_models
from .errors import BundleNotFoundError
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
        help="list catalog voices without installing model assets",
        description="List catalog voices without installing model assets. Catalog metadata may be refreshed or read offline.",
    )
    voices.add_argument("--model", default="supertonic-3")
    voices.add_argument("--offline", action="store_true", help="use cached catalog metadata only")
    voices.add_argument("--refresh-catalog", action="store_true", help="refresh catalog metadata")
    voices.add_argument("--cache-dir", type=Path)
    voices.add_argument("--catalog-path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "voices":
        models = discover_models(
            offline=args.offline,
            refresh=args.refresh_catalog,
            cache_dir=args.cache_dir,
            catalog_path=args.catalog_path,
        )
        requested = args.model.removeprefix("supertonic:")
        model = next(
            (item for item in models if requested in (item.id, item.ref, *item.aliases)),
            None,
        )
        if model is None:
            raise BundleNotFoundError(f"Unknown Supertonic model {args.model!r}")
        for voice_id in model.voice_ids:
            print(voice_id)
        return 0

    with SupertonicRuntime.from_pretrained(args.model) as runtime:
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
