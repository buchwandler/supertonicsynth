#!/usr/bin/env python
"""
Script to build documentation for supertonicsynth.

This script builds the Sphinx documentation for the supertonicsynth package.
It can be run using:
    python docs/make.py [option]

Options:
    clean   - clean the build directory
    html    - build HTML documentation
    dirhtml - build HTML documentation with directory structure
    all     - build all documentation formats
    help    - show help message
"""

import shutil
import subprocess
import sys
from pathlib import Path


def main():
    """Run the script."""
    sphinx_build = "sphinx-build"
    source_dir = Path(__file__).resolve().parent
    build_dir = source_dir / "_build"

    target = "html" if len(sys.argv) < 2 else sys.argv[1]

    if target == "clean":
        if build_dir.exists():
            print(f"Cleaning {build_dir}...")
            shutil.rmtree(build_dir)
        return 0

    if target == "help":
        print(__doc__)
        return 0

    # Set of valid targets
    valid_targets = {
        "html",
        "dirhtml",
        "latex",
        "latexpdf",
        "text",
        "man",
        "changes",
        "linkcheck",
        "doctest",
        "all",
    }

    if target not in valid_targets:
        print(f"Unknown target: {target}")
        print("Use 'help' target for help")
        return 1

    build_dir.mkdir(parents=True, exist_ok=True)
    formats = ["html", "dirhtml", "latex"] if target == "all" else [target]
    for fmt in formats:
        output_dir = build_dir / fmt
        cmd = [sphinx_build, "-b", fmt, str(source_dir), str(output_dir)]
        print(f"Building {fmt} documentation...")
        subprocess.run(cmd, check=True)

    if target == "all":
        print(f"Build finished. Documentation is in {build_dir}")
    else:
        print(f"Build finished. Documentation is in {build_dir / target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
