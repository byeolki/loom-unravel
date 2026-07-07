from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import run_m1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="unravel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the M1 rough part-separation pipeline")
    run_parser.add_argument("input", type=Path)
    run_parser.add_argument("-o", "--output", type=Path, required=True)

    args = parser.parse_args(argv)

    if args.command == "run":
        args.output.mkdir(parents=True, exist_ok=True)
        document = run_m1(args.input, args.output)
        print(f"wrote {len(document.parts)} parts to {args.output}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
