from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import run_m1
from .validation import run_validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="unravel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the part-separation pipeline")
    run_parser.add_argument("input", type=Path)
    run_parser.add_argument("-o", "--output", type=Path, required=True)
    run_parser.add_argument(
        "--inpaint", action="store_true", help="fill occluded regions with diffusion inpainting"
    )

    validate_parser = subparsers.add_parser(
        "validate", help="run the pipeline over a directory of test illustrations"
    )
    validate_parser.add_argument("input_dir", type=Path)
    validate_parser.add_argument("-o", "--output", type=Path, required=True)
    validate_parser.add_argument(
        "--inpaint", action="store_true", help="fill occluded regions with diffusion inpainting"
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        args.output.mkdir(parents=True, exist_ok=True)
        document = run_m1(args.input, args.output, inpaint=args.inpaint)
        print(f"wrote {len(document.parts)} parts to {args.output}")
        return 0

    if args.command == "validate":
        summary = run_validation(args.input_dir, args.output, inpaint=args.inpaint)
        print(
            f"{summary.parts_pass_count}/{summary.total} images passed "
            f"({summary.parts_pass_rate:.0%}), meets_parts_criterion="
            f"{summary.meets_parts_criterion}"
        )
        for result in summary.results:
            if not result.success:
                print(f"  FAIL {result.filename}: missing={result.missing_labels} error={result.error}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
