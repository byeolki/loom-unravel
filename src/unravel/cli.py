from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import run_m1
from .validation import load_validation_report, run_validation

INPAINT_HELP = "fill occluded regions with LaMa inpainting"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="unravel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the part-separation pipeline")
    run_parser.add_argument("input", type=Path)
    run_parser.add_argument("-o", "--output", type=Path, required=True)
    run_parser.add_argument("--inpaint", action="store_true", help=INPAINT_HELP)

    validate_parser = subparsers.add_parser(
        "validate", help="run the pipeline over a directory of test illustrations"
    )
    validate_parser.add_argument("input_dir", type=Path)
    validate_parser.add_argument("-o", "--output", type=Path, required=True)
    validate_parser.add_argument("--inpaint", action="store_true", help=INPAINT_HELP)

    report_parser = subparsers.add_parser(
        "report", help="re-summarize a validation_report.json after filling in artifact_review"
    )
    report_parser.add_argument("output_dir", type=Path)

    args = parser.parse_args(argv)

    if args.command == "run":
        args.output.mkdir(parents=True, exist_ok=True)
        document = run_m1(args.input, args.output, inpaint=args.inpaint)
        print(f"wrote {len(document.parts)} parts to {args.output}")
        return 0

    if args.command == "validate":
        summary = run_validation(args.input_dir, args.output, inpaint=args.inpaint)
        print_summary(summary)
        return 0

    if args.command == "report":
        summary = load_validation_report(args.output_dir)
        print_summary(summary)
        return 0

    return 1


def print_summary(summary) -> None:
    print(
        f"parts: {summary.parts_pass_count}/{summary.total} passed "
        f"({summary.parts_pass_rate:.0%}), meets_parts_criterion={summary.meets_parts_criterion}"
    )
    for result in summary.results:
        if not result.success:
            print(f"  FAIL {result.filename}: missing={result.missing_labels} error={result.error}")

    if summary.artifact_reviewed_count == 0:
        print(
            "artifacts: not yet reviewed — fill in artifact_review (true/false) per image in "
            "validation_report.json, then run `unravel report <output_dir>`"
        )
    else:
        print(
            f"artifacts: {summary.artifact_pass_count}/{summary.artifact_reviewed_count} reviewed "
            f"images passed ({summary.artifact_pass_rate:.0%}), "
            f"meets_artifact_criterion={summary.meets_artifact_criterion}"
        )


if __name__ == "__main__":
    sys.exit(main())
