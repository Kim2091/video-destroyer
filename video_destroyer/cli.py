import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .run_store import RunError, RunStore
from .workflows import create, import_pairs, resume
from .workflows.common import WorkflowError, report_existing_run, validate_existing_run


def _print_summary(result):
    print(f"Dataset ready: {result['dataset']}")
    print(f"Clip pairs discovered: {result['pairs']}")
    print(f"Clip pairs accepted: {result['pairs_accepted']}")
    print(f"Frame sequences accepted: {result['sequences_accepted']}")
    print(f"Frame sequences rejected: {result['sequences_rejected']}")
    print("Validation: passed")
    print(f"Report: {result['report']}")


def build_parser():
    parser = argparse.ArgumentParser(prog="video-destroyer", description="Build validated paired video-frame datasets.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    create_parser = subcommands.add_parser("create", help="Create HR/LR pairs from source videos.")
    create_parser.add_argument("input", help="Source video file or directory")
    create_parser.add_argument("--output", required=True, help="New run directory")
    create_parser.add_argument("--config", help="Version 2 processing configuration")
    create_parser.add_argument("--fail-on-rejection", action="store_true", help="Return failure after writing reports when items are rejected")

    import_parser = subcommands.add_parser("import-pairs", help="Import matched HR/LR clips.")
    import_parser.add_argument("--hr", required=True, help="HR clip root")
    import_parser.add_argument("--lr", required=True, help="LR clip root")
    import_parser.add_argument("--output", required=True, help="New run directory")
    import_parser.add_argument("--config", help="Version 2 processing configuration")
    import_parser.add_argument("--materialize", choices=("copy", "hardlink"), help="Make run-owned input clips instead of referencing inputs")
    import_parser.add_argument("--fail-on-rejection", action="store_true", help="Return failure after writing reports when items are rejected")

    resume_parser = subcommands.add_parser("resume", help="Resume an existing run")
    resume_parser.add_argument("run")
    validate_parser = subcommands.add_parser("validate", help="Validate a finalized dataset")
    validate_parser.add_argument("run")
    report_parser = subcommands.add_parser("report", help="Rewrite a run summary report")
    report_parser.add_argument("run")
    subcommands.add_parser("gui", help="Launch the optional desktop interface")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "gui":
            from .gui import main as gui_main
            return gui_main()
        if args.command == "create":
            config = load_config(args.config, "create")
            if args.fail_on_rejection:
                config["runtime"]["fail_on_rejection"] = True
            result = create.start(args.input, args.output, config)
            _print_summary(result)
            return 1 if result["strict_failure"] else 0
        if args.command == "import-pairs":
            config = load_config(args.config, "import-pairs")
            if args.fail_on_rejection:
                config["runtime"]["fail_on_rejection"] = True
            result = import_pairs.start(args.hr, args.lr, args.output, config, args.materialize)
            _print_summary(result)
            return 1 if result["strict_failure"] else 0
        store = RunStore.open(args.run)
        if args.command == "resume":
            result = resume.start(args.run)
            _print_summary(result)
            return 1 if result["strict_failure"] else 0
        if args.command == "validate":
            errors = validate_existing_run(store)
            if errors:
                print("Validation failed: " + "; ".join(errors), file=sys.stderr)
                return 1
            print(f"Validation passed: {Path(args.run).resolve() / 'dataset'}")
            return 0
        result = report_existing_run(store)
        _print_summary(result)
        return 0
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except (RunError, WorkflowError, ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Unexpected processing error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
