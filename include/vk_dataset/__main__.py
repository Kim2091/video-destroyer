"""
    python -m vk_dataset run <job.json> [--work-dir DIR]
    python -m vk_dataset check <job.json>
    python -m vk_dataset nodes
    python -m vk_dataset stages
    python -m vk_dataset rename <dataset>
    python -m vk_dataset preview

run prints the JSON-lines protocol (see __init__.py); check loads a job and
prints {"result": "ok"} or {"error": ...} without running it; nodes prints
every node type's declaration for the UI, in stage order; stages prints the
stages of a job (docs.py). Every command first applies the inference settings
the app passes in VK_DATASET_RUNTIME (runtime_settings.py). rename renumbers the sequences of a dataset folder
after some were removed (see rename.py). preview serves frames of any node's
output on stdin and stdout until stdin closes (see preview.py).
"""

import argparse
import json
import sys
import traceback

from . import Graph, node_specs, run
from .registry import stages
from .rename import rename_sequences
from .runner import emit_json
from .runtime_settings import apply_runtime_settings


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vk_dataset")
    commands = parser.add_subparsers(dest="command", required=True)
    run_cmd = commands.add_parser("run")
    run_cmd.add_argument("job")
    run_cmd.add_argument("--work-dir")
    check_cmd = commands.add_parser("check")
    check_cmd.add_argument("job")
    commands.add_parser("nodes")
    commands.add_parser("stages")
    rename_cmd = commands.add_parser("rename")
    rename_cmd.add_argument("dataset")
    commands.add_parser("preview")
    args = parser.parse_args(argv)
    apply_runtime_settings()
    if args.command == "preview":
        from .preview import serve
        return serve()

    try:
        if args.command == "nodes":
            emit_json({"result": node_specs()})
        elif args.command == "stages":
            emit_json({"result": stages()})
        elif args.command == "rename":
            emit_json({"result": rename_sequences(args.dataset, emit=emit_json)})
        elif args.command == "check":
            Graph.load(args.job)
            emit_json({"result": "ok"})
        else:
            emit_json({"result": run(args.job, work_dir=args.work_dir)})
    except Exception as e:
        sys.stderr.write(traceback.format_exc())
        emit_json({"error": str(e)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
