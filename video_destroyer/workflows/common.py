import json
import os
import shutil
from pathlib import Path

from ..curation import curate_sequences
from ..extraction import extract_pairs
from ..manifests import atomic_json_write, atomic_jsonl_write, read_jsonl
from ..models import PairRecord, SequenceRecord
from ..validation import validate_dataset, validate_pairs


class WorkflowError(RuntimeError):
    pass


def resolve_pair_paths(store, record):
    if record.ownership == "owned":
        return store.root / record.hr_path, store.root / record.lr_path
    inputs = store.run["inputs"]
    return Path(inputs["hr_root"]) / record.hr_path, Path(inputs["lr_root"]) / record.lr_path


def _records(store):
    return [PairRecord.from_dict(value) for value in read_jsonl(store.root / "pairs.jsonl")]


def _sequences(store):
    return [SequenceRecord.from_dict(value) for value in read_jsonl(store.root / "sequences.jsonl")]


def _write_summary(store, pairs, sequences, validation_errors=None, validation=None):
    accepted_pairs = sum(pair.status == "validated" for pair in pairs)
    rejected_pairs = sum(pair.status == "rejected" for pair in pairs)
    accepted_sequences = sum(sequence.status == "accepted" for sequence in sequences)
    rejected_sequences = sum(sequence.status == "rejected" for sequence in sequences)
    if validation is None:
        validation = "passed" if not validation_errors else "failed"
    dataset = store.root / "dataset"
    lines = [
        f"Dataset ready: {dataset}" if validation.startswith("passed") else f"Dataset not ready ({validation})",
        f"Clip pairs discovered: {len(pairs)}",
        f"Clip pairs accepted: {accepted_pairs}",
        f"Clip pairs rejected: {rejected_pairs}",
        f"Frame sequences accepted: {accepted_sequences}",
        f"Frame sequences rejected: {rejected_sequences}",
        f"Validation: {validation}",
        f"Report: {store.root / 'reports' / 'summary.txt'}",
    ]
    if validation_errors:
        lines.extend(f"Validation error: {error}" for error in validation_errors)
    (store.root / "reports" / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "pairs": len(pairs), "pairs_accepted": accepted_pairs, "pairs_rejected": rejected_pairs,
        "sequences_accepted": accepted_sequences, "sequences_rejected": rejected_sequences,
        "dataset": str(dataset), "report": str(store.root / "reports" / "summary.txt"),
        "strict_failure": bool(rejected_pairs or rejected_sequences),
        "validation": validation,
    }


def run_dataset_workflow(store, discover=None):
    """Run or resume the shared validation, extraction, curation, and finalization stages."""
    config = store.run["config"]
    stage = None
    pairs = _records(store)
    sequences = _sequences(store)
    try:
        if not store.completed("discovery"):
            stage = "discovery"
            store.begin(stage)
            if discover is None:
                raise WorkflowError("Cannot resume before discovery without a discoverer")
            pairs = discover()
            if not pairs:
                raise WorkflowError("No matching clip pairs were discovered")
            atomic_jsonl_write(store.root / "pairs.jsonl", pairs)
            store.finish(stage, {"pairs_discovered": len(pairs)})

        if not store.completed("pair_validation"):
            stage = "pair_validation"
            store.begin(stage)
            scale, rejected = validate_pairs(
                pairs, lambda pair: resolve_pair_paths(store, pair), config["validation"]["expected_scale"]
            )
            atomic_jsonl_write(store.root / "pairs.jsonl", pairs)
            atomic_json_write(store.root / "reports" / "pair-validation.json", {
                "scale": str(scale) if scale is not None else None,
                "accepted": sum(pair.status == "validated" for pair in pairs),
                "rejected": rejected,
                "pairs": [pair.to_dict() for pair in pairs],
            })
            store.finish(stage, {"pairs_accepted": sum(pair.status == "validated" for pair in pairs), "pairs_rejected": rejected})
            if not any(pair.status == "validated" for pair in pairs):
                raise WorkflowError("No valid clip pairs remain after preflight validation")

        if not store.completed("extraction"):
            stage = "extraction"
            store.begin(stage)
            # A prior interrupted extraction may have left only this stage's files.
            shutil.rmtree(store.root / ".work" / "frames", ignore_errors=True)
            shutil.rmtree(store.root / ".work" / "decoded", ignore_errors=True)
            sequences = extract_pairs(pairs, lambda pair: resolve_pair_paths(store, pair), store.root / ".work", config["extract"])
            atomic_jsonl_write(store.root / "pairs.jsonl", pairs)
            atomic_jsonl_write(store.root / "sequences.jsonl", sequences)
            store.finish(stage, {"sequences_extracted": len(sequences)})
            if not sequences:
                raise WorkflowError("No frame sequences were extracted")

        if not store.completed("curation"):
            stage = "curation"
            store.begin(stage)
            sequences = curate_sequences(
                sequences, store.root / ".work", config["curate"], config["validation"]["retain_rejected"], store.root / "rejected"
            )
            atomic_jsonl_write(store.root / "sequences.jsonl", sequences)
            store.finish(stage, {"sequences_accepted": sum(item.status == "accepted" for item in sequences), "sequences_rejected": sum(item.status == "rejected" for item in sequences)})
            if not any(item.status == "accepted" for item in sequences):
                raise WorkflowError("No valid frame sequences remain after curation")

        candidate = store.root / ".work" / "accepted"
        if not store.completed("dataset_validation"):
            stage = "dataset_validation"
            store.begin(stage)
            errors = validate_dataset(candidate, sequences, config["extract"]["sequence_length"], config["validation"]["expected_scale"])
            atomic_json_write(store.root / "reports" / "dataset-validation.json", {"passed": not errors, "errors": errors})
            if errors:
                raise WorkflowError("Final dataset validation failed: " + "; ".join(errors))
            store.finish(stage)

        if not store.completed("finalization"):
            stage = "finalization"
            store.begin(stage)
            dataset = store.root / "dataset"
            if dataset.exists():
                # An interruption can occur after the atomic rename and before state is
                # persisted. Treat that result as finalized only if it still validates.
                if candidate.exists():
                    raise WorkflowError("Dataset directory already exists; refusing to overwrite it")
                errors = validate_dataset(dataset, sequences, config["extract"]["sequence_length"], config["validation"]["expected_scale"])
                if errors:
                    raise WorkflowError("Existing finalized dataset failed validation: " + "; ".join(errors))
            else:
                os.replace(candidate, dataset)
            store.finish(stage)

        result = _write_summary(store, pairs, sequences)
        store.complete(result["pairs_rejected"] or result["sequences_rejected"])
        result["strict_failure"] = result["strict_failure"] and config["runtime"]["fail_on_rejection"]
        return result
    except Exception as error:
        if stage:
            store.fail(stage, error)
        _write_summary(store, pairs, sequences, [str(error)])
        raise


def validate_existing_run(store):
    sequences = _sequences(store)
    config = store.run["config"]
    errors = validate_dataset(store.root / "dataset", sequences, config["extract"]["sequence_length"], config["validation"]["expected_scale"])
    atomic_json_write(store.root / "reports" / "dataset-validation.json", {"passed": not errors, "errors": errors})
    _write_summary(store, _records(store), sequences, errors)
    return errors


def report_existing_run(store):
    """Rewrite the summary from stored manifests without revalidating anything."""
    recorded = store.root / "reports" / "dataset-validation.json"
    errors, validation = None, "not checked"
    if recorded.is_file():
        try:
            result = json.loads(recorded.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            result = None
        if isinstance(result, dict):
            errors = list(result.get("errors") or ())
            validation = "passed (recorded)" if result.get("passed") else "failed (recorded)"
    return _write_summary(store, _records(store), _sequences(store), errors, validation)
