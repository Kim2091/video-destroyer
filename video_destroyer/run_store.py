"""Persistent run state, immutable resolved configuration, and atomic updates."""

from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import __version__
from .manifests import atomic_json_write


class RunError(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.run_file = self.root / "run.yaml"
        self.state_file = self.root / "state.json"

    @classmethod
    def create(cls, root, workflow, config, inputs):
        store = cls(root)
        if store.root.exists() and any(store.root.iterdir()):
            raise RunError(f"Output directory already exists and is not empty: {store.root}. Use resume instead.")
        store.root.mkdir(parents=True, exist_ok=True)
        for relative in ("reports", "logs", ".work/clips/hr", ".work/clips/lr", ".work/frames/hr", ".work/frames/lr", ".work/accepted/hr", ".work/accepted/lr"):
            (store.root / relative).mkdir(parents=True, exist_ok=True)
        (store.root / "logs" / "run.log").touch()
        resolved = {"workflow": workflow, "inputs": inputs, "config": config}
        store.run_file.write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
        store.run = resolved
        store.state = {
            "workflow": workflow,
            "status": "interrupted",
            "started_at": _now(),
            "ended_at": None,
            "tool_version": __version__,
            "stages": {},
            "counters": {},
        }
        store._save_state()
        return store

    @classmethod
    def open(cls, root):
        store = cls(root)
        if not store.run_file.is_file() or not store.state_file.is_file():
            raise RunError(f"Not a Video Destroyer run: {store.root}")
        try:
            store.run = yaml.safe_load(store.run_file.read_text(encoding="utf-8"))
            import json
            store.state = json.loads(store.state_file.read_text(encoding="utf-8"))
        except Exception as error:
            raise RunError(f"Unable to load run metadata: {error}") from error
        return store

    def _save_state(self):
        atomic_json_write(self.state_file, self.state)

    def begin(self, stage):
        self.state["status"] = "interrupted"
        self.state["ended_at"] = None
        self.state["stages"][stage] = {"status": "running", "started_at": _now(), "ended_at": None}
        self._save_state()

    def finish(self, stage, counters=None):
        entry = self.state["stages"].setdefault(stage, {})
        entry.update({"status": "completed", "ended_at": _now()})
        if counters:
            self.state["counters"].update(counters)
        self._save_state()

    def fail(self, stage, error):
        entry = self.state["stages"].setdefault(stage, {})
        entry.update({"status": "failed", "ended_at": _now(), "error": str(error)})
        self.state["status"] = "failed"
        self.state["ended_at"] = _now()
        self._save_state()

    def complete(self, rejected):
        self.state["status"] = "completed_with_rejections" if rejected else "completed"
        self.state["ended_at"] = _now()
        self._save_state()

    def completed(self, stage):
        return self.state.get("stages", {}).get(stage, {}).get("status") == "completed"
