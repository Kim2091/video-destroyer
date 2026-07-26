from pathlib import Path

from ..pairing import discover_import_pairs
from ..run_store import RunStore
from .common import run_dataset_workflow


def start(hr_root, lr_root, output, config, materialize=None):
    inputs = {"hr_root": str(Path(hr_root).resolve()), "lr_root": str(Path(lr_root).resolve()), "materialize": materialize}
    store = RunStore.create(output, "import-pairs", config, inputs)
    return run_dataset_workflow(store, lambda: discover_import_pairs(hr_root, lr_root, materialize, store.root))
