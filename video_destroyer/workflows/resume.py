from ..run_store import RunStore
from ..pairing import discover_import_pairs
from .common import run_dataset_workflow
from .create import discover as discover_created


def start(run):
    store = RunStore.open(run)
    # Inputs and the resolved config are intentionally loaded from run.yaml only.
    if store.run["workflow"] == "import-pairs":
        inputs = store.run["inputs"]
        discover = lambda: discover_import_pairs(inputs["hr_root"], inputs["lr_root"], inputs.get("materialize"), store.root)
    elif store.run["workflow"] == "create":
        discover = lambda: discover_created(store)
    else:
        raise ValueError(f"Unknown workflow in run metadata: {store.run['workflow']}")
    return run_dataset_workflow(store, discover)
