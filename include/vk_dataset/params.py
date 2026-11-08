"""
Random settings, drawn the same way every time from the job's seed.

A param declared `random` takes either a value or one sampler:

    {"uniform": [low, high]}            a float in [low, high]
    {"int": [low, high]}                an int in [low, high], both included
    {"choice": [a, b, ...]}             one of the values
    {"choice": [a, b], "weights": [3, 1]}

Every node gets its own generator, seeded from the job seed, the video, the
chunk and the node's id. So a chunk's draws do not depend on how many chunks
came before it or on which other nodes are enabled, and rerunning a job with
the same seed reproduces each chunk exactly.
"""

import hashlib
import random

SAMPLER_KEYS = ("uniform", "int", "choice")


def is_sampler(value):
    return isinstance(value, dict) and any(k in value for k in SAMPLER_KEYS)


def rng_for(seed, *parts):
    """A random.Random seeded from the job seed and the given key parts."""
    key = "/".join(str(p) for p in (seed, *parts)).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))


def sample(value, rng):
    """Draws one value from a sampler; any other value is returned as is."""
    if not is_sampler(value):
        return value
    if "uniform" in value:
        low, high = value["uniform"]
        return rng.uniform(float(low), float(high))
    if "int" in value:
        low, high = value["int"]
        return rng.randint(int(low), int(high))
    options = list(value["choice"])
    if not options:
        raise ValueError("a choice sampler needs at least one option")
    weights = value.get("weights")
    if weights is not None:
        if len(weights) != len(options):
            raise ValueError("a choice sampler needs one weight per option")
        return rng.choices(options, weights=[float(w) for w in weights])[0]
    return rng.choice(options)


def weighted_pick(table, rng, weight_key="probability"):
    """
    Picks one key of a table whose entries carry a weight, such as the codec
    table {"h264": {"probability": 0.6, ...}, ...}. Entries with weight 0 are
    never picked; returns None when every weight is 0.
    """
    names = [n for n, entry in table.items() if float(entry.get(weight_key, 0)) > 0]
    if not names:
        return None
    return rng.choices(names, weights=[float(table[n][weight_key]) for n in names])[0]
