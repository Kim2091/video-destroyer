"""
Settings driven by wires: a value node's output linked into a setting of
another node, so several nodes can share one random draw or derive a
setting from another.

A link is an ordinary edge whose target port is "param:" and the setting's
name:

    {"from": "pick.value", "to": "comp.param:profile"}

Only value outputs link into settings, and only settings of the kinds in
LINKABLE_KINDS take a link. A linked value replaces the setting's own value
or sampler for the chunk, and the runner records it as it records a draw.

What a value node can give is worked out when the graph loads where it is
knowable (a range, or a list of options), so a pick that a dropdown does not
offer, or a range past a setting's limits, is refused before anything runs.
Everything else is checked when the value arrives, by coerce().
"""

import math

from .params import is_sampler

PARAM_PREFIX = "param:"
LINKABLE_KINDS = ("float", "int", "bool", "str", "choice", "value")
# Ranges of whole numbers longer than this are not checked option by option.
CHECKED_RANGE = 1000


class LinkError(ValueError):
    pass


def param_port(port):
    """The setting a "param:<name>" port names, or None for an ordinary port."""
    return port[len(PARAM_PREFIX):] if isinstance(port, str) and port.startswith(PARAM_PREFIX) else None


def number(value):
    """A value as a float, or None when it is not a number (a bool is not one)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def same(a, b):
    """Whether two values are equal, comparing numbers as numbers and anything else by its text."""
    x, y = number(a), number(b)
    if x is not None and y is not None:
        return abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))
    return _text(a) == _text(b)


def _text(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


def _choices_text(param):
    return ", ".join(str(c) for c in param.choices)


def _limits(param, value, where):
    if param.min is not None and value < param.min:
        raise LinkError(f"{where} got {value:g}, below its least, {param.min:g}")
    if param.max is not None and value > param.max:
        raise LinkError(f"{where} got {value:g}, past its most, {param.max:g}")


def coerce(param, value, where):
    """
    A linked value as the setting takes it: a number for a number (rounded
    for a whole-number setting), one of a dropdown's own options, a bool, or
    text. `where` names the node and setting in the error.
    """
    if param.kind in ("float", "int"):
        n = number(value)
        if n is None:
            raise LinkError(f"{where} takes a number, but got {value!r}")
        result = float(n) if param.kind == "float" else int(round(n))
        _limits(param, result, where)
        return result
    if param.kind == "choice":
        for choice in param.choices:
            if same(choice, value):
                return choice
        raise LinkError(f"{where} got {value!r}, which is not one of its options: {_choices_text(param)}")
    if param.kind == "bool":
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("true", "on", "yes", "1"):
            return True
        if text in ("false", "off", "no", "0"):
            return False
        raise LinkError(f"{where} takes on or off, but got {value!r}")
    if param.kind == "str":
        return value if isinstance(value, str) else _text(value)
    return value


# -- what a value node can give ------------------------------------------------
#
# A value node type may declare `describe(get)`, which returns what its value
# can be: ("range", low, high, whole) for any number in [low, high] (whole
# numbers only when `whole`), ("set", (a, b, ...)) for one of those values, or
# None when it cannot tell. get(name) is the same for one of its own settings:
# the source's description when the setting is linked, else ("set", (value,)).


def describe(graph, node_id, seen=()):
    """What the value node `node_id` can give, or None when it cannot be told before running."""
    node = graph.nodes[node_id]
    if not node.enabled or node.type.describe is None or node_id in seen:
        return None

    def get(name):
        if name in node.links:
            return describe(graph, node.links[name][0], (*seen, node_id))
        param = node.type.param(name)
        return _described(node.params.get(name, param.default if param else None))

    return node.type.describe(get)


def _described(value):
    """A setting's own value as a description: one value, or what its sampler draws from."""
    if not is_sampler(value):
        return ("set", (value,))
    if "choice" in value:
        weights = value.get("weights")
        options = value.get("choice") or []
        if isinstance(weights, list) and len(weights) == len(options):
            return ("set", tuple(o for o, w in zip(options, weights) if number(w)))
        return ("set", tuple(options))
    low, high = value.get("uniform") or value.get("int") or (None, None)
    if number(low) is None or number(high) is None:
        return None
    return ("range", number(low), number(high), "int" in value)


def check(param, info, where, source):
    """Raises LinkError when a value described by `info` can never suit the setting."""
    if info is None:
        return
    if info[0] == "set":
        values = info[1]
        if not values:
            raise LinkError(f"{source} has no options to give {where}")
        for value in values:
            coerce(param, value, f"{where}, from {source},")
        return
    _, low, high, whole = info
    if param.kind in ("float", "int"):
        _limits(param, low, f"{where}, from {source},")
        _limits(param, high, f"{where}, from {source},")
        return
    if param.kind == "choice":
        if not whole:
            raise LinkError(f"{where} takes one of {_choices_text(param)}, but {source} gives any number "
                            f"from {low:g} to {high:g}")
        if high - low <= CHECKED_RANGE:
            for n in range(math.ceil(low), math.floor(high) + 1):
                coerce(param, n, f"{where}, from {source},")
        return
    if param.kind == "bool":
        raise LinkError(f"{where} takes on or off, but {source} gives a number")
